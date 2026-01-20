package scorer

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/jellydator/ttlcache/v3"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/requestcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

const (
	// InputSizeType is the type of the InputSize scorer.
	InputSizeType = "input-size-scorer"

	// defaultInputSizeRequestTimeout defines the default timeout for open requests to be
	// considered stale and removed from the cache.
	defaultInputSizeRequestTimeout = defaultRequestTimeout
)

// InputSizeParameters defines the parameters for the InputSize scorer.
type InputSizeParameters struct {
	// RequestTimeout defines the timeout for requests.
	// Once the request is "in-flight" for this duration, it is considered to
	// be timed out and dropped.
	// This field accepts duration strings like "30s", "1m", "2h".
	RequestTimeout string `json:"requestTimeout"`
}

// inputSizeEntry represents a single request in the cache with its input size
type inputSizeEntry struct {
	PodName   string
	RequestID string
	InputSize uint64
}

// String returns a string representation of the request entry.
func (r *inputSizeEntry) String() string {
	return fmt.Sprintf("%s.%s", r.PodName, r.RequestID)
}

// compile-time type assertion
var _ framework.Scorer = &InputSize{}
var _ requestcontrol.PreRequest = &InputSize{}
var _ requestcontrol.ResponseComplete = &InputSize{}

// InputSizeFactory defines the factory function for the InputSize scorer.
func InputSizeFactory(name string, rawParameters json.RawMessage, handle plugins.Handle) (plugins.Plugin, error) {
	parameters := InputSizeParameters{}
	if rawParameters != nil {
		if err := json.Unmarshal(rawParameters, &parameters); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' scorer - %w", InputSizeType, err)
		}
	}

	return NewInputSize(handle.Context(), &parameters).WithName(name), nil
}

// NewInputSize creates a new InputSize scorer.
func NewInputSize(ctx context.Context, params *InputSizeParameters) *InputSize {
	requestTimeout := defaultInputSizeRequestTimeout
	logger := log.FromContext(ctx)

	if params != nil && params.RequestTimeout != "" {
		paramsRequestTimeout, err := time.ParseDuration(params.RequestTimeout)
		if err != nil || paramsRequestTimeout <= 0 {
			logger.Error(err, "Invalid request timeout duration, using default request timeout")
		} else {
			requestTimeout = paramsRequestTimeout
			logger.Info("Using request timeout", "requestTimeout", requestTimeout)
		}
	}

	// cache for individual requests with their own TTL
	requestCache := ttlcache.New[string, *inputSizeEntry](
		ttlcache.WithTTL[string, *inputSizeEntry](requestTimeout),
		ttlcache.WithDisableTouchOnHit[string, *inputSizeEntry](),
	)

	scorer := &InputSize{
		typedName:     plugins.TypedName{Type: InputSizeType},
		requestCache:  requestCache,
		podInputSizes: make(map[string]uint64, 2),
		mutex:         &sync.RWMutex{},
	}

	// callback to decrement input size when requests expire
	// most requests will be removed in ResponseComplete, but this ensures
	// that we don't leak pod input sizes if ResponseComplete is not called
	requestCache.OnEviction(func(_ context.Context, reason ttlcache.EvictionReason,
		item *ttlcache.Item[string, *inputSizeEntry]) {
		if reason == ttlcache.EvictionReasonExpired {
			scorer.decrementPodInputSize(item.Value().PodName, item.Value().InputSize)
		}
	})

	go cleanCachePeriodically(ctx, requestCache, requestTimeout)

	return scorer
}

// InputSize keeps track of total input size for active requests per pod.
type InputSize struct {
	typedName plugins.TypedName

	// requestCache stores individual request entries with unique composite keys (podName.requestID)
	requestCache *ttlcache.Cache[string, *inputSizeEntry]

	// podInputSizes maintains fast lookup for total input size per pod
	podInputSizes map[string]uint64
	mutex         *sync.RWMutex
}

// TypedName returns the typed name of the plugin.
func (s *InputSize) TypedName() plugins.TypedName {
	return s.typedName
}

// WithName sets the name of the plugin.
func (s *InputSize) WithName(name string) *InputSize {
	s.typedName.Name = name
	return s
}

// Score scores the given pods based on the total input size of active requests
// being served by each pod. The score is normalized using the formula:
// Score_p = (Max - InputSize_p) / Max
// Pods with lower input sizes get higher scores. Pods not in the cache are
// treated as having input size 0 (highest score).
func (s *InputSize) Score(ctx context.Context, _ *types.CycleState, _ *types.LLMRequest,
	pods []types.Pod) map[types.Pod]float64 {
	s.mutex.RLock()
	podInputSizes := make(map[string]uint64, len(s.podInputSizes))
	var maxSize uint64
	for podName, size := range s.podInputSizes {
		podInputSizes[podName] = size
		if size > maxSize {
			maxSize = size
		}
	}
	s.mutex.RUnlock()

	scoredPodsMap := make(map[types.Pod]float64, len(pods))
	for _, pod := range pods {
		podName := pod.GetPod().NamespacedName.String()
		size := podInputSizes[podName] // defaults to 0 if not found
		if maxSize == 0 {
			// No tracked input sizes, all pods get equal score
			scoredPodsMap[pod] = 1.0
		} else if size == 0 {
			// Pod has no active requests, highest score
			scoredPodsMap[pod] = 1.0
		} else {
			// Score_p = (Max - InputSize_p) / Max
			scoredPodsMap[pod] = float64(maxSize-size) / float64(maxSize)
		}
	}

	log.FromContext(ctx).V(logutil.DEBUG).Info("Scored pods by input size", "scores", scoredPodsMap)
	return scoredPodsMap
}

// PreRequest is called before a request is sent to the target pod.
// It creates a new request entry in the cache with its input size and
// increments the pod's total input size.
func (s *InputSize) PreRequest(ctx context.Context, request *types.LLMRequest,
	schedulingResult *types.SchedulingResult) {
	debugLogger := log.FromContext(ctx).V(logutil.DEBUG)

	inputSize := calculateInputSize(request)

	for _, profileResult := range schedulingResult.ProfileResults {
		if profileResult == nil || profileResult.TargetPods == nil || len(profileResult.TargetPods) == 0 {
			continue
		}

		// create request entry for first pod only. TODO: support fallback pods
		entry := &inputSizeEntry{
			PodName:   profileResult.TargetPods[0].GetPod().NamespacedName.String(),
			RequestID: request.RequestId,
			InputSize: inputSize,
		}

		// add to request cache with TTL
		s.requestCache.Set(entry.String(), entry, 0) // Use default TTL
		s.incrementPodInputSize(entry.PodName, inputSize)

		debugLogger.Info("Added request to input size cache", "requestEntry", entry.String(), "inputSize", inputSize)
	}
}

// ResponseComplete is called after a response is sent to the client.
// It removes the specific request entry from the cache and decrements
// the pod's total input size.
func (s *InputSize) ResponseComplete(ctx context.Context, request *types.LLMRequest,
	_ *requestcontrol.Response, targetPod *backend.Pod) {
	debugLogger := log.FromContext(ctx).V(logutil.DEBUG).WithName("InputSize.ResponseComplete")
	if targetPod == nil {
		debugLogger.Info("Skipping ResponseComplete because targetPod is nil")
		return
	}

	entryKey := fmt.Sprintf("%s.%s", targetPod.NamespacedName.String(), request.RequestId)

	if item, found := s.requestCache.GetAndDelete(entryKey); found {
		s.decrementPodInputSize(item.Value().PodName, item.Value().InputSize)
		debugLogger.Info("Removed request from input size cache", "requestEntry", entryKey)
	} else {
		debugLogger.Info("Request not found in input size cache", "requestEntry", entryKey)
	}
}

// incrementPodInputSize increments the total input size for a pod.
func (s *InputSize) incrementPodInputSize(podName string, inputSize uint64) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	s.podInputSizes[podName] = safeAdd(s.podInputSizes[podName], inputSize)
}

// decrementPodInputSize decrements the total input size for a pod and removes
// the entry if size reaches zero.
func (s *InputSize) decrementPodInputSize(podName string, inputSize uint64) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	if currentSize, exists := s.podInputSizes[podName]; exists {
		if inputSize >= currentSize {
			delete(s.podInputSizes, podName)
		} else {
			s.podInputSizes[podName] = currentSize - inputSize
		}
	}
}

// calculateInputSize calculates the input size from an LLM request.
// For ChatCompletions: sum of message contents, tools, documents, and chat template
// For Completions: length of the prompt
func calculateInputSize(request *types.LLMRequest) uint64 {
	if request == nil || request.Body == nil {
		return 0
	}

	if request.Body.ChatCompletions != nil {
		var totalSize uint64

		for _, msg := range request.Body.ChatCompletions.Messages {
			totalSize = safeAdd(totalSize, uint64(len(msg.Content.PlainText())))
		}

		if len(request.Body.ChatCompletions.Tools) > 0 {
			if toolsJSON, err := json.Marshal(request.Body.ChatCompletions.Tools); err == nil {
				totalSize = safeAdd(totalSize, uint64(len(toolsJSON)))
			}
		}

		if len(request.Body.ChatCompletions.Documents) > 0 {
			if docsJSON, err := json.Marshal(request.Body.ChatCompletions.Documents); err == nil {
				totalSize = safeAdd(totalSize, uint64(len(docsJSON)))
			}
		}

		// Add chat template size
		totalSize = safeAdd(totalSize, uint64(len(request.Body.ChatCompletions.ChatTemplate)))

		return totalSize
	}

	if request.Body.Completions != nil {
		return uint64(len(request.Body.Completions.Prompt))
	}

	return 0
}

// safeAdd adds two uint64 values with overflow protection.
// If the addition overflowed, it returns math.MaxUint64 instead.
func safeAdd(a, b uint64) uint64 {
	if a > math.MaxUint64-b {
		return math.MaxUint64
	}
	return a + b
}

