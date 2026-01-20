package scorer

import (
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	k8stypes "k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/requestcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"

	"github.com/llm-d/llm-d-inference-scheduler/test/utils"
)

func TestInputSizeScorer_Score(t *testing.T) {
	podA := &types.PodMetrics{
		Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod-a", Namespace: "default"}},
		MetricsState: &backendmetrics.MetricsState{
			WaitingQueueSize: 2,
		},
	}
	podB := &types.PodMetrics{
		Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod-b", Namespace: "default"}},
		MetricsState: &backendmetrics.MetricsState{
			WaitingQueueSize: 0,
		},
	}
	podC := &types.PodMetrics{
		Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod-c", Namespace: "default"}},
		MetricsState: &backendmetrics.MetricsState{
			WaitingQueueSize: 15,
		},
	}

	tests := []struct {
		name       string
		setupCache func(*InputSize)
		input      []types.Pod
		wantScores map[types.Pod]float64
	}{
		{
			name: "no pods in cache",
			setupCache: func(_ *InputSize) {
				// Cache is empty
			},
			input: []types.Pod{podA, podB, podC},
			wantScores: map[types.Pod]float64{
				podA: 1,
				podB: 1,
				podC: 1,
			},
		},
		{
			name: "all pods in cache with different input sizes",
			setupCache: func(s *InputSize) {
				s.mutex.Lock()
				s.podInputSizes["default/pod-a"] = 300  // middle
				s.podInputSizes["default/pod-b"] = 100  // min tracked
				s.podInputSizes["default/pod-c"] = 1100 // max
				s.mutex.Unlock()
			},
			input: []types.Pod{podA, podB, podC},
			// Score_p = (Max - InputSize_p) / Max (Min implicitly 0)
			// pod-a: (1100 - 300) / 1100 = 800/1100 ≈ 0.7272...
			// pod-b: (1100 - 100) / 1100 = 1000/1100 ≈ 0.9090...
			// pod-c: (1100 - 1100) / 1100 = 0/1100 = 0.0
			wantScores: map[types.Pod]float64{
				podA: 800.0 / 1100.0,
				podB: 1000.0 / 1100.0,
				podC: 0.0,
			},
		},
		{
			name: "some pods in cache",
			setupCache: func(s *InputSize) {
				s.mutex.Lock()
				s.podInputSizes["default/pod-a"] = 400
				s.podInputSizes["default/pod-c"] = 100
				// pod-b not in cache (size 0)
				s.mutex.Unlock()
			},
			input: []types.Pod{podA, podB, podC},
			// Score_p = (Max - InputSize_p) / Max
			// pod-a: (400 - 400) / 400 = 0/400 = 0.0
			// pod-b: size 0 (not in cache), gets 1.0
			// pod-c: (400 - 100) / 400 = 300/400 = 0.75
			wantScores: map[types.Pod]float64{
				podA: 0.0,
				podB: 1.0,
				podC: 0.75,
			},
		},
		{
			name: "all pods have same input size",
			setupCache: func(s *InputSize) {
				s.mutex.Lock()
				s.podInputSizes["default/pod-a"] = 500
				s.podInputSizes["default/pod-b"] = 500
				s.podInputSizes["default/pod-c"] = 500
				s.mutex.Unlock()
			},
			input: []types.Pod{podA, podB, podC},
			// All pods have same input size = max, so all get score 0.0
			// This is consistent with ActiveRequest scorer behavior
			wantScores: map[types.Pod]float64{
				podA: 0.0,
				podB: 0.0,
				podC: 0.0,
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ctx := utils.NewTestContext(t)

			scorer := NewInputSize(ctx, nil)
			test.setupCache(scorer)

			got := scorer.Score(ctx, nil, nil, test.input)

			if diff := cmp.Diff(test.wantScores, got); diff != "" {
				t.Errorf("Unexpected output (-want +got): %v", diff)
			}
		})
	}
}

func TestInputSizeScorer_PreRequest(t *testing.T) {
	ctx := utils.NewTestContext(t)
	scorer := NewInputSize(ctx, nil)

	podA := &types.PodMetrics{
		Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod-a", Namespace: "default"}},
		MetricsState: &backendmetrics.MetricsState{
			WaitingQueueSize: 2,
		},
	}

	request := &types.LLMRequest{
		RequestId: "test-request-1",
		Body: &types.LLMRequestBody{
			Completions: &types.CompletionsRequest{
				Prompt: "Hello, world!", // 13 characters
			},
		},
	}

	schedulingResult := &types.SchedulingResult{
		ProfileResults: map[string]*types.ProfileRunResult{
			"test-profile": {
				TargetPods: []types.Pod{podA},
			},
		},
	}

	// First request
	scorer.PreRequest(ctx, request, schedulingResult)

	// Check cache and pod input sizes
	compositeKey := "default/pod-a.test-request-1"
	if !scorer.requestCache.Has(compositeKey) {
		t.Errorf("Expected request to be in cache with key %s", compositeKey)
	}

	scorer.mutex.RLock()
	size := scorer.podInputSizes["default/pod-a"]
	scorer.mutex.RUnlock()
	if size != 13 {
		t.Errorf("Expected pod-a input size to be 13, got %d", size)
	}

	// Second request with different ID to same pod
	request2 := &types.LLMRequest{
		RequestId: "test-request-2",
		Body: &types.LLMRequestBody{
			Completions: &types.CompletionsRequest{
				Prompt: "Hello!", // 6 characters
			},
		},
	}
	schedulingResult2 := &types.SchedulingResult{
		ProfileResults: map[string]*types.ProfileRunResult{
			"test-profile": {
				TargetPods: []types.Pod{podA},
			},
		},
	}

	scorer.PreRequest(ctx, request2, schedulingResult2)

	// Check incremented size
	scorer.mutex.RLock()
	size = scorer.podInputSizes["default/pod-a"]
	scorer.mutex.RUnlock()
	if size != 19 { // 13 + 6
		t.Errorf("Expected pod-a input size to be 19, got %d", size)
	}

	// Check both requests are in cache
	compositeKey2 := "default/pod-a.test-request-2"
	if !scorer.requestCache.Has(compositeKey2) {
		t.Errorf("Expected second request to be in cache with key %s", compositeKey2)
	}
}

func TestInputSizeScorer_PreRequestChatCompletions(t *testing.T) {
	ctx := utils.NewTestContext(t)
	scorer := NewInputSize(ctx, nil)

	podA := &types.PodMetrics{
		Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod-a", Namespace: "default"}},
	}

	request := &types.LLMRequest{
		RequestId: "test-request-chat",
		Body: &types.LLMRequestBody{
			ChatCompletions: &types.ChatCompletionsRequest{
				Messages: []types.Message{
					{Role: "user", Content: types.Content{Raw: "Hello"}},         // 5 characters
					{Role: "assistant", Content: types.Content{Raw: "Hi there"}}, // 8 characters
					{Role: "user", Content: types.Content{Raw: "How are you?"}},  // 12 characters
				},
			},
		},
	}

	schedulingResult := &types.SchedulingResult{
		ProfileResults: map[string]*types.ProfileRunResult{
			"test-profile": {
				TargetPods: []types.Pod{podA},
			},
		},
	}

	scorer.PreRequest(ctx, request, schedulingResult)

	scorer.mutex.RLock()
	size := scorer.podInputSizes["default/pod-a"]
	scorer.mutex.RUnlock()

	var expectedSize uint64 = 5 + 8 + 12 // 25
	if size != expectedSize {
		t.Errorf("Expected pod-a input size to be %d, got %d", expectedSize, size)
	}
}

func TestInputSizeScorer_ResponseComplete(t *testing.T) {
	ctx := utils.NewTestContext(t)

	scorer := NewInputSize(ctx, nil)

	request := &types.LLMRequest{
		RequestId: "test-request-1",
		Body: &types.LLMRequestBody{
			Completions: &types.CompletionsRequest{
				Prompt: "Hello, world!",
			},
		},
	}

	podA := &types.PodMetrics{
		Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod-a", Namespace: "default"}},
		MetricsState: &backendmetrics.MetricsState{
			WaitingQueueSize: 2,
		},
	}
	// Setup initial state: add request through PreRequest
	schedulingResult := &types.SchedulingResult{
		ProfileResults: map[string]*types.ProfileRunResult{
			"test-profile": {
				TargetPods: []types.Pod{podA},
			},
		},
	}

	scorer.PreRequest(ctx, request, schedulingResult)

	// Verify initial state
	compositeKey := "default/pod-a.test-request-1"
	if !scorer.requestCache.Has(compositeKey) {
		t.Fatal("Request should be in cache before ResponseComplete")
	}

	scorer.mutex.RLock()
	initialSize := scorer.podInputSizes["default/pod-a"]
	scorer.mutex.RUnlock()
	if initialSize != 13 {
		t.Fatalf("Expected initial size to be 13, got %d", initialSize)
	}

	// Call ResponseComplete
	scorer.ResponseComplete(ctx, request, &requestcontrol.Response{}, podA.GetPod())

	// Check request is removed from cache
	if scorer.requestCache.Has(compositeKey) {
		t.Errorf("Request should be removed from cache after ResponseComplete")
	}

	// Check pod input size is decremented and removed (since it was the only request)
	scorer.mutex.RLock()
	_, exists := scorer.podInputSizes["default/pod-a"]
	scorer.mutex.RUnlock()
	if exists {
		t.Errorf("Pod should be removed from podInputSizes when size reaches 0")
	}
}

func TestInputSizeScorer_TTLExpiration(t *testing.T) {
	ctx := utils.NewTestContext(t)

	// Use very short timeout for test
	params := &InputSizeParameters{RequestTimeout: "1s"}
	scorer := NewInputSize(ctx, params) // 1 second timeout

	request := &types.LLMRequest{
		RequestId: "test-request-ttl",
		Body: &types.LLMRequestBody{
			Completions: &types.CompletionsRequest{
				Prompt: "Test prompt",
			},
		},
	}

	podA := &types.PodMetrics{
		Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod-a", Namespace: "default"}},
	}

	schedulingResult := &types.SchedulingResult{
		ProfileResults: map[string]*types.ProfileRunResult{
			"test-profile": {
				TargetPods: []types.Pod{podA},
			},
		},
	}

	// Add request
	scorer.PreRequest(ctx, request, schedulingResult)

	// Verify request is added
	scorer.mutex.RLock()
	initialSize := scorer.podInputSizes["default/pod-a"]
	scorer.mutex.RUnlock()
	if initialSize != 11 { // "Test prompt" = 11 characters
		t.Fatalf("Expected initial size to be 11, got %d", initialSize)
	}

	// Wait for TTL expiration
	time.Sleep(2 * time.Second)

	// Trigger cleanup
	scorer.requestCache.DeleteExpired()

	// Check that pod input size is decremented due to TTL expiration
	scorer.mutex.RLock()
	_, exists := scorer.podInputSizes["default/pod-a"]
	scorer.mutex.RUnlock()
	if exists {
		t.Errorf("Pod should be removed from podInputSizes after TTL expiration")
	}
}

func TestNewInputSizeScorer_InvalidTimeout(t *testing.T) {
	ctx := utils.NewTestContext(t)

	params := &InputSizeParameters{RequestTimeout: "invalid"}
	scorer := NewInputSize(ctx, params)

	// Should use default timeout when invalid value is provided
	if scorer == nil {
		t.Error("Expected scorer to be created even with invalid timeout")
	}
}

func TestInputSizeScorer_TypedName(t *testing.T) {
	ctx := utils.NewTestContext(t)

	scorer := NewInputSize(ctx, nil)

	typedName := scorer.TypedName()
	if typedName.Type != InputSizeType {
		t.Errorf("Expected type %s, got %s", InputSizeType, typedName.Type)
	}
}

func TestInputSizeScorer_WithName(t *testing.T) {
	ctx := utils.NewTestContext(t)

	scorer := NewInputSize(ctx, nil)
	testName := "test-scorer"

	scorer = scorer.WithName(testName)

	if scorer.TypedName().Name != testName {
		t.Errorf("Expected name %s, got %s", testName, scorer.TypedName().Name)
	}
}

func TestCalculateInputSize(t *testing.T) {
	tests := []struct {
		name     string
		request  *types.LLMRequest
		expected uint64
	}{
		{
			name:     "nil request",
			request:  nil,
			expected: 0,
		},
		{
			name: "nil body",
			request: &types.LLMRequest{
				RequestId: "test",
				Body:      nil,
			},
			expected: 0,
		},
		{
			name: "completions request",
			request: &types.LLMRequest{
				RequestId: "test",
				Body: &types.LLMRequestBody{
					Completions: &types.CompletionsRequest{
						Prompt: "Hello, world!",
					},
				},
			},
			expected: 13,
		},
		{
			name: "chat completions request",
			request: &types.LLMRequest{
				RequestId: "test",
				Body: &types.LLMRequestBody{
					ChatCompletions: &types.ChatCompletionsRequest{
						Messages: []types.Message{
							{Role: "user", Content: types.Content{Raw: "Hello"}},
							{Role: "assistant", Content: types.Content{Raw: "Hi"}},
						},
					},
				},
			},
			expected: 7, // 5 + 2
		},
		{
			name: "empty completions",
			request: &types.LLMRequest{
				RequestId: "test",
				Body: &types.LLMRequestBody{
					Completions: &types.CompletionsRequest{
						Prompt: "",
					},
				},
			},
			expected: 0,
		},
		{
			name: "empty chat completions",
			request: &types.LLMRequest{
				RequestId: "test",
				Body: &types.LLMRequestBody{
					ChatCompletions: &types.ChatCompletionsRequest{
						Messages: []types.Message{},
					},
				},
			},
			expected: 0,
		},
		{
			name: "chat completions with tools",
			request: &types.LLMRequest{
				RequestId: "test",
				Body: &types.LLMRequestBody{
					ChatCompletions: &types.ChatCompletionsRequest{
						Messages: []types.Message{
							{Role: "user", Content: types.Content{Raw: "Hello"}}, // 5 chars
						},
						Tools: []interface{}{
							map[string]interface{}{
								"type": "function",
								"function": map[string]interface{}{
									"name": "get_weather",
								},
							},
						},
					},
				},
			},
			// 5 (message) + len of JSON-serialized tools
			// {"type":"function","function":{"name":"get_weather"}} = 53 chars in JSON array = [...]
			expected: 5 + 55, // approximate, the exact size depends on JSON encoding
		},
		{
			name: "chat completions with chat template",
			request: &types.LLMRequest{
				RequestId: "test",
				Body: &types.LLMRequestBody{
					ChatCompletions: &types.ChatCompletionsRequest{
						Messages: []types.Message{
							{Role: "user", Content: types.Content{Raw: "Hi"}}, // 2 chars
						},
						ChatTemplate: "custom_template", // 15 chars
					},
				},
			},
			expected: 2 + 15, // 17
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := calculateInputSize(test.request)
			if got != test.expected {
				t.Errorf("calculateInputSize() = %d, want %d", got, test.expected)
			}
		})
	}
}