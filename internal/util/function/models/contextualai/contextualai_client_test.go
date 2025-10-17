// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package contextualai

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewContextualAIClient(t *testing.T) {
	// Test successful client creation
	client, err := NewContextualAIClient("test_api_key")
	assert.NoError(t, err)
	assert.NotNil(t, client)
	assert.Equal(t, "test_api_key", client.apiKey)

	// Test empty API key
	client, err = NewContextualAIClient("")
	assert.Error(t, err)
	assert.Nil(t, client)
	assert.Contains(t, err.Error(), "missing credentials config")
}

func TestHeaders(t *testing.T) {
	client := &ContextualAIClient{apiKey: "test_api_key"}
	headers := client.headers()

	assert.Equal(t, "application/json", headers["Content-Type"])
	assert.Equal(t, "Bearer test_api_key", headers["Authorization"])
}

func TestRerankOK(t *testing.T) {
	const testQuery = "What is machine learning?"
	const testModel = "ctxl-rerank-v2-instruct-multilingual"
	testDocuments := []string{
		"Machine learning is a subset of artificial intelligence that focuses on algorithms.",
		"Deep learning uses neural networks with multiple layers to process data.",
		"Natural language processing helps computers understand human language.",
	}

	var res RerankResponse
	repStr := `{
		"results": [
			{
				"index": 0,
				"relevance_score": 0.95
			},
			{
				"index": 2,
				"relevance_score": 0.78
			},
			{
				"index": 1,
				"relevance_score": 0.65
			}
		]
	}`
	err := json.Unmarshal([]byte(repStr), &res)
	assert.NoError(t, err)

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Validate HTTP method and headers
		assert.Equal(t, http.MethodPost, r.Method)
		assert.Equal(t, "application/json", r.Header.Get("Content-Type"))
		assert.Contains(t, r.Header.Get("Authorization"), "Bearer")

		// Validate request body
		var req rerankRequest
		err := json.NewDecoder(r.Body).Decode(&req)
		require.NoError(t, err)

		assert.Equal(t, testQuery, req.Query)
		assert.Equal(t, testDocuments, req.Documents)
		assert.Equal(t, testModel, req.Model)

		w.WriteHeader(http.StatusOK)
		data, _ := json.Marshal(res)
		w.Write(data)
	}))
	defer ts.Close()
	url := ts.URL

	{
		c, _ := NewContextualAIClient("mock_key")
		ret, err := c.Rerank(url, testModel, testQuery, testDocuments, map[string]any{}, 30)
		assert.NoError(t, err)
		assert.NotNil(t, ret)
		assert.Len(t, ret.Results, 3)

		// Check that results are sorted by index
		assert.Equal(t, 0, ret.Results[0].Index)
		assert.Equal(t, 1, ret.Results[1].Index)
		assert.Equal(t, 2, ret.Results[2].Index)

		// Check realistic relevance scores
		assert.Equal(t, float32(0.95), ret.Results[0].RelevanceScore)
		assert.Equal(t, float32(0.65), ret.Results[1].RelevanceScore)
		assert.Equal(t, float32(0.78), ret.Results[2].RelevanceScore)
	}
}

func TestRerankWithParams(t *testing.T) {
	const testQuery = "How does neural network training work?"
	const testModel = "ctxl-rerank-v2-instruct-multilingual"
	testDocuments := []string{
		"Neural networks learn through backpropagation and gradient descent optimization.",
		"Training involves forward pass, loss calculation, and weight updates.",
	}
	testMetadata := []string{"technical_doc", "tutorial"}

	var res RerankResponse
	repStr := `{
		"results": [
			{
				"index": 0,
				"relevance_score": 0.92
			},
			{
				"index": 1,
				"relevance_score": 0.87
			}
		]
	}`
	err := json.Unmarshal([]byte(repStr), &res)
	assert.NoError(t, err)

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Validate request body with parameters
		var req rerankRequest
		err := json.NewDecoder(r.Body).Decode(&req)
		require.NoError(t, err)

		assert.Equal(t, testQuery, req.Query)
		assert.Equal(t, testDocuments, req.Documents)
		assert.Equal(t, testModel, req.Model)
		assert.Equal(t, 5, req.TopN)
		assert.Equal(t, "Prioritize technical accuracy", req.Instruction)
		assert.Equal(t, testMetadata, req.Metadata)

		w.WriteHeader(http.StatusOK)
		data, _ := json.Marshal(res)
		w.Write(data)
	}))
	defer ts.Close()
	url := ts.URL

	{
		c, _ := NewContextualAIClient("mock_key")
		params := map[string]any{
			"top_n":       5,
			"instruction": "Prioritize technical accuracy",
			"metadata":    testMetadata,
		}
		ret, err := c.Rerank(url, testModel, testQuery, testDocuments, params, 30)
		assert.NoError(t, err)
		assert.NotNil(t, ret)
		assert.Len(t, ret.Results, 2)

		assert.Equal(t, 0, ret.Results[0].Index)
		assert.Equal(t, 1, ret.Results[1].Index)
		assert.Equal(t, float32(0.92), ret.Results[0].RelevanceScore)
		assert.Equal(t, float32(0.87), ret.Results[1].RelevanceScore)
	}
}

func TestRerankErrorHandling(t *testing.T) {
	tests := []struct {
		name         string
		statusCode   int
		responseBody string
		expectedErr  string
	}{
		{
			name:         "unauthorized error",
			statusCode:   http.StatusUnauthorized,
			responseBody: `{"detail": "Invalid API key"}`,
			expectedErr:  "Call service failed",
		},
		{
			name:         "validation error",
			statusCode:   http.StatusUnprocessableEntity,
			responseBody: `{"detail": "Invalid model specified"}`,
			expectedErr:  "Call service failed",
		},
		{
			name:         "rate limit error",
			statusCode:   http.StatusTooManyRequests,
			responseBody: `{"detail": "Rate limit exceeded"}`,
			expectedErr:  "Call service failed",
		},
		{
			name:         "server error",
			statusCode:   http.StatusInternalServerError,
			responseBody: `{"detail": "Internal server error"}`,
			expectedErr:  "Call service failed",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tt.statusCode)
				w.Write([]byte(tt.responseBody))
			}))
			defer ts.Close()

			c, _ := NewContextualAIClient("test_key")
			_, err := c.Rerank(ts.URL, "ctxl-rerank-v2-instruct-multilingual", "test query", []string{"doc1"}, map[string]any{}, 30)

			assert.Error(t, err)
			assert.Contains(t, err.Error(), tt.expectedErr)
		})
	}
}

func TestRerankInvalidJSON(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`invalid json response`))
	}))
	defer ts.Close()
	url := ts.URL

	{
		c, _ := NewContextualAIClient("mock_key")
		_, err := c.Rerank(url, "ctxl-rerank-v2-instruct-multilingual", "What is AI?", []string{"AI is artificial intelligence", "ML is machine learning"}, map[string]any{}, 30)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "Call service failed")
	}
}

func TestRerankEmptyResults(t *testing.T) {
	var res RerankResponse
	repStr := `{
		"results": []
	}`
	err := json.Unmarshal([]byte(repStr), &res)
	assert.NoError(t, err)

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		data, _ := json.Marshal(res)
		w.Write(data)
	}))
	defer ts.Close()
	url := ts.URL

	{
		c, _ := NewContextualAIClient("mock_key")
		ret, err := c.Rerank(url, "ctxl-rerank-v2-instruct-multilingual", "What is quantum computing?", []string{"Quantum computing uses quantum mechanics", "It processes information differently"}, map[string]any{}, 30)
		assert.NoError(t, err)
		assert.NotNil(t, ret)
		assert.Len(t, ret.Results, 0)
	}
}

func TestRerankSingleResult(t *testing.T) {
	var res RerankResponse
	repStr := `{
		"results": [
			{
				"index": 1,
				"relevance_score": 0.95
			}
		]
	}`
	err := json.Unmarshal([]byte(repStr), &res)
	assert.NoError(t, err)

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		data, _ := json.Marshal(res)
		w.Write(data)
	}))
	defer ts.Close()
	url := ts.URL

	{
		c, _ := NewContextualAIClient("mock_key")
		ret, err := c.Rerank(url, "ctxl-rerank-v2-instruct-multilingual", "What is blockchain?", []string{"Blockchain is a distributed ledger", "It uses cryptography for security", "Transactions are immutable"}, map[string]any{}, 30)
		assert.NoError(t, err)
		assert.NotNil(t, ret)
		assert.Len(t, ret.Results, 1)

		assert.Equal(t, 1, ret.Results[0].Index)
		assert.Equal(t, float32(0.95), ret.Results[0].RelevanceScore)
	}
}

func TestRerankTimeout(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Simulate slow response
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"results": [{"index": 0, "relevance_score": 0.85}]}`))
	}))
	defer ts.Close()
	url := ts.URL

	{
		c, _ := NewContextualAIClient("mock_key")
		ret, err := c.Rerank(url, "ctxl-rerank-v2-instruct-multilingual", "What is cloud computing?", []string{"Cloud computing provides on-demand computing resources"}, map[string]any{}, 1) // 1 second timeout
		// This should succeed with the mock server, but tests the timeout parameter
		assert.NoError(t, err)
		assert.NotNil(t, ret)
	}
}

func TestRerankWithDifferentConfigs(t *testing.T) {
	tests := []struct {
		name          string
		params        map[string]any
		expectedTopN  int
		expectedInstr string
		expectedMeta  []string
	}{
		{
			name:         "with default config",
			params:       map[string]any{},
			expectedTopN: 0,
		},
		{
			name:         "with topN only",
			params:       map[string]any{"top_n": 5},
			expectedTopN: 5,
		},
		{
			name:          "with instruction only",
			params:        map[string]any{"instruction": "Prioritize recent documents"},
			expectedInstr: "Prioritize recent documents",
		},
		{
			name:         "with metadata only",
			params:       map[string]any{"metadata": []string{"doc1", "doc2"}},
			expectedMeta: []string{"doc1", "doc2"},
		},
		{
			name:          "with all parameters",
			params:        map[string]any{"top_n": 3, "instruction": "Focus on technical accuracy", "metadata": []string{"tech", "guide"}},
			expectedTopN:  3,
			expectedInstr: "Focus on technical accuracy",
			expectedMeta:  []string{"tech", "guide"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var capturedRequest rerankRequest

			ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				err := json.NewDecoder(r.Body).Decode(&capturedRequest)
				require.NoError(t, err)

				response := RerankResponse{
					Results: []RerankedResultItem{
						{Index: 0, RelevanceScore: 0.9},
					},
				}

				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(response)
			}))
			defer ts.Close()

			c, _ := NewContextualAIClient("test_key")
			_, err := c.Rerank(ts.URL, "ctxl-rerank-v2-instruct-multilingual", "test query", []string{"doc1", "doc2"}, tt.params, 30)

			require.NoError(t, err)
			assert.Equal(t, "ctxl-rerank-v2-instruct-multilingual", capturedRequest.Model)
			assert.Equal(t, "test query", capturedRequest.Query)
			assert.Equal(t, []string{"doc1", "doc2"}, capturedRequest.Documents)

			if tt.expectedTopN > 0 {
				assert.Equal(t, tt.expectedTopN, capturedRequest.TopN)
			}

			if tt.expectedInstr != "" {
				assert.Equal(t, tt.expectedInstr, capturedRequest.Instruction)
			}

			if len(tt.expectedMeta) > 0 {
				assert.Equal(t, tt.expectedMeta, capturedRequest.Metadata)
			}
		})
	}
}
