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
	"fmt"
	"sort"

	"github.com/milvus-io/milvus/internal/util/function/models"
)

type ContextualAIClient struct {
	apiKey string
}

func NewContextualAIClient(apiKey string) (*ContextualAIClient, error) {
	if apiKey == "" {
		return nil, fmt.Errorf("missing credentials config or configure the %s environment variable in the Milvus service", models.ContextualAIAKEnvStr)
	}
	return &ContextualAIClient{apiKey: apiKey}, nil
}

func (c *ContextualAIClient) headers() map[string]string {
	return map[string]string{
		"Content-Type":  "application/json",
		"Authorization": fmt.Sprintf("Bearer %s", c.apiKey),
	}
}


type rerankRequest struct {
	Model       string   `json:"model"`
	Query       string   `json:"query"`
	Documents   []string `json:"documents"`
	TopN        int      `json:"top_n,omitempty"`
	Instruction string   `json:"instruction,omitempty"`
	Metadata    []string `json:"metadata,omitempty"`
}

type RerankedResultItem struct {
	Index          int     `json:"index"`
	RelevanceScore float32 `json:"relevance_score"`
}

type RerankResponse struct {
	Results []RerankedResultItem `json:"results"`
}

type contextualAIRerank struct {
	apiKey string
	url    string
}

func newContextualAIRerank(apiKey string, url string) *contextualAIRerank {
	return &contextualAIRerank{apiKey: apiKey, url: url}
}

func (c *contextualAIRerank) rerank(modelName string, query string, texts []string, headers map[string]string, params map[string]any, timeoutSec int64) (*RerankResponse, error) {
	r := rerankRequest{
		Model:     modelName,
		Query:     query,
		Documents: texts,
	}
	if params != nil {
		if v, ok := params["top_n"]; ok {
			if n, ok2 := v.(int); ok2 {
				r.TopN = n
			}
		}
		if v, ok := params["instruction"]; ok {
			if s, ok2 := v.(string); ok2 {
				r.Instruction = s
			}
		}
		if v, ok := params["metadata"]; ok {
			if m, ok2 := v.([]string); ok2 {
				r.Metadata = m
			}
		}
	}

	res, err := models.PostRequest[RerankResponse](r, c.url, headers, timeoutSec)
	if err != nil {
		return nil, err
	}
	sort.Slice(res.Results, func(i, j int) bool { return res.Results[i].Index < res.Results[j].Index })
	return res, nil
}

func (c *ContextualAIClient) Rerank(url string, modelName string, query string, texts []string, params map[string]any, timeoutSec int64) (*RerankResponse, error) {
	client := newContextualAIRerank(c.apiKey, url)
	return client.rerank(modelName, query, texts, c.headers(), params, timeoutSec)
}
