# SAGE Prompt 设计

## 核心原则

1. **明确输出格式**：指定 "YES/NO"、"SKIP/EXECUTE" 等
2. **提供必要上下文**：但避免冗长
3. **简洁提问**：一个 prompt 一个任务
4. **示例引导**：给出期望输出样例

## 1. 时效性检测

```python
TEMPORAL_DETECTION_PROMPT = """
Does this query require real-time or very recent information?

Query: "{query}"

Consider:
- Time-sensitive keywords: "latest", "recent", "current", "today", "2026"
- Event queries: "match result", "news", "update"
- Static queries: "principle", "definition", "history"

Output format (first line only):
TEMPORAL / STATIC

Answer:"""
```

## 2. 查询相似度

```python
QUERY_SIMILARITY_PROMPT = """
Are these two queries asking for similar information?

Query 1: "{query1}"
Query 2: "{query2}"

Consider:
- Same intent? (analyzing vs comparing vs explaining)
- Same entity? (EAGLE vs Medusa)
- Same aspect? (performance vs principle vs deployment)

Output format (first line only):
SIMILAR / DIFFERENT

Answer:"""
```

## 3. Stage 1 跳过判断

```python
SKIP_DECISION_PROMPT = """
Can we skip this search hop based on previous results?

User Query: "{query}"

Previous Results:
{previous_results}

Current Hop:
  Tool: {hop.tool}
  Query: "{hop.query}"
  Purpose: {hop.purpose}

Rules:
1. If previous results fully answer this hop's query → SKIP
2. If this hop adds essential new info → EXECUTE
3. If this is the first hop → EXECUTE
4. If uncertain or this hop is foundational → EXECUTE

Output format (first line only):
SKIP / EXECUTE / UNCERTAIN

Decision:"""
```

## 4. Stage 2 重要性评分

```python
IMPORTANCE_SCORING_PROMPT = """
Predict the final answer and rate each hop's importance.

User Query: "{query}"

Previous Results:
{previous_results}

Remaining Hops to Evaluate:
{format_hops(uncertain_hops)}

Task:
1. Generate a speculative answer to the query (based on your knowledge)
2. For each hop, predict what it would find
3. Rate importance (0-100):
   - 80-100: ESSENTIAL (critical info)
   - 40-79: USEFUL (adds value but not critical)
   - 0-39: REDUNDANT (covered or not needed)

Output format:
Speculative Answer: [your complete answer]

Hop {id} Prediction: [what this hop would find]
Hop {id} Importance: [score 0-100]
Hop {id} Reasoning: [why this score]

Response:"""
```

## 5. 结果验证

```python
VERIFICATION_PROMPT = """
Compare these two results for consistency.

API Result:
{api_result}

Speculated Result:
{spec_result}

Question: Are they consistent enough to use the speculation?

Consider:
- Factual accuracy: Do they agree on key facts?
- Completeness: Does speculation cover the main points?
- Specificity: Is speculation specific enough?

Output format:
Score: [0-100]
Reasoning: [brief explanation]

Response:"""
```

## 6. 实体映射提取

```python
ENTITY_MAPPING_PROMPT = """
Extract entity mapping between pattern and query.

Pattern: "{pattern}"
Query: "{query}"

Identify placeholders in pattern (e.g., {{ENTITY}}, {{ASPECT}})
and their corresponding values in query.

Output format (JSON):
{{"ENTITY": "value1", "ASPECT": "value2"}}

Mapping:"""
```

## 7. 依赖推断

```python
DEPENDENCY_INFERENCE_PROMPT = """
Does this new hop depend on existing hops?

Existing Hops:
{format_hops(existing_hops)}

New Hop:
  Tool: {new_hop.tool}
  Query: "{new_hop.query}"

Output dependencies (JSON):
{{
  "hop_id": dependency_strength (0.0-1.0)
}}

If no dependencies, output: {{}}

Dependencies:"""
```

## Prompt 优化技巧

### 1. 控制输出长度

```python
# 快速判断：限制输出
llm.generate(prompt, max_new_tokens=5)

# 深度分析：允许长输出
llm.generate(prompt, max_new_tokens=500)
```

### 2. 使用结构化输出

```python
# 不好：自由文本
"Explain if these are similar."

# 好：结构化
"Output: SIMILAR / DIFFERENT"
"First line only: YES / NO"
```

### 3. 提供决策规则

```python
# 不好：开放问题
"Should we skip this hop?"

# 好：明确规则
"Rules:
 1. If X → SKIP
 2. If Y → EXECUTE
 3. If uncertain → EXECUTE"
```

### 4. 示例驱动

```python
prompt = """
Task: Extract entities

Example:
  Query: "EAGLE performance analysis"
  Output: {{"ENTITY": "EAGLE", "ASPECT": "performance"}}

Your turn:
  Query: "{query}"
  Output:"""
```

## 常见问题

### Q: 如何处理 LLM 不按格式输出？

A: 解析时宽容：
```python
output = llm.generate(prompt)
# 提取第一行
first_line = output.strip().split('\n')[0]
# 提取关键词
if "SKIP" in first_line:
    return "SKIP"
elif "EXECUTE" in first_line:
    return "EXECUTE"
else:
    return "UNCERTAIN"  # 默认保守
```

### Q: 如何调试 prompt？

A: 记录输入输出：
```python
logger.debug(f"Prompt: {prompt}")
logger.debug(f"Output: {output}")
logger.debug(f"Parsed: {parsed_result}")
```

### Q: 如何优化 prompt 性能？

A:
1. **减少输入长度**：只提供必要上下文
2. **限制输出长度**：`max_new_tokens`
3. **批处理**：多个判断合并为一个 prompt
4. **缓存**：相同 prompt 复用结果

### Q: 如何提高 prompt 准确率？

A:
1. **明确规则**：提供决策标准
2. **Few-shot**：给出示例
3. **Chain-of-thought**：要求解释推理过程
4. **Fine-tune**：在特定任务上微调 LLM
