# SAGE 核心机制详解

## 1. Hop 重要性判断

### 两阶段策略

**Stage 1: 快速二分类**

适用：~70% 的 hops（明确情况）

```python
def stage1_decision(hop, context):
    if hop.index == 0:
        return "EXECUTE"  # 第一个 hop 必须执行
    
    # 检查后续依赖
    if has_dependents(hop):
        return "EXECUTE"  # 有依赖者不轻易剪枝
    
    # LLM 判断
    prompt = f"""
    Can we skip this hop based on previous context?
    
    Context: {format_context(context)}
    Hop: {hop.query}
    
    Output: SKIP / EXECUTE / UNCERTAIN
    """
    
    output = llm_judge(prompt, max_new_tokens=5)
    decision = output.strip().split()[0]
    
    return decision
```

**Stage 2: 深度评分**

适用：~30% 的 hops（不确定情况）

```python
def stage2_scoring(uncertain_hops, context):
    prompt = f"""
    Predict the final answer and rate each hop's importance.
    
    Context: {format_context(context)}
    Hops to evaluate: {format_hops(uncertain_hops)}
    
    Output format:
    Speculation: [your prediction]
    
    Hop {id} Prediction: [what it would find]
    Hop {id} Importance: [score 0-100]
    """
    
    output = llm_judge(prompt, max_new_tokens=500)
    evaluation = parse_output(output)
    
    # 决策 + 缓存
    for hop_eval in evaluation["hops"]:
        if hop_eval["importance"] >= 60:
            # 保留
            keep(hop_eval["id"])
            # 缓存 speculation
            speculation_cache[hop_eval["id"]] = hop_eval["prediction"]
        else:
            # 剪枝
            prune(hop_eval["id"])
    
    return evaluation
```

### 为什么两阶段？

**成本分析**：
- Stage 1: 20ms/hop
- Stage 2: 150ms（一次评估多个 hops）

如果 70% 在 Stage 1 解决：
- 平均成本: 0.7 × 20 + 0.3 × 150 = 59ms
- vs 全用 Stage 2: 150ms
- 节省: 91ms

## 2. 依赖关系

### 建立方式

**离线学习**（构建图时）：

```python
def extract_dependencies(execution_log):
    dependencies = {}
    
    for i, hop in enumerate(execution_log):
        for prev_idx in range(i):
            prev_hop = execution_log[prev_idx]
            
            # 数据依赖：query 中使用了 prev result
            if hop.query.contains(prev_hop.result[:50]):
                add_dependency(hop, prev_hop, "data", 0.9)
            
            # 语义依赖：reasoning 中引用了 prev result
            elif hop.reasoning.references(prev_hop.result):
                add_dependency(hop, prev_hop, "semantic", 0.6)
    
    return dependencies
```

**在线推断**（新 hop 时）：

```python
def infer_dependency(new_hop, existing_hops):
    prompt = f"""
    Does this new hop depend on existing hops?
    
    Existing: {format_hops(existing_hops)}
    New: {new_hop.query}
    
    Output dependencies in JSON:
    {{"hop1": 0.8, "hop3": 0.5}}
    """
    
    output = llm_judge(prompt)
    return parse_json(output)
```

### 依赖类型

**数据依赖**（强）：
```
hop2 的 query 直接使用 hop1 的 result

例：
  hop1: "search EAGLE github url"  
        → "https://github.com/..."
  hop2: "get stars for https://github.com/..."
        → 直接使用 URL

强度: 0.8-1.0
剪枝规则: hop1 被剪 → hop2 必须剪
```

**语义依赖**（弱）：
```
hop3 的 reasoning 需要理解 hop1 的内容

例：
  hop1: "EAGLE technical principle"
        → "uses draft model for speculation..."
  hop3: "EAGLE performance"  
        → reasoning 中引用 "speculation" 概念

强度: 0.3-0.7
剪枝规则: hop1 被剪 → hop3 独立决策（可能保留）
```

### 剪枝时的依赖检查

```python
def safe_pruning(skip_set, dependencies):
    # 迭代检查连带剪枝
    changed = True
    while changed:
        changed = False
        
        for hop in plan:
            if hop in skip_set:
                continue
            
            # 检查依赖的 hop 是否被剪
            for dep_hop, dep_type, strength in dependencies[hop]:
                if dep_hop in skip_set:
                    # 强数据依赖 → 连带剪枝
                    if dep_type == "data" and strength > 0.7:
                        skip_set.add(hop)
                        changed = True
                        break
    
    return skip_set
```

## 3. 推理图匹配

### 三层匹配

**Layer 1: Query 匹配**

```python
def query_match(new_query, graph):
    prompt = f"""
    Are these queries similar in intent?
    
    Pattern: {graph.pattern}
    Query: {new_query}
    
    If similar, extract entity mapping:
    {{ENTITY}} → value
    
    Output: SIMILAR/DIFFERENT + mapping
    """
    
    output = llm_judge(prompt)
    
    if "SIMILAR" in output:
        mapping = extract_mapping(output)
        return (graph, mapping)
    return None
```

**Layer 2: Structure 匹配**

```python
def structure_match(new_plan, graph):
    # 抽象化新 plan
    new_intents = [abstract_intent(hop) for hop in new_plan]
    graph_intents = [node.intent for node in graph.nodes]
    
    # 最长公共子序列
    lcs = longest_common_subsequence(new_intents, graph_intents)
    score = len(lcs) / max(len(new_intents), len(graph_intents))
    
    return score > 0.7
```

**Layer 3: Hop 匹配**

```python
def hop_match(new_hop, graph_node, entity_mapping):
    # 实例化 graph node
    instantiated = graph_node.query_template
    for k, v in entity_mapping.items():
        instantiated = instantiated.replace(f"{{{k}}}", v)
    
    # 语义相似度
    score = llm_semantic_similarity(new_hop.query, instantiated)
    
    return score > 0.8
```

### 实例化流程

```python
def instantiate_graph(graph, entity_mapping):
    plan = []
    
    # 实例化 nodes
    for node in graph.nodes.values():
        query = node.query_template
        for placeholder, value in entity_mapping.items():
            query = query.replace(f"{{{placeholder}}}", value)
        
        hop = Hop(id=node.id, query=query, tool=node.tool)
        plan.append(hop)
    
    # 复用 dependencies
    dependencies = {}
    for (src, dst), edge in graph.edges.items():
        if dst not in dependencies:
            dependencies[dst] = []
        dependencies[dst].append((src, edge.type, edge.strength))
    
    return plan, dependencies
```

## 4. Speculation 执行

### 并行模式

```python
async def execute_hop(hop, context):
    # 检查缓存
    cached_spec = speculation_cache.get(hop.id)
    
    # 启动 API 和 Speculation
    api_task = call_api(hop.tool, hop.query)
    
    if cached_spec:
        spec_result = cached_spec  # 0ms
    else:
        spec_task = generate_speculation(hop, context)
        spec_result = await spec_task  # 150ms
    
    api_result = await api_task  # 300ms
    
    # Judge 验证
    score = llm_judge.verify(api_result, spec_result)
    
    if score >= 75:
        return spec_result
    else:
        return await reasoning(api_result, context)
```

### Speculation 复用

**问题**：Stage 2 生成的 speculation 被丢弃

**方案**：缓存并在执行时复用

```python
# Stage 2
for hop_eval in evaluation["hops"]:
    if hop_eval["importance"] >= 60:
        speculation_cache[hop_eval["id"]] = hop_eval["prediction"]

# 执行时
cached = speculation_cache.get(hop.id)
if cached:
    use cached  # 节省 150ms GPU + 200 tokens
```

**质量保证**：Judge 验证
- Stage 2 时 context 可能不完整
- speculation 质量可能不够
- Judge 分数低 → 用 API（有保障）

## 5. LLM Judge

### 统一接口

```python
class LLMJudge:
    def __init__(self, model):
        self.model = model
    
    def judge(self, prompt, max_new_tokens=50):
        return self.model.generate(prompt, max_new_tokens=max_new_tokens)
    
    # 时效性
    def is_temporal(self, query):
        return "YES" in self.judge(TEMPORAL_PROMPT.format(query=query))
    
    # 相似度
    def is_similar(self, q1, q2):
        return "YES" in self.judge(SIMILARITY_PROMPT.format(q1=q1, q2=q2))
    
    # Hop 决策
    def should_skip(self, hop, context):
        return self.judge(SKIP_PROMPT.format(hop=hop, context=context))
    
    # 重要性
    def evaluate(self, hops, context):
        return self.judge(IMPORTANCE_PROMPT.format(...), max_new_tokens=500)
    
    # 验证
    def verify(self, api, spec):
        output = self.judge(VERIFY_PROMPT.format(api=api, spec=spec))
        return int(extract_score(output))
```

### Prompt 设计原则

1. **明确输出格式**：指定 "YES/NO" 或 "score"
2. **提供上下文**：包含必要信息
3. **简洁提问**：避免冗长描述
4. **示例引导**：给出期望输出样例

## 6. 完整流程案例

### 查询："分析 EAGLE 的部署挑战"

```
[时效性检测]
  LLM: "STATIC"
  
[图匹配]
  找到: "analyze_{ENTITY}_challenges"
  映射: {ENTITY} → "EAGLE"
  
[实例化]
  plan: [hop1, hop2, hop3, hop4]
  dependencies: hop3→hop1, hop4→hop2
  
[Stage 1 剪枝]
  hop1: EXECUTE
  hop2: EXECUTE
  hop3: UNCERTAIN
  hop4: EXECUTE
  
[Stage 2 评分]
  hop3 importance: 72 → 保留
  cache["hop3"] = "EAGLE 约 2.5x..."
  
[执行]
  hop1: API || 新 spec → Judge → 用 spec
  hop2: API || 新 spec → Judge → 用 API
  hop3: API || 缓存 spec → Judge → 用缓存 spec（节省 150ms）
  hop4: API || 新 spec → Judge → 用 spec
  
[综合]
  生成最终答案
```

## 7. 关键问题

### Judge 阈值如何确定？

当前：固定 75

问题：
- 太低：接受质量差的 speculation
- 太高：很少使用 speculation

可能方案：
- 根据 hop 类型调整（原理 vs 数据）
- 统计历史准确率
- 多轮验证

### 依赖关系如何验证？

当前：离线学习 + 在线推断

问题：
- 历史数据可能不准
- LLM 判断可能有误

可能方案：
- 执行后验证实际依赖
- 更新依赖强度（加权平均）
- 人工标注关键模式

### Speculation 质量如何保证？

当前：Judge 验证

统计估计：
- ~70% 通过验证
- ~30% 使用 API
- ~5% Judge 误判

可能方案：
- 多轮验证
- 置信度输出
- 事后反馈调整
