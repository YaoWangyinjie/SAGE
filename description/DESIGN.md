# SAGE 技术方案

## 1. 核心问题

Multi-hop Research Agent 延迟高：
- Planning: 生成多步计划
- Execution: 串行执行每个 hop (API + Reasoning)
- Synthesis: 综合所有结果

## 2. 解决方案

三个优化方向：

### 2.1 降低 Tool Call 成本

**语义缓存**：
- 问题：字符串精确匹配命中率低
- 方案：LLM 判断语义相似度
```python
def semantic_cache_lookup(query):
    for cached_query in cache:
        if llm_judge.is_similar(query, cached_query):
            return cache[cached_query]
    return None
```

**推理图复用**：
- 问题：每次都重新生成 plan 和分析依赖
- 方案：缓存抽象推理图，跨查询复用
```python
graph = {
    "pattern": "analyze_{ENTITY}_challenges",
    "nodes": [
        {"intent": "get_principle({ENTITY})"},
        {"intent": "get_issues({ENTITY})"}
    ],
    "dependencies": [("n2", "n1", "semantic", 0.5)]
}

# 新查询: "Medusa 的挑战"
# 实体映射: {ENTITY} → "Medusa"
# 复用依赖关系
```

### 2.2 减少 Tool Call 次数

**智能剪枝**：
- 问题：不是所有 hop 都必要
- 方案：LLM 两阶段评估

Stage 1 (快速，~70%情况)：
```
Prompt: "这个 hop 可以跳过吗？"
Output: SKIP / EXECUTE / UNCERTAIN
Cost: 20ms
```

Stage 2 (深度，~30%情况)：
```
Prompt: "推测最终答案，评估每个 hop 的重要性"
Output: {
    "speculation": "...",
    "hops": [
        {"id": "hop3", "importance": 72, "prediction": "..."}
    ]
}
Decision: importance ≥ 60 → 保留
Cost: 150ms
Side-effect: 缓存 prediction 供执行时复用
```

**依赖检查**：
```python
if prune("hop2"):
    # 检查谁依赖 hop2
    for dep_hop, strength in graph.get_dependents("hop2"):
        if strength > 0.7:  # 强依赖
            prune(dep_hop)  # 连带剪枝
```

### 2.3 并行执行

**推测执行**：
```
API (300ms) || Local Speculation (150ms)
           ↓
        Judge (20ms)
           ↓
    score ≥ 75: use speculation
    score < 75: use API + reasoning
```

**Speculation 复用**：
- 问题：Stage 2 生成的 speculation 被丢弃，执行时重新生成
- 方案：缓存 Stage 2 的 speculation
```python
# Stage 2
speculation_cache["hop3"] = prediction

# 执行时
cached = speculation_cache.get("hop3")
if cached:
    spec = cached  # 0ms, 直接用
else:
    spec = generate_new()  # 150ms
```

收益：
- 节省 GPU 时间
- 节省 token 成本
- Judge 保证质量

## 3. 核心机制

### 3.1 LLM Judge

统一判断接口：

```python
class LLMJudge:
    def __call__(self, prompt, max_tokens=50):
        return self.llm.generate(prompt, max_new_tokens=max_tokens)
    
    # 时效性检测
    def is_temporal(self, query):
        prompt = f"Does this query need real-time info? {query}\nOutput: YES/NO"
        return "YES" in self(prompt)
    
    # 查询相似度  
    def is_similar(self, q1, q2):
        prompt = f"Are these similar?\n1: {q1}\n2: {q2}\nOutput: YES/NO"
        return "YES" in self(prompt)
    
    # Hop 判断
    def should_skip(self, hop, context):
        prompt = f"Can we skip this hop?\nContext: {context}\nHop: {hop}\nOutput: SKIP/EXECUTE/UNCERTAIN"
        return self(prompt).split()[0]
    
    # 重要性评分
    def evaluate_importance(self, hops, context):
        prompt = f"Predict answer and rate each hop's importance (0-100):\n{format_hops(hops)}"
        return parse_output(self(prompt, max_tokens=500))
    
    # 结果验证
    def verify(self, api_result, speculation):
        prompt = f"Are these consistent?\nAPI: {api_result}\nSpec: {speculation}\nScore (0-100):"
        return int(self(prompt))
```

### 3.2 推理图

**数据结构**：

```python
@dataclass
class Node:
    id: str
    intent: str  # "get_principle({ENTITY})"
    query_template: str  # "{ENTITY} technical principle"

@dataclass  
class Edge:
    source: str
    target: str
    type: str  # "data" | "semantic"
    strength: float  # 0-1

@dataclass
class ReasoningGraph:
    pattern: str  # "analyze_{ENTITY}_challenges"
    nodes: Dict[str, Node]
    edges: Dict[Tuple[str, str], Edge]
```

**匹配流程**：

```python
def match_graph(query, graph_db):
    candidates = []
    
    for graph in graph_db:
        # LLM 判断语义相似度
        if llm_judge.is_similar(query, graph.pattern):
            # 提取实体映射
            mapping = extract_entity_mapping(query, graph.pattern)
            candidates.append((graph, mapping))
    
    if not candidates:
        return None
    
    # 选择最佳匹配
    best = max(candidates, key=lambda x: similarity_score(x))
    return best
```

**实例化**：

```python
def instantiate_plan(graph, entity_mapping):
    plan = []
    
    for node in graph.nodes.values():
        # 替换实体
        query = node.query_template
        for placeholder, value in entity_mapping.items():
            query = query.replace(f"{{{placeholder}}}", value)
        
        hop = Hop(
            id=node.id,
            query=query,
            tool=node.tool
        )
        plan.append(hop)
    
    # 复用依赖关系
    dependencies = {}
    for (src, dst), edge in graph.edges.items():
        dependencies[dst] = dependencies.get(dst, [])
        dependencies[dst].append((src, edge.type, edge.strength))
    
    return plan, dependencies
```

### 3.3 剪枝流程

```python
def prune_plan(plan, context, dependencies):
    to_skip = set()
    uncertain = []
    
    # Stage 1: 快速判断
    for hop in plan:
        if hop.index == 0:
            continue  # 第一个 hop 不剪
        
        decision = llm_judge.should_skip(hop, context)
        
        if decision == "SKIP":
            to_skip.add(hop.id)
        elif decision == "UNCERTAIN":
            uncertain.append(hop)
    
    # Stage 2: 深度评分
    if uncertain:
        evaluation = llm_judge.evaluate_importance(uncertain, context)
        
        for hop_eval in evaluation["hops"]:
            if hop_eval["importance"] >= 60:
                # 保留并缓存 speculation
                speculation_cache[hop_eval["id"]] = hop_eval["prediction"]
            else:
                to_skip.add(hop_eval["id"])
    
    # 依赖检查
    changed = True
    while changed:
        changed = False
        for hop_id in list(plan):
            if hop_id in to_skip:
                continue
            
            # 检查依赖的 hop 是否被剪枝
            for dep_hop, dep_type, strength in dependencies.get(hop_id, []):
                if dep_hop in to_skip and strength > 0.7:
                    to_skip.add(hop_id)
                    changed = True
                    break
    
    return [h for h in plan if h.id not in to_skip]
```

### 3.4 执行流程

```python
async def execute_hop(hop, context):
    # 1. 检查 speculation 缓存
    cached_spec = speculation_cache.get(hop.id)
    
    # 2. 启动 API
    api_task = asyncio.create_task(call_api(hop.tool, hop.query))
    
    # 3. Speculation
    if cached_spec:
        spec_result = cached_spec
    else:
        spec_result = await generate_speculation(hop, context)
    
    # 4. 等待 API
    api_result = await api_task
    
    # 5. Judge 验证
    score = llm_judge.verify(api_result, spec_result)
    
    # 6. 决策
    if score >= 75:
        return spec_result
    else:
        return await reasoning(api_result, context)
```

## 4. 关键问题

### 4.1 Judge 标准

**问题**：多少分算"一致"？

**当前方案**：固定阈值 75
- 太低 (50)：接受质量差的 speculation
- 太高 (90)：很少使用 speculation

**可能改进**：
- 根据 hop 类型动态调整（技术原理 vs 数值数据）
- 统计历史准确率自适应
- 多轮验证（不确定时再判断一次）

### 4.2 依赖关系准确性

**问题**：如何确保依赖图准确？

**当前方案**：
- 离线学习：分析历史执行 log
- 在线推断：LLM 判断新 hop 的依赖

**可能问题**：
- 历史 log 可能不全
- LLM 判断可能不准

**可能改进**：
- 执行后验证（实际是否使用了依赖的 hop）
- 更新依赖强度（加权平均）
- 人工标注关键模式

### 4.3 时效性处理

**问题**：缓存的信息可能过期

**当前方案**：
- LLM 判断查询是否需要实时信息
- TEMPORAL → 跳过缓存
- STATIC → 使用缓存

**可能问题**：
- 判断不准（"最新比赛"被判为 STATIC）
- 部分信息过期（技术原理 vs 版本号）

**可能改进**：
- 分层处理（不同信息不同 TTL）
- 时间戳标记（"2024-03 的数据"）
- 主动失效（检测到新事件时清除相关缓存）

### 4.4 图匹配粒度

**问题**：匹配太严格 vs 太宽松

太严格：
- "分析 EAGLE 挑战" vs "EAGLE 的问题"
- 语义相同但被判为不同

太宽松：
- "分析 EAGLE 挑战" vs "EAGLE 性能对比"
- 不同任务但被判为相同

**当前方案**：LLM 语义判断

**可能改进**：
- 结构匹配（不只看 query，还看 plan 结构）
- 分层缓存（粗粒度 + 细粒度）
- 部分复用（只复用部分 hops）

### 4.5 Speculation 质量

**问题**：Stage 2 时 context 不完整，speculation 可能不准

**当前方案**：Judge 验证，不准就用 API

**统计估计**：
- ~70% 情况 speculation 够用
- ~30% 情况需要用 API
- Judge 误判率 ~5%

**可能改进**：
- 多轮验证（分数接近阈值时再判断）
- 置信度输出（"我 80% 确定它们一致"）
- 事后反馈（用户纠正 → 调整策略）

## 5. 实现优先级

### P0: 核心功能
- [ ] LLM Judge 基础框架
- [ ] Stage 1+2 剪枝
- [ ] Speculation 缓存复用
- [ ] 基本的图匹配

### P1: 质量保障
- [ ] 依赖检查（避免误剪枝）
- [ ] Judge 验证
- [ ] 时效性检测

### P2: 性能优化
- [ ] 推理图复用
- [ ] 并行执行
- [ ] 批处理 LLM 调用

### P3: 高级功能
- [ ] 自适应阈值
- [ ] 依赖学习和更新
- [ ] 多层缓存

## 6. 测试策略

### 单元测试
- LLM Judge 各个功能
- 图匹配逻辑
- 依赖检查算法

### 集成测试
- 完整流程走通
- 边界情况（空 plan、全剪枝等）

### 质量测试
- 人工评估答案质量
- 对比 baseline（vanilla agent）
- 统计 Judge 准确率

### 性能测试
- 测量各模块延迟
- 找性能瓶颈
- 优化热点路径

## 7. 未来方向

### 7.1 Fine-tune Judge

训练数据：
- 正例：人工标注的"一致"样本对
- 负例：人工标注的"不一致"样本对

预期收益：
- 延迟降低（18ms → 5ms）
- 准确率提升（85% → 92%）

### 7.2 多层缓存

```
L1: Exact match (< 1ms)
L2: Semantic match (20ms)  
L3: Graph match (50ms)
```

### 7.3 流式执行

边执行边判断是否需要后续 hop：
```
hop1 执行 → 判断是否够了 → 不够 → hop2
                      ↓
                    够了 → 提前返回
```

### 7.4 主动学习

收集人工反馈：
- "这个 hop 应该跳过"
- "这两个结果不一致"

更新策略：
- 调整判断 prompt
- 更新依赖强度
- Fine-tune Judge
