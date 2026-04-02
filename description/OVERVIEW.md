# SAGE 高层架构

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                      SAGE 系统                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐      ┌──────────────┐               │
│  │ LLM Judge    │◄─────┤ Reasoning    │               │
│  │              │      │ Graph DB     │               │
│  └──────────────┘      └──────────────┘               │
│         │                      │                       │
│         ↓                      ↓                       │
│  ┌──────────────────────────────────────┐             │
│  │         核心引擎                      │             │
│  ├──────────────────────────────────────┤             │
│  │ • 时效性检测                         │             │
│  │ • 图匹配与复用                       │             │
│  │ • 智能剪枝                           │             │
│  │ • 推测执行                           │             │
│  │ • 结果验证                           │             │
│  └──────────────────────────────────────┘             │
│         │                      │                       │
│         ↓                      ↓                       │
│  ┌──────────────┐      ┌──────────────┐               │
│  │ Spec Cache   │      │ Tool Calls   │               │
│  └──────────────┘      └──────────────┘               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## 核心组件

### 1. LLM Judge

统一的判断接口，所有决策都通过它：

```python
class LLMJudge:
    def temporal_detection(query) -> "STATIC" | "TEMPORAL"
    def query_similarity(q1, q2) -> "SIMILAR" | "DIFFERENT"  
    def hop_decision(hop, context) -> "SKIP" | "EXECUTE" | "UNCERTAIN"
    def importance_scoring(hops, context) -> {hop_id: score}
    def verify_speculation(api, spec) -> score
```

### 2. Reasoning Graph

抽象推理图，缓存历史成功路径：

```python
class ReasoningGraph:
    nodes = {
        "n1": {
            "intent": "get_principle({ENTITY})",
            "query_template": "{ENTITY} technical principle"
        },
        "n2": {
            "intent": "get_performance({ENTITY})",
            "query_template": "{ENTITY} benchmark"
        }
    }
    
    edges = {
        ("n2", "n1"): {
            "type": "semantic",
            "strength": 0.5
        }
    }
```

### 3. Speculation Cache

简单的 dict，缓存 Stage 2 生成的 speculation：

```python
speculation_cache = {
    "hop3": "EAGLE achieves 2.5-3x speedup...",
    "hop4": "Common issues include..."
}
```

## 核心流程

### 1. 图匹配

```
输入: 新查询 "Medusa 的性能"

匹配过程:
  1. 遍历 graph_db
  2. LLM 判断语义相似度
  3. 提取实体映射: {ENTITY} → "Medusa"
  4. 实例化 plan
  5. 复用依赖关系
```

### 2. 智能剪枝

```
输入: 4 hops plan

Stage 1 (逐个判断):
  hop1: EXECUTE (第一个 hop)
  hop2: EXECUTE (明确重要)
  hop3: UNCERTAIN (不确定) ← 进入 Stage 2
  hop4: EXECUTE (明确重要)

Stage 2 (深度评分):
  输入: [hop3]
  LLM 生成:
    - Speculation: "..."
    - hop3 importance: 72
  决策: 72 ≥ 60 → 保留
  缓存: speculation_cache["hop3"] = speculation

输出: [hop1, hop2, hop3, hop4]
```

### 3. 推测执行

```
执行 hop3:
  
  1. 检查缓存
     cached_spec = speculation_cache.get("hop3")
     
  2. 启动任务
     API ║ cached_spec (0ms, 直接读!)
     
  3. Judge 验证
     score = judge(api_result, cached_spec)
     
  4. 决策
     if score ≥ 75:
         use cached_spec  # 节省 GPU + token
     else:
         use api_result + reasoning
```

## 依赖关系

### 表示方式

```python
# 抽象表示（存储在图中）
dependency = {
    "source": "n1",
    "target": "n3", 
    "type": "semantic",
    "strength": 0.5
}

# 含义: n3 的推理需要 n1 的信息（弱依赖）
```

### 剪枝时的检查

```python
if hop_to_prune in ["hop2"]:
    # 检查谁依赖 hop2
    dependents = graph.get_dependents("hop2")
    
    for (dep_hop, dep_type, strength) in dependents:
        if strength > 0.7 and dep_type == "data":
            # 强数据依赖 → 连带剪枝
            also_prune(dep_hop)
```

## LLM 调用时机

### 时效性检测
```
输入: 用户查询
调用: 1 次
成本: ~18ms
```

### 图匹配
```
输入: 查询 + 候选图
调用: 每个候选图 1 次（通常 5-10 个）
成本: ~100ms
```

### Stage 1 剪枝
```
输入: 单个 hop + context
调用: 每个 hop 1 次（~3-4 次）
成本: ~60ms
```

### Stage 2 剪枝
```
输入: 所有 UNCERTAIN hops + context
调用: 1 次（批处理）
成本: ~150ms
副作用: 生成 speculation 并缓存
```

### Judge 验证
```
输入: API result + speculation
调用: 每个 hop 1 次（~3-4 次）
成本: ~60ms
```

**总 LLM 调用**：~10 次/查询

## 数据流

```
查询: "分析 EAGLE 的部署挑战"
  ↓
[时效性检测] LLM → "STATIC"
  ↓
[图匹配] LLM → 找到 "analyze_deployment" 图
  ↓
[实例化] {ENTITY} → "EAGLE"
  生成 plan: [hop1, hop2, hop3, hop4]
  复用依赖: hop3→hop1, hop4→hop2
  ↓
[Stage 1] LLM × 4
  hop1: EXECUTE
  hop2: EXECUTE  
  hop3: UNCERTAIN
  hop4: EXECUTE
  ↓
[Stage 2] LLM × 1
  hop3 importance: 72 → 保留
  speculation_cache["hop3"] = "..."
  ↓
[执行 hop1] 
  API ║ 新生成 speculation
  Judge → 使用 speculation
  ↓
[执行 hop2]
  API ║ 新生成 speculation
  Judge → 使用 API
  ↓
[执行 hop3]
  API ║ 缓存的 speculation (0ms!)
  Judge → 使用缓存 speculation
  节省: 150ms GPU + 200 tokens
  ↓
[执行 hop4]
  API ║ 新生成 speculation
  Judge → 使用 speculation
  ↓
[综合] 生成最终答案
```

## 关键设计决策

### 为什么 Stage 2 speculation 要缓存？

原因：
1. 已经生成过（150ms + 200 tokens）
2. 执行时可能用得上
3. Judge 会验证质量（有保障）
4. 实现简单（几行代码）

风险：
1. Stage 2 时 context 可能不完整
2. 质量可能不如执行时重新生成

应对：
- Judge 验证（不够好就用 API）
- 最差情况等同于没优化

### 为什么用依赖图？

原因：
1. 剪枝时避免破坏依赖
2. 并行执行时指导调度
3. 跨查询复用结构

实现：
- 离线学习（历史执行分析）
- 在线推断（LLM 判断新 hop）

### 为什么全用 LLM？

对比规则系统：
- 规则：需要大量阈值，难调参
- LLM：改 prompt 即可，零样本

成本：
- 每次 LLM 调用 ~18ms
- 总成本 ~180ms（可接受）

优化：
- 批处理（多个判断合并）
- Fine-tune（降低延迟）
- 缓存（复用判断结果）
