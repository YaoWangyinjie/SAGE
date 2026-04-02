# SAGE: Speculative Agent with Graph-based Execution

Multi-hop Research Agent 加速框架

---

## 核心思路

传统 Research Agent 是串行执行：Planning → hop1 → hop2 → ... → Synthesis

SAGE 通过三个技术实现加速：

1. **语义缓存**：LLM 判断查询相似度，复用历史路径
2. **智能剪枝**：LLM 评估 hop 重要性，减少不必要的搜索
3. **推测执行**：API 和本地推理并行，LLM Judge 验证质量

## 核心设计原则

**LLM-First**：所有判断统一用 LLM，不用规则系统

- 时效性检测 → LLM 判断
- 查询相似度 → LLM 判断  
- Hop 重要性 → LLM 判断
- 结果验证 → LLM Judge

好处：简单、灵活、可解释、易维护

## 关键优化

### 1. Speculation 复用

**问题**：Stage 2 剪枝时生成的 speculation 被丢弃，执行时重新生成

**方案**：直接缓存 Stage 2 的 speculation，执行时复用
- 节省 GPU 时间和 token 成本
- Judge 保证质量（不够好就用 API）
- 实现简单（几行代码）

### 2. 推理图复用

**问题**：每次都重新分析依赖关系

**方案**：缓存抽象推理图，基于语义匹配复用
- 图中存储抽象结构（不存具体内容）
- 实体映射：{ENTITY} → "EAGLE" / "Medusa"
- 依赖关系跨查询复用

### 3. LLM 判断机制

**两阶段剪枝**：

Stage 1 (快速)：
```
LLM: "这个 hop 可以跳过吗？"
输出: SKIP / EXECUTE / UNCERTAIN
成本: 20ms
```

Stage 2 (深度)：
```
LLM: "生成推测答案，评估每个 hop 的重要性"
输出: Speculation + Importance (0-100)
决策: importance ≥ 60 → 保留
成本: 150ms
副作用: 缓存 speculation 供执行时复用
```

### 4. 并行执行

```
API (300ms) || Local Speculation (150ms)
           ↓
        Judge (20ms)
           ↓
    使用 Speculation or API
```

## 文件说明

- `README.md` - 项目概述（本文件）
- `DESIGN.md` - 完整技术方案
- `OVERVIEW.md` - 高层架构
- `CORE_MECHANISMS.md` - 核心机制详解
- `TECHNICAL_DETAILS.md` - 实现细节
- `FLOWCHARTS.md` - 流程图
- `PROMPTS.md` - Prompt 设计
- `SIMPLE_SPECULATION_REUSE.md` - Speculation 复用方案

## 核心流程

```
查询输入
  ↓
时效性检测 (LLM)
  ↓
图匹配 (语义相似度)
  ↓
Planning (实例化或新建)
  ↓
剪枝 (LLM 两阶段)
  ├─ Stage 1: 快速判断
  └─ Stage 2: 深度评分 + 缓存 speculation
  ↓
执行 (并行 + speculation 复用)
  ↓
综合结果
```

## 关键决策

1. **为什么用 LLM 而不是规则？**
   - 规则需要大量阈值调参
   - LLM 理解语义，更灵活准确
   - 改 prompt 比改规则简单

2. **为什么复用 Stage 2 speculation？**
   - 已经生成过，不复用就浪费
   - Judge 会验证质量
   - 不够好就用 API（有保障）

3. **为什么用推理图而不是直接缓存？**
   - 直接缓存只能精确匹配
   - 推理图存抽象结构，可跨实体复用
   - 命中率更高

## 待解决问题

1. Judge 的判断标准（相似度多少算通过？）
2. 依赖关系的准确性（如何验证和更新？）
3. 时效性信息的处理（缓存 TTL？）
4. 系统负载高时的调度策略
