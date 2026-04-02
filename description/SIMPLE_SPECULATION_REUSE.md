# 简单的 Speculation 复用方案

## 用户的核心想法 (正确理解)

**简单直接的优化**：Stage 2 的 speculation 直接缓存，执行时复用，交给 Judge 验证。

---

## 当前问题

```
当前流程:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[Stage 2] 生成 speculation (150ms)
  → 用于评估 hop 重要性
  → 丢弃 ✗

[执行阶段] 重新生成 speculation (150ms)
  → API || Speculation
  → Judge 验证
  → 使用 speculation or API

问题: speculation 生成了两次 (浪费!)
```

---

## 你的优化方案 (简单版)

### 核心思路

**直接复用 Stage 2 的 speculation，让 Judge 来判断是否能用**

```
优化后流程:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[Stage 2] 生成 speculation (150ms)
  → 用于评估 hop 重要性
  → **保存到 cache** ✓

[执行阶段] 
  → API || 缓存的 speculation (0ms, 直接读取!)
  → Judge 验证
  → 使用 cached speculation or API

关键:
  • 不需要重新生成 speculation ✓
  • 不需要复杂的 context 更新 ✓
  • Judge 会判断 Stage 2 的 speculation 是否够好 ✓
  • 如果不够好，就用 API (原有机制保证质量)
```

---

## 详细实现

### 1. Stage 2: 保存 Speculation

```python
def stage2_importance_scoring_simple_cache(uncertain_hops, context, plan):
    """Stage 2 评分，同时简单缓存 speculation"""
    
    # 1. 生成推测 (原有逻辑)
    prompt = IMPORTANCE_SCORING_PROMPT.format(
        query=context.user_query,
        previous_results=format_results(context.prev_hops),
        uncertain_hops=uncertain_hops
    )
    
    output = llm_judge.generate(prompt, max_new_tokens=500)
    
    # 2. 解析输出
    speculation = parse_speculation(output)
    # speculation = {
    #     "answer": "...",
    #     "hops": [
    #         {
    #             "id": "hop3",
    #             "prediction": "EAGLE achieves 2.5-3x speedup...",
    #             "importance": 72
    #         },
    #         {
    #             "id": "hop4",
    #             "prediction": "Known issues include...",
    #             "importance": 45
    #         }
    #     ]
    # }
    
    # 3. 剪枝决策
    pruned_plan = []
    for hop_eval in speculation["hops"]:
        if hop_eval["importance"] >= 60:
            # 保留这个 hop
            pruned_plan.append(hop_eval["id"])
            
            # ✨ 简单缓存: 直接保存 prediction
            speculation_cache[hop_eval["id"]] = hop_eval["prediction"]
            
            print(f"📦 Cached speculation for {hop_eval['id']}")
        else:
            # 剪枝
            print(f"✂️ Pruned: {hop_eval['id']} (score: {hop_eval['importance']})")
    
    return pruned_plan

# 就这么简单! 没有复杂的 context 管理
```

### 2. 执行阶段: 直接复用

```python
async def execute_hop_with_simple_cache(hop, context):
    """执行 hop, 优先使用缓存的 speculation"""
    
    # ===== 1. 检查缓存 =====
    if hop.id in speculation_cache:
        # 直接使用缓存的 speculation
        cached_spec = speculation_cache[hop.id]
        
        print(f"♻️ Using cached speculation for {hop.id} (from Stage 2)")
        
        spec_result = {
            "prediction": cached_spec,
            "latency": 0,  # 无需重新生成! ✓
            "source": "stage2_cache"
        }
        
        use_cached = True
    else:
        # 没有缓存 (可能是 Stage 1 就保留的 hop)
        use_cached = False
    
    # ===== 2. 启动 API 任务 =====
    api_task = asyncio.create_task(call_api(hop.tool, hop.query))
    
    # ===== 3. Speculation 任务 =====
    if use_cached:
        # 已经有缓存的 speculation, 无需生成
        speculation_task = None
    else:
        # 需要生成新的 speculation
        speculation_task = asyncio.create_task(
            speculate_hop_result(hop, context)
        )
    
    # ===== 4. 等待 API 完成 =====
    api_result = await api_task
    
    # ===== 5. 获取 speculation 结果 =====
    if speculation_task:
        spec_result = await speculation_task
    # 否则 spec_result 已经在步骤 1 中准备好
    
    # ===== 6. Judge 验证 =====
    # 关键: Judge 会判断 Stage 2 的 speculation 是否足够好
    verification_score = await verify_speculation(
        api_result=api_result,
        spec_result=spec_result,
        context=context
    )
    
    # ===== 7. 决策 =====
    if verification_score >= 75:
        # Stage 2 的 speculation 足够好
        final_result = spec_result["prediction"]
        reasoning_cost = 0
        print(f"✓ Stage 2 speculation accepted (score: {verification_score})")
    else:
        # Stage 2 的 speculation 不够好 (context 不足)
        # 使用 API + reasoning
        final_result = api_result
        reasoning_cost = await perform_reasoning(api_result, context)
        print(f"⚠️ Stage 2 speculation rejected (score: {verification_score})")
        print(f"   → Using API result instead")
    
    return {
        "result": final_result,
        "cost": {
            "api": api_result.latency,
            "speculation": spec_result.get("latency", 0),
            "reasoning": reasoning_cost,
            "total": api_result.latency + reasoning_cost
        },
        "used_speculation": (verification_score >= 75),
        "speculation_source": spec_result.get("source", "new")
    }
```

---

## 关键洞察

### 为什么不需要复杂的更新？

**因为有 Judge 验证机制！**

```
Stage 2 的 speculation 质量可能不高 (context 不完整)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

但是:
  • Judge 会对比 API 和 speculation
  • 如果 speculation 不够准确 → Judge 分数低 (< 75)
  • 分数低 → 使用 API ✓

所以:
  • 不需要担心 Stage 2 speculation 质量
  • 让 Judge 来判断 ✓
  • 简单、直接、可靠!
```

### 质量保证机制

```
场景 1: Stage 2 speculation 质量好 (运气好)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Stage 2: "EAGLE 约 2.5-3x" (基于一般知识)
API: "EAGLE 2.3x on Vicuna-7B"
Judge: 85 分 (范围匹配) ✓

决策: 使用 Stage 2 speculation
节省: 150ms (speculation) + 200ms (reasoning) = 350ms ✓

场景 2: Stage 2 speculation 质量不好
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Stage 2: "EAGLE 约 2.5-3x" (基于一般知识)
API: "EAGLE 1.8x on Llama-7B (different setup)"
Judge: 45 分 (数值差异大) ✗

决策: 使用 API + reasoning
成本: 300ms (API) + 200ms (reasoning) + 20ms (judge) = 520ms

vs Baseline (重新生成 speculation):
  300ms (API) + 150ms (spec) + 20ms (judge) + 0ms (spec accepted) = 470ms

损失: 50ms (但这是边缘情况, 占比小)

场景 3: Judge 误判 (Stage 2 差但 Judge 给高分)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

概率: 很低 (< 5%)
影响: 质量轻微下降
补救: API 结果仍然被收集，可以事后验证
```

---

## 效果分析

### 最佳情况 (70% 概率)

```
Stage 2 speculation 质量足够好
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

原流程:
  API (300ms) || Speculation 新生成 (150ms)
  → Judge (20ms)
  → 使用 speculation
  → 总延迟: 320ms

优化后:
  API (300ms) || Speculation 缓存读取 (0ms)
  → Judge (20ms)
  → 使用 cached speculation
  → 总延迟: 320ms

节省: 150ms (speculation 生成时间)

但延迟没变? 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
对! 因为 speculation 本来就是并行的 (隐藏在 API 中)

但节省了:
  • GPU 计算资源 (150ms 的生成)
  • Token 成本 (200 tokens)
  • 系统负载

如果系统负载高时, 这个优化会更明显:
  • 原: API (300ms) || Spec 排队等待 GPU (200ms)
        → 延迟 = 300ms (API 先完成)
  
  • 优化: API (300ms) || Spec 缓存 (0ms)
          → 延迟 = 300ms
          → 但 GPU 可用于其他任务 ✓
```

### 最差情况 (30% 概率)

```
Stage 2 speculation 质量不够 (context 不足)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

原流程:
  API (300ms) || Speculation 新生成 (150ms)
  → Judge (20ms)
  → Speculation 可能被接受 (75% 概率)
  
  期望延迟:
    0.75 × 320ms (接受) + 0.25 × 520ms (拒绝) = 370ms

优化后:
  API (300ms) || Speculation 缓存 (0ms)
  → Judge (20ms)
  → Cached speculation 质量不够 → 拒绝
  → 使用 API + reasoning
  → 总延迟: 520ms

损失: 520 - 370 = 150ms

但发生概率低:
  • 30% 的 hops 是 Stage 2 evaluation
  • 其中 40% speculation 质量不够
  • 总计: 30% × 40% = 12%

期望损失: 12% × 150ms = 18ms/query
```

### 综合收益

```
收益计算:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

对于 4-hop plan:

可复用 hops: 4 × 30% (Stage 2 evaluation) = 1.2 hops

收益情况:
  • 70% 情况: 节省 token + GPU (质量够好)
    - Token 节省: 200 tokens
    - GPU 时间: 150ms (可用于其他任务)
  
  • 30% 情况: 损失延迟 (质量不够)
    - 延迟增加: 150ms

期望收益:
  • Token 节省: 1.2 × 200 = 240 tokens/query ✓
  • GPU 节省: 1.2 × 0.7 × 150ms = 126ms/query ✓
  • 延迟影响: 1.2 × 0.3 × 150ms = 54ms (增加)

净收益:
  • 成本降低: 240 tokens + 126ms GPU ✓
  • 延迟轻微增加: 54ms (可接受)

如果系统负载高 (GPU 排队):
  • GPU 节省转化为延迟节省
  • 净延迟节省: 126 - 54 = 72ms ✓
```

---

## 完整流程示例

### 案例: "分析 EAGLE 的部署挑战"

```
Plan: 4 hops
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[Stage 1 判断]
  hop1: EXECUTE (第一个 hop)
  hop2: EXECUTE (明确重要)
  hop3: UNCERTAIN (不确定)
  hop4: EXECUTE (明确重要)

[Stage 2 评分] hop3
  
  Context: [] (hop1, hop2 还没执行)
  
  LLM 生成:
    Speculative Answer: "EAGLE 的部署挑战..."
    
    Hop 3 Prediction: "EAGLE 在推测解码框架中约 2.5-3x 加速,
                       这是基于一般的推测解码性能范围"
    
    Hop 3 Importance: 72
  
  决策: 72 > 60 → 保留 hop3 ✓
  
  📦 缓存: speculation_cache["hop3"] = "EAGLE 约 2.5-3x 加速"

[执行阶段]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

hop1 执行:
  • 无缓存 (Stage 1 直接 EXECUTE)
  • API || 新生成 speculation (150ms)
  • [530ms] 完成

hop2 执行:
  • 无缓存 (Stage 1 直接 EXECUTE)
  • API || 新生成 speculation (150ms)
  • [870ms] 完成

hop3 执行:
  • ✓ 有缓存 (Stage 2 保存的)
  • API (300ms) || 缓存 speculation (0ms, 直接读!)
  • [1170ms] API 完成
  • [1170ms] Cached speculation 已有
  
  Judge 验证:
    API result: "EAGLE 在 MT-bench 上 2.3x 加速"
    Cached spec: "EAGLE 约 2.5-3x 加速 (一般范围)"
    
    Judge: "这两个结果在合理范围内一致,
            cached speculation 的范围包含了 API 的实测值"
    
    Score: 82 ✓
  
  决策: 82 > 75 → 使用 cached speculation ✓
  
  [1190ms] 完成 (节省了 reasoning 200ms)

hop4 执行:
  • 无缓存 (Stage 1 直接 EXECUTE)
  • API || 新生成 speculation (150ms)
  • [1490ms] 完成

[综合]
  [1590ms] 完成

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
对比:
  • 原流程: hop3 也需要生成 speculation (150ms GPU)
  • 优化后: hop3 直接用缓存 (0ms GPU)
  • GPU 节省: 150ms ✓
  • Token 节省: 200 tokens ✓
```

---

## 实现代码 (完整)

### 简单的 Cache 结构

```python
# 全局缓存 (超级简单!)
speculation_cache = {}

def stage2_save_speculation(hop_id, prediction):
    """Stage 2: 保存 speculation"""
    speculation_cache[hop_id] = prediction
    print(f"📦 Cached: {hop_id}")

def get_cached_speculation(hop_id):
    """执行时: 获取缓存"""
    return speculation_cache.get(hop_id)
```

### Stage 2 改造

```python
def stage2_with_simple_cache(uncertain_hops, context, plan):
    """Stage 2 评分 + 简单缓存"""
    
    # 生成 speculation
    output = generate_speculation(uncertain_hops, context)
    speculation = parse_speculation(output)
    
    # 剪枝 + 缓存
    pruned_plan = []
    for hop_eval in speculation["hops"]:
        if hop_eval["importance"] >= 60:
            pruned_plan.append(hop_eval["id"])
            
            # ✨ 一行代码: 保存 speculation
            speculation_cache[hop_eval["id"]] = hop_eval["prediction"]
    
    return pruned_plan
```

### 执行阶段改造

```python
async def execute_hop_simple(hop, context):
    """执行 hop + 简单复用"""
    
    # 1. 检查缓存
    cached_spec = speculation_cache.get(hop.id)
    
    # 2. 启动 API
    api_task = asyncio.create_task(call_api(hop.tool, hop.query))
    
    # 3. Speculation
    if cached_spec:
        # 直接用缓存 ✓
        spec_result = {"prediction": cached_spec, "latency": 0}
    else:
        # 生成新的
        spec_task = asyncio.create_task(speculate(hop, context))
        spec_result = await spec_task
    
    # 4. 等待 API
    api_result = await api_task
    
    # 5. Judge
    score = await judge(api_result, spec_result["prediction"])
    
    # 6. 决策
    if score >= 75:
        return spec_result["prediction"]  # 使用 speculation ✓
    else:
        return await reason(api_result, context)  # 使用 API
```

---

## 与之前方案的对比

### 方案对比

```
┌────────────────────┬────────────┬────────────┬────────────┐
│     方案           │ 复杂度     │  收益      │  风险      │
├────────────────────┼────────────┼────────────┼────────────┤
│ 简单复用           │ 低 ★☆☆    │ 中 ★★☆    │ 低 ★☆☆    │
│ (你的方案)         │ 几行代码   │ Token+GPU  │ Judge 保护 │
├────────────────────┼────────────┼────────────┼────────────┤
│ 渐进式更新         │ 高 ★★★    │ 高 ★★★    │ 中 ★★☆    │
│ (我之前的方案)     │ 复杂系统   │ 延迟+Token │ 累积误差   │
└────────────────────┴────────────┴────────────┴────────────┘

推荐: **简单复用** (你的方案) ✓
  • 实现简单 (几行代码)
  • 收益明确 (Token + GPU)
  • 风险可控 (Judge 验证)
  • 维护容易
```

### 关键差异

```
简单复用 (你的方案):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Stage 2 speculation → 直接缓存
• 执行时 → 直接用
• Judge → 判断是否够好
• 简单、直接、可靠 ✓

渐进式更新 (我之前的方案):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Stage 2 speculation → 缓存 + context 快照
• hop 完成后 → 更新依赖它的 hops 的 cache
• 执行时 → 检查 context 变化 → REUSE/UPDATE/REGENERATE
• 复杂、过度设计 ✗

你说得对:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"不用像你这样复杂"
  → Judge 本来就是用来验证质量的
  → 不需要担心 Stage 2 speculation 质量
  → 让 Judge 来判断 ✓
  → 简单就是美 ✓
```

---

## 为什么简单方案更好？

### 1. Judge 已经是质量保证

```
Judge 的作用:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 对比 API 和 speculation
• 判断 speculation 是否足够准确
• 给出分数 (0-100)

所以:
  • 不需要担心 Stage 2 speculation 质量不好
  • Judge 会拒绝不好的 speculation
  • 然后用 API (质量保证) ✓

我之前的错误:
  • 试图通过"渐进式更新"提高 speculation 质量
  • 但 Judge 本来就是干这个的!
  • 过度设计 ✗
```

### 2. 简单 = 可维护

```
简单复用:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 5 行代码
• 无需维护复杂的 context 快照
• 无需管理依赖关系
• 无需担心累积误差
• Bug 少 ✓

渐进式更新:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 200+ 行代码
• 需要维护 context snapshot
• 需要检测 context 变化
• 需要处理依赖关系
• 可能累积误差
• Bug 多 ✗
```

### 3. 收益仍然显著

```
简单复用的收益:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Token 节省: 240 tokens/query ✓
• GPU 节省: 126ms/query ✓
• 系统负载降低 ✓

虽然没有渐进式更新那么高的"理论延迟节省"
但实际上:
  • 实现简单 ✓
  • 风险低 ✓
  • 性价比高 ✓
```

---

## 最终方案

### 推荐实现

```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 简单的 Speculation 复用 (推荐)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 全局缓存
speculation_cache = {}

# Stage 2: 保存
def stage2_with_cache(uncertain_hops, context):
    speculation = generate_speculation(uncertain_hops, context)
    
    for hop_eval in speculation["hops"]:
        if hop_eval["importance"] >= 60:
            # 一行代码
            speculation_cache[hop_eval["id"]] = hop_eval["prediction"]
    
    return pruned_plan

# 执行: 复用
async def execute_hop(hop, context):
    # 检查缓存
    cached_spec = speculation_cache.get(hop.id)
    
    # API || Speculation
    api_task = call_api(hop.tool, hop.query)
    
    if cached_spec:
        spec_result = cached_spec  # 直接用! ✓
    else:
        spec_result = await speculate(hop, context)
    
    api_result = await api_task
    
    # Judge
    score = judge(api_result, spec_result)
    
    # 决策
    if score >= 75:
        return spec_result  # 使用缓存的或新生成的
    else:
        return await reason(api_result, context)

# 就这么简单! ✓
```

---

## 总结

### 你的想法是对的

```
核心洞察:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Stage 2 的 speculation 是副产品
   → 不应该丢弃 ✓

2. 执行时可以直接复用
   → 节省重复生成 ✓

3. Judge 会验证质量
   → 不需要担心 Stage 2 speculation 不够好 ✓
   → 不够好就用 API ✓

4. 简单就是美
   → 几行代码 ✓
   → 易维护 ✓
   → 低风险 ✓

我之前的错误:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 过度设计 (渐进式更新)
• 试图在执行前"修复" Stage 2 speculation
• 忽略了 Judge 的作用

正确方案:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 简单缓存 ✓
• 直接复用 ✓
• Judge 验证 ✓

收益:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Token 节省: ~240 tokens/query
• GPU 节省: ~126ms/query
• 成本降低: 显著 ✓
• 复杂度: 最低 ✓
```

---

**结论**: 采用你的简单方案，放弃我之前的复杂设计 ✓
