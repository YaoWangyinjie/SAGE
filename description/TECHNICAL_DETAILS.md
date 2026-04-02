# SAGE 技术实现细节

核心数据结构、算法和关键实现。

---

## 1. 核心数据结构

### 1.1 Reasoning Graph

```python
class ReasoningGraph:
    """推理图：缓存历史查询和推理路径"""
    
    def __init__(self):
        self.queries = {}      # query_id → QueryNode
        self.paths = {}        # path_id → PathNode
        self.index = {}        # concept → [query_ids]
        self.llm_judge = UnifiedLLMJudge()
    
    def find_similar(self, query: str) -> Optional[PathNode]:
        """查找相似的历史路径"""
        # Step 1: 提取概念
        concepts = extract_concepts(query)
        
        # Step 2: 基于概念查找候选
        candidates = self._find_candidates_by_concepts(concepts)
        
        # Step 3: LLM 判断相似度
        for cand_id in candidates:
            cand_query = self.queries[cand_id].text
            
            judgment = self.llm_judge.similarity_judge(
                query1=query,
                query2=cand_query
            )
            
            if judgment == "SIMILAR":
                path_id = self._find_path_by_query(cand_id)
                return self.paths[path_id]
        
        return None
    
    def instantiate_path(self, path: PathNode, new_query: str) -> List[Hop]:
        """实例化路径到新查询（实体替换）"""
        old_entities = extract_entities(path.original_query)
        new_entities = extract_entities(new_query)
        entity_mapping = match_entities(old_entities, new_entities)
        
        new_hops = []
        for hop in path.hops:
            new_hop_query = hop.query
            for old_entity, new_entity in entity_mapping.items():
                new_hop_query = new_hop_query.replace(old_entity, new_entity)
            
            new_hops.append(Hop(
                tool=hop.tool,
                query=new_hop_query,
                purpose=hop.purpose
            ))
        
        return new_hops
```

### 1.2 Semantic Tool Cache

```python
class SemanticToolCache:
    """语义工具缓存，使用 LLM 判断相似度"""
    
    def __init__(self, llm_judge: UnifiedLLMJudge):
        self.cache = {}  # cache_key → CacheEntry
        self.llm_judge = llm_judge
    
    def get(self, tool: str, query: str, context: dict) -> Optional[dict]:
        """查询缓存"""
        # Step 1: 规范化查询
        normalized = self._normalize_query(query)
        
        # Step 2: 查找候选
        candidates = self._find_candidates(tool, normalized)
        
        if not candidates:
            return None
        
        # Step 3: LLM 判断相似度
        for cache_entry in candidates[:3]:
            if self._is_expired(cache_entry):
                continue
            
            judgment = self.llm_judge.tool_cache_similarity(
                cached_tool=tool,
                cached_query=cache_entry.query,
                cached_context=cache_entry.context,
                current_tool=tool,
                current_query=query,
                current_context=context
            )
            
            if judgment == "REUSABLE":
                cache_entry.access_count += 1
                cache_entry.last_access = now()
                return cache_entry.result
        
        return None
    
    def put(self, tool: str, query: str, context: dict, 
            result: dict, ttl: int):
        """添加到缓存"""
        cache_key = self._generate_key(tool, query)
        compressed = self._compress_result(result)
        
        self.cache[cache_key] = CacheEntry(
            tool=tool,
            query=query,
            context=context,
            result=result,
            compressed_summary=compressed,
            timestamp=now(),
            ttl=ttl,
            access_count=0
        )
```

### 1.3 Speculation Cache（简单复用）

```python
# 全局缓存 (Stage 2 → Execution 复用)
speculation_cache = {}

def stage2_save_speculation(hop_id: str, prediction: str):
    """Stage 2: 保存 speculation"""
    speculation_cache[hop_id] = prediction

def get_cached_speculation(hop_id: str) -> Optional[str]:
    """执行时: 获取缓存的 speculation"""
    return speculation_cache.get(hop_id)
```

### 1.4 数据类型定义

```python
@dataclass
class QueryNode:
    text: str
    concepts: List[str]
    metadata: dict
    timestamp: datetime
    access_count: int = 0

@dataclass
class PathNode:
    query_id: str
    hops: List[Hop]
    original_query: str
    success_rate: float = 1.0
    avg_latency: float = 0.0
    execution_count: int = 0

@dataclass
class Hop:
    tool: str
    query: str
    purpose: str
    parent_query: str
    params: dict = None
    dependencies: List[int] = None

@dataclass
class CacheEntry:
    tool: str
    query: str
    context: dict
    result: dict
    compressed_summary: dict
    timestamp: datetime
    ttl: int
    access_count: int = 0
    last_access: datetime = None

@dataclass
class JudgeResult:
    decision: str
    confidence: float
    score: int = None
    reasoning: str = None
    latency: float = 0.0
```

---

## 2. UnifiedLLMJudge 实现

```python
class UnifiedLLMJudge:
    """统一的 LLM Judge，处理所有判断任务"""
    
    def __init__(self, model_name: str = "gpt2-small"):
        self.model = load_model(model_name)
        self.tokenizer = load_tokenizer(model_name)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1. 时效性检测
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def temporal_detect(self, query: str) -> str:
        """判断查询的时效性类型"""
        prompt = f"""
        Query: "{query}"
        
        Is this query asking for:
        REALTIME: Real-time/breaking information (e.g., sports scores, stock prices)
        RECENT: Recent but not real-time (e.g., last month's events)
        STATIC: Timeless information (e.g., concepts, principles)
        
        Output (one word): REALTIME / RECENT / STATIC
        
        Type:"""
        
        output = self._generate(prompt, max_tokens=1)
        return output.strip().upper()
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 2. 查询相似度判断
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def similarity_judge(self, query1: str, query2: str) -> str:
        """判断两个查询是否相似"""
        prompt = f"""
        Query 1: "{query1}"
        Query 2: "{query2}"
        
        Are they asking for the same type of information?
        Output (one word): SIMILAR / DIFFERENT
        
        Answer:"""
        
        output = self._generate(prompt, max_tokens=1)
        return output.strip().upper()
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 3. Hop Skip 判断 (Stage 1)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def hop_skip_judge(self, hop: Hop, context: List[dict]) -> str:
        """判断 hop 是否可以跳过"""
        prompt = f"""
        Previous results:
        {format_context(context)}
        
        Current hop: {hop.query}
        
        Can we skip this hop?
        SKIP: Previous results already contain this information
        EXECUTE: We need to execute this hop
        UNCERTAIN: Not sure
        
        Output (one word): SKIP / EXECUTE / UNCERTAIN
        
        Decision:"""
        
        output, logprobs = self._generate_with_logprobs(prompt, max_tokens=1)
        decision = output.strip().upper()
        
        # 使用 logprobs 判断置信度
        if logprobs[0] < -1.0:
            return "UNCERTAIN"
        
        return decision
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 4. Hop 重要性评分 (Stage 2)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def hop_importance_scoring(self, uncertain_hops: List[Hop], 
                               context: List[dict]) -> dict:
        """为 uncertain hops 评分 + 生成 speculation"""
        prompt = f"""
        User query: "{context['user_query']}"
        
        Previous results:
        {format_context(context['prev_hops'])}
        
        Hops to evaluate:
        {format_hops(uncertain_hops)}
        
        Task:
        1. Generate speculative answer to the user query
        2. For each hop, predict what it would find
        3. Rate importance (0-100)
        
        Output format:
        Speculative Answer: [answer]
        
        Hop 1 Prediction: [prediction]
        Hop 1 Importance: [score]
        
        ...
        
        Response:"""
        
        output = self._generate(prompt, max_tokens=500)
        return self._parse_speculation(output)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 5. 结果验证
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def result_verify(self, api_result: dict, 
                      speculation_result: str, 
                      threshold: int = 75) -> dict:
        """验证 speculation 是否足够好"""
        prompt = f"""
        API Result:
        {api_result['snippet']}
        
        Speculation:
        {speculation_result}
        
        Rate consistency (0-100):
        
        Score:"""
        
        output = self._generate(prompt, max_tokens=3)
        score = self._parse_score(output)
        
        return {
            "score": score,
            "use_speculation": score >= threshold,
            "latency": self.last_latency
        }
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 6. Tool Cache 相似度
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def tool_cache_similarity(self, cached_tool: str, cached_query: str,
                              cached_context: dict, current_tool: str,
                              current_query: str, current_context: dict) -> str:
        """判断缓存是否可复用"""
        prompt = f"""
        Cached: {cached_tool}("{cached_query}")
        Current: {current_tool}("{current_query}")
        
        Can we reuse cached result?
        Output (one word): REUSABLE / NOT_REUSABLE
        
        Answer:"""
        
        output = self._generate(prompt, max_tokens=1)
        return output.strip().upper()
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 辅助方法
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def _generate(self, prompt: str, max_tokens: int) -> str:
        """生成输出"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text[len(prompt):].strip()
    
    def _generate_with_logprobs(self, prompt: str, max_tokens: int) -> Tuple[str, List[float]]:
        """生成输出 + logprobs"""
        # 实现略
        pass
    
    def _parse_score(self, text: str) -> int:
        """解析分数"""
        import re
        numbers = re.findall(r'\d+', text)
        if numbers:
            score = int(numbers[0])
            if 0 <= score <= 100:
                return score
        return 50  # 默认中等分数
    
    def _parse_speculation(self, text: str) -> dict:
        """解析 speculation 输出"""
        # 实现略
        pass
```

---

## 3. 并行执行

### 3.1 Speculative Executor

```python
class SpeculativeExecutor:
    """推测执行引擎"""
    
    def __init__(self, llm_judge: UnifiedLLMJudge, 
                 local_model: LocalModel):
        self.llm_judge = llm_judge
        self.local_model = local_model
        self.threshold = 75
    
    async def execute_hop(self, hop: Hop, context: List[dict]) -> dict:
        """执行单个 hop，带推测"""
        
        # Step 1: 检查 speculation cache
        cached_spec = speculation_cache.get(hop.id)
        
        # Step 2: 并行启动 API 和 speculation
        api_task = asyncio.create_task(self._execute_api(hop))
        
        if cached_spec:
            # 使用缓存的 speculation
            spec_result = cached_spec
        else:
            # 生成新的 speculation
            spec_task = asyncio.create_task(
                self._local_speculation(hop, context)
            )
            spec_result = await spec_task
        
        # Step 3: 等待 API
        api_result = await api_task
        
        # Step 4: LLM Judge 验证
        judge_result = await self.llm_judge.result_verify(
            api_result=api_result,
            speculation_result=spec_result,
            threshold=self.threshold
        )
        
        # Step 5: 决策
        if judge_result["use_speculation"]:
            return {
                "result": spec_result,
                "method": "speculation",
                "score": judge_result["score"]
            }
        else:
            # 使用 API + reasoning
            final_result = await self._regenerate(api_result, context)
            return {
                "result": final_result,
                "method": "regenerated",
                "score": judge_result["score"]
            }
    
    async def _execute_api(self, hop: Hop) -> dict:
        """执行 API 调用"""
        result = await api_client.call(
            tool=hop.tool,
            query=hop.query,
            params=hop.params
        )
        return result
    
    async def _local_speculation(self, hop: Hop, context: List[dict]) -> str:
        """本地推测"""
        prompt = f"""
        Based on previous results:
        {format_context(context)}
        
        Predict what this search would find:
        "{hop.query}"
        
        Prediction:"""
        
        result = await self.local_model.generate(prompt, max_tokens=200)
        return result.strip()
    
    async def _regenerate(self, api_result: dict, context: List[dict]) -> str:
        """基于 API 结果重新生成推理"""
        prompt = f"""
        Search result: {api_result}
        Context: {format_context(context)}
        Generate reasoning:
        """
        result = await self.local_model.generate(prompt)
        return result.strip()
```

### 3.2 Prefetching（可选优化）

```python
class PrefetchingExecutor(SpeculativeExecutor):
    """支持预取的执行器"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefetch_cache = {}
    
    async def execute_plan(self, plan: List[Hop], context: List[dict]):
        """执行完整计划，带预取"""
        results = []
        
        for i, hop in enumerate(plan):
            # 检查预取缓存
            if hop.query in self.prefetch_cache:
                api_result = await self.prefetch_cache[hop.query]
                del self.prefetch_cache[hop.query]
            else:
                api_result = None
            
            # 执行当前 hop
            result = await self.execute_hop_with_prefetch(
                hop, context, prefetched_api=api_result
            )
            
            results.append(result)
            context.append(result)
            
            # 预取下一个 hop
            if i + 1 < len(plan):
                next_hop = plan[i + 1]
                prefetch_task = asyncio.create_task(
                    self._execute_api(next_hop)
                )
                self.prefetch_cache[next_hop.query] = prefetch_task
        
        return results
```

---

## 4. 核心算法

### 4.1 概念提取

```python
def extract_concepts(text: str) -> List[str]:
    """提取关键概念（NER + 关键词）"""
    concepts = []
    
    # 1. 命名实体识别
    entities = ner_model.extract(text)
    concepts.extend([e.text for e in entities])
    
    # 2. 关键词提取
    keywords = extract_keywords(text, top_k=5)
    concepts.extend(keywords)
    
    # 3. 意图识别
    intent = classify_intent(text)
    concepts.append(f"intent:{intent}")
    
    # 去重和规范化
    concepts = list(set([c.lower() for c in concepts]))
    return concepts

def extract_keywords(text: str, top_k: int = 5) -> List[str]:
    """基于 TF-IDF 提取关键词"""
    tokens = tokenize(text)
    tokens = [t for t in tokens if t not in STOP_WORDS]
    tf = Counter(tokens)
    return [word for word, _ in tf.most_common(top_k)]

def classify_intent(text: str) -> str:
    """分类查询意图"""
    text_lower = text.lower()
    if any(w in text_lower for w in ["分析", "analyze"]):
        return "analyze"
    elif any(w in text_lower for w in ["对比", "compare"]):
        return "compare"
    elif any(w in text_lower for w in ["解释", "explain"]):
        return "explain"
    else:
        return "general"
```

### 4.2 实体匹配

```python
def match_entities(old_entities: List[str], 
                   new_entities: List[str]) -> dict:
    """在两个实体列表之间建立映射（用于路径实例化）"""
    mapping = {}
    
    old_by_type = group_by_type(old_entities)
    new_by_type = group_by_type(new_entities)
    
    for entity_type, old_list in old_by_type.items():
        if entity_type in new_by_type:
            new_list = new_by_type[entity_type]
            for i, old_entity in enumerate(old_list):
                if i < len(new_list):
                    mapping[old_entity] = new_list[i]
    
    return mapping

def group_by_type(entities: List[str]) -> dict:
    """按类型分组实体"""
    grouped = {}
    for entity in entities:
        entity_type = ner_model.classify(entity)
        if entity_type not in grouped:
            grouped[entity_type] = []
        grouped[entity_type].append(entity)
    return grouped
```

---

## 5. 缓存优化

### 5.1 动态 TTL

```python
def calculate_dynamic_ttl(query_type: str, update_history: List[datetime]) -> int:
    """计算动态 TTL"""
    # Base TTL
    base_ttl = {
        "REALTIME": 0,        # 不缓存
        "RECENT": 259200,     # 3 天
        "STATIC": 7776000     # 90 天
    }[query_type]
    
    if query_type == "REALTIME":
        return 0
    
    # Freshness factor
    if not update_history:
        freshness_factor = 1.0
    else:
        intervals = [
            (update_history[i+1] - update_history[i]).days
            for i in range(len(update_history) - 1)
        ]
        avg_interval = sum(intervals) / len(intervals)
        
        if avg_interval < 7:
            freshness_factor = 0.2
        elif avg_interval < 30:
            freshness_factor = 0.5
        else:
            freshness_factor = 1.0
    
    ttl = int(base_ttl * freshness_factor)
    return max(ttl, 3600)  # 最少 1 小时
```

### 5.2 结果压缩

```python
def compress_search_result(result: dict, max_length: int = 500) -> dict:
    """压缩搜索结果"""
    key_facts = extract_key_facts(result["snippet"])
    entities = extract_entities(result["snippet"])
    summary = summarize_text(result["snippet"], max_length=max_length)
    
    return {
        "key_facts": key_facts[:5],
        "main_entities": entities[:10],
        "summary": summary,
        "metadata": {
            "title": result.get("title", "")[:100],
            "url": result.get("url", ""),
            "timestamp": result.get("timestamp")
        }
    }
```

### 5.3 缓存淘汰策略

```python
MAX_CACHE_SIZE = 10000

def eviction_policy(cache: dict) -> None:
    """LRU + 访问频率混合淘汰"""
    if len(cache) <= MAX_CACHE_SIZE:
        return
    
    scores = {}
    for key, entry in cache.items():
        lru_score = (now() - entry.last_access).total_seconds()
        freq_score = entry.access_count
        scores[key] = freq_score * 1000 - lru_score
    
    sorted_keys = sorted(scores.items(), key=lambda x: x[1])
    to_evict = sorted_keys[:int(MAX_CACHE_SIZE * 0.1)]
    
    for key, _ in to_evict:
        del cache[key]
```

---

## 6. 错误处理

### 6.1 LLM Judge 失败处理

```python
async def robust_llm_judge(prompt: str, max_retries: int = 2) -> dict:
    """鲁棒的 LLM Judge，带重试和 fallback"""
    for attempt in range(max_retries):
        try:
            result = await llm_judge.judge(prompt)
            if validate_output(result):
                return result
        except Exception as e:
            logger.error(f"LLM Judge error: {e}")
            if attempt == max_retries - 1:
                break
            continue
    
    # Fallback 到保守策略
    logger.error("LLM Judge failed, using fallback")
    return {
        "decision": "EXECUTE",  # 保守
        "confidence": 0.0,
        "fallback": True
    }

def validate_output(result: dict) -> bool:
    """验证 LLM 输出格式"""
    required_fields = ["decision", "confidence"]
    for field in required_fields:
        if field not in result:
            return False
    if result["decision"] not in VALID_DECISIONS:
        return False
    if not (0.0 <= result["confidence"] <= 1.0):
        return False
    return True
```

### 6.2 API 调用失败处理

```python
async def robust_api_call(hop: Hop, max_retries: int = 1) -> dict:
    """鲁棒的 API 调用，带重试"""
    for attempt in range(max_retries + 1):
        try:
            result = await api_client.call(
                tool=hop.tool,
                query=hop.query,
                timeout=5.0
            )
            if result and "error" not in result:
                return result
        except TimeoutError:
            logger.warning(f"API timeout, attempt {attempt+1}")
            continue
        except Exception as e:
            logger.error(f"API error: {e}")
            break
    
    # API 失败，标记为降级
    logger.warning("API failed, attempting fallback")
    return {
        "error": "API_FAILED",
        "fallback_to_speculation": True
    }
```

---

## 7. 配置管理

```python
SAGE_CONFIG = {
    # LLM Judge 配置
    "judge": {
        "model": "gpt2-small",
        "quantize": True,
        "batch_size": 8,
        "max_tokens": 10,
        "temperature": 0.3
    },
    
    # 阈值配置
    "thresholds": {
        "verification": 75,      # 验证阈值
        "pruning": 60,           # 剪枝阈值 (Stage 2)
        "confidence": 0.8        # 置信度阈值 (Stage 1)
    },
    
    # 缓存配置
    "cache": {
        "max_size": 10000,
        "ttl": {
            "REALTIME": 0,
            "RECENT": 259200,
            "STATIC": 7776000
        },
        "compression": True
    },
    
    # 执行配置
    "execution": {
        "enable_prefetch": True,
        "enable_speculation": True,
        "parallel_hops": True,
        "max_concurrent_apis": 5
    }
}
```

---

## 8. 关键技术决策

### 8.1 为什么用 GPT-2 Small 作为 Judge？

- 模型小 (124M) → 延迟低
- 足够的理解能力 → 判断任务相对简单
- 开源可控 → 无 API 成本

### 8.2 为什么用 LLM 而非 Embedding？

Embedding 问题：
- 阈值难调
- 语义理解有限（"EAGLE deployment" vs "Medusa deployment" 相似度高但实体不同）
- 无法理解上下文

LLM 优势：
- 自然理解语义
- 上下文感知
- 灵活可调（改 prompt 即可）
- 统一接口

---

**文档版本**: v2.0  
**最后更新**: 2026-03-16
