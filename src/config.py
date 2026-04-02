"""
SAGE Configuration — 所有超参数与常量集中管理。
修改此文件即可调整系统行为，无需触碰业务逻辑。
"""

# ============================================================
# 1. 模型配置 (Model Configuration)
# ============================================================

# 本地小模型（Speculative Model），负责投机采样 / 快速判断
# 支持任意 HuggingFace 模型 ID 或本地路径
LOCAL_MODEL_ID = "gpt2"               # 替换为量化后的小模型，如 "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LOCAL_MODEL_QUANTIZED = False         # 是否使用 4-bit / 8-bit 量化（需要 bitsandbytes）
LOCAL_MODEL_DEVICE = "cpu"            # "cuda" | "mps" | "cpu"
LOCAL_MODEL_MAX_NEW_TOKENS_FAST = 10  # Stage1 快速判断最大生成 token 数
LOCAL_MODEL_MAX_NEW_TOKENS_DEEP = 500 # Stage2 深度评分最大生成 token 数
LOCAL_MODEL_TEMPERATURE = 0.3         # 生成温度（越低越确定）
LOCAL_MODEL_BATCH_SIZE = 8            # Stage2 批处理大小

# 大模型 API（Oracle Model），负责工具调用 & 最终推理
ORACLE_API_BASE = "https://api.openai.com/v1"   # OpenAI 兼容接口 base URL
ORACLE_API_KEY_ENV = "OPENAI_API_KEY"            # 从环境变量读取 API Key
ORACLE_MODEL_NAME = "gpt-4o-mini"                # 替换为实际使用的模型
ORACLE_MAX_TOKENS = 1024                         # Oracle 单次最大回复 tokens
ORACLE_TEMPERATURE = 0.2                         # Oracle 生成温度
ORACLE_TIMEOUT_SECONDS = 30                      # API 请求超时（秒）
ORACLE_MAX_RETRIES = 1                           # API 失败最大重试次数

# ============================================================
# 2. 判断阈值 (Judge Thresholds)
# ============================================================

# 投机结果验证分数阈值：Judge 评分 >= 此值时采用本地模型结果
VERIFY_SCORE_THRESHOLD = 75          # 0–100 分制

# Stage2 重要性分数阈值：低于此值的 hop 被剪枝
IMPORTANCE_SCORE_THRESHOLD = 60      # 0–100 分制

# Stage1 置信度阈值：logprob 置信度低于此值时送入 Stage2
STAGE1_CONFIDENCE_THRESHOLD = 0.8   # 0–1，越高越严格

# 依赖边强度阈值：强度 >= 此值视为"强数据依赖"，触发级联剪枝
DEPENDENCY_CASCADE_THRESHOLD = 0.7  # 0–1

# 图匹配层级阈值
GRAPH_QUERY_SIMILARITY_THRESHOLD = 0.7   # Layer1：查询语义相似度
GRAPH_STRUCTURAL_MATCH_THRESHOLD = 0.7   # Layer2：LCS 结构匹配度
GRAPH_HOP_SIMILARITY_THRESHOLD = 0.8     # Layer3：hop 级语义相似度

# 工具缓存相似度判断：LLM 判断 REUSABLE / NOT_REUSABLE 的内部参考
TOOL_CACHE_SIMILARITY_HINT = 0.85   # 仅用于 prompt 参考，不做数值比较

# ============================================================
# 3. 缓存配置 (Cache Configuration)
# ============================================================

# 推测缓存（speculation cache）：保存 Stage2 生成的预测，供执行阶段复用
SPEC_CACHE_MAX_SIZE = 1000          # 最多缓存多少条 hop 预测

# 工具结果语义缓存（semantic tool cache）
TOOL_CACHE_MAX_SIZE = 10000         # 缓存条目上限，超出时触发 LRU 淘汰
TOOL_CACHE_EVICT_RATIO = 0.1        # 淘汰比例（每次淘汰 10%）

# 动态 TTL（秒）：根据 LLM 判断的查询时效性设置缓存过期时间
TTL_REALTIME = 0                    # 实时查询：不缓存
TTL_RECENT = 3 * 24 * 3600         # 近期查询：3 天
TTL_STATIC = 90 * 24 * 3600        # 静态查询：90 天

# 缓存结果压缩：摘要最大字符数
CACHE_SUMMARY_MAX_CHARS = 500

# ============================================================
# 4. 推理图配置 (Reasoning Graph Configuration)
# ============================================================

# 图数据库持久化路径（JSON 格式）
GRAPH_DB_PATH = "data/graphs/reasoning_graph.json"

# 实体占位符格式
ENTITY_PLACEHOLDER_FORMAT = "{{{}}}"   # 如 {ENTITY}, {PAPER}, {AUTHOR}

# LCS 匹配时允许的最大 hop 数量差异
MAX_HOP_COUNT_DIFF = 2

# ============================================================
# 5. 执行配置 (Execution Configuration)
# ============================================================

# 是否启用预取（look-ahead prefetch）：执行 hop_i 时预取 hop_{i+1} 的 API 调用
ENABLE_PREFETCH = True

# 是否启用并行投机执行（API 调用 & 本地推测同时跑）
ENABLE_PARALLEL_SPECULATION = True

# hop 执行超时（秒）：单个 hop 超时后标记为失败
HOP_EXECUTION_TIMEOUT = 60

# ============================================================
# 6. 日志配置 (Logging Configuration)
# ============================================================

LOG_LEVEL = "INFO"                  # DEBUG | INFO | WARNING | ERROR
LOG_DIR = "data/logs"
LOG_FILE = "sage.log"
ENABLE_PROMPT_LOGGING = True        # 记录每次 LLM 的 prompt 和输出（调试用）

# ============================================================
# 7. 杂项 (Miscellaneous)
# ============================================================

# 依赖关系离线学习：从执行日志中统计依赖，低于此次数视为噪声
MIN_EXECUTION_COUNT_FOR_DEPENDENCY = 5

# 最终合成回答时最大 tokens
SYNTHESIS_MAX_TOKENS = 2048

# 系统默认语言提示（用于 prompt 构造）
SYSTEM_LANGUAGE = "zh"              # "zh" | "en"
