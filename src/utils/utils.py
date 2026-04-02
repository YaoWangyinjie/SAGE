"""
SAGE Utils — 工具函数集合。

包含：
- extract_concepts：NER + TF-IDF 关键词提取，供图匹配使用
- match_entities：占位符到实体的类型对齐
- compress_result：结果压缩摘要
- setup_logging：日志初始化
- make_hops_description：供依赖推断 prompt 使用
"""

import logging
import os
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================
# 日志初始化
# ============================================================

def setup_logging(config) -> None:
    """根据 config 初始化全局日志配置"""
    os.makedirs(config.LOG_DIR, exist_ok=True)
    log_path = os.path.join(config.LOG_DIR, config.LOG_FILE)

    handlers = [logging.StreamHandler()]
    try:
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    except Exception as e:
        print(f"[Utils] 无法创建日志文件 {log_path}: {e}")

    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    logger.info(f"[Utils] 日志初始化完成，日志级别={config.LOG_LEVEL}")


# ============================================================
# 文本处理
# ============================================================

# 常见中英文停用词（轻量版）
_STOPWORDS = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都",
    "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你",
    "会", "着", "没有", "看", "好", "自己", "这",
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "need", "dare", "ought", "used", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "up", "about", "into",
    "through", "during", "including", "until", "against", "and",
    "or", "but", "if", "as", "what", "which", "who", "this", "that",
}


def extract_concepts(text: str) -> List[str]:
    """
    从文本中提取关键概念。
    策略：简单分词 → 去停用词 → 按长度和频率排序。
    扩展点：可替换为 jieba + TF-IDF 或 spaCy NER。

    Returns: 关键词列表（降序排列）
    """
    # 简单分词：按空格 / 标点分割
    tokens = re.split(r"[\s，。！？、；：""''【】\(\)\[\]{},./;:'\"\-]+", text)
    # 过滤停用词和长度 < 2 的词
    concepts = [
        t.strip()
        for t in tokens
        if t.strip() and t.strip().lower() not in _STOPWORDS and len(t.strip()) >= 2
    ]
    # 去重保序
    seen = set()
    unique = []
    for c in concepts:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def compress_result(result: str, max_chars: int) -> str:
    """截断并压缩工具结果"""
    if len(result) <= max_chars:
        return result
    return result[:max_chars] + "…（已截断）"


def make_hops_description(hops) -> str:
    """
    生成所有 hop 的结构化文本描述，供依赖推断 prompt 使用。
    """
    lines = []
    for hop in hops:
        deps_str = ", ".join(hop.dependencies) if hop.dependencies else "无"
        lines.append(
            f"[{hop.hop_id}]\n"
            f"  问题: {hop.query}\n"
            f"  工具: {hop.tool_name}\n"
            f"  已知依赖: {deps_str}"
        )
    return "\n\n".join(lines)


# ============================================================
# 实体类型匹配
# ============================================================

# 简单实体类型规则（可扩展为模型驱动）
_ENTITY_TYPE_PATTERNS: List[Tuple[str, List[str]]] = [
    ("PAPER",  [r"[A-Z]{2,}(?:-\w+)?",           # 全大写缩写如 EAGLE, LLAMA
                r"\w+\s+\d+(\.\d+)?"]),            # 带版本号如 LLaMA 2
    ("AUTHOR", [r"[A-Z][a-z]+\s+[A-Z][a-z]+"]),   # 姓名如 John Smith
    ("ORG",    [r"(Google|Meta|OpenAI|Baidu|Microsoft|Apple|Amazon|NVIDIA)"]),
    ("DATE",   [r"\d{4}年?|\d{4}-\d{2}-\d{2}"]),
    ("NUMBER", [r"\d+[\.,]?\d*%?[KMBT]?"]),
]


def infer_entity_type(entity_value: str) -> str:
    """
    根据实体值推断其类型（粗糙规则匹配）。
    Returns: 类型字符串如 "PAPER" | "AUTHOR" | "ORG" | "DATE" | "NUMBER" | "UNKNOWN"
    """
    for type_name, patterns in _ENTITY_TYPE_PATTERNS:
        for pat in patterns:
            if re.search(pat, entity_value):
                return type_name
    return "UNKNOWN"


def match_entities(
    new_query_entities: List[str],
    placeholder_types: List[str],
) -> Dict[str, str]:
    """
    将从新查询中提取的实体按类型对齐到占位符。
    返回 {占位符名: 实体值} 映射。

    Parameters
    ----------
    new_query_entities : 从新查询中提取的实体值列表
    placeholder_types  : 占位符类型列表（如 ["PAPER", "ASPECT"]）
    """
    mapping: Dict[str, str] = {}
    used = set()

    for ph_type in placeholder_types:
        for entity in new_query_entities:
            if entity in used:
                continue
            inferred = infer_entity_type(entity)
            if inferred == ph_type or ph_type == "ENTITY":
                mapping[ph_type] = entity
                used.add(entity)
                break

    return mapping


# ============================================================
# 计划格式化（调试 / 日志输出）
# ============================================================

def format_plan(plan) -> str:
    """将 HopPlan 格式化为可读字符串（调试用）"""
    lines = [f"=== HopPlan: {plan.original_query[:60]} ==="]
    for hop in plan.hops:
        status = "SKIP" if hop.skipped else "EXEC"
        result_preview = (hop.result or "")[:80].replace("\n", " ")
        lines.append(
            f"  [{status}] {hop.hop_id}: {hop.query[:50]}"
            + (f" → {result_preview}..." if result_preview else "")
        )
    return "\n".join(lines)
