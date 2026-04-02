"""
SAGE 核心数据结构定义。
所有模块共享此文件中的 dataclass，确保类型一致性。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ============================================================
# 枚举类型 (Enums)
# ============================================================

class TemporalType(str, Enum):
    """查询时效性分类"""
    REALTIME = "REALTIME"   # 实时数据，不缓存
    RECENT   = "RECENT"     # 近期数据，短期缓存
    STATIC   = "STATIC"     # 静态数据，长期缓存


class Stage1Decision(str, Enum):
    """Stage1 快速剪枝判断结果"""
    SKIP      = "SKIP"       # 直接跳过该 hop
    EXECUTE   = "EXECUTE"    # 必须执行
    UNCERTAIN = "UNCERTAIN"  # 不确定，送入 Stage2 精判


class SimilarityDecision(str, Enum):
    """查询 / 工具结果语义相似度判断"""
    SIMILAR   = "SIMILAR"
    DIFFERENT = "DIFFERENT"


class VerifyDecision(str, Enum):
    """投机结果验证决策"""
    USE_SPECULATION = "USE_SPECULATION"
    USE_API         = "USE_API"


# ============================================================
# Hop（推理步骤）相关
# ============================================================

@dataclass
class Hop:
    """
    一个 hop 表示 multi-hop 推理链中的一个步骤。
    """
    hop_id: str                          # 唯一标识，如 "hop_1"
    query: str                           # 该步骤需要回答的子问题
    tool_name: str                       # 调用的工具名称，如 "web_search"
    tool_args: Dict[str, Any]            # 工具调用参数
    dependencies: List[str] = field(default_factory=list)
    # ↑ 依赖的 hop_id 列表（数据依赖 or 语义依赖）

    # 执行结果（执行后填充）
    result: Optional[str] = None
    result_summary: Optional[str] = None  # 压缩摘要
    execution_time_ms: Optional[float] = None
    skipped: bool = False
    fallback_to_speculation: bool = False  # API 失败后退回投机结果


@dataclass
class HopPlan:
    """
    完整的 hop 执行计划（pruning 前 / 后均用此结构）。
    """
    original_query: str
    hops: List[Hop]
    created_at: float = field(default_factory=time.time)

    def get_hop(self, hop_id: str) -> Optional[Hop]:
        for h in self.hops:
            if h.hop_id == hop_id:
                return h
        return None

    def active_hops(self) -> List[Hop]:
        return [h for h in self.hops if not h.skipped]


# ============================================================
# Judge 相关
# ============================================================

@dataclass
class JudgeResult:
    """LLM Judge 的通用返回结构"""
    decision: str                     # 枚举值字符串
    confidence: float = 1.0          # 置信度 0–1（基于 logprob 或 fallback）
    score: Optional[int] = None      # 数值评分 0–100（验证/重要性场景）
    reasoning: Optional[str] = None  # 可选的推理解释
    speculation: Optional[str] = None  # Stage2 生成的预测答案
    raw_output: Optional[str] = None   # 原始模型输出（调试用）


# ============================================================
# 缓存相关
# ============================================================

@dataclass
class CacheEntry:
    """工具结果语义缓存条目"""
    key: str                    # 查询指纹（由 tool_name + args 哈希）
    result_summary: str         # 压缩后的工具结果摘要
    temporal_type: TemporalType
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: int = 0                # 有效时长（秒），0 = 不缓存

    def is_expired(self) -> bool:
        if self.ttl == 0:
            return True
        return (time.time() - self.created_at) > self.ttl

    def touch(self):
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class SpeculationCacheEntry:
    """Stage2 推测预测缓存条目（hop 级别）"""
    hop_id: str
    prediction: str           # 本地模型生成的预测答案
    importance_score: int     # 对应的重要性分数 0–100
    created_at: float = field(default_factory=time.time)


# ============================================================
# 推理图相关
# ============================================================

@dataclass
class GraphNode:
    """推理图中的节点（抽象化的 hop 意图模板）"""
    node_id: str
    intent_template: str       # 如 "获取 {ENTITY} 的部署挑战"
    query_template: str        # 如 "What are the deployment challenges of {ENTITY}?"
    tool_name: str
    tool_args_template: Dict[str, Any] = field(default_factory=dict)
    # tool_args_template 中可含 {ENTITY} 等占位符


@dataclass
class GraphEdge:
    """推理图中的有向边（hop 间依赖关系）"""
    source_id: str
    target_id: str
    dependency_type: str    # "data" | "semantic"
    strength: float         # 0–1，越高越强
    learned_count: int = 0  # 从执行日志中学习到的共现次数


@dataclass
class ReasoningPath:
    """推理图中的一条完整路径（对应一类查询模板）"""
    path_id: str
    query_template: str          # 抽象查询模板
    nodes: List[GraphNode]       # 有序 hop 节点列表
    edges: List[GraphEdge]       # 边列表
    entity_types: List[str]      # 涉及的实体类型，如 ["PAPER", "AUTHOR"]
    use_count: int = 0
    avg_success_rate: float = 1.0


@dataclass
class EntityMapping:
    """占位符到实体值的映射"""
    placeholder: str    # 如 "{ENTITY}"
    value: str          # 如 "EAGLE"
    entity_type: str    # 如 "PAPER"
