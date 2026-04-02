"""
SAGE Cache 模块 — 双层缓存系统。

1. SpeculationCache：Stage2 投机预测缓存（hop_id → prediction）
   - 极简 dict，Stage2 生成后立即写入，执行阶段 0ms 读取
   - 避免执行阶段重复生成 speculation

2. SemanticToolCache：工具结果语义缓存
   - LLM 判断相似度决定是否复用
   - 动态 TTL（REALTIME=0, RECENT=3天, STATIC=90天）
   - LRU + 访问频率混合淘汰策略
"""

import hashlib
import json
import logging
import time
from collections import OrderedDict
from typing import Any, Dict, Optional

from src.data_structures import CacheEntry, SpeculationCacheEntry, TemporalType

logger = logging.getLogger(__name__)


# ============================================================
# 1. 投机预测缓存（极简实现）
# ============================================================

class SpeculationCache:
    """
    保存 Stage2 阶段为每个 hop 生成的预测答案。
    执行阶段读取 0ms，避免重复 GPU 推断。

    生命周期：单次 query 内有效（每次查询前应 reset）。
    """

    def __init__(self, config):
        self.config = config
        self._store: Dict[str, SpeculationCacheEntry] = {}

    def save(self, hop_id: str, prediction: str, importance_score: int):
        """Stage2 评分后调用，保存预测"""
        if len(self._store) >= self.config.SPEC_CACHE_MAX_SIZE:
            # 超出上限时简单删除最旧的一条
            oldest_key = next(iter(self._store))
            del self._store[oldest_key]

        self._store[hop_id] = SpeculationCacheEntry(
            hop_id=hop_id,
            prediction=prediction,
            importance_score=importance_score,
        )
        logger.debug(f"[SpecCache] 保存 {hop_id}: score={importance_score}")

    def get(self, hop_id: str) -> Optional[str]:
        """执行阶段调用，读取缓存预测（0ms）"""
        entry = self._store.get(hop_id)
        if entry:
            logger.debug(f"[SpecCache] 命中 {hop_id}")
            return entry.prediction
        return None

    def reset(self):
        """每次新查询前清空（单查询生命周期）"""
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


# ============================================================
# 2. 工具结果语义缓存
# ============================================================

class SemanticToolCache:
    """
    工具调用结果的语义缓存。

    - 哈希键：tool_name + sorted(tool_args) → SHA256 前 16 位
    - LLM 判断相似度：tool_name + args 语义层面是否等价
    - 动态 TTL：由 LLM temporal_detect 决定缓存时长
    - 淘汰策略：LRU-ordered dict + access_count 加权评分
    """

    def __init__(self, config, judge):
        self.config = config
        self.judge = judge
        # OrderedDict 保持插入顺序，用于 LRU 淘汰
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()

    # ----------------------------------------------------------
    # 核心 API
    # ----------------------------------------------------------

    def get(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
    ) -> Optional[str]:
        """
        查询缓存。
        先精确键匹配（快速路径），再 LLM 语义相似度匹配（慢速路径）。
        返回缓存的结果摘要，或 None（未命中）。
        """
        # 快速路径：精确键匹配
        key = self._make_key(tool_name, tool_args)
        if key in self._store:
            entry = self._store[key]
            if entry.is_expired():
                del self._store[key]
                logger.debug(f"[ToolCache] 过期删除: {key[:8]}")
                return None
            entry.touch()
            self._store.move_to_end(key)  # LRU 更新
            logger.debug(f"[ToolCache] 精确命中: {key[:8]}")
            return entry.result_summary

        # 慢速路径：LLM 语义匹配（遍历非过期条目）
        now = time.time()
        for cached_key, entry in list(self._store.items()):
            if entry.is_expired():
                del self._store[cached_key]
                continue
            age_hours = (now - entry.created_at) / 3600
            decision = self.judge.tool_cache_similarity(
                new_tool_name=tool_name,
                new_tool_args=tool_args,
                cached_tool_name=entry.key.split(":")[0],   # 编码时前缀存了 tool_name
                cached_tool_args={},                          # 简化：全量语义判断
                cached_summary=entry.result_summary,
                cache_age_hours=age_hours,
            )
            if decision == "REUSABLE":
                entry.touch()
                logger.debug(f"[ToolCache] 语义命中: {cached_key[:8]}")
                return entry.result_summary

        return None

    def put(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        result: str,
        temporal_type: TemporalType,
    ):
        """将工具调用结果写入缓存"""
        if temporal_type == TemporalType.REALTIME:
            logger.debug("[ToolCache] REALTIME 查询，跳过缓存")
            return

        ttl = {
            TemporalType.RECENT: self.config.TTL_RECENT,
            TemporalType.STATIC: self.config.TTL_STATIC,
        }.get(temporal_type, self.config.TTL_STATIC)

        key = self._make_key(tool_name, tool_args)
        summary = self._compress(result)
        entry = CacheEntry(
            key=f"{tool_name}:{key}",
            result_summary=summary,
            temporal_type=temporal_type,
            ttl=ttl,
        )
        self._store[key] = entry
        self._store.move_to_end(key)

        if len(self._store) > self.config.TOOL_CACHE_MAX_SIZE:
            self._evict()

        logger.debug(f"[ToolCache] 写入: {key[:8]} ttl={ttl}s temporal={temporal_type.value}")

    # ----------------------------------------------------------
    # 工具函数
    # ----------------------------------------------------------

    @staticmethod
    def _make_key(tool_name: str, tool_args: Dict) -> str:
        """生成缓存键：tool_name + 排序后 args 的 SHA256 前 16 位"""
        serialized = json.dumps({"tool": tool_name, "args": tool_args}, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]

    def _compress(self, result: str) -> str:
        """压缩工具结果为摘要，截断到最大字符数"""
        if len(result) <= self.config.CACHE_SUMMARY_MAX_CHARS:
            return result
        return result[: self.config.CACHE_SUMMARY_MAX_CHARS] + "…（已截断）"

    def _evict(self):
        """LRU + 访问频率混合淘汰策略"""
        evict_count = max(1, int(self.config.TOOL_CACHE_MAX_SIZE * self.config.TOOL_CACHE_EVICT_RATIO))

        # 按 (last_accessed + access_count * 权重) 升序排列，淘汰分数最低的
        scored = sorted(
            self._store.items(),
            key=lambda kv: kv[1].last_accessed + kv[1].access_count * 3600,
        )
        for k, _ in scored[:evict_count]:
            del self._store[k]
        logger.info(f"[ToolCache] 淘汰 {evict_count} 条，剩余 {len(self._store)} 条")

    def stats(self) -> Dict:
        """返回缓存统计信息"""
        now = time.time()
        active = sum(1 for e in self._store.values() if not e.is_expired())
        return {
            "total_entries": len(self._store),
            "active_entries": active,
            "expired_entries": len(self._store) - active,
        }
