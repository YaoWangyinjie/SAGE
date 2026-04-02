"""
SAGE Pruner — 两阶段智能剪枝 + 依赖级联检查。

流程：
  Stage1（~20ms/hop）：快速二分类 SKIP / EXECUTE / UNCERTAIN
      → CERTAIN 直接决定；UNCERTAIN 送入 Stage2
  Stage2（~150ms/batch）：深度重要性评分 + 生成 speculation
      → score < IMPORTANCE_SCORE_THRESHOLD → SKIP；同时写入 SpeculationCache
  依赖级联检查：被 SKIP 的 hop 的强数据依赖 hop 也一起 SKIP，迭代直到收敛

设计约定：
  - 第一个 hop（hop_1）始终 EXECUTE（不可跳过）
  - 有其他 hop 依赖它的 hop 始终 EXECUTE（依赖守护）
  - Stage2 的 speculation 写入 SpeculationCache 供执行阶段复用
"""

import logging
from typing import Dict, List, Set

from src.data_structures import Hop, HopPlan, Stage1Decision
from src.judge.llm_judge import robust_llm_judge

logger = logging.getLogger(__name__)


class TwoStagePruner:
    """
    两阶段剪枝器。
    """

    def __init__(self, config, judge, spec_cache):
        """
        Parameters
        ----------
        config     : src.config 模块
        judge      : UnifiedLLMJudge
        spec_cache : SpeculationCache
        """
        self.config = config
        self.judge = judge
        self.spec_cache = spec_cache

    # ----------------------------------------------------------
    # 主入口
    # ----------------------------------------------------------

    def prune(self, plan: HopPlan) -> HopPlan:
        """
        对 HopPlan 执行完整的两阶段剪枝 + 依赖级联检查。
        修改 hop.skipped 字段（in-place），返回同一个 plan 对象。
        """
        hops = plan.hops
        if not hops:
            return plan

        # 预计算：哪些 hop 有其他 hop 依赖它（被依赖集合）
        depended_by: Set[str] = self._compute_depended_set(hops)

        # 初始化所有为 EXECUTE
        skip_map: Dict[str, bool] = {h.hop_id: False for h in hops}

        # ---- Stage 1 ----
        uncertain_hops: List[Hop] = []
        context_summary = ""  # 逐步积累（此阶段暂无执行结果，用空串）

        for hop in hops:
            # 守护规则：第一个 hop 或被其他 hop 依赖的 hop → 强制 EXECUTE
            if hop.hop_id == hops[0].hop_id or hop.hop_id in depended_by:
                logger.debug(f"[Pruner.S1] {hop.hop_id} 被守护，强制 EXECUTE")
                continue

            decision, confidence = robust_llm_judge(
                self.judge.hop_skip_judge,
                plan.original_query,
                hop.hop_id,
                hop.query,
                context_summary,
                fallback=("UNCERTAIN", 0.5),
            )
            logger.debug(
                f"[Pruner.S1] {hop.hop_id}: decision={decision} conf={confidence:.2f}"
            )

            if decision == Stage1Decision.SKIP and confidence >= self.config.STAGE1_CONFIDENCE_THRESHOLD:
                skip_map[hop.hop_id] = True
            elif decision == Stage1Decision.EXECUTE and confidence >= self.config.STAGE1_CONFIDENCE_THRESHOLD:
                pass  # 保持 EXECUTE
            else:
                uncertain_hops.append(hop)

        # ---- Stage 2 ----
        if uncertain_hops:
            self._stage2(plan, uncertain_hops, skip_map, context_summary)

        # ---- 依赖级联检查 ----
        self._cascade_prune(hops, skip_map)

        # 写回 hop.skipped
        for hop in hops:
            hop.skipped = skip_map[hop.hop_id]

        skipped_ids = [h.hop_id for h in hops if h.skipped]
        active_ids = [h.hop_id for h in hops if not h.skipped]
        logger.info(
            f"[Pruner] 剪枝完成: 保留 {len(active_ids)} hop{active_ids}，"
            f"跳过 {len(skipped_ids)} hop{skipped_ids}"
        )
        return plan

    # ----------------------------------------------------------
    # Stage 2 深度评分
    # ----------------------------------------------------------

    def _stage2(
        self,
        plan: HopPlan,
        uncertain_hops: List[Hop],
        skip_map: Dict[str, bool],
        context_summary: str,
    ):
        """批量评分 uncertain hops，并将 speculation 写入缓存"""
        all_hops_summary = self._make_hops_summary(plan.hops)

        results = robust_llm_judge(
            self.judge.hop_importance_scoring,
            plan.original_query,
            uncertain_hops,
            context_summary,
            all_hops_summary,
            fallback=[],
        )
        if not results:
            logger.warning("[Pruner.S2] 批量评分失败，保守保留所有 uncertain hops")
            return

        for item in results:
            hop_id = item["hop_id"]
            score = item.get("score", 70)
            speculation = item.get("speculation") or item.get("prediction", "")

            # 写入 SpeculationCache（Stage2 副作用，供执行阶段复用）
            if speculation:
                self.spec_cache.save(hop_id, speculation, score)

            if score < self.config.IMPORTANCE_SCORE_THRESHOLD:
                skip_map[hop_id] = True
                logger.debug(f"[Pruner.S2] {hop_id} score={score} → SKIP")
            else:
                logger.debug(f"[Pruner.S2] {hop_id} score={score} → EXECUTE")

    # ----------------------------------------------------------
    # 依赖级联剪枝
    # ----------------------------------------------------------

    def _cascade_prune(self, hops: List[Hop], skip_map: Dict[str, bool]):
        """
        若 hop A 被跳过，且 hop B 对 A 有强数据依赖，则 B 也被跳过。
        迭代直到无新变化（收敛）。
        """
        changed = True
        while changed:
            changed = False
            for hop in hops:
                if skip_map.get(hop.hop_id, False):
                    continue  # 已跳过
                for dep_id in hop.dependencies:
                    if not skip_map.get(dep_id, False):
                        continue  # 依赖未被跳过
                    # 依赖被跳过了，检查边强度
                    strength = self._get_dependency_strength(hops, dep_id, hop.hop_id)
                    if strength >= self.config.DEPENDENCY_CASCADE_THRESHOLD:
                        skip_map[hop.hop_id] = True
                        changed = True
                        logger.debug(
                            f"[Pruner.Cascade] {hop.hop_id} 因 {dep_id}(strength={strength:.2f}) "
                            f"被级联剪枝"
                        )
                        break

    # ----------------------------------------------------------
    # 工具函数
    # ----------------------------------------------------------

    @staticmethod
    def _compute_depended_set(hops: List[Hop]) -> Set[str]:
        """计算所有被其他 hop 依赖的 hop_id 集合"""
        depended: Set[str] = set()
        for hop in hops:
            for dep_id in hop.dependencies:
                depended.add(dep_id)
        return depended

    @staticmethod
    def _get_dependency_strength(hops: List[Hop], source_id: str, target_id: str) -> float:
        """
        获取两个 hop 之间的依赖强度。
        当前实现：若 target 的 dependencies 包含 source，默认强度 0.85（数据依赖）。
        扩展点：从 ReasoningGraph 的 GraphEdge 读取精确强度。
        """
        for hop in hops:
            if hop.hop_id == target_id and source_id in hop.dependencies:
                return 0.85  # 默认数据依赖强度
        return 0.0

    @staticmethod
    def _make_hops_summary(hops: List[Hop]) -> str:
        """生成所有 hop 的简短描述，供 Stage2 prompt 使用"""
        lines = []
        for hop in hops:
            lines.append(f"[{hop.hop_id}] {hop.query} (工具: {hop.tool_name})")
        return "\n".join(lines)
