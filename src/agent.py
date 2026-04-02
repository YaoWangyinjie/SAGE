"""
SAGE Agent — 顶层编排器。

完整执行流程（对应 FLOWCHARTS.md）：
  1. temporal_detect：判断查询时效性
  2. graph_match：在推理图中查找相似历史路径 → 复用或新建 HopPlan
  3. dependency_infer：推断 / 更新 hop 间依赖关系
  4. prune：两阶段剪枝 + 依赖级联检查
  5. execute：投机并行执行所有 active hops（含预取）
  6. synthesize：调用 Oracle 综合回答
  7. learn：从本次执行结果更新推理图

SAGEAgent 是系统唯一的对外入口，其他模块均通过此类调用。
"""

import asyncio
import logging
import time
from typing import Optional

import src.config as config
from src.cache.cache import SemanticToolCache, SpeculationCache
from src.data_structures import Hop, HopPlan, TemporalType
from src.executor.executor import OracleAPIClient, PrefetchingExecutor
from src.graph.reasoning_graph import ReasoningGraphDB
from src.judge.llm_judge import UnifiedLLMJudge, robust_llm_judge
from src.pruner.pruner import TwoStagePruner
from src.utils.utils import format_plan, make_hops_description, setup_logging

logger = logging.getLogger(__name__)


class SAGEAgent:
    """
    SAGE 主 Agent。
    管理所有子模块的生命周期，提供 query() 接口。
    """

    def __init__(self):
        setup_logging(config)
        logger.info("[SAGE] 初始化中...")

        # 核心组件
        self.judge = UnifiedLLMJudge(config)
        self.spec_cache = SpeculationCache(config)
        self.tool_cache = SemanticToolCache(config, self.judge)
        self.graph_db = ReasoningGraphDB(config, self.judge)
        self.pruner = TwoStagePruner(config, self.judge, self.spec_cache)
        self.oracle = OracleAPIClient(config)
        self.executor = PrefetchingExecutor(
            config=config,
            judge=self.judge,
            oracle_client=self.oracle,
            spec_cache=self.spec_cache,
            tool_cache=self.tool_cache,
        )

        logger.info("[SAGE] 初始化完成")

    # ----------------------------------------------------------
    # 主入口
    # ----------------------------------------------------------

    async def query(self, user_query: str) -> str:
        """
        处理用户查询，返回最终回答。

        Parameters
        ----------
        user_query : 用户自然语言问题

        Returns
        -------
        str : 综合推理后的最终答案
        """
        t_start = time.time()
        logger.info(f"[SAGE] 开始处理查询: {user_query[:80]}")

        # 每次新查询前重置投机缓存（单查询生命周期）
        self.spec_cache.reset()

        # ---- Step 1: 时效性检测 ----
        temporal_type_str = robust_llm_judge(
            self.judge.temporal_detect, user_query, fallback="STATIC"
        )
        temporal_type = TemporalType(temporal_type_str)
        logger.info(f"[SAGE] 时效性: {temporal_type.value}")

        # ---- Step 2: 图匹配（尝试复用历史推理路径）----
        plan = await self._build_plan(user_query, temporal_type)

        # ---- Step 3: 依赖关系推断（更新 hop.dependencies）----
        self._infer_and_update_dependencies(plan)

        # ---- Step 4: 两阶段剪枝 ----
        self.pruner.prune(plan)
        logger.info(f"[SAGE] 剪枝后:\n{format_plan(plan)}")

        # ---- Step 5: 投机执行 ----
        results = await self.executor.execute_plan(plan)

        # ---- Step 6: 合成最终答案 ----
        answer = await self._synthesize(plan, results)

        # ---- Step 7: 从本次执行学习 ----
        self.graph_db.learn_from_execution(plan)
        self.graph_db.save()

        elapsed = (time.time() - t_start) * 1000
        logger.info(f"[SAGE] 查询完成，总耗时 {elapsed:.0f}ms")
        return answer

    def query_sync(self, user_query: str) -> str:
        """同步版本的 query，供非 async 场景调用"""
        return asyncio.run(self.query(user_query))

    # ----------------------------------------------------------
    # 内部步骤
    # ----------------------------------------------------------

    async def _build_plan(self, user_query: str, temporal_type: TemporalType) -> HopPlan:
        """
        尝试从推理图中复用历史路径；若无匹配，则调用 Oracle 生成新计划。
        """
        match_result = self.graph_db.find_similar_path(user_query)
        if match_result:
            path, mappings = match_result
            plan = self.graph_db.instantiate_path(path, mappings, user_query)
            logger.info(
                f"[SAGE] 图匹配成功，复用路径 {path.path_id}，"
                f"实例化 {len(plan.hops)} 个 hop"
            )
            return plan

        # 无图匹配 → Oracle 生成 hop 计划
        logger.info("[SAGE] 图匹配失败，调用 Oracle 生成新计划")
        plan = await self._oracle_generate_plan(user_query)
        return plan

    async def _oracle_generate_plan(self, user_query: str) -> HopPlan:
        """
        让 Oracle 模型将查询分解为多跳子问题。
        返回初始 HopPlan（依赖关系待后续推断）。
        """
        prompt = (
            f"请将以下问题分解为多个子问题（hop），每个 hop 对应一次工具调用。\n"
            f"问题：{user_query}\n\n"
            f"以 JSON 数组格式输出，每个元素包含：\n"
            f"- hop_id: 如 hop_1, hop_2 ...\n"
            f"- query: 该步骤需要回答的子问题\n"
            f"- tool_name: 工具名称（web_search / knowledge_base / calculator / ...）\n"
            f"- tool_args: 工具参数字典\n"
            f"- dependencies: 依赖的 hop_id 列表（可为空）\n\n"
            f"直接输出 JSON 数组："
        )
        import json

        try:
            raw = await self.oracle._chat(prompt)
            # 提取 JSON 数组
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start >= 0 and end > start:
                hops_data = json.loads(raw[start:end])
                hops = [
                    Hop(
                        hop_id=h.get("hop_id", f"hop_{i+1}"),
                        query=h.get("query", ""),
                        tool_name=h.get("tool_name", "web_search"),
                        tool_args=h.get("tool_args", {}),
                        dependencies=h.get("dependencies", []),
                    )
                    for i, h in enumerate(hops_data)
                ]
                return HopPlan(original_query=user_query, hops=hops)
        except Exception as e:
            logger.error(f"[SAGE] Oracle 生成计划失败: {e}")

        # Fallback：单跳计划
        logger.warning("[SAGE] 退化为单跳计划")
        return HopPlan(
            original_query=user_query,
            hops=[
                Hop(
                    hop_id="hop_1",
                    query=user_query,
                    tool_name="web_search",
                    tool_args={"query": user_query},
                )
            ],
        )

    def _infer_and_update_dependencies(self, plan: HopPlan):
        """
        使用 LLM Judge 推断 hop 间依赖关系，更新 hop.dependencies。
        仅对 dependencies 为空的 hop 执行推断（避免覆盖图匹配结果）。
        """
        needs_inference = any(not h.dependencies for h in plan.hops if not h.skipped)
        if not needs_inference:
            return

        hops_desc = make_hops_description(plan.hops)
        dep_map = robust_llm_judge(
            self.judge.infer_dependencies, hops_desc, fallback=None
        )
        if not dep_map:
            return

        # dep_map 格式：{"hop_2": {"hop_1": 0.85}, ...}
        for hop in plan.hops:
            if hop.hop_id in dep_map and not hop.dependencies:
                strong_deps = [
                    dep_id
                    for dep_id, strength in dep_map[hop.hop_id].items()
                    if strength >= config.DEPENDENCY_CASCADE_THRESHOLD
                ]
                hop.dependencies = strong_deps
                if strong_deps:
                    logger.debug(f"[SAGE] {hop.hop_id} 推断依赖: {strong_deps}")

    async def _synthesize(self, plan: HopPlan, results: list) -> str:
        """将各 hop 结果综合为最终答案"""
        hop_results_parts = []
        for hop, result in zip(plan.active_hops(), results):
            if result:
                hop_results_parts.append(f"步骤 {hop.hop_id}（{hop.query}）：\n{result}")

        if not hop_results_parts:
            return "抱歉，未能获取到足够的信息来回答该问题。"

        hop_results_str = "\n\n".join(hop_results_parts)
        try:
            return await self.oracle.synthesize(plan.original_query, hop_results_str)
        except Exception as e:
            logger.error(f"[SAGE] 合成答案失败: {e}")
            return hop_results_str  # Fallback：直接拼接

    async def close(self):
        """释放资源"""
        await self.oracle.aclose()
        logger.info("[SAGE] 资源已释放")
