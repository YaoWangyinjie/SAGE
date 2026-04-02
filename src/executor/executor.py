"""
SAGE Executor — 投机执行器。

核心思路（类投机采样）：
  1. 并行启动 API 工具调用（~300ms）和本地 speculation 生成（~150ms 或 0ms 命中缓存）
  2. 等待两者完成后，由 LLM Judge 验证 speculation 与 API 结果的一致性
  3. 若 verify_score >= VERIFY_SCORE_THRESHOLD → 使用 speculation（节省后续推理 token）
     否则 → 使用 API 结果，通过 Oracle 模型进一步推理

PrefetchingExecutor：
  在执行 hop_i 时，预取 hop_{i+1} 的 API 调用，进一步压缩端到端延迟。

容错设计：
  - API 失败 → 回退到 speculation（标记 fallback_to_speculation=True）
  - speculation 失败 → 直接使用 API 结果（跳过验证）
  - 整体超时 → 标记 hop 失败，继续后续 hop
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional

import httpx

from src.data_structures import Hop, HopPlan, TemporalType, VerifyDecision
from src.judge.llm_judge import robust_llm_judge

logger = logging.getLogger(__name__)


# ============================================================
# Oracle API 客户端（工具调用 / 最终推理）
# ============================================================

class OracleAPIClient:
    """
    调用大模型 API（OpenAI 兼容接口）执行工具调用和最终推理。
    """

    def __init__(self, config):
        self.config = config
        api_key = os.environ.get(config.ORACLE_API_KEY_ENV, "")
        self._client = httpx.AsyncClient(
            base_url=config.ORACLE_API_BASE,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=config.ORACLE_TIMEOUT_SECONDS,
        )

    async def call_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """
        通过 Oracle 模型执行工具调用。
        此处实现为向 Oracle 发送 prompt，让其模拟工具执行并返回结果。
        真实部署中可替换为直接调用工具 API（如 Bing Search API）。
        """
        prompt = (
            f"请执行以下工具调用并返回结果：\n"
            f"工具：{tool_name}\n"
            f"参数：{tool_args}\n"
            f"直接返回工具执行结果，不要有多余解释。"
        )
        return await self._chat(prompt)

    async def reason(self, query: str, context: str) -> str:
        """基于工具结果进行推理，生成该 hop 的答案"""
        prompt = (
            f"问题：{query}\n\n"
            f"参考信息：\n{context}\n\n"
            f"请根据参考信息回答问题，简洁准确。"
        )
        return await self._chat(prompt)

    async def synthesize(self, original_query: str, hop_results: str) -> str:
        """最终综合回答合成"""
        from src.prompts import SYNTHESIS_PROMPT
        prompt = SYNTHESIS_PROMPT.format(
            original_query=original_query,
            hop_results=hop_results,
        )
        return await self._chat(prompt, max_tokens=self.config.SYNTHESIS_MAX_TOKENS)

    async def _chat(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """发送 chat completion 请求"""
        payload = {
            "model": self.config.ORACLE_MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens or self.config.ORACLE_MAX_TOKENS,
            "temperature": self.config.ORACLE_TEMPERATURE,
        }
        for attempt in range(self.config.ORACLE_MAX_RETRIES + 1):
            try:
                resp = await self._client.post("/chat/completions", json=payload)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            except Exception as e:
                logger.warning(f"[Oracle] 第 {attempt + 1} 次请求失败: {e}")
                if attempt < self.config.ORACLE_MAX_RETRIES:
                    await asyncio.sleep(0.5)
        raise RuntimeError(f"Oracle API 请求失败，已重试 {self.config.ORACLE_MAX_RETRIES} 次")

    async def aclose(self):
        await self._client.aclose()


# ============================================================
# 投机执行器
# ============================================================

class SpeculativeExecutor:
    """
    单 hop 投机执行器。

    执行逻辑（parallel speculation）：
      1. 异步启动 API 工具调用任务
      2. 从 SpeculationCache 读取预测（0ms）或用本地模型生成（~150ms）
      3. 等待 API 结果完成
      4. Judge 验证一致性
      5. 根据验证分数决定使用 speculation 还是 API 结果
    """

    def __init__(self, config, judge, oracle_client, spec_cache, tool_cache):
        self.config = config
        self.judge = judge
        self.oracle = oracle_client
        self.spec_cache = spec_cache
        self.tool_cache = tool_cache

    async def execute_hop(self, hop: Hop, context: str, original_query: str) -> str:
        """
        执行单个 hop，返回该 hop 的答案字符串。
        结果同时写入 hop.result。
        """
        if hop.skipped:
            logger.debug(f"[Exec] {hop.hop_id} 已跳过")
            return ""

        t0 = time.time()

        # -- Step 1：检查工具缓存 --
        cached_result = self.tool_cache.get(hop.tool_name, hop.tool_args)
        if cached_result:
            logger.info(f"[Exec] {hop.hop_id} 工具缓存命中，直接使用")
            hop.result = cached_result
            hop.result_summary = cached_result[:200]
            hop.execution_time_ms = (time.time() - t0) * 1000
            return cached_result

        # -- Step 2：并行启动 API 调用 + speculation --
        api_task = asyncio.create_task(self._call_api_safe(hop))

        # Speculation：优先从缓存读取（0ms），否则跳过本地生成（简化：直接等 API）
        cached_speculation = self.spec_cache.get(hop.hop_id)
        if cached_speculation:
            logger.debug(f"[Exec] {hop.hop_id} speculation 缓存命中（0ms）")
            speculation = cached_speculation
        else:
            speculation = None
            logger.debug(f"[Exec] {hop.hop_id} 无 speculation 缓存")

        # -- Step 3：等待 API 结果 --
        try:
            api_result = await asyncio.wait_for(
                api_task, timeout=self.config.HOP_EXECUTION_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.error(f"[Exec] {hop.hop_id} API 超时")
            api_result = None
        except Exception as e:
            logger.error(f"[Exec] {hop.hop_id} API 失败: {e}")
            api_result = None

        # -- Step 4：API 失败回退到 speculation --
        if api_result is None:
            if speculation:
                hop.fallback_to_speculation = True
                hop.result = speculation
                hop.result_summary = speculation[:200]
                logger.warning(f"[Exec] {hop.hop_id} API 失败，回退到 speculation")
            else:
                hop.result = f"[{hop.hop_id} 执行失败]"
                logger.error(f"[Exec] {hop.hop_id} API 失败且无 speculation，标记失败")
            hop.execution_time_ms = (time.time() - t0) * 1000
            return hop.result or ""

        # -- Step 5：Judge 验证 --
        final_result = api_result
        if speculation and self.config.ENABLE_PARALLEL_SPECULATION:
            verify_score = robust_llm_judge(
                self.judge.result_verify,
                original_query,
                hop.query,
                speculation,
                api_result,
                fallback=0,
            )
            logger.debug(f"[Exec] {hop.hop_id} verify_score={verify_score}")
            if verify_score >= self.config.VERIFY_SCORE_THRESHOLD:
                final_result = speculation
                logger.info(
                    f"[Exec] {hop.hop_id} 使用 speculation（score={verify_score}），"
                    f"节省后续推理"
                )
            else:
                # 使用 Oracle 基于 API 结果推理
                try:
                    final_result = await self.oracle.reason(hop.query, api_result)
                except Exception as e:
                    logger.warning(f"[Exec] {hop.hop_id} Oracle.reason 失败，直接用 API 原始结果: {e}")
                    final_result = api_result
        else:
            # 无 speculation：直接用 Oracle 推理
            try:
                final_result = await self.oracle.reason(hop.query, api_result)
            except Exception as e:
                logger.warning(f"[Exec] {hop.hop_id} Oracle.reason 失败: {e}")
                final_result = api_result

        # -- Step 6：写入缓存 --
        temporal_type_str = robust_llm_judge(
            self.judge.temporal_detect, hop.query, fallback="STATIC"
        )
        temporal_type = TemporalType(temporal_type_str)
        self.tool_cache.put(hop.tool_name, hop.tool_args, final_result, temporal_type)

        hop.result = final_result
        hop.result_summary = final_result[:self.config.CACHE_SUMMARY_MAX_CHARS]
        hop.execution_time_ms = (time.time() - t0) * 1000
        logger.info(f"[Exec] {hop.hop_id} 完成，耗时 {hop.execution_time_ms:.0f}ms")
        return final_result

    async def _call_api_safe(self, hop: Hop) -> Optional[str]:
        """安全的 API 工具调用包装"""
        try:
            return await self.oracle.call_tool(hop.tool_name, hop.tool_args)
        except Exception as e:
            logger.error(f"[Exec._call_api] {hop.hop_id} 工具调用异常: {e}")
            return None


# ============================================================
# 预取执行器（继承投机执行器）
# ============================================================

class PrefetchingExecutor(SpeculativeExecutor):
    """
    在执行 hop_i 时，同时预取 hop_{i+1} 的 API 调用（look-ahead prefetch）。
    减少串行等待时间。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prefetch_tasks: Dict[str, asyncio.Task] = {}

    async def execute_plan(self, plan: HopPlan) -> List[str]:
        """
        顺序执行 plan 中所有 active hops，启用预取。
        返回各 hop 结果列表。
        """
        active_hops = plan.active_hops()
        results: List[str] = []
        context_parts: List[str] = []

        for i, hop in enumerate(active_hops):
            # 预取下一个 hop 的 API 调用
            if self.config.ENABLE_PREFETCH and i + 1 < len(active_hops):
                next_hop = active_hops[i + 1]
                if next_hop.hop_id not in self._prefetch_tasks:
                    self._prefetch_tasks[next_hop.hop_id] = asyncio.create_task(
                        self._call_api_safe(next_hop)
                    )
                    logger.debug(f"[Prefetch] 预取 {next_hop.hop_id}")

            context_summary = "\n".join(context_parts[-3:])  # 最近 3 个 hop 结果作为上下文
            result = await self.execute_hop(hop, context_summary, plan.original_query)
            results.append(result)
            if result:
                context_parts.append(f"[{hop.hop_id}] {result[:200]}")

        # 清理未使用的预取任务
        for task in self._prefetch_tasks.values():
            if not task.done():
                task.cancel()
        self._prefetch_tasks.clear()

        return results
