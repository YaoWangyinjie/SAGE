"""
SAGE LLM Judge — 统一的 LLM 判断接口。

职责：
- 使用本地小模型（Local Model）进行快速二分类 / 评分
- 提供 temporal_detect / similarity_judge / hop_skip_judge 等统一方法
- 所有方法都有容错逻辑：失败时回退到保守默认值
- 支持 logprob-based 置信度估计

模型加载策略：
- 本地小模型：HuggingFace transformers，懒加载（首次调用时初始化）
- 设计为可插拔：替换 _generate / _generate_with_logprobs 即可接入其他推理后端
"""

import json
import logging
import re
import time
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class UnifiedLLMJudge:
    """
    统一 LLM Judge，封装本地小模型推断逻辑。
    所有方法在失败时均回退到保守默认值，确保系统可用性。
    """

    def __init__(self, config):
        """
        Parameters
        ----------
        config : module
            src.config 模块，直接传入以避免循环导入
        """
        self.config = config
        self._model = None
        self._tokenizer = None
        self._model_loaded = False

    # ----------------------------------------------------------
    # 私有：模型懒加载
    # ----------------------------------------------------------

    def _ensure_model_loaded(self):
        """首次调用时加载本地小模型（懒加载）"""
        if self._model_loaded:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            logger.info(f"[Judge] 加载本地模型: {self.config.LOCAL_MODEL_ID}")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.LOCAL_MODEL_ID,
                trust_remote_code=True,
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            load_kwargs: Dict = {
                "device_map": self.config.LOCAL_MODEL_DEVICE
                if self.config.LOCAL_MODEL_DEVICE != "cpu"
                else None,
                "trust_remote_code": True,
            }
            if self.config.LOCAL_MODEL_QUANTIZED:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )

            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.LOCAL_MODEL_ID,
                **load_kwargs,
            )
            if self.config.LOCAL_MODEL_DEVICE == "cpu":
                self._model = self._model.to("cpu")
            self._model.eval()
            self._model_loaded = True
            logger.info("[Judge] 本地模型加载完成")
        except Exception as e:
            logger.error(f"[Judge] 本地模型加载失败: {e}，将使用 fallback 逻辑")
            self._model_loaded = True  # 标记为已尝试，避免反复重试

    # ----------------------------------------------------------
    # 私有：推断核心
    # ----------------------------------------------------------

    def _generate(self, prompt: str, max_new_tokens: int) -> str:
        """
        使用本地小模型生成文本。
        返回生成的新 token 字符串（不含 prompt 本身）。
        """
        self._ensure_model_loaded()
        if self._model is None or self._tokenizer is None:
            return ""

        try:
            import torch

            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            )
            if self.config.LOCAL_MODEL_DEVICE not in ("cpu", None):
                inputs = {k: v.to(self.config.LOCAL_MODEL_DEVICE) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=self.config.LOCAL_MODEL_TEMPERATURE,
                    do_sample=self.config.LOCAL_MODEL_TEMPERATURE > 0,
                    pad_token_id=self._tokenizer.pad_token_id,
                )

            input_len = inputs["input_ids"].shape[1]
            new_tokens = outputs[0][input_len:]
            return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        except Exception as e:
            logger.warning(f"[Judge] _generate 失败: {e}")
            return ""

    def _generate_with_logprobs(
        self, prompt: str, max_new_tokens: int = 5
    ) -> Tuple[str, float]:
        """
        生成文本并返回首个 token 的置信度（softmax 概率）。
        Returns: (generated_text, confidence)
        """
        self._ensure_model_loaded()
        if self._model is None or self._tokenizer is None:
            return "", 0.5

        try:
            import torch
            import torch.nn.functional as F

            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            )
            if self.config.LOCAL_MODEL_DEVICE not in ("cpu", None):
                inputs = {k: v.to(self.config.LOCAL_MODEL_DEVICE) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=self.config.LOCAL_MODEL_TEMPERATURE,
                    do_sample=False,  # greedy for logprob
                    pad_token_id=self._tokenizer.pad_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            input_len = inputs["input_ids"].shape[1]
            new_tokens = outputs.sequences[0][input_len:]
            text = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            # 首个 token 的 softmax 最大概率作为置信度
            if outputs.scores:
                first_score = outputs.scores[0][0]
                probs = F.softmax(first_score, dim=-1)
                confidence = float(probs.max().item())
            else:
                confidence = 0.5

            return text, confidence
        except Exception as e:
            logger.warning(f"[Judge] _generate_with_logprobs 失败: {e}")
            return "", 0.5

    # ----------------------------------------------------------
    # 私有：解析工具
    # ----------------------------------------------------------

    @staticmethod
    def _parse_decision(text: str, valid_choices: List[str], default: str) -> str:
        """
        从模型输出中提取决策词。
        宽松解析：在输出首行中搜索任意有效决策词（大小写不敏感）。
        """
        first_line = text.split("\n")[0].upper().strip()
        for choice in valid_choices:
            if choice in first_line:
                return choice
        # 全文再搜一次
        upper_text = text.upper()
        for choice in valid_choices:
            if choice in upper_text:
                return choice
        logger.debug(f"[Judge] 无法解析决策，使用默认值 '{default}'，原始输出: {text!r}")
        return default

    @staticmethod
    def _parse_score(text: str, default: int = 50) -> int:
        """从模型输出中提取 0-100 的整数分数"""
        # 优先匹配 "SCORE: 85" 格式
        match = re.search(r"SCORE\s*:\s*(\d{1,3})", text, re.IGNORECASE)
        if match:
            return max(0, min(100, int(match.group(1))))
        # 退化：找任意独立数字
        numbers = re.findall(r"\b(\d{1,3})\b", text)
        for n in numbers:
            val = int(n)
            if 0 <= val <= 100:
                return val
        return default

    @staticmethod
    def _parse_speculation(text: str) -> Optional[str]:
        """从 Stage2 输出中提取 SPECULATION 字段"""
        match = re.search(r"SPECULATION\s*:\s*(.+?)(?:\n|PREDICTION|SCORE|$)", text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def _parse_json_safe(text: str) -> Optional[dict]:
        """安全解析 JSON，失败返回 None"""
        try:
            # 找到第一个 { 到最后一个 }
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except Exception:
            pass
        return None

    # ----------------------------------------------------------
    # 公开 API
    # ----------------------------------------------------------

    def temporal_detect(self, query: str) -> str:
        """
        检测查询的时效性。
        Returns: "REALTIME" | "RECENT" | "STATIC"
        """
        from src.prompts import TEMPORAL_DETECTION_PROMPT
        prompt = TEMPORAL_DETECTION_PROMPT.format(query=query)

        t0 = time.time()
        output = self._generate(prompt, max_new_tokens=self.config.LOCAL_MODEL_MAX_NEW_TOKENS_FAST)
        elapsed = (time.time() - t0) * 1000
        logger.debug(f"[Judge.temporal_detect] {elapsed:.0f}ms | output={output!r}")

        return self._parse_decision(output, ["REALTIME", "RECENT", "STATIC"], default="STATIC")

    def similarity_judge(self, new_query: str, historical_query: str) -> str:
        """
        判断两个查询语义是否相似，可复用历史路径。
        Returns: "SIMILAR" | "DIFFERENT"
        """
        from src.prompts import QUERY_SIMILARITY_PROMPT
        prompt = QUERY_SIMILARITY_PROMPT.format(
            new_query=new_query,
            historical_query=historical_query,
        )
        output = self._generate(prompt, max_new_tokens=self.config.LOCAL_MODEL_MAX_NEW_TOKENS_FAST)
        return self._parse_decision(output, ["SIMILAR", "DIFFERENT"], default="DIFFERENT")

    def hop_skip_judge(
        self,
        original_query: str,
        hop_id: str,
        hop_query: str,
        context_summary: str,
    ) -> Tuple[str, float]:
        """
        Stage1 快速判断：该 hop 是否可以跳过。
        Returns: (decision, confidence)
            decision: "SKIP" | "EXECUTE" | "UNCERTAIN"
            confidence: 0–1 基于 logprob
        """
        from src.prompts import SKIP_DECISION_PROMPT
        prompt = SKIP_DECISION_PROMPT.format(
            original_query=original_query,
            hop_id=hop_id,
            hop_query=hop_query,
            context_summary=context_summary or "（暂无上下文）",
        )
        text, confidence = self._generate_with_logprobs(
            prompt, max_new_tokens=self.config.LOCAL_MODEL_MAX_NEW_TOKENS_FAST
        )
        decision = self._parse_decision(text, ["SKIP", "EXECUTE", "UNCERTAIN"], default="UNCERTAIN")
        return decision, confidence

    def hop_importance_scoring(
        self,
        original_query: str,
        hops_to_evaluate: list,  # List[Hop]
        context_summary: str,
        all_hops_summary: str,
    ) -> List[Dict]:
        """
        Stage2 批量重要性评分 + 生成投机预测。
        Returns: List of dicts with keys: hop_id, score, speculation, prediction
        """
        from src.prompts import IMPORTANCE_SCORING_PROMPT
        results = []

        # 按 batch_size 分批处理
        batch_size = self.config.LOCAL_MODEL_BATCH_SIZE
        for i in range(0, len(hops_to_evaluate), batch_size):
            batch = hops_to_evaluate[i : i + batch_size]
            for hop in batch:
                prompt = IMPORTANCE_SCORING_PROMPT.format(
                    original_query=original_query,
                    hop_id=hop.hop_id,
                    hop_query=hop.query,
                    tool_name=hop.tool_name,
                    tool_args=json.dumps(hop.tool_args, ensure_ascii=False),
                    context_summary=context_summary or "（暂无上下文）",
                    all_hops_summary=all_hops_summary,
                )
                text = self._generate(
                    prompt, max_new_tokens=self.config.LOCAL_MODEL_MAX_NEW_TOKENS_DEEP
                )
                score = self._parse_score(text, default=70)  # 默认保留
                speculation = self._parse_speculation(text)
                # 提取 PREDICTION 字段
                pred_match = re.search(
                    r"PREDICTION\s*:\s*(.+?)(?:\n|SCORE|$)", text, re.IGNORECASE | re.DOTALL
                )
                prediction = pred_match.group(1).strip() if pred_match else ""

                results.append(
                    {
                        "hop_id": hop.hop_id,
                        "score": score,
                        "speculation": speculation,
                        "prediction": prediction,
                        "raw_output": text,
                    }
                )
        return results

    def result_verify(
        self,
        original_query: str,
        hop_query: str,
        speculation: str,
        api_result: str,
    ) -> int:
        """
        验证投机预测与 API 结果的一致性。
        Returns: 一致性分数 0–100
        """
        from src.prompts import VERIFICATION_PROMPT
        prompt = VERIFICATION_PROMPT.format(
            original_query=original_query,
            hop_query=hop_query,
            speculation=speculation,
            api_result=api_result[:1000],  # 截断避免超长
        )
        text = self._generate(prompt, max_new_tokens=self.config.LOCAL_MODEL_MAX_NEW_TOKENS_FAST)
        return self._parse_score(text, default=50)

    def tool_cache_similarity(
        self,
        new_tool_name: str,
        new_tool_args: Dict,
        cached_tool_name: str,
        cached_tool_args: Dict,
        cached_summary: str,
        cache_age_hours: float,
    ) -> str:
        """
        判断新工具调用是否可以复用缓存结果。
        Returns: "REUSABLE" | "NOT_REUSABLE"
        """
        from src.prompts import TOOL_CACHE_SIMILARITY_PROMPT
        prompt = TOOL_CACHE_SIMILARITY_PROMPT.format(
            new_tool_name=new_tool_name,
            new_tool_args=json.dumps(new_tool_args, ensure_ascii=False),
            cached_tool_name=cached_tool_name,
            cached_tool_args=json.dumps(cached_tool_args, ensure_ascii=False),
            cache_age_hours=cache_age_hours,
            cached_summary=cached_summary[:300],
        )
        output = self._generate(prompt, max_new_tokens=self.config.LOCAL_MODEL_MAX_NEW_TOKENS_FAST)
        return self._parse_decision(output, ["REUSABLE", "NOT_REUSABLE"], default="NOT_REUSABLE")

    def extract_entity_mapping(
        self,
        new_query: str,
        placeholders: List[str],
        query_template: str,
    ) -> Optional[Dict[str, str]]:
        """
        从新查询提取实体，映射到历史路径占位符。
        Returns: {"ENTITY": "EAGLE", ...} or None
        """
        from src.prompts import ENTITY_MAPPING_PROMPT
        prompt = ENTITY_MAPPING_PROMPT.format(
            new_query=new_query,
            placeholders=", ".join(placeholders),
            query_template=query_template,
        )
        text = self._generate(
            prompt, max_new_tokens=self.config.LOCAL_MODEL_MAX_NEW_TOKENS_FAST * 10
        )
        return self._parse_json_safe(text)

    def infer_dependencies(self, hops_description: str) -> Optional[Dict]:
        """
        推断 hop 之间的依赖关系强度。
        Returns: {"hop_2": {"hop_1": 0.85}, ...} or None
        """
        from src.prompts import DEPENDENCY_INFERENCE_PROMPT
        prompt = DEPENDENCY_INFERENCE_PROMPT.format(hops_description=hops_description)
        text = self._generate(
            prompt, max_new_tokens=self.config.LOCAL_MODEL_MAX_NEW_TOKENS_DEEP
        )
        return self._parse_json_safe(text)

    def graph_query_similarity(self, new_query: str, path_query_template: str, entity_types: List[str]) -> str:
        """
        Layer1：判断新查询与图路径模板是否同类。
        Returns: "SIMILAR" | "DIFFERENT"
        """
        from src.prompts import GRAPH_QUERY_SIMILARITY_PROMPT
        prompt = GRAPH_QUERY_SIMILARITY_PROMPT.format(
            new_query=new_query,
            path_query_template=path_query_template,
            entity_types=", ".join(entity_types),
        )
        output = self._generate(prompt, max_new_tokens=self.config.LOCAL_MODEL_MAX_NEW_TOKENS_FAST * 5)
        return self._parse_decision(output, ["SIMILAR", "DIFFERENT"], default="DIFFERENT")


# ----------------------------------------------------------
# 容错包装（全局函数，供其他模块安全调用）
# ----------------------------------------------------------

def robust_llm_judge(judge_fn, *args, max_retries: int = 2, fallback=None, **kwargs):
    """
    对任意 judge 方法进行重试包装。
    失败时返回 fallback 值（默认为 None，调用方自行处理）。
    """
    for attempt in range(max_retries + 1):
        try:
            return judge_fn(*args, **kwargs)
        except Exception as e:
            logger.warning(f"[robust_llm_judge] 第 {attempt + 1} 次尝试失败: {e}")
            if attempt < max_retries:
                time.sleep(0.05 * (attempt + 1))
    logger.error(f"[robust_llm_judge] 所有重试均失败，返回 fallback: {fallback!r}")
    return fallback
