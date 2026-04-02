"""
SAGE Reasoning Graph Database — 推理图存储与匹配。

职责：
- 持久化存储抽象推理路径（带占位符的 hop 模板）
- 三层匹配：Layer1 LLM 查询相似度 → Layer2 LCS 结构匹配 → Layer3 hop 语义相似度
- 路径实例化：将占位符替换为具体实体，生成可执行 HopPlan
- 依赖关系离线学习：从执行日志中统计共现次数，更新边强度
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

from src.data_structures import (
    EntityMapping,
    GraphEdge,
    GraphNode,
    Hop,
    HopPlan,
    ReasoningPath,
)

logger = logging.getLogger(__name__)


class ReasoningGraphDB:
    """
    推理图数据库。
    图以 JSON 文件持久化，运行时全量加载到内存。
    """

    def __init__(self, config, judge):
        """
        Parameters
        ----------
        config : module  — src.config
        judge  : UnifiedLLMJudge
        """
        self.config = config
        self.judge = judge
        self._paths: Dict[str, ReasoningPath] = {}  # path_id → ReasoningPath
        self._load()

    # ----------------------------------------------------------
    # 持久化
    # ----------------------------------------------------------

    def _load(self):
        """从磁盘加载图数据"""
        path = self.config.GRAPH_DB_PATH
        if not os.path.exists(path):
            logger.info(f"[Graph] 图数据库文件不存在，从空图开始: {path}")
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for p in data.get("paths", []):
                rp = self._deserialize_path(p)
                self._paths[rp.path_id] = rp
            logger.info(f"[Graph] 加载了 {len(self._paths)} 条推理路径")
        except Exception as e:
            logger.error(f"[Graph] 加载图数据失败: {e}")

    def save(self):
        """将当前图持久化到磁盘"""
        os.makedirs(os.path.dirname(self.config.GRAPH_DB_PATH), exist_ok=True)
        data = {"paths": [self._serialize_path(p) for p in self._paths.values()]}
        with open(self.config.GRAPH_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.debug(f"[Graph] 图已持久化，共 {len(self._paths)} 条路径")

    @staticmethod
    def _serialize_path(p: ReasoningPath) -> dict:
        return {
            "path_id": p.path_id,
            "query_template": p.query_template,
            "entity_types": p.entity_types,
            "use_count": p.use_count,
            "avg_success_rate": p.avg_success_rate,
            "nodes": [
                {
                    "node_id": n.node_id,
                    "intent_template": n.intent_template,
                    "query_template": n.query_template,
                    "tool_name": n.tool_name,
                    "tool_args_template": n.tool_args_template,
                }
                for n in p.nodes
            ],
            "edges": [
                {
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                    "dependency_type": e.dependency_type,
                    "strength": e.strength,
                    "learned_count": e.learned_count,
                }
                for e in p.edges
            ],
        }

    @staticmethod
    def _deserialize_path(data: dict) -> ReasoningPath:
        nodes = [
            GraphNode(
                node_id=n["node_id"],
                intent_template=n["intent_template"],
                query_template=n["query_template"],
                tool_name=n["tool_name"],
                tool_args_template=n.get("tool_args_template", {}),
            )
            for n in data.get("nodes", [])
        ]
        edges = [
            GraphEdge(
                source_id=e["source_id"],
                target_id=e["target_id"],
                dependency_type=e["dependency_type"],
                strength=e["strength"],
                learned_count=e.get("learned_count", 0),
            )
            for e in data.get("edges", [])
        ]
        return ReasoningPath(
            path_id=data["path_id"],
            query_template=data["query_template"],
            entity_types=data.get("entity_types", []),
            nodes=nodes,
            edges=edges,
            use_count=data.get("use_count", 0),
            avg_success_rate=data.get("avg_success_rate", 1.0),
        )

    # ----------------------------------------------------------
    # 三层匹配
    # ----------------------------------------------------------

    def find_similar_path(self, new_query: str) -> Optional[Tuple[ReasoningPath, List[EntityMapping]]]:
        """
        对新查询进行三层图匹配，返回最佳匹配路径和实体映射，或 None。

        Layer1: LLM 查询语义相似度（SIMILAR / DIFFERENT）
        Layer2: LCS 结构匹配度（hop 意图模板序列）
        Layer3: hop 级语义相似度（细粒度验证）
        """
        if not self._paths:
            return None

        candidates: List[Tuple[ReasoningPath, float]] = []

        # ------ Layer 1: LLM 粗筛 ------
        for path in self._paths.values():
            decision = self.judge.graph_query_similarity(
                new_query=new_query,
                path_query_template=path.query_template,
                entity_types=path.entity_types,
            )
            if decision == "SIMILAR":
                candidates.append((path, 0.0))
            logger.debug(f"[Graph.L1] path={path.path_id} decision={decision}")

        if not candidates:
            return None

        # ------ Layer 2: LCS 结构匹配 ------
        # 将 new_query 简单拆成意图 token 序列（此处用 words 近似，可替换为 NLP 提取）
        query_tokens = new_query.lower().split()
        best_path: Optional[ReasoningPath] = None
        best_score = 0.0

        for path, _ in candidates:
            path_tokens = [n.intent_template.lower() for n in path.nodes]
            lcs_score = self._lcs_ratio(query_tokens, path_tokens)
            logger.debug(f"[Graph.L2] path={path.path_id} lcs={lcs_score:.2f}")
            if lcs_score >= self.config.GRAPH_STRUCTURAL_MATCH_THRESHOLD and lcs_score > best_score:
                best_score = lcs_score
                best_path = path

        if best_path is None:
            # Layer2 阈值未达到，退化取 Layer1 第一个候选（宽松模式）
            best_path = candidates[0][0]
            logger.debug("[Graph.L2] 未达阈值，使用 Layer1 首候选")

        # ------ Layer 3: 实体映射提取 ------
        placeholders = self._extract_placeholders(best_path)
        entity_mapping_dict = self.judge.extract_entity_mapping(
            new_query=new_query,
            placeholders=placeholders,
            query_template=best_path.query_template,
        )
        if not entity_mapping_dict:
            logger.warning(f"[Graph.L3] 实体映射提取失败，跳过路径 {best_path.path_id}")
            return None

        mappings = [
            EntityMapping(
                placeholder=self.config.ENTITY_PLACEHOLDER_FORMAT.format(k),
                value=v,
                entity_type=k,
            )
            for k, v in entity_mapping_dict.items()
        ]
        return best_path, mappings

    # ----------------------------------------------------------
    # 路径实例化
    # ----------------------------------------------------------

    def instantiate_path(
        self,
        path: ReasoningPath,
        mappings: List[EntityMapping],
        original_query: str,
    ) -> HopPlan:
        """
        将模板路径 + 实体映射 → 具体可执行的 HopPlan。
        """
        hops: List[Hop] = []
        for i, node in enumerate(path.nodes):
            hop_id = f"hop_{i + 1}"

            # 替换 query_template 中的占位符
            query = node.query_template
            tool_args = json.dumps(node.tool_args_template)
            for m in mappings:
                query = query.replace(m.placeholder, m.value)
                tool_args = tool_args.replace(m.placeholder, m.value)

            # 从边中恢复依赖关系（目标节点依赖哪些源节点）
            deps = []
            for edge in path.edges:
                if edge.target_id == node.node_id:
                    # 找到 source 对应的 hop_id
                    for j, src_node in enumerate(path.nodes):
                        if src_node.node_id == edge.source_id:
                            deps.append(f"hop_{j + 1}")
                            break

            hops.append(
                Hop(
                    hop_id=hop_id,
                    query=query,
                    tool_name=node.tool_name,
                    tool_args=json.loads(tool_args),
                    dependencies=deps,
                )
            )

        path.use_count += 1
        return HopPlan(original_query=original_query, hops=hops)

    # ----------------------------------------------------------
    # 图学习（离线依赖统计）
    # ----------------------------------------------------------

    def learn_from_execution(self, plan: HopPlan):
        """
        从完成执行的 HopPlan 中学习依赖关系。
        如果匹配到已有路径，更新边权；否则尝试抽象为新路径并存储。
        """
        # 统计哪些 hop 实际执行了
        executed = [h for h in plan.hops if not h.skipped]
        if len(executed) < 2:
            return

        # 简化：此处仅记录日志，完整实现需 NLP 抽象和模板化
        logger.info(
            f"[Graph.learn] 从查询 '{plan.original_query[:50]}' 学习，"
            f"执行了 {len(executed)}/{len(plan.hops)} 个 hop"
        )

    def add_path(self, path: ReasoningPath):
        """手动添加或更新路径"""
        self._paths[path.path_id] = path
        logger.info(f"[Graph] 添加路径: {path.path_id}")

    # ----------------------------------------------------------
    # 工具函数
    # ----------------------------------------------------------

    @staticmethod
    def _lcs_ratio(seq1: List[str], seq2: List[str]) -> float:
        """计算两个序列的 LCS 长度比（相对于较长序列）"""
        if not seq1 or not seq2:
            return 0.0
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        lcs_len = dp[m][n]
        return lcs_len / max(m, n)

    @staticmethod
    def _extract_placeholders(path: ReasoningPath) -> List[str]:
        """从路径的模板中提取所有唯一占位符名称（不含花括号）"""
        import re
        placeholders = set()
        for node in path.nodes:
            for match in re.finditer(r"\{([A-Z_]+)\}", node.query_template):
                placeholders.add(match.group(1))
            for v in node.tool_args_template.values():
                if isinstance(v, str):
                    for match in re.finditer(r"\{([A-Z_]+)\}", v):
                        placeholders.add(match.group(1))
        return list(placeholders)
