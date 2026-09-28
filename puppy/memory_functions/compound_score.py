"""
复合打分模块 (Compound Score)

本模块负责将相似度、新近性、重要性三类分数融合为最终检索排序分数。

改动说明（论文核心创新 - Regime-aware Memory Retrieval）：
  原始版本使用固定加法 merge_score = similarity + (recency + importance/100)，
  无法区分不同市场环境下各维度的重要程度。
  改进后 merge_score 接受 alpha/beta/gamma 三个权重参数，
  由上层 BrainDB 根据市场状态动态传入，实现
  「同一查询在不同市场条件下获得不同检索排序」的能力。
"""


class LinearCompoundScore:
    """线性复合打分器"""

    def recency_and_importance_score(
        self, recency_score: float, importance_score: float
    ) -> float:
        """计算新近性与重要性的部分复合分（用于衰减/迁移阶段，不涉及检索权重）"""
        importance_score = min(importance_score, 100)
        return recency_score + importance_score / 100

    # ── 改动：merge_score 支持自适应三权重 ──────────────────────
    #   alpha : 语义相似度权重
    #   beta  : 新近性权重
    #   gamma : 重要性权重
    #   默认值 1.0/1.0/1.0 等价于原始行为，保证向后兼容。
    def merge_score(
        self,
        similarity_score: float,
        recency_and_importance: float,  # 保留旧参数名以兼容，不再使用
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        recency_score: float = 0.0,
        importance_score: float = 0.0,
    ) -> float:
        """
        将三维分数按市场状态权重加权求和。

        Args:
            similarity_score: FAISS 返回的余弦相似度
            recency_and_importance: （兼容旧调用，内部不再使用）
            alpha: 相似度权重，由市场状态决定
            beta:  新近性权重，由市场状态决定
            gamma: 重要性权重，由市场状态决定
            recency_score: 当前记忆条目的新近性分数
            importance_score: 当前记忆条目的重要性分数

        Returns:
            加权后的最终检索分数
        """
        return (
            alpha * similarity_score
            + beta * recency_score
            + gamma * (importance_score / 100)
        )