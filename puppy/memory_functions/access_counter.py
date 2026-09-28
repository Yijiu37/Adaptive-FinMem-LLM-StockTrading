"""
记忆重要性更新模块 (Access Counter → Importance Score Change)

改动说明（论文小创新1 - 连续反馈强化）：
  原始版本：importance_score += access_counter * 5
    access_counter 仅累加离散 ±1，无法区分"轻微有效"与"显著有效"。
  改进后：
    1. access_counter 累积的是连续浮点反馈值（由 Portfolio.get_feedback_response 返回）。
    2. 更新公式变为：
         new_score = importance_score + eta * access_counter + lam * |access_counter|
       其中 eta 控制方向性强化，lam 控制被引用次数的额外奖励。
    3. 结果 clip 到 [0, 100]，保证分数不越界。
"""

import numpy as np


class LinearImportanceScoreChange:
    """基于连续反馈强度的重要性分数更新器"""

    def __init__(self, eta: float = 5.0, lam: float = 1.0):
        """
        Args:
            eta: 反馈方向强化系数（正反馈加分、负反馈减分）
            lam: 累积访问奖励系数（无论正负，被引用越多额外奖励越大）
        """
        self.eta = eta
        self.lam = lam

    def __call__(self, access_counter: float, importance_score: float) -> float:
        # access_counter 现在是累积连续反馈值（如 0.73, -0.42 等）
        new_score = (
            importance_score
            + self.eta * access_counter
            + self.lam * abs(access_counter)
        )
        return float(np.clip(new_score, 0, 100))