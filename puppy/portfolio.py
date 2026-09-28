"""
投资组合管理模块 (Portfolio)

改动说明：
  1.【小创新1 - 连续反馈强化】get_feedback_response()
      原始：根据窗口累计收益正负返回离散 -1/0/1。
      改进：使用 tanh(R / σ) 将收益标准化到 (-1, 1) 区间，
            保留反馈的「强度」信息，使下游记忆重要性更新更细粒度。
  2.【核心创新前置 - 市场状态识别】新增 get_market_regime()
      基于短窗口收益率均值（趋势）和滚动波动率（风险）
      将当前市场划分为 bull / bear / range / high_vol / neutral，
      供 BrainDB 在检索时选择对应的权重配置。
"""

import polars as pl
import numpy as np
from datetime import date
from annotated_types import Gt
from typing import Dict, Annotated, Union
from pydantic import BaseModel


class PriceStructure(BaseModel):
    price: Annotated[float, Gt(0)]


class Portfolio:
    def __init__(self, symbol: str, lookback_window_size: int = 7) -> None:
        self.cur_date = None
        self.symbol = symbol
        self.action_series = {}
        self.market_price = None
        self.day_count = 0
        self.date_series = []
        self.holding_shares = 0
        self.market_price_series = np.array([])
        self.portfolio_share_series = np.array([])
        self.lookback_window_size = lookback_window_size

    def update_market_info(self, new_market_price_info: float, cur_date: date) -> None:
        PriceStructure.model_validate({"price": new_market_price_info})
        self.market_price = new_market_price_info
        self.cur_date = cur_date
        self.date_series.append(cur_date)
        self.day_count += 1
        self.market_price_series = np.append(
            self.market_price_series, new_market_price_info
        )

    def record_action(self, action: Dict[str, int]) -> None:
        self.holding_shares += action["direction"]
        self.action_series[self.cur_date] = action["direction"]

    def get_action_df(self) -> pl.DataFrame:
        temp_dict = {"date": [], "symbol": [], "direction": []}
        for date in self.action_series:
            temp_dict["date"].append(date)
            temp_dict["symbol"].append(self.symbol)
            temp_dict["direction"].append(self.action_series[date])
        return pl.DataFrame(temp_dict)

    def update_portfolio_series(self) -> None:
        self.portfolio_share_series = np.append(
            self.portfolio_share_series, self.holding_shares
        )

        # ── 改动：连续反馈替代离散反馈 ─────────────────────────────
    def get_feedback_response(self) -> Union[Dict[str, Union[float, date]], None]:
        """
        计算滚动窗口内的风险调整连续反馈值。

        原始实现返回 {feedback: -1/0/1}；
        改进后返回 {feedback: tanh(R/σ)}，值域 (-1, 1)。
          - R   = 窗口内策略累计盈亏
          - σ   = 窗口内策略盈亏标准差（+ ε 防除零）
          - tanh 将无界收益压缩到有界区间，兼顾方向与强度。
        """
        if self.day_count <= self.lookback_window_size:
            return None

        price_diff = np.diff(self.market_price_series)
        shares = self.portfolio_share_series[:-1]

        # 对齐长度（与原始逻辑一致）
        if len(price_diff) != len(shares):
            pnl = (price_diff[:-1] * shares)[-self.lookback_window_size :]
        else:
            pnl = (price_diff * shares)[-self.lookback_window_size :]

        R = np.sum(pnl)                            # 窗口累计收益
        sigma = np.std(pnl) + 1e-8                 # 窗口波动率（防除零）

        import os
        if os.environ.get("FINMEM_ORIGINAL") == "1" or os.environ.get("FINMEM_NOCFR") == "1":
            # 原版: 离散反馈 -1/0/1
            if R > 0:
                continuous_feedback = 1.0
            elif R < 0:
                continuous_feedback = -1.0
            else:
                continuous_feedback = 0.0
        else:
            print("触发 CFR！")
            continuous_feedback = float(np.tanh(R / sigma))

        return {
            "feedback": continuous_feedback,
            "date": self.date_series[-self.lookback_window_size],
        }

    def get_moment(self, moment_window: int = 3) -> Union[Dict[str, int], None]:
        if self.day_count <= moment_window:
            return None

        temp = np.cumsum((np.diff(self.market_price_series))[-moment_window:])[-1]

        if temp > 0:
            return {"moment": 1, "date": self.date_series[-moment_window]}
        elif temp < 0:
            return {"moment": -1, "date": self.date_series[-moment_window]}
        else:
            return {"moment": 0, "date": self.date_series[-moment_window]}

    # ── 新增：市场状态识别 ────────────────────────────────────
    def get_market_regime(
        self,
        short_window: int = 5,
        vol_window: int = 10,
        vol_threshold: float = 0.03,
        trend_threshold: float = 0.005,
    ) -> str:
        """
        基于近期收益率的趋势与波动率判定当前市场状态。

        判定逻辑（优先级从高到低）：
          1. 波动率 > vol_threshold  → "high_vol"
          2. 短窗均值收益 > trend_threshold  → "bull"
          3. 短窗均值收益 < -trend_threshold → "bear"
          4. 否则 → "range"
          * 数据不足时返回 "neutral"（等价于不做状态调整）

        Args:
            short_window: 计算趋势强度的短窗口天数
            vol_window: 计算波动率的滚动窗口天数
            vol_threshold: 高波动阈值
            trend_threshold: 趋势阈值（绝对值）

        Returns:
            市场状态字符串，取值为 bull / bear / range / high_vol / neutral
        """
        import os
        if os.environ.get("FINMEM_ORIGINAL") == "1" or os.environ.get("FINMEM_NORGR") == "1":
            return "neutral"
        else:
            print("触发 RGR！")

        if self.day_count < vol_window + 1:
            return "neutral"

        prices = self.market_price_series[-(vol_window + 1) :]
        returns = np.diff(prices) / (prices[:-1] + 1e-12)   # 日收益率序列
        vol = float(np.std(returns))                         # 滚动波动率
        trend = float(np.mean(returns[-short_window:]))      # 短窗口趋势强度

        if vol > vol_threshold:
            return "high_vol"
        elif trend > trend_threshold:
            return "bull"
        elif trend < -trend_threshold:
            return "bear"
        else:
            return "range"