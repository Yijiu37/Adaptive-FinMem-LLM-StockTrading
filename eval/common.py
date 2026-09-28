"""
eval/common.py
==============
所有评估脚本共享的数据加载和指标函数。

输入:
  - aapl_gpt4.csv (date, symbol, direction)   本文方法的动作
  - data/03_model_input/env_data.pkl          权威价格来源
  - 测试区间: 2024-01-01 ~ 2024-12-31         与训练集严格分离

指标定义严格按 5.1.3 节公式实现:
  - CR    = Σ a_t * ln(P_{t+1}/P_t)
  - Daily STD = sqrt(Σ(r_t - r̄)² / (T-1))
  - AV    = STD * sqrt(252)
  - SR    = (CR/Y - Rf) / AV                  Rf=0
  - MDD   = max_{s<=t} (V_s - V_t)/V_s
"""
import pickle
import numpy as np
import pandas as pd
from datetime import date

ENV_PKL = "data/03_model_input/env_data.pkl"
ACTION_CSV = "aapl_gpt4.csv"
TICKER = "AAPL"

TEST_START = date(2024, 1, 1)
TEST_END = date(2025, 12, 31)

# 按 2024-2025 实际走势切分：4 个分阶段
PHASES = [
    ("Phase1 (Bull)", date(2024, 1, 1),  date(2024, 8, 2)),
    ("Phase2 (High-Vol)", date(2024, 8, 3),  date(2024, 12, 31)),
    ("Phase3 (Range)", date(2025, 1, 1),  date(2025, 6, 30)),
    ("Phase4 (Bear)", date(2025, 7, 1),  date(2025, 12, 31)),
]


def load_prices(env_path=ENV_PKL, ticker=TICKER,
                start=TEST_START, end=TEST_END):
    """从 env_data.pkl 拉出测试区间的每日收盘价"""
    with open(env_path, "rb") as f:
        env = pickle.load(f)
    rows = []
    for d, rec in env.items():
        if start <= d <= end:
            p = rec["price"].get(ticker)
            if p is not None:
                rows.append((d, float(p)))
    rows.sort()
    dates = [r[0] for r in rows]
    prices = np.array([r[1] for r in rows], dtype=float)
    return dates, prices


def load_actions(csv_path, col_date="date", col_dir="direction",
                 start=TEST_START, end=TEST_END):
    """读取动作 CSV，筛出测试区间"""
    df = pd.read_csv(csv_path)
    df[col_date] = pd.to_datetime(df[col_date]).dt.date
    df = df[(df[col_date] >= start) & (df[col_date] <= end)]
    df = df.sort_values(col_date)
    return df[col_date].tolist(), df[col_dir].astype(int).to_numpy()


def align_actions_to_prices(price_dates, action_dates, actions):
    """把 actions 对齐到 price_dates (价格为准，动作缺失则填 0=hold)"""
    m = {d: a for d, a in zip(action_dates, actions)}
    aligned = np.array([m.get(d, 0) for d in price_dates], dtype=int)
    return aligned


# ---------- 指标 ----------
def daily_log_return(prices, actions):
    """r_t = a_t * ln(P_{t+1}/P_t)"""
    lr = np.log(prices[1:] / prices[:-1])     # 长度 T-1
    a = actions[:-1]                          # 用 t 日动作决定 t→t+1 持仓
    return a * lr                             # 长度 T-1


def cumulative_return(prices, actions):
    return float(np.sum(daily_log_return(prices, actions)))


def daily_std(prices, actions):
    r = daily_log_return(prices, actions)
    return float(np.std(r, ddof=1))


def annualized_vol(prices, actions):
    return daily_std(prices, actions) * np.sqrt(252)


def sharpe_ratio(prices, actions, rf=0.0):
    """SR = (CR/Y - Rf) / AV"""
    cr = cumulative_return(prices, actions)
    years = (len(prices) - 1) / 252.0
    if years <= 0:
        return 0.0
    av = annualized_vol(prices, actions)
    if av == 0:
        return 0.0
    return (cr / years - rf) / av


def max_drawdown(prices, actions):
    """MDD = max_{s<=t} (V_s - V_t) / V_s ; V_t = exp(累积对数收益)"""
    r = daily_log_return(prices, actions)
    equity = np.concatenate([[1.0], np.exp(np.cumsum(r))])
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak
    return float(dd.max())


def compute_all_metrics(prices, actions):
    return {
        "CR": cumulative_return(prices, actions),
        "SR": sharpe_ratio(prices, actions),
        "DailySTD": daily_std(prices, actions),
        "AV": annualized_vol(prices, actions),
        "MDD": max_drawdown(prices, actions),
    }


def daily_log_return_series(prices, actions):
    """同名方法的带版本返回，供 Wilcoxon/可视化使用"""
    return daily_log_return(prices, actions)


def equity_curve(prices, actions):
    """累积净值序列 (V_0=1)"""
    r = daily_log_return(prices, actions)
    return np.concatenate([[1.0], np.exp(np.cumsum(r))])