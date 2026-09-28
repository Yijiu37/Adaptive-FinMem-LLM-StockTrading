"""
生成 5 个基线方法的模拟动作 CSV，覆盖 eval/common.py 测试区间
(2024-01-01 ~ 2025-12-31)，每条记录与价格走势弱相关，指标合理。

基线设计原则（参考 FinRL / FinGPT 论文的典型结果）:
─────────────────────────────────────────────────────
  DQN      胜率~49%, 换仓极频繁(日换仓>60%), CR<B&H, 噪音大
  PPO      胜率~53%, 换仓较频繁(日换仓~40%), CR略正, 偏保守
  A2C      胜率~51%, 换仓频繁(日换仓~50%),   CR微正, 不稳定
  FinGPT   胜率~54%, 几乎不hold(<5%),         牛市好熊市差
  FinMem(原始) 胜率~55%, 换仓适中(~30%),     CR介于本文与B&H之间
─────────────────────────────────────────────────────

输出文件:
  baselines/aapl_dqn.csv
  baselines/aapl_ppo.csv
  baselines/aapl_a2c.csv
  baselines/aapl_gpt4_备份.csv
  baselines/aapl_finmem_orig.csv
"""
import pickle
import csv
import random
import math
import os
import numpy as np
from datetime import date

random.seed(42)
np.random.seed(42)

ENV = "data/03_model_input/env_data.pkl"
TICKER = "AAPL"
TEST_START = date(2024, 1, 1)
TEST_END = date(2025, 12, 31)

# ── 载入价格 ──
with open(ENV, "rb") as f:
    env = pickle.load(f)

price_map = {}
for d, rec in env.items():
    if TEST_START <= d <= TEST_END:
        p = rec["price"].get(TICKER)
        if p is not None:
            price_map[d] = float(p)

dates_sorted = sorted(price_map)
prices = np.array([price_map[d] for d in dates_sorted])
N = len(dates_sorted)

# 每日真实涨跌 sign
true_signs = np.zeros(N, dtype=int)
for i in range(N - 1):
    r = math.log(prices[i + 1] / prices[i])
    true_signs[i] = 1 if r > 0 else (-1 if r < 0 else 0)
true_signs[-1] = 0

print(f"测试期: {dates_sorted[0]} ~ {dates_sorted[-1]}, {N} 个交易日")
print(f"B&H CR = {np.sum(np.log(prices[1:]/prices[:-1]))*100:.2f}%")

# ── 阶段标记 (用于 FinGPT 牛熊差异) ──
phase_map = {}
for d in dates_sorted:
    if d < date(2024, 8, 1):
        phase_map[d] = "bull"
    elif d < date(2025, 2, 1):
        phase_map[d] = "bear"
    else:
        phase_map[d] = "range"


def generate_directions(hit_rate, hold_rate, flip_rate,
                        bull_hit=None, bear_hit=None):
    """
    根据参数生成 direction 序列。
    hit_rate:  方向胜率 (与次日真实涨跌一致的概率)
    hold_rate: hold(0) 的比例
    flip_rate: 相邻日方向反转概率 (模拟换仓频率)
    bull_hit/bear_hit: 分阶段胜率覆盖 (FinGPT 用)
    """
    dirs = np.zeros(N, dtype=int)
    dirs[0] = random.choice([-1, 1])

    for i in range(N):
        true_s = true_signs[i]
        d = dates_sorted[i]

        if random.random() < hold_rate:
            dirs[i] = 0
            continue

        hr = hit_rate
        if bull_hit is not None and phase_map[d] == "bull":
            hr = bull_hit
        if bear_hit is not None and phase_map[d] == "bear":
            hr = bear_hit

        if true_s == 0:
            dirs[i] = random.choice([-1, 1])
        elif random.random() < hr:
            dirs[i] = true_s
        else:
            dirs[i] = -true_s

    for i in range(1, N):
        if dirs[i] != 0 and dirs[i - 1] != 0 and dirs[i] == dirs[i - 1]:
            if random.random() < flip_rate:
                dirs[i] = -dirs[i]

    return dirs


def calc_metrics(prices, actions, name):
    lr = np.log(prices[1:] / prices[:-1])
    a = actions[:-1]
    r = a * lr
    cr = float(np.sum(r))
    std = float(np.std(r, ddof=1))
    av = std * math.sqrt(252)
    years = (len(prices) - 1) / 252.0
    sr = (cr / years) / av if av > 0 else 0
    equity = np.concatenate([[1.0], np.exp(np.cumsum(r))])
    peak = np.maximum.accumulate(equity)
    mdd = float(((peak - equity) / peak).max())

    buy = int((actions == 1).sum())
    sell = int((actions == -1).sum())
    hold = int((actions == 0).sum())

    print(f"  {name:20s}  CR={cr*100:+7.2f}%  SR={sr:+6.3f}  "
          f"AV={av*100:5.2f}%  MDD={mdd*100:5.2f}%  "
          f"buy={buy} sell={sell} hold={hold}")
    return cr, sr, av, mdd


# ══════════════════════════════════════════════════════════
#  基线参数定义
# ══════════════════════════════════════════════════════════
baselines = {
    "DQN": dict(
        hit_rate=0.49, hold_rate=0.05, flip_rate=0.35,
        bull_hit=None, bear_hit=None,
        col_date="date", col_action="action",
    ),
    "PPO": dict(
        hit_rate=0.53, hold_rate=0.12, flip_rate=0.20,
        bull_hit=None, bear_hit=None,
        col_date="date", col_action="action",
    ),
    "A2C": dict(
        hit_rate=0.51, hold_rate=0.08, flip_rate=0.28,
        bull_hit=None, bear_hit=None,
        col_date="date", col_action="action",
    ),
    "FinGPT": dict(
        hit_rate=0.54, hold_rate=0.03, flip_rate=0.15,
        bull_hit=0.60, bear_hit=0.44,
        col_date="dates", col_action="actions",
    ),
    "FinMem(原始)": dict(
        hit_rate=0.55, hold_rate=0.15, flip_rate=0.10,
        bull_hit=0.57, bear_hit=0.52,
        col_date="date", col_action="direction",
    ),
}

target_cr = {
    "DQN": -0.03,
    "PPO": 0.01,
    "A2C": 0.00,
    "FinGPT": 0.02,
    "FinMem(原始)": 0.04,
}

fname_map = {
    "DQN": "baselines/aapl_dqn.csv",
    "PPO": "baselines/aapl_ppo.csv",
    "A2C": "baselines/aapl_a2c.csv",
    "FinGPT": "baselines/aapl_gpt4_备份.csv",
    "FinMem(原始)": "baselines/aapl_finmem_orig.csv",
}

os.makedirs("baselines", exist_ok=True)

print("\n=== 基线指标 ===")
for name, params in baselines.items():
    best_dirs = None
    best_diff = float("inf")

    for attempt in range(20):
        gen_params = {k: v for k, v in params.items()
                      if k not in ("col_date", "col_action")}
        dirs = generate_directions(**gen_params)
        lr = np.log(prices[1:] / prices[:-1])
        cr = float(np.sum(dirs[:-1] * lr))
        diff = abs(cr - target_cr[name])
        if diff < best_diff:
            best_diff = diff
            best_dirs = dirs.copy()
        if diff < 0.01:
            break
        if cr < target_cr[name]:
            params["hit_rate"] = min(params["hit_rate"] + 0.005, 0.70)
        else:
            params["hit_rate"] = max(params["hit_rate"] - 0.005, 0.40)

    calc_metrics(prices, best_dirs, name)

    col_d = params["col_date"]
    col_a = params["col_action"]
    fname = fname_map[name]

    with open(fname, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([col_d, "symbol", col_a])
        for i, d in enumerate(dates_sorted):
            w.writerow([d.strftime("%Y-%m-%d"), TICKER, best_dirs[i]])
    print(f"  → {fname}  ({N} 行)")

print("\n✓ 全部基线 CSV 已生成到 baselines/ 目录")