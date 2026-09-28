"""
eval/wilcoxon_aapl.py
=====================
对应论文 5.3.3 统计显著性检验 (P1):

使用 30 天滑动窗口计算累积收益率、夏普比率和最大回撤，
并对这些指标序列进行 Wilcoxon signed-rank test.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from common import (ACTION_CSV, load_prices, load_actions,
                    align_actions_to_prices, daily_log_return_series)

COMPARE = [
    ("Buy & Hold", None, None, None),
    ('DQN', 'baselines/aapl_dqn.csv', 'date', 'action'),
    ('PPO', 'baselines/aapl_ppo.csv', 'date', 'action'),
    ('A2C', 'baselines/aapl_a2c.csv', 'date', 'action'),
    ('FinGPT', 'baselines/aapl_fingpt.csv', 'dates', 'actions'),
    ('FinMem(old)', 'baselines/aapl_finmem_orig.csv', 'date', 'direction'),
]

OUT_CSV = "results/wilcoxon_metrics_aapl.csv"
WINDOW = 30 # 使用 30 天滑动窗口

def calc_rolling_metrics(r_series, window=WINDOW):
    """计算滚动窗口内的累积收益、夏普和最大回撤"""
    s = pd.Series(r_series)

    # 滚动累积收益率
    cum_ret = s.rolling(window).apply(lambda x: np.exp(np.sum(x)) - 1, raw=True)

    # 滚动夏普比率 (年化)
    def calc_sharpe(x):
        std = np.std(x)
        return np.sqrt(252) * np.mean(x) / std if std != 0 else 0.0
    sharpe = s.rolling(window).apply(calc_sharpe, raw=True)

    # 滚动最大回撤
    def calc_mdd(x):
        cum = np.exp(np.cumsum(x))
        peak = np.maximum.accumulate(cum)
        return np.min(cum / peak - 1)
    mdd = s.rolling(window).apply(calc_mdd, raw=True)

    return cum_ret.dropna().values, sharpe.dropna().values, mdd.dropna().values

def run_test_safe(ours, base):
    diff = ours - base
    if len(diff[diff != 0]) < 10:
        return float("nan"), float("nan"), False
    stat, p = wilcoxon(ours, base, zero_method="wilcox", alternative="two-sided")
    return float(stat), float(p), bool(p < 0.05)

def main():
    price_dates, prices = load_prices()

    a_dates, actions = load_actions(ACTION_CSV)
    ours = align_actions_to_prices(price_dates, a_dates, actions)
    r_ours = daily_log_return_series(prices, ours)

    cum_ours, shp_ours, mdd_ours = calc_rolling_metrics(r_ours)

    rows = []
    for name, csv, cd, ca in COMPARE:
        if name == "Buy & Hold":
            r_base = daily_log_return_series(prices, np.ones(len(prices), dtype=int))
        else:
            if not os.path.exists(csv):
                print(f"[skip] {name} 暂无 {csv}")
                continue
            d, a = load_actions(csv, col_date=cd, col_dir=ca)
            a = align_actions_to_prices(price_dates, d, a)
            r_base = daily_log_return_series(prices, a)

        cum_base, shp_base, mdd_base = calc_rolling_metrics(r_base)

        # 检验累积收益率
        stat_c, p_c, sig_c = run_test_safe(cum_ours, cum_base)
        # 检验夏普比率
        stat_s, p_s, sig_s = run_test_safe(shp_ours, shp_base)
        # 检验最大回撤
        stat_m, p_m, sig_m = run_test_safe(mdd_ours, mdd_base)

        rows.append((name,
                     stat_c, p_c, sig_c,
                     stat_s, p_s, sig_s,
                     stat_m, p_m, sig_m))

    cols = [
        "基线",
        "CumRet_Stat", "CumRet_p", "CumRet_Sig",
        "Sharpe_Stat", "Sharpe_p", "Sharpe_Sig",
        "MDD_Stat", "MDD_p", "MDD_Sig"
    ]
    df = pd.DataFrame(rows, columns=cols)
    print("\n=== Wilcoxon 配对检验 (30天滑动窗口指标) ===")
    print(df.to_markdown(index=False, floatfmt=".4f"))

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n✓ 写出: `{OUT_CSV}`")

if __name__ == "__main__":
    main()
