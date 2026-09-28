"""
eval/visualize_aapl.py
======================
对应论文 5.7 累计收益曲线 (P0):

画出测试期内各方法的 equity curve (V_0=1).
根据命令行传入的 --methods 动态加载对比基线，
并根据 --output_dir 动态保存图表。
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from common import (ACTION_CSV, load_prices, load_actions,
                    align_actions_to_prices, equity_curve)

# 方法对应的 CSV 配置字典：(csv_path, date_column, action_column)
METHOD_CFG = {
    'DQN':         ('baselines/aapl_dqn.csv',               'date',  'action'),
    'PPO':         ('baselines/aapl_ppo.csv',               'date',  'action'),
    'A2C':         ('baselines/aapl_a2c.csv',               'date',  'action'),
    'FinGPT':      ('baselines/aapl_fingpt.csv',            'dates', 'actions'),
    'FinMem(old)': ('baselines/aapl_finmem_orig.csv',       'date',  'direction'),
    'Full(noCFR)': ('baselines/aapl_finmem_noCFR.csv',       'date',  'direction'),
    'Full(noETR)': ('baselines/aapl_finmem_noETR.csv',       'date',  'direction'),
    'Full(noRGR)': ('baselines/aapl_finmem_noRGR.csv',       'date',  'direction'),
    'Full(CFR)':   ('baselines/aapl_finmem_only_CFR.csv',   'date',  'direction'),
    'Full(ETR)':   ('baselines/aapl_finmem_only_ETR.csv',   'date',  'direction'),
    'Full(RGR)':   ('baselines/aapl_finmem_only_RGR.csv',   'date',  'direction'),
}

def main():
    parser = argparse.ArgumentParser(description="Visualize AAPL Equity Curve")
    parser.add_argument("--methods", nargs="+", default=["FinMem(old)", "Full"], help="要评估的方法列表")
    parser.add_argument("--output_dir", type=str, default="results", help="图表保存路径")
    args = parser.parse_args()

    price_dates, prices = load_prices()

    curves = {}
    curves["Buy & Hold"] = equity_curve(prices, np.ones(len(prices), dtype=int))

    # 本文完整方法 Full（硬编码走 ACTION_CSV）
    if "Full" in args.methods:
        d, a = load_actions(ACTION_CSV)
        ours = align_actions_to_prices(price_dates, d, a)
        curves["Full"] = equity_curve(prices, ours)

    # 遍历外部传入的其他对应方法
    for name in args.methods:
        if name == "Full":
            continue
        if name not in METHOD_CFG:
            print(f"[skip] 未定义此方法的配置文件: {name}")
            continue

        csv, cd, ca = METHOD_CFG[name]
        if not os.path.exists(csv):
            print(f"[skip] {name} 暂无 {csv}")
            continue

        d, a = load_actions(csv, col_date=cd, col_dir=ca)
        a = align_actions_to_prices(price_dates, d, a)
        curves[name] = equity_curve(prices, a)

    # ---- 画图 ----
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = pd.to_datetime(price_dates)

    style = {
        "Buy & Hold": dict(linewidth=1.6, linestyle="--", color="gray"),
        "Full":       dict(linewidth=2.2, color="#c0392b"),
    }
    for name, eq in curves.items():
        s = style.get(name, dict(linewidth=1.5))
        ax.plot(x, eq, label=name, **s)

    ax.axhline(1.0, color="black", linewidth=0.6, alpha=0.5)
    ax.set_title("AAPL profit", fontsize=13)
    ax.set_xlabel("date")
    ax.set_ylabel("profit (start=1.0)")
    ax.legend(loc="best", fontsize=10)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()

    # ---- 动态保存图表 ----
    os.makedirs(args.output_dir, exist_ok=True)
    out_png = os.path.join(args.output_dir, "cumret_aapl.png")
    plt.savefig(out_png, dpi=150)
    print(f"✓ 图已保存: {out_png}")

if __name__ == "__main__":
    main()
