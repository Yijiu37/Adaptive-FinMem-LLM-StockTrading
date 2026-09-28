"""
eval/regime_analysis.py
=======================
对应论文 5.5 市场分阶段分析 (P1):

将测试期切分为若干子阶段, 各阶段独立计算 CR / SR / MDD.
默认按 PHASES 切成三段 (涨/跌/震);
也可以根据实际价格走势手动改 common.PHASES.
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import (ACTION_CSV, PHASES, load_prices, load_actions,
                    align_actions_to_prices, compute_all_metrics)

# 各种方法的配置字典
METHOD_CFG = {
    'DQN':         ('baselines/aapl_dqn.csv',               'date',  'action'),
    'PPO':         ('baselines/aapl_ppo.csv',               'date',  'action'),
    'A2C':         ('baselines/aapl_a2c.csv',               'date',  'action'),
    'FinGPT':      ('baselines/aapl_fingpt.csv',            'dates', 'actions'),
    'FinMem(old)': ('baselines/aapl_finmem_orig.csv',       'date',  'direction'),
    'Full(noCFR)': ('baselines/aapl_finmem_noCFR.csv',      'date',  'direction'),
    'Full(noETR)': ('baselines/aapl_finmem_noETR.csv',      'date',  'direction'),
    'Full(noRGR)': ('baselines/aapl_finmem_noRGR.csv',      'date',  'direction'),
    'Full(CFR)':   ('baselines/aapl_finmem_only_CFR.csv',   'date',  'direction'),
    'Full(ETR)':   ('baselines/aapl_finmem_only_ETR.csv',   'date',  'direction'),
    'Full(RGR)':   ('baselines/aapl_finmem_only_RGR.csv',   'date',  'direction'),
}

def slice_phase(price_dates, prices, actions, lo, hi):
    mask = np.array([(d >= lo) and (d <= hi) for d in price_dates])
    return prices[mask], actions[mask]

def main():
    parser = argparse.ArgumentParser(description="Regime Analysis for AAPL")
    parser.add_argument("--methods", nargs="+", default=["FinMem(old)", "Full"], help="要评估的方法列表")
    parser.add_argument("--output_dir", type=str, default="results", help="结果保存路径")
    args = parser.parse_args()

    price_dates, prices = load_prices()

    method_actions = {
        "Buy & Hold": np.ones(len(prices), dtype=int),
    }

    # 根据传入的方法动态加载
    for name in args.methods:
        if name == "Full":
            # 本文完整方法 Full（硬编码走 ACTION_CSV）
            d, a = load_actions(ACTION_CSV)
            method_actions["Full"] = align_actions_to_prices(price_dates, d, a)
        else:
            if name not in METHOD_CFG:
                print(f"[skip] 未定义此方法的配置文件: {name}")
                continue

            csv, cd, ca = METHOD_CFG[name]
            if not os.path.exists(csv):
                print(f"[skip] {name} 暂无 {csv}")
                continue

            d, a = load_actions(csv, col_date=cd, col_dir=ca)
            method_actions[name] = align_actions_to_prices(price_dates, d, a)

    # ---- 计算每阶段指标 ----
    rows = []
    for ph_name, lo, hi in PHASES:
        ph_idx = [i for i, d in enumerate(price_dates) if lo <= d <= hi]
        if len(ph_idx) < 5:
            print(f"[warn] {ph_name} too few trading days ({len(ph_idx)}), skip")
            continue

        sub_prices = prices[ph_idx]
        bh_ret = np.log(sub_prices[-1] / sub_prices[0])

        for m, a in method_actions.items():
            sub_a = a[ph_idx]
            metrics = compute_all_metrics(sub_prices, sub_a)
            rows.append({
                "phase": ph_name,
                "Interval": f"{lo} ~ {hi}",
                "Trading Day": len(ph_idx),
                "method": m,
                "CR": metrics["CR"],
                "SR": metrics["SR"],
                "MDD": metrics["MDD"],
                "B&H Log Returns": bh_ret,
            })

    if not rows:
        print("[warn] 没有计算任何阶段的数据，可能是 PHASES 日期或价格数据不匹配。")
        return

    df = pd.DataFrame(rows)
    print("\n=== Regime Metrics ===")
    print(df.to_markdown(index=False, floatfmt=".4f"))

    # ---- 动态保存指标文件 ----
    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, "regime_metrics_aapl.csv")
    out_png = os.path.join(args.output_dir, "regime_aapl.png")

    df.to_csv(out_csv, index=False)

    # ---- 画分阶段 CR 柱状图 ----
    # 确保存在不同方法进行分组，防止只有一种方法时报错
    if len(df['method'].unique()) > 0:
        pivot = df.pivot(index="phase", columns="method", values="CR")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        pivot.plot(kind="bar", ax=ax, edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_ylabel("CR profit (log return)")
        ax.set_title("AAPL compare of CR across different market regimes", fontsize=13)
        ax.legend(loc="best")
        ax.grid(True, axis="y", alpha=0.3)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)

    print(f"\n✓ 写出: {out_csv}")
    print(f"✓ 写出: {out_png}")

if __name__ == "__main__":
    main()
