"""
eval/metrics_aapl.py
====================
对应论文 5.3.1 主对比实验 (P0):

读取对应的 CSV 并计算打印 5 个指标。
根据命令行传入的 --methods 动态加载对比基线，
并根据 --output_dir 动态保存 CSV 表格。
"""
import os
import argparse
import pandas as pd
from common import (TICKER, TEST_START, TEST_END, ACTION_CSV,
                    load_prices, load_actions, align_actions_to_prices,
                    compute_all_metrics)

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

def fmt(v, kind):
    if kind in ("CR", "MDD"):
        return f"{v*100:.2f}%"
    if kind == "SR":
        return f"{v:.3f}"
    return f"{v*100:.4f}%"

def main():
    parser = argparse.ArgumentParser(description="Compute Metrics for AAPL")
    parser.add_argument("--methods", nargs="+", default=["FinMem(old)", "Full"], help="要评估的方法列表")
    parser.add_argument("--output_dir", type=str, default="results", help="指标结果保存路径")
    args = parser.parse_args()

    price_dates, prices = load_prices()
    print(f"[info] 测试期: {TEST_START} ~ {TEST_END}, "
          f"{len(prices)} 个交易日, "
          f"{TICKER} 价格区间 {prices.min():.2f}~{prices.max():.2f}")

    rows = {}
    rows["Buy & Hold"] = compute_all_metrics(prices, [1] * len(prices))

    # 根据传入的 arguments 构建数据
    for name in args.methods:
        if name == "Full":
            # 本文完整方法 Full（硬编码走 ACTION_CSV）
            a_dates, actions = load_actions(ACTION_CSV)
            actions = align_actions_to_prices(price_dates, a_dates, actions)
            rows["Full"] = compute_all_metrics(prices, actions)
        else:
            if name not in METHOD_CFG:
                print(f"[skip] 未定义此方法的配置文件: {name}")
                continue

            csv, cd, ca = METHOD_CFG[name]
            if not os.path.exists(csv):
                print(f"[skip] {name} 暂无 {csv}")
                continue

            d, a = load_actions(csv, col_date=cd, col_dir=ca)
            a = align_actions_to_prices(price_dates, d, a)
            rows[name] = compute_all_metrics(prices, a)

    df = pd.DataFrame(rows).T
    # 如果某些指标不存在（可能因为全为 0 导致），过滤一下防止报错
    cols_to_use = [c for c in ["CR", "SR", "DailySTD", "AV", "MDD"] if c in df.columns]
    df = df[cols_to_use]

    print("\n=== 主对比指标 (原始数值) ===")
    print(df.round(6).to_string())

    # markdown 友好版
    print("\n=== 主对比指标 (论文可读) ===")
    pretty = df.copy()
    for col in cols_to_use:
        pretty[col] = df[col].apply(lambda v: fmt(v, col))
    print(pretty.to_markdown())

    # ---- 动态保存指标文件 ----
    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, "metrics_aapl.csv")
    df.to_csv(out_csv)
    print(f"\n✓ 写出: {out_csv}")

if __name__ == "__main__":
    main()
