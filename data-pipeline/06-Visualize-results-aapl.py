"""
06-Visualize-results-aapl.py
============================
基于本地 env_data.pkl 价格 + aapl_gpt4_备份.csv 动作，
画出 AAPL 2015-07 ~ 2015-12 测试期的累计收益曲线。

用法:
    cd /Users/liuhanyu.666/FinMem
    .venv/bin/python data-pipeline/06-Visualize-results-aapl.py
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date, datetime

# ============ 配置区 ============
ENV_PKL = "data/03_model_input/env_data.pkl"
TICKER = "AAPL"
TEST_START = date(2015, 7, 1)
TEST_END = date(2015, 12, 31)

# 你的结果文件
RESULT_CSV = "aapl_gpt4_备份.csv"
COL_DATE = "date"
COL_DIR = "direction"

OUT_PNG = "results/cumret_aapl_visualize.png"
# ================================


def load_prices_from_pkl(env_path, ticker, start, end):
    """从 env_data.pkl 加载测试区间的每日收盘价"""
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
    prices = [r[1] for r in rows]
    return dates, prices


def load_actions_from_csv(csv_path, col_date, col_dir, start, end):
    """读取动作 CSV 并过滤到测试区间"""
    df = pd.read_csv(csv_path)
    df[col_date] = pd.to_datetime(df[col_date]).dt.date
    df = df[(df[col_date] >= start) & (df[col_date] <= end)]
    df = df.sort_values(col_date)
    return df[col_date].tolist(), df[col_dir].astype(int).tolist()


def align_actions(price_dates, action_dates, actions):
    """将动作对齐到价格日期（缺失填0=hold）"""
    m = dict(zip(action_dates, actions))
    return [m.get(d, 0) for d in price_dates]


def reward_list(price, actions):
    """计算累计对数收益序列"""
    reward = 0
    rw_list = [0]
    for i in range(len(price) - 1):
        reward += actions[i] * np.log(price[i + 1] / price[i])
        rw_list.append(reward)
    return rw_list


def main():
    # 1. 加载价格
    price_dates, prices = load_prices_from_pkl(ENV_PKL, TICKER, TEST_START, TEST_END)
    print(f"[info] 测试期: {TEST_START} ~ {TEST_END}, {len(prices)} 个交易日")
    print(f"[info] 价格区间: {min(prices):.2f} ~ {max(prices):.2f}")

    # 2. 加载动作并对齐
    a_dates, actions = load_actions_from_csv(RESULT_CSV, COL_DATE, COL_DIR, TEST_START, TEST_END)
    aligned = align_actions(price_dates, a_dates, actions)

    print(f"[info] 动作分布: buy={aligned.count(1)}, sell={aligned.count(-1)}, hold={aligned.count(0)}")

    # 3. 计算累计收益
    bh_actions = [1] * len(prices)
    bh_rw = reward_list(prices, bh_actions)
    full_rw = reward_list(prices, aligned)

    # 4. 画图
    dates = [datetime.combine(d, datetime.min.time()) for d in price_dates]

    labels = ["Buy & Hold", "FinMem (Ours)"]
    return_lists = [bh_rw, full_rw]
    colors = ["#000", "#d14749"]
    linestyles = ["-.", "-"]
    linewidths = [2.5, 3.5]
    alphas = [0.66, 1.0]

    fig, ax = plt.subplots(figsize=(14, 7))

    for returns, label, color, ls, alpha, lw in zip(
        return_lists, labels, colors, linestyles, alphas, linewidths
    ):
        ax.plot(dates, returns, label=label, color=color,
                linestyle=ls, alpha=alpha, linewidth=lw)

    ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
    ax.set_xlabel("Date", fontsize=14)
    ax.set_ylabel("Cumulative Log Return", fontsize=14)
    ax.set_title(f"{TICKER} Cumulative Return (2015-07 ~ 2015-12)", fontsize=16)
    ax.legend(fontsize=13, frameon=True)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()
    plt.tight_layout()

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150)
    print(f"\n✓ 图已保存: {OUT_PNG}")
    print(f"  B&H 最终累计收益: {bh_rw[-1]:.4f}")
    print(f"  FinMem (Ours) 最终累计收益: {full_rw[-1]:.4f}")


if __name__ == "__main__":
    main()