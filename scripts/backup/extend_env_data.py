"""
scripts/extend_env_data.py
==========================
为 data/03_model_input/env_data.pkl 生成 2010-01-01 ~ 2025-12-31 的
多行情阶段仿真数据，并保留原真实数据区间 (2014-01-02 ~ 2015-12-31)。

改进点（相比上一版）：
  1. 按规划好的"行情阶段表"分段生成价格 —— 牛/熊/高波动/震荡交替出现，
     接近真实市场的非平稳特性。
  2. 新闻按当天所处的阶段类型挑模板，情绪与价格方向自洽（牛市多利好、熊市多利空、
     高波动多不确定信号）。
  3. 在阶段切换点采用"软过渡"：用前后两段参数的加权平均构造过渡日，避免硬拐点。
  4. 保持 pickle 结构 100% 兼容 —— 每日仍然是
     {"price": {TICKER: float}, "news": {TICKER: [str]}, "filing_q": {}, "filing_k": {}}

输出:
  - data/03_model_input/env_data.pkl      （覆盖）
  - data/03_model_input/env_data.pkl.bak  （首次运行生成的备份，不会重复覆盖）
  - data/03_model_input/phase_schedule.csv（本次生成的阶段表，便于复查/作图）
"""
import os
import csv
import pickle
import shutil
from datetime import date, timedelta

import numpy as np

ENV_PKL = "data/03_model_input/env_data.pkl"
BACKUP_PKL = "data/03_model_input/env_data.pkl.bak"
PHASE_CSV = "data/03_model_input/phase_schedule.csv"
TICKER = "AAPL"

# ------------------ 目标区间 ------------------
FULL_START = date(2010, 1, 1)
FULL_END = date(2025, 12, 31)

# ------------------ 四类行情参数 (年化) ------------------
# drift = 年化收益率, vol = 年化波动率
REGIME_PARAMS = {
    "bull":     dict(drift=+0.35, vol=0.22),   # 单边上涨
    "bear":     dict(drift=-0.30, vol=0.30),   # 单边下跌
    "high_vol": dict(drift=+0.05, vol=0.55),   # 高波动，方向不定
    "range":    dict(drift=+0.02, vol=0.14),   # 震荡盘整
}

# ------------------ 预设行情时间线 ------------------
# 说明: 这是一个"近似历史+合理交替"的阶段表
# - 2010 ~ 2013 前半真实 AAPL 经历了牛市+调整
# - 2020 疫情 + 2022 熊市 + 2023 AI 反弹 等均被尊重
# - 中间会避开真实数据区间 (2014-01-02 ~ 2015-12-31)
SCHEDULE = [
    ("2010-01-01", "2010-05-31", "bull"),
    ("2010-06-01", "2010-08-31", "range"),
    ("2010-09-01", "2011-05-31", "bull"),
    ("2011-06-01", "2011-10-31", "high_vol"),   # 欧债危机
    ("2011-11-01", "2012-09-30", "bull"),       # iPhone 周期
    ("2012-10-01", "2013-06-30", "bear"),       # $aapl 从 $100→$55
    ("2013-07-01", "2014-01-01", "range"),
    # 2014-01-02 ~ 2015-12-31 真实数据区间，跳过
    ("2016-01-01", "2017-12-31", "bull"),
    ("2018-01-01", "2018-03-31", "range"),
    ("2018-04-01", "2018-10-31", "bull"),
    ("2018-11-01", "2018-12-31", "bear"),       # 年底急跌
    ("2019-01-01", "2019-12-31", "bull"),
    ("2020-01-01", "2020-02-28", "range"),
    ("2020-03-01", "2020-04-15", "high_vol"),   # 疫情恐慌
    ("2020-04-16", "2020-12-31", "bull"),
    ("2021-01-01", "2021-12-31", "bull"),       # 持续牛市
    ("2022-01-01", "2022-10-31", "bear"),       # 加息熊市
    ("2022-11-01", "2023-01-31", "range"),
    ("2023-02-01", "2023-12-31", "bull"),       # AI 行情
    ("2024-01-01", "2024-07-31", "bull"),
    ("2024-08-01", "2024-09-30", "high_vol"),   # 日元套息逆转
    ("2024-10-01", "2025-06-30", "bull"),
    ("2025-07-01", "2025-09-30", "range"),
    ("2025-10-01", "2025-12-31", "bear"),       # 假设的年末回调
]

# ------------------ 新闻模板 (按情绪分类) ------------------
NEWS_BY_REGIME = {
    "bull": [
        "$ aapl breaks key resistance; analysts raise price target to ${target}",
        "$ aapl quarterly revenue beats estimates, iphone sales up {pct}% yoy",
        "apple unveils next-gen chip roadmap; $ aapl up {pct}% in pre-market",
        "$ aapl services segment hits record high, margins expand further",
        "$ aapl announces ${buyback}b share buyback and dividend hike",
        "$ aapl upgraded to overweight by major bank, target ${target}",
        "apple pay volume hits new record, $ aapl fintech story strengthens",
        "$ aapl wearables revenue crosses new quarterly milestone",
        "analyst: $ aapl services ltv continues to expand, model intact",
        "$ aapl ai strategy gains traction with enterprise customers",
    ],
    "bear": [
        "$ aapl trims production orders as inventories normalize",
        "regulatory probe into app store fees weighs on $ aapl sentiment",
        "$ aapl vision pro shipments disappoint, shares slip {pct}%",
        "greater china sales soften {pct}% - fx and competition cited ($ aapl)",
        "macro headwinds drag tech sector lower; $ aapl retraces {pct}%",
        "antitrust lawsuit updates: $ aapl faces fresh scrutiny in eu",
        "$ aapl downgraded by major bank, price target cut to ${target}",
        "weak guidance prompts $ aapl to break below 50dma, support at ${target}",
        "$ aapl supplier warns on demand; shares down {pct}% aftermarket",
        "fund managers trim $ aapl exposure amid rate-hike concerns",
    ],
    "high_vol": [
        "wall st. volatility spikes; $ aapl swings {pct}% intraday on macro news",
        "$ aapl halted briefly on unusual order flow; resumes with heavy volume",
        "options premium surges on $ aapl ahead of fed decision",
        "rumor: apple targeting foldable iphone launch in {year} ($ aapl)",
        "mixed signals: $ aapl services beat but hardware guide light",
        "$ aapl gap down on macro shock, buyers step in at ${target}",
        "volatility etfs see inflows; $ aapl among most-traded mega caps",
        "divergent analyst calls create tug-of-war in $ aapl",
        "large dark-pool prints flagged on $ aapl, direction unclear",
        "$ aapl correlates tightly with vix this week, intraday noise high",
    ],
    "range": [
        "$ aapl consolidates near ${target} as traders wait for catalyst",
        "volume dries up on $ aapl; market digesting last earnings print",
        "$ aapl dividend yield attracts income-focused funds amid volatility",
        "wall st. eyes $ aapl as defensive mega-cap in choppy tape",
        "sector rotation leaves $ aapl range-bound this week",
        "options flow neutral on $ aapl; iv compresses to multi-month low",
        "$ aapl holds support at ${target}, resistance at ${res}",
        "analysts maintain hold on $ aapl pending product launch",
        "$ aapl sideways as market awaits macro data",
        "retail sentiment on $ aapl balanced between bulls and bears",
    ],
}

# 真实区间如果出现需要同时合成新闻的情况（默认不会发生）用通用池
GENERIC_NEWS = NEWS_BY_REGIME["range"]


# ------------------ 工具函数 ------------------
def is_weekday(d):
    return d.weekday() < 5


def iter_weekdays(start, end):
    cur = start
    while cur <= end:
        if is_weekday(cur):
            yield cur
        cur += timedelta(days=1)


def gbm_path(n, drift_annual, vol_annual, rng):
    """给定年化漂移/波动率, 生成 n 个日度对数收益率"""
    daily_drift = drift_annual / 252
    daily_vol = vol_annual / np.sqrt(252)
    return rng.normal(daily_drift, daily_vol, size=n)


def parse_date(s):
    return date.fromisoformat(s)


def synth_news(rng, regime):
    """基于 regime 情绪合成 3-6 条新闻, 整体拼成一条大字符串"""
    pool = NEWS_BY_REGIME.get(regime, GENERIC_NEWS)
    n = int(rng.integers(3, 7))
    items = []
    for _ in range(n):
        tmpl = pool[int(rng.integers(0, len(pool)))]
        text = tmpl.format(
            target=int(rng.integers(100, 260)),
            res=int(rng.integers(120, 280)),
            pct=round(float(rng.uniform(0.5, 8.0)), 1),
            buyback=int(rng.integers(50, 120)),
            gen=int(rng.integers(12, 20)),
            year=int(rng.integers(2024, 2028)),
        )
        items.append(text)
    return [" | ".join(items)]


# ------------------ 主逻辑 ------------------
def build_day_regime_map():
    """日期 -> 阶段标签"""
    m = {}
    for s, e, r in SCHEDULE:
        ds, de = parse_date(s), parse_date(e)
        for d in iter_weekdays(ds, de):
            m[d] = r
    return m


def generate_synthetic_segment(real_anchor_price, days, regime_map, seed,
                               direction="forward"):
    """
    在给定交易日序列 days 上, 根据每日 regime 生成价格路径。
    direction='forward': 从 real_anchor_price 往后生成
    direction='backward': 终点对齐 real_anchor_price
    """
    rng = np.random.default_rng(seed)
    if len(days) == 0:
        return np.array([])

    # 收集每日的日度漂移/波动率
    daily_drift = np.zeros(len(days))
    daily_vol = np.zeros(len(days))
    for i, d in enumerate(days):
        regime = regime_map.get(d, "range")
        p = REGIME_PARAMS[regime]
        daily_drift[i] = p["drift"] / 252
        daily_vol[i] = p["vol"] / np.sqrt(252)

    # 软过渡: 对相邻 5 日做简单平滑, 避免阶段硬拐点
    kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    if len(daily_drift) >= 5:
        daily_drift = np.convolve(daily_drift, kernel, mode="same")
        daily_vol = np.convolve(daily_vol, kernel, mode="same")

    shocks = rng.normal(daily_drift, daily_vol)
    log_path = np.cumsum(shocks)

    if direction == "forward":
        prices = real_anchor_price * np.exp(log_path)
    else:
        # backward: 让最后一项 = real_anchor_price
        fwd = np.exp(log_path)
        p0 = real_anchor_price / fwd[-1]
        prices = p0 * fwd
    return prices


def main():
    print("[1] 加载原始 env_data.pkl ...")
    with open(ENV_PKL, "rb") as f:
        env = pickle.load(f)
    orig_keys = sorted(env.keys())
    orig_start, orig_end = orig_keys[0], orig_keys[-1]
    print(f"    原始范围: {orig_start} ~ {orig_end} ({len(orig_keys)} 条)")

    if not os.path.exists(BACKUP_PKL):
        shutil.copy(ENV_PKL, BACKUP_PKL)
        print(f"    已备份到 {BACKUP_PKL}")

    price_start_real = env[orig_start]["price"][TICKER]
    price_end_real = env[orig_end]["price"][TICKER]

    # 2) 构造阶段映射
    regime_map = build_day_regime_map()
    print(f"[2] 阶段表展开: 覆盖 {len(regime_map)} 个交易日")

    # 3) 收集前/后段交易日
    pre_days = [d for d in iter_weekdays(FULL_START, orig_start - timedelta(days=1))]
    post_days = [d for d in iter_weekdays(orig_end + timedelta(days=1), FULL_END)]
    print(f"[3] 前段 {len(pre_days)} 天, 后段 {len(post_days)} 天")

    # 4) 生成价格
    pre_prices = generate_synthetic_segment(
        price_start_real, pre_days, regime_map, seed=42, direction="backward")
    post_prices = generate_synthetic_segment(
        price_end_real, post_days, regime_map, seed=123, direction="forward")

    # 5) 合成新闻并组装 env
    print("[4] 合成情绪化新闻 ...")
    rng_news = np.random.default_rng(2024)

    new_env = {}
    for d, p in zip(pre_days, pre_prices):
        regime = regime_map.get(d, "range")
        new_env[d] = {
            "price": {TICKER: float(p)},
            "news": {TICKER: synth_news(rng_news, regime)},
            "filing_q": {},
            "filing_k": {},
        }
    for d in orig_keys:
        new_env[d] = env[d]
    for d, p in zip(post_days, post_prices):
        regime = regime_map.get(d, "range")
        new_env[d] = {
            "price": {TICKER: float(p)},
            "news": {TICKER: synth_news(rng_news, regime)},
            "filing_q": {},
            "filing_k": {},
        }

    # 6) 打印阶段统计 + 抽样
    keys_sorted = sorted(new_env.keys())
    print(f"\n[5] 合成完成: 共 {len(new_env)} 条, 范围 {keys_sorted[0]} ~ {keys_sorted[-1]}")
    # 衔接跳变
    p_pre_end = new_env[pre_days[-1]]["price"][TICKER] if pre_days else price_start_real
    p_real_begin = new_env[orig_start]["price"][TICKER]
    jump1 = abs(p_real_begin - p_pre_end) / p_pre_end * 100
    p_real_end = new_env[orig_end]["price"][TICKER]
    p_post_start = new_env[post_days[0]]["price"][TICKER] if post_days else price_end_real
    jump2 = abs(p_post_start - p_real_end) / p_real_end * 100
    print(f"    衔接跳变: 前段→真实 {jump1:.2f}%,  真实→后段 {jump2:.2f}%")

    # 按阶段统计
    from collections import Counter
    c = Counter()
    for d, r in regime_map.items():
        if d in new_env:
            c[r] += 1
    real_days = sum(1 for d in orig_keys if d not in regime_map)
    print(f"    阶段分布 (仅合成部分): {dict(c)}")
    print(f"    真实数据: {real_days} 天未打标签（保持原样）")

    # 7) 导出阶段表
    with open(PHASE_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start", "end", "regime"])
        for s, e, r in SCHEDULE:
            w.writerow([s, e, r])
    print(f"[6] 阶段表已写出: {PHASE_CSV}")

    # 8) 写回 pickle
    with open(ENV_PKL, "wb") as f:
        pickle.dump(new_env, f)
    print(f"[7] 写回 {ENV_PKL}  ✓")

    # 9) 抽样展示
    print("\n" + "=" * 70)
    print("抽样展示: 每个阶段取 2 天 (价格 + 新闻前 180 字)")
    print("=" * 70)
    seen = {}
    sample_list = []
    for d in keys_sorted:
        r = regime_map.get(d)
        if r and seen.get(r, 0) < 2:
            seen[r] = seen.get(r, 0) + 1
            sample_list.append((d, r))
    for d, r in sample_list:
        rec = new_env[d]
        price = rec["price"][TICKER]
        news_text = rec["news"][TICKER][0]
        print(f"\n【{r.upper():8s}】 {d}   price = ${price:.2f}")
        print(f"  news: {news_text[:180]}...")


if __name__ == "__main__":
    main()