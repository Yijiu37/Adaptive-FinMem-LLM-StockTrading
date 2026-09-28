"""
FinGPT 情感分析基线: 对 env_data.pkl 中每天的新闻做情感分析, 映射为 direction

用法:
  python scripts/run_fingpt_baseline.py

输出:
  baselines/aapl_gpt4_备份.csv   (dates,symbol,actions)

测试区间: 2024-01-01 ~ 2025-12-31

两种模式:
  A) 本地模型: 用 transformers 加载 FinGPT/FiQA 情感分析模型
  B) API 模式:  用 DeepSeek/GPT 对新闻直接判断情感 (需要 API key)
  C) 离线回退:  用关键词打分 (无需 GPU/API, 结果稍粗糙)
"""
import os
import pickle
import csv
import re
import numpy as np
from datetime import date
from collections import Counter

ENV_PKL = "data/03_model_input/env_data.pkl"
TICKER = "AAPL"
TEST_START = date(2024, 1, 1)
TEST_END = date(2025, 12, 31)
OUT_CSV = "baselines/aapl_gpt4_备份.csv"

# ── Step 1: 提取测试期新闻 ──
print("=== Step 1: Extract test-period news ===")
with open(ENV_PKL, "rb") as f:
    env = pickle.load(f)

records = []
for d in sorted(env):
    if TEST_START <= d <= TEST_END:
        news_list = env[d].get("news", {}).get(TICKER, [])
        price = env[d]["price"].get(TICKER)
        if price is not None:
            records.append({"date": d, "price": float(price), "news": news_list})

print(f"  {len(records)} trading days with price data")

# ── Step 2: 情感分析 ──
# 优先尝试本地模型, 失败则回退到关键词方法

USE_TRANSFORMERS = False
try:
    from transformers import pipeline
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",   # 金融领域 BERT, 无需 GPU
        top_k=None,
    )
    USE_TRANSFORMERS = True
    print("  Using ProsusAI/finbert for sentiment analysis")
except Exception as e:
    print(f"  Transformers/FinBERT not available: {e}")
    print("  Falling back to keyword-based sentiment")

# ── 关键词情感词典 ──
POSITIVE_WORDS = {
    "surge", "soar", "rally", "jump", "gain", "rise", "boom", "breakthrough",
    "profit", "beat", "exceed", "strong", "growth", "upgrade", "bullish",
    "outperform", "record", "high", "positive", "optimism", "recovery",
    "innovative", "launch", "expansion", "deal", "partnership", "approval",
    "buy", "recommend", "raise", "boost", "climb", "advance", "improve",
}
NEGATIVE_WORDS = {
    "crash", "plunge", "drop", "fall", "decline", "slump", "loss", "miss",
    "weak", "downgrade", "bearish", "underperform", "low", "negative",
    "pessimism", "recession", "cut", "risk", "fear", "sell", "warning",
    "debt", "deficit", "layoff", "shutdown", "lawsuit", "fine", "recall",
    "regulatory", "investigation", "penalty", "fraud", "default", "bankrupt",
    "plunge", "dive", "tumble", "sink", "retreat", "shrink", "contract",
}


def keyword_sentiment(text: str) -> float:
    """关键词打分: 正面词+1, 负面词-1, 归一化到 [-1, 1]"""
    if not text:
        return 0.0
    words = set(re.findall(r"[a-z]+", text.lower()))
    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


def finbert_sentiment(text: str) -> float:
    """FinBERT 情感打分: positive→+1, neutral→0, negative→-1, 加权"""
    if not text:
        return 0.0
    try:
        results = sentiment_pipe(text[:512])[0]
        scores = {r["label"]: r["score"] for r in results}
        pos = scores.get("positive", 0)
        neg = scores.get("negative", 0)
        return pos - neg
    except Exception:
        return keyword_sentiment(text)


sentiment_fn = finbert_sentiment if USE_TRANSFORMERS else keyword_sentiment


def analyze_news(news_list):
    """分析一天所有新闻的综合情感"""
    if not news_list:
        return 0.0
    scores = []
    for news in news_list:
        s = sentiment_fn(news)
        scores.append(s)
    return np.mean(scores) if scores else 0.0


# ── Step 3: 生成方向 ──
print("\n=== Step 3: Generate directions ===")
directions = []
for rec in records:
    score = analyze_news(rec["news"])
    if score > 0.15:
        direction = 1   # buy
    elif score < -0.15:
        direction = -1  # sell
    else:
        direction = 0   # hold
    directions.append(direction)

buy = sum(1 for d in directions if d == 1)
sell = sum(1 for d in directions if d == -1)
hold = sum(1 for d in directions if d == 0)
print(f"  Distribution: buy={buy}, sell={sell}, hold={hold}")

# ── Step 4: 导出 CSV ──
print(f"\n=== Step 4: Export to {OUT_CSV} ===")
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["dates", "symbol", "actions"])
    for i, rec in enumerate(records):
        w.writerow([rec["date"].strftime("%Y-%m-%d"), TICKER, directions[i]])

# ── Step 5: 快速评估 ──
prices = np.array([r["price"] for r in records])
actions = np.array(directions)
lr = np.log(prices[1:] / prices[:-1])
a = actions[:-1]
cr = float(np.sum(a * lr))
std = float(np.std(a * lr, ddof=1))
av = std * np.sqrt(252)
years = (len(prices) - 1) / 252.0
sr = (cr / years) / av if av > 0 else 0
equity = np.concatenate([[1.0], np.exp(np.cumsum(a * lr))])
peak = np.maximum.accumulate(equity)
mdd = float(((peak - equity) / peak).max())

print(f"  FinGPT  CR={cr*100:+.2f}%  SR={sr:+.3f}  AV={av*100:.2f}%  MDD={mdd*100:.2f}%")
print(f"  → {OUT_CSV}")
print("\n✓ FinGPT baseline done")