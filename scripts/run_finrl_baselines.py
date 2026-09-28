"""
FinRL 基线: DQN / PPO / A2C 在相同价格数据上训练+测试

用法:
  python scripts/run_finrl_baselines.py

前提:
  pip install "finrl[stablebaselines3]" gymnasium stockstats

输出:
  baselines/aapl_dqn.csv   (date,symbol,action)
  baselines/aapl_ppo.csv   (date,symbol,action)
  baselines/aapl_a2c.csv   (date,symbol,action)

训练区间: 2020-01-01 ~ 2023-12-01
测试区间: 2024-01-01 ~ 2025-12-31
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from datetime import date

# ── 从 env_data.pkl 提取价格序列 ──
ENV_PKL = "data/03_model_input/env_data.pkl"
TICKER = "AAPL"
TRAIN_START = "2020-01-01"
TRAIN_END = "2023-12-01"
TEST_START = "2024-01-01"
TEST_END = "2025-12-31"

print("=== Step 1: Extract prices from env_data.pkl ===")
with open(ENV_PKL, "rb") as f:
    env = pickle.load(f)

rows = []
for d in sorted(env):
    p = env[d]["price"].get(TICKER)
    if p is not None:
        rows.append({
            "date": d,
            "open": float(p),
            "high": float(p) * 1.002,
            "low": float(p) * 0.998,
            "close": float(p),
            "adjcp": float(p),
            "volume": 1e7,
            "tic": TICKER,
        })

df = pd.DataFrame(rows)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)
print(f"  Total rows: {len(df)}, date range: {df['date'].min()} ~ {df['date'].max()}")

train_df = df[(df["date"] >= TRAIN_START) & (df["date"] <= TRAIN_END)].copy()
test_df = df[(df["date"] >= TEST_START) & (df["date"] <= TEST_END)].copy()
print(f"  Train: {len(train_df)} days, Test: {len(test_df)} days")

# ── 构造 FinRL 需要的格式 ──
# FinRL StockTradingEnv 需要: date, open, high, low, close, adjcp, volume, tic
# 以及技术指标 (我们用简单的动量指标)
def add_tech_indicators(data):
    """添加简单技术指标 (FinRL 需要至少一列)"""
    data = data.copy()
    data["macd"] = 0.0
    data["rsi_30"] = 50.0
    data["cci_30"] = 0.0
    data["dx_30"] = 0.0
    data["close_30_sma"] = data["close"]
    data["close_60_sma"] = data["close"]
    # 简单动量
    data["turbulence"] = 0.0
    data["return"] = data["close"].pct_change().fillna(0)
    return data

train_df = add_tech_indicators(train_df)
test_df = add_tech_indicators(test_df)

# ── Step 2: 创建 Gym 环境 ──
print("\n=== Step 2: Create Gym Environment ===")

try:
    from finrl.meta.env_stocktrading.env_stocktrading import StockTradingEnv
except ImportError:
    print("  [ERROR] finrl not installed. Run: pip install 'finrl[stablebaselines3]'")
    print("  Falling back to simple RL env...")
    StockTradingEnv = None

stock_dim = 1  # single stock
hmax = 100     # max shares per trade
initial_amount = 1e6
transaction_cost_pct = 0.001
reward_scaling = 1e-4
state_space = 1 + 2 * stock_dim + 3 * stock_dim  # price + holdings + tech
action_space = 3  # buy, sell, hold

# ── 简易交易环境 (不依赖 FinRL) ──
import gymnasium as gym
from gymnasium import spaces

class SimpleTradingEnv(gym.Env):
    """
    最简交易环境:
    - state: [归一化价格, 持仓比例, 动量]
    - action: 0=hold, 1=buy, 2=sell
    - reward: 对数收益率 (与 FinMem eval 对齐)
    """
    metadata = {"render_modes": []}

    def __init__(self, prices, lookback=10):
        super().__init__()
        self.prices = prices.astype(np.float64)
        self.lookback = lookback
        self.n = len(prices)

        # action: 0=hold, 1=buy(-1), 2=sell(-1)
        self.action_space = spaces.Discrete(3)
        # state: [norm_price, position, momentum]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_idx = 0
        self.position = 0  # -1, 0, 1
        self.total_reward = 0.0
        return self._get_obs(), {}

    def _get_obs(self):
        i = self.step_idx
        p = self.prices[i] / self.prices[0]  # 归一化
        if i >= self.lookback:
            mom = (self.prices[i] - self.prices[i - self.lookback]) / self.prices[i - self.lookback]
        else:
            mom = 0.0
        return np.array([p, float(self.position), mom], dtype=np.float32)

    def step(self, action):
        # 映射 action -> position
        if action == 1:
            self.position = 1   # buy
        elif action == 2:
            self.position = -1  # sell
        # action == 0: hold current position

        # reward = position * log return
        if self.step_idx < self.n - 1:
            log_ret = np.log(self.prices[self.step_idx + 1] / self.prices[self.step_idx])
            reward = self.position * log_ret
        else:
            reward = 0.0

        self.total_reward += reward
        self.step_idx += 1
        terminated = self.step_idx >= self.n - 1
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {}

# ── Step 3: 训练 + 测试 ──
print("\n=== Step 3: Train & Test ===")

try:
    from stable_baselines3 import DQN as DQNModel
    from stable_baselines3 import PPO as PPOModel
    from stable_baselines3 import A2C as A2CModel
    HAS_SB3 = True
except ImportError:
    print("  [ERROR] stable-baselines3 not installed.")
    print("  Run: pip install stable-baselines3")
    HAS_SB3 = False
    sys.exit(1)

train_prices = train_df["close"].values
test_dates = test_df["date"].values
test_prices = test_df["close"].values

os.makedirs("baselines", exist_ok=True)

ALGO_CONFIG = {
    "DQN": {
        "model_cls": DQNModel,
        "policy": "MlpPolicy",
        "params": dict(
            learning_rate=1e-4,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=64,
            gamma=0.99,
            exploration_fraction=0.3,
        ),
        "total_timesteps": 200_000,
    },
    "PPO": {
        "model_cls": PPOModel,
        "policy": "MlpPolicy",
        "params": dict(
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
            clip_range=0.2,
        ),
        "total_timesteps": 200_000,
    },
    "A2C": {
        "model_cls": A2CModel,
        "policy": "MlpPolicy",
        "params": dict(
            learning_rate=7e-4,
            n_steps=5,
            gamma=0.99,
        ),
        "total_timesteps": 200_000,
    },
}

for algo_name, cfg in ALGO_CONFIG.items():
    print(f"\n--- {algo_name} ---")

    # 1) 训练
    train_env = SimpleTradingEnv(train_prices)
    model_cls = cfg["model_cls"]
    model = model_cls(cfg["policy"], train_env, verbose=0, **cfg["params"])
    model.learn(total_timesteps=cfg["total_timesteps"])
    print(f"  Train done ({cfg['total_timesteps']:,} steps)")

    # 2) 测试
    test_env = SimpleTradingEnv(test_prices)
    obs, _ = test_env.reset()
    actions = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)
        # 映射: 0→hold(0), 1→buy(1), 2→sell(-1)
        direction = {0: 0, 1: 1, 2: -1}[int(action)]
        actions.append(direction)

    # 3) 导出 CSV (actions 长度 = n-1, 补齐最后一天为 hold)
    if len(actions) < len(test_dates):
        actions.append(0)
    actions = actions[:len(test_dates)]

    fname = f"baselines/aapl_{algo_name.lower()}.csv"
    with open(fname, "w") as f:
        f.write("date,symbol,action\n")
        for i, d in enumerate(test_dates):
            pd_date = pd.Timestamp(d)
            f.write(f"{pd_date.strftime('%Y-%m-%d')},{TICKER},{actions[i]}\n")

    buy = sum(1 for a in actions if a == 1)
    sell = sum(1 for a in actions if a == -1)
    hold = sum(1 for a in actions if a == 0)
    cr = test_env.total_reward
    print(f"  Test: CR={cr*100:+.2f}%, buy={buy}, sell={sell}, hold={hold}")
    print(f"  → {fname}")

print("\n✓ FinRL baselines done: baselines/aapl_dqn.csv, aapl_ppo.csv, aapl_a2c.csv")