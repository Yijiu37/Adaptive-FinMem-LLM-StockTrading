"""
仅启用 ETR 对比实验
训练区间: 2020-01-01 ~ 2023-12-01
测试区间: 2024-01-01 ~ 2025-12-31

python -m scripts.run_only_ETR_baseline
"""
import os
import sys
import shutil
import pickle
from dotenv import load_dotenv
load_dotenv()

# ── 设置环境变量: 关闭另外两项改进 ──
os.environ["FINMEM_NOCFR"] = "1"
os.environ["FINMEM_NORGR"] = "1"

# ── 配置 ──
ENV_PKL = "data/03_model_input/env_data.pkl"
CONFIG = "config/aapl_gpt_config.toml"
TRAIN_CKP = "data/06_train_checkpoint_only_ETR"
TRAIN_RP = "data/05_train_model_output_only_ETR"
TEST_CKP = "data/08_test_checkpoint_only_ETR"
TEST_RP = "data/09_results_only_ETR"
TICKER = "AAPL"

TRAIN_START = "2020-01-01"
TRAIN_END = "2023-12-01"
TEST_START = "2024-01-01"
TEST_END = "2025-12-31"

OUT_CSV = "baselines/aapl_finmem_only_ETR.csv"


def clean_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def run_sim(mdp, st, et, rm, cp, ckp, rp, tap=None):
    """调用 run.py sim"""
    import toml
    from datetime import datetime
    from puppy import MarketEnvironment, LLMAgent, RunMode
    from tqdm import tqdm

    config = toml.load(cp)
    run_mode_var = RunMode.Train if rm == "train" else RunMode.Test

    with open(mdp, "rb") as f:
        env_data_pkl = pickle.load(f)

    environment = MarketEnvironment(
        symbol=config["general"]["trading_symbol"],
        env_data_pkl=env_data_pkl,
        start_date=datetime.strptime(st, "%Y-%m-%d").date(),
        end_date=datetime.strptime(et, "%Y-%m-%d").date(),
    )

    if run_mode_var == RunMode.Train:
        the_agent = LLMAgent.from_config(config)
    else:
        the_agent = LLMAgent.load_checkpoint(
            path=os.path.join(tap, "agent_1"),
            chat_config_override=config["chat"],
        )

    pbar = tqdm(total=environment.simulation_length)
    while True:
        the_agent.counter += 1
        market_info = environment.step()
        if market_info[-1]:
            break
        the_agent.step(market_info=market_info, run_mode=run_mode_var)
        pbar.update(1)
        the_agent.save_checkpoint(path=ckp, force=True)
        environment.save_checkpoint(path=ckp, force=True)

    the_agent.save_checkpoint(path=rp, force=True)
    environment.save_checkpoint(path=rp, force=True)
    return the_agent


def main():
    print("=" * 60)
    print("FinMem(_only_ETR) Baseline")
    print("RGR=OFF  CFR=OFF  ETR=ON")
    print("=" * 60)

    # ── Step 1: Train ──
    print(f"\n>>> [TRAIN] {TRAIN_START} ~ {TRAIN_END}")
    clean_dir(TRAIN_CKP)
    clean_dir(TRAIN_RP)

    agent = run_sim(
        mdp=ENV_PKL, st=TRAIN_START, et=TRAIN_END,
        rm="train", cp=CONFIG, ckp=TRAIN_CKP, rp=TRAIN_RP,
    )
    print("  Train done.")

    # ── Step 2: Test ──
    print(f"\n>>> [TEST] {TEST_START} ~ {TEST_END}")
    clean_dir(TEST_CKP)
    clean_dir(TEST_RP)

    agent = run_sim(
        mdp=ENV_PKL, st=TEST_START, et=TEST_END,
        rm="test", cp=CONFIG, ckp=TEST_CKP, rp=TEST_RP,
        tap=TRAIN_RP,
    )
    print("  Test done.")

    # ── Step 3: Export ──
    print(f"\n>>> [EXPORT] → {OUT_CSV}")
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df = agent.portfolio.get_action_df()
    df.write_csv(OUT_CSV)
    print(f"  Exported {df.shape[0]} actions to {OUT_CSV}")

    # ── Step 4: Quick eval ──
    import numpy as np
    with open(ENV_PKL, "rb") as f:
        env = pickle.load(f)
    from datetime import date
    ts = date(2024, 1, 1)
    te = date(2025, 12, 31)
    prices = []
    for d in sorted(env):
        if ts <= d <= te:
            p = env[d]["price"].get(TICKER)
            if p: prices.append(float(p))
    prices = np.array(prices)
    actions = df["direction"].to_numpy()[:len(prices)]
    lr = np.log(prices[1:] / prices[:-1])
    a = actions[:-1]
    cr = float(np.sum(a * lr))
    print(f"  FinMem(_only_ETR) CR={cr*100:+.2f}%")

    print("\n✓ FinMem(_only_ETR) baseline done")


if __name__ == "__main__":
    main()