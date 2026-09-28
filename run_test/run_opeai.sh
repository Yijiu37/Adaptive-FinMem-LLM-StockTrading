#!/bin/bash
# AAPL 实验主控脚本
#
# 使用方式:
#   bash run_opeai.sh train                     # 只跑训练 (2020-01-01 ~ 2023-12-31 )
#   bash run_opeai.sh test                      # 只跑测试 (2024-01-01 ~ 2025-12-31 )
#   bash run_opeai.sh all                       # 训练 + 测试 + 评估 一条龙
#   bash run_opeai.sh train-resume              # 训练断点续跑
#   bash run_opeai.sh test-resume               # 测试断点续跑
#   bash run_opeai.sh eval                      # 仅跑评估 (假设 csv 已存在)
#
#   nohup bash run_opeai.sh test > test_run.log 2>&1 &   # 后台运行
#
# 前置条件:
#   - .env 中 OPENAI_API_KEY 余额充足 (跑前先 python scripts/verify_api.py)
#   - data/03_model_input/env_data.pkl 已生成
#   - 跑测试时 data/05_train_model_output/agent_1/ 必须存在 (训练产物)

set -e

# 切到脚本所在目录, 自动激活 venv (兼容 nohup 后台启动)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
if [ -z "$VIRTUAL_ENV" ] && [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

ENV_PKL="data/03_model_input/env_data.pkl"
CONFIG="config/aapl_gpt_config.toml"
TRAIN_CKP="data/06_train_checkpoint"
TRAIN_RP="data/05_train_model_output"
TEST_CKP="data/08_test_checkpoint"
TEST_RP="data/09_results"

run_train() {
  echo ">>> [TRAIN] 启动训练 (2020-01-01 ~ 2023-12-01)"
  python run.py sim \
    -mdp "$ENV_PKL" \
    -st  2020-01-01 \
    -et  2023-12-01 \
    -rm  train \
    -cp  "$CONFIG" \
    -ckp "$TRAIN_CKP" \
    -rp  "$TRAIN_RP"
}

run_train_resume() {
  echo ">>> [TRAIN] 续跑训练"
  python run.py sim-checkpoint \
    -ckp "$TRAIN_CKP" \
    -rp  "$TRAIN_RP" \
    -cp  "$CONFIG" \
    -rm  train
}

run_test() {
  echo ">>> [TEST] 启动测试 (2024-01-01 ~ 2024-12-31)"
  python run.py sim \
    -mdp "$ENV_PKL" \
    -st  2024-01-01 \
    -et  2025-12-31 \
    -rm  test \
    -cp  "$CONFIG" \
    -tap "$TRAIN_RP" \
    -ckp "$TEST_CKP" \
    -rp  "$TEST_RP"
}

run_test_resume() {
  echo ">>> [TEST] 续跑测试"
  python run.py sim-checkpoint \
    -ckp "$TEST_CKP" \
    -rp  "$TEST_RP" \
    -cp  "$CONFIG" \
    -rm  test
}

run_eval() {
  echo ""
  echo ">>> [EXPORT] 导出动作 CSV"
  python save_file.py
  echo ""
  echo ">>> [EVAL] 跑评估 (P0+P1)"
  PYTHONPATH=eval python eval/metrics_aapl.py
  PYTHONPATH=eval python eval/wilcoxon_aapl.py
  PYTHONPATH=eval python eval/visualize_aapl.py
  PYTHONPATH=eval python eval/regime_analysis.py
}

case "${1:-}" in
  train)         run_train ;;
  train-resume)  run_train_resume ;;
  test)          run_test ;;
  test-resume)   run_test_resume ;;
  eval)          run_eval ;;
  all)           run_train; run_test; run_eval ;;
  *)
    echo "用法: bash run_opeai.sh {train|train-resume|test|test-resume|eval|all}"
    exit 1
    ;;
esac