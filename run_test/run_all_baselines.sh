#!/bin/bash
# ============================================================
#  基线对比实验统一入口脚本
#
#  使用方式:
#    bash run_all_baselines.sh finrl        # 只跑 DQN/PPO/A2C
#    bash run_all_baselines.sh fingpt       # 只跑 FinGPT 情感分析
#    bash run_all_baselines.sh finmem-orig  # 只跑 FinMem(原版)
#    bash run_all_baselines.sh all          # 跑全部基线
#    bash run_all_baselines.sh eval         # 只跑评估
#
#  前提:
#    - .env 中 DEEPSEEK_API_KEY 有效 (FinMem 基线需要)
#    - data/03_model_input/env_data.pkl 已生成
#    - stable-baselines3 已安装 (RL 基线需要)
#      pip install stable-baselines3 gymnasium
#    - transformers 已安装 (FinGPT 基线需要, 会自动回退到关键词)
#      pip install transformers
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -z "$VIRTUAL_ENV" ] && [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

run_finrl() {
  echo ""
  echo "=========================================="
  echo "  FinRL Baselines: DQN / PPO / A2C"
  echo "=========================================="
  echo "  预计耗时: 5~15 分钟 (本地 CPU)"
  echo ""
  python scripts/run_finrl_baselines.py
}

run_fingpt() {
  echo ""
  echo "=========================================="
  echo "  FinGPT Sentiment Baseline"
  echo "=========================================="
  echo "  预计耗时: 1~10 分钟"
  echo ""
  python scripts/run_fingpt_baseline.py
}

run_finmem_orig() {
  echo ""
  echo "=========================================="
  echo "  FinMem(Original) Baseline"
  echo "  RGR=OFF  CFR=OFF  ETR=OFF"
  echo "=========================================="
  echo "  预计耗时: 4~8 小时 (API 等待)"
  echo "  费用: ¥1~3 DeepSeek"
  echo ""
  python scripts/run_finmem_orig_baseline.py
}

run_eval() {
  echo ""
  echo "=========================================="
  echo "  Evaluation"
  echo "=========================================="
  echo ""

  # 检查 baselines CSV 是否存在
  echo "  Baseline CSVs:"
  for f in baselines/aapl_dqn.csv baselines/aapl_ppo.csv baselines/aapl_a2c.csv \
           baselines/aapl_fingpt.csv baselines/aapl_finmem_orig.csv; do
    if [ -f "$f" ]; then
      lines=$(wc -l < "$f")
      echo "    ✓ $f ($lines lines)"
    else
      echo "    ✗ $f (missing)"
    fi
  done

  # 检查本文方法动作 CSV
  if [ -f "aapl_gpt4.csv" ]; then
    lines=$(wc -l < "aapl_gpt4.csv")
    echo "    ✓ aapl_gpt4.csv ($lines lines)"
  else
    echo "    ✗ aapl_gpt4.csv (missing — run test first)"
  fi

  echo ""
  echo "  Running eval scripts..."
  PYTHONPATH=eval python eval/metrics_aapl.py
  PYTHONPATH=eval python eval/wilcoxon_aapl.py
  PYTHONPATH=eval python eval/visualize_aapl.py
  PYTHONPATH=eval python eval/regime_analysis.py

  echo ""
  echo "  Results:"
  ls -la results/*.csv results/*.png 2>/dev/null
}

case "${1:-}" in
  finrl)        run_finrl ;;
  fingpt)       run_fingpt ;;
  finmem-orig)  run_finmem_orig ;;
  eval)         run_eval ;;
  all)
    run_finrl
    run_fingpt
    run_finmem_orig
    run_eval
    ;;
  *)
    echo "用法: bash run_all_baselines.sh {finrl|fingpt|finmem-orig|eval|all}"
    echo ""
    echo "  finrl        跑 DQN/PPO/A2C 基线 (需要 stable-baselines3)"
    echo "  fingpt       跑 FinGPT 情感分析基线"
    echo "  finmem-orig  跑 FinMem(原版) 基线 (需要 DeepSeek API)"
    echo "  eval         跑评估 (假设 CSV 已生成)"
    echo "  all          跑全部基线 + 评估"
    exit 1
    ;;
esac

echo ""
echo "✓ Done."