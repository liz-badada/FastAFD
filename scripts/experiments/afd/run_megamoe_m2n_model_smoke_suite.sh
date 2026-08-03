#!/usr/bin/env bash
set -euo pipefail

runner=$(dirname "${BASH_SOURCE[0]}")/run_megamoe_m2n_model_benchmark.sh
export RESULTS_DIR=${RESULTS_DIR:-/workspace/results/megamoe-multimodel-smoke-suite}
export SEQUENCES_PER_AG_RANK=${SEQUENCES_PER_AG_RANK:-16}
export MICROBATCHES=${MICROBATCHES:-2}
export MTP_NEXTN=${MTP_NEXTN:-0}
export LAYERS=${LAYERS:-1}
export WARMUPS=${WARMUPS:-2}
export ITERATIONS=${ITERATIONS:-10}

for model in qwen3_235b_fp8 qwen3_235b_fp4 minimax_m25_fp8 minimax_m25_fp4 \
  minimax_m3_fp8 minimax_m3_fp4 \
  deepseek_v4_flash_fp8 deepseek_v4_flash_fp4; do
  MODEL=${model} AG_SIZE=4 EG_SIZE=4 "${runner}"
done

for model in deepseek_v4_pro_fp8 deepseek_v4_pro_fp4; do
  MODEL=${model} AG_SIZE=2 EG_SIZE=6 "${runner}"
done
