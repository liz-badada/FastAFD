#!/usr/bin/env bash
set -euo pipefail

runner=$(dirname "${BASH_SOURCE[0]}")/run_megamoe_colocated_model_benchmark.sh
export RESULTS_DIR=${RESULTS_DIR:-/workspace/results/megamoe-colocated-multimodel-smoke-suite}
export EP_SIZE=${EP_SIZE:-8}
export TOKENS_PER_RANK=${TOKENS_PER_RANK:-16}
export MTP_NEXTN=${MTP_NEXTN:-0}
export LAYERS=${LAYERS:-1}
export WARMUPS=${WARMUPS:-2}
export ITERATIONS=${ITERATIONS:-10}

for model in qwen3_235b_fp4 minimax_m25_fp4 minimax_m3_fp4 \
  deepseek_v4_flash_fp4 deepseek_v4_pro_fp4; do
  MODEL=${model} "${runner}"
done
