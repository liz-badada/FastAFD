#!/usr/bin/env bash
set -euo pipefail

runner=$(dirname "${BASH_SOURCE[0]}")/run_megamoe_colocated_model_benchmark.sh
export RESULTS_DIR=${RESULTS_DIR:-/workspace/results/megamoe-colocated-multimodel}
export EP_SIZE=${EP_SIZE:-8}
export BACKEND=${BACKEND:-both}
export LAYERS=${LAYERS:-0}
export ROUTING=${ROUTING:-balanced}
export WEIGHT_SLOTS=${WEIGHT_SLOTS:-2}
export WARMUPS=${WARMUPS:-30}
export ITERATIONS=${ITERATIONS:-30}

read -r -a models <<< "${MODELS:-qwen3_235b_fp4 minimax_m25_fp4 minimax_m3_fp4 deepseek_v4_flash_fp4 deepseek_v4_pro_fp4}"
read -r -a batches <<< "${TOKENS_PER_RANK_GRID:-8 16 32 48 64 96 128 192}"
read -r -a nextn_grid <<< "${MTP_NEXTN_GRID:-0 1 2 3}"

for model in "${models[@]}"; do
  for tokens in "${batches[@]}"; do
    for nextn in "${nextn_grid[@]}"; do
      MODEL=${model} TOKENS_PER_RANK=${tokens} MTP_NEXTN=${nextn} "${runner}"
    done
  done
done
