#!/usr/bin/env bash
set -euo pipefail

runner=$(dirname "${BASH_SOURCE[0]}")/run_megamoe_m2n_model_benchmark.sh
export RESULTS_DIR=${RESULTS_DIR:-/workspace/results/megamoe-m2n-multimodel}
export LAYERS=${LAYERS:-0}
export ROUTING=${ROUTING:-balanced}
export WARMUPS=${WARMUPS:-5}
export ITERATIONS=${ITERATIONS:-30}

read -r -a models <<< "${MODELS:-qwen3_235b_fp4 minimax_m25_fp4 minimax_m3_fp4 deepseek_v4_flash_fp4 deepseek_v4_pro_fp4}"
read -r -a splits <<< "${AFD_SPLIT_GRID:-4:4}"
read -r -a batches <<< "${SEQUENCES_PER_AG_RANK_GRID:-8 16 32 48 64 96 128 192}"
read -r -a nextn_grid <<< "${MTP_NEXTN_GRID:-0 1 2 3}"
read -r -a microbatch_grid <<< "${MICROBATCH_GRID:-1 2 4}"

for model in "${models[@]}"; do
  for split in "${splits[@]}"; do
    IFS=: read -r ag_size eg_size <<< "${split}"
    for sequences in "${batches[@]}"; do
      for nextn in "${nextn_grid[@]}"; do
        for microbatches in "${microbatch_grid[@]}"; do
          if (( sequences % microbatches != 0 )); then
            continue
          fi
          MODEL=${model} \
          AG_SIZE=${ag_size} \
          EG_SIZE=${eg_size} \
          SEQUENCES_PER_AG_RANK=${sequences} \
          MTP_NEXTN=${nextn} \
          MICROBATCHES=${microbatches} \
          "${runner}"
        done
      done
    done
  done
done
