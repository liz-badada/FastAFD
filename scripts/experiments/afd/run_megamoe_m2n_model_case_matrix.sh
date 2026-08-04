#!/usr/bin/env bash
set -euo pipefail

runner=$(dirname "${BASH_SOURCE[0]}")/run_megamoe_m2n_model_benchmark.sh
: "${CASE_MATRIX:?Set CASE_MATRIX to a whitespace-separated case matrix}"

while read -r model split sequences nextn microbatches backend remainder; do
  if [[ -z ${model:-} || ${model:0:1} == "#" ]]; then
    continue
  fi
  if [[ -n ${remainder:-} ]]; then
    echo "Unexpected extra fields in CASE_MATRIX row: ${model} ${split} ${sequences} ${nextn} ${microbatches} ${backend} ${remainder}" >&2
    exit 2
  fi
  if [[ ! ${split:-} =~ ^[1-9][0-9]*:[1-9][0-9]*$ ]]; then
    echo "Invalid A:F split in CASE_MATRIX: ${split:-<missing>}" >&2
    exit 2
  fi
  IFS=: read -r ag_size eg_size <<< "${split}"
  MODEL=${model} \
  AG_SIZE=${ag_size} \
  EG_SIZE=${eg_size} \
  SEQUENCES_PER_AG_RANK=${sequences} \
  MTP_NEXTN=${nextn} \
  MICROBATCHES=${microbatches} \
  BACKEND=${backend} \
  "${runner}"
done < "${CASE_MATRIX}"
