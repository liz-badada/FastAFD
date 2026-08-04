#!/usr/bin/env bash
set -euo pipefail

runner=$(dirname "${BASH_SOURCE[0]}")/run_megamoe_colocated_model_benchmark.sh
: "${CASE_MATRIX:?Set CASE_MATRIX to a whitespace-separated case matrix}"

while read -r model tokens nextn backend remainder; do
  if [[ -z ${model:-} || ${model:0:1} == "#" ]]; then
    continue
  fi
  if [[ -n ${remainder:-} ]]; then
    echo "Unexpected extra fields in CASE_MATRIX row: ${model} ${tokens} ${nextn} ${backend} ${remainder}" >&2
    exit 2
  fi
  if [[ ! ${tokens:-} =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid tokens-per-rank in CASE_MATRIX: ${tokens:-<missing>}" >&2
    exit 2
  fi
  MODEL=${model} \
  TOKENS_PER_RANK=${tokens} \
  MTP_NEXTN=${nextn} \
  BACKEND=${backend} \
  "${runner}"
done < "${CASE_MATRIX}"
