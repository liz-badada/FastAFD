#!/usr/bin/env bash
set -euo pipefail

workspace=${WORKSPACE:-/workspace}
fastafd_root=${FASTAFD_ROOT:-${workspace}/FastAFD-megamoe-multimodel}
results_dir=${RESULTS_DIR:-${workspace}/results/megamoe-colocated-multimodel}
model=${MODEL:?set MODEL to an FP4 key from megamoe_model_profiles.py}
ep_size=${EP_SIZE:-8}
tokens_per_rank=${TOKENS_PER_RANK:-64}
mtp_nextn=${MTP_NEXTN:-0}
layers=${LAYERS:-0}
routing=${ROUTING:-balanced}
warmups=${WARMUPS:-5}
iterations=${ITERATIONS:-30}

mkdir -p "${workspace}/runtime_home" "${workspace}/cache/deepgemm-multimodel" "${results_dir}"
export HOME="${workspace}/runtime_home"
export XDG_CACHE_HOME="${workspace}/cache"
export MINISGL_DEEPGEMM_BUILD_DIR="${workspace}/cache/deepgemm-multimodel"
export PYTHONPATH="${workspace}/python_deps:${fastafd_root}/python"
export MEASUREMENT_CONTAINER_IMAGE=${MEASUREMENT_CONTAINER_IMAGE:-unknown}

suffix="${model}_${ep_size}ep_t${tokens_per_rank}_n${mtp_nextn}_${routing}"
torchrun --standalone --nproc-per-node="${ep_size}" \
  "${fastafd_root}/scripts/experiments/afd/benchmark_megamoe_colocated_models.py" \
  --model "${model}" \
  --output "${results_dir}/${suffix}.json" \
  --ep-size "${ep_size}" \
  --tokens-per-rank "${tokens_per_rank}" \
  --mtp-nextn "${mtp_nextn}" \
  --layers "${layers}" \
  --routing "${routing}" \
  --warmups "${warmups}" \
  --iterations "${iterations}"
