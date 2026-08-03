#!/usr/bin/env bash
set -euo pipefail

workspace=${WORKSPACE:-/workspace}
fastafd_root=${FASTAFD_ROOT:-${workspace}/FastAFD-megamoe-multimodel}
results_dir=${RESULTS_DIR:-${workspace}/results/megamoe-multimodel}
model=${MODEL:?set MODEL to a key from megamoe_model_profiles.py}
ag_size=${AG_SIZE:-4}
eg_size=${EG_SIZE:-4}
sequences_per_ag_rank=${SEQUENCES_PER_AG_RANK:-96}
mtp_nextn=${MTP_NEXTN:-0}
microbatches=${MICROBATCHES:-2}
layers=${LAYERS:-0}
routing=${ROUTING:-balanced}
expected_tokens_per_lane=${EXPECTED_TOKENS_PER_LANE:-}
prefetch_mib=${PREFETCH_MIB:-0}
warmups=${WARMUPS:-5}
iterations=${ITERATIONS:-30}

mkdir -p "${workspace}/runtime_home" "${workspace}/cache/deepgemm-multimodel" "${results_dir}"
export HOME="${workspace}/runtime_home"
export XDG_CACHE_HOME="${workspace}/cache"
export MINISGL_DEEPGEMM_BUILD_DIR="${workspace}/cache/deepgemm-multimodel"
export PYTHONPATH="${workspace}/python_deps:${fastafd_root}/python"
export MEASUREMENT_CONTAINER_IMAGE=${MEASUREMENT_CONTAINER_IMAGE:-unknown}

suffix="${model}_${ag_size}a${eg_size}f_s${sequences_per_ag_rank}_n${mtp_nextn}_mb${microbatches}_${routing}"
args=(
  --model "${model}"
  --output "${results_dir}/${suffix}.json"
  --ag-size "${ag_size}"
  --eg-size "${eg_size}"
  --sequences-per-ag-rank "${sequences_per_ag_rank}"
  --mtp-nextn "${mtp_nextn}"
  --microbatches "${microbatches}"
  --layers "${layers}"
  --routing "${routing}"
  --prefetch-mib "${prefetch_mib}"
  --warmups "${warmups}"
  --iterations "${iterations}"
)
if [[ -n "${expected_tokens_per_lane}" ]]; then
  args+=(--expected-tokens-per-lane "${expected_tokens_per_lane}")
fi

torchrun --standalone --nproc-per-node="$((ag_size + eg_size))" \
  "${fastafd_root}/scripts/experiments/afd/benchmark_megamoe_m2n_models.py" \
  "${args[@]}"
