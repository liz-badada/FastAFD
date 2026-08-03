#!/usr/bin/env bash
#SBATCH --account=beta-users
#SBATCH --qos=interactive-isolated-short
#SBATCH --partition=b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=768G
#SBATCH --time=01:30:00
#SBATCH --job-name=megamoe-model
#SBATCH --exclude=umbriel-b200-042

set -euo pipefail

host_root=${HOST_ROOT:-/home/scratch.jinyanc_wwfo/fastafd_measure_20260721}
image=${CONTAINER_IMAGE:-/home/scratch.jinyanc_wwfo/enroot_images/sglang_deepseek_v4_blackwell.sqsh}
run_script=${RUN_SCRIPT:-run_megamoe_m2n_model_benchmark.sh}
source_root=${SOURCE_ROOT:-${host_root}/FastAFD-megamoe-multimodel}
case "${run_script}" in
  run_megamoe_m2n_model_benchmark.sh|run_megamoe_m2n_model_smoke_suite.sh|run_megamoe_colocated_model_benchmark.sh|run_megamoe_colocated_model_smoke_suite.sh) ;;
  *)
    echo "Unsupported RUN_SCRIPT: ${run_script}" >&2
    exit 2
    ;;
esac
source_commit=$(git -C "${source_root}" rev-parse HEAD)
source_branch=$(git -C "${source_root}" branch --show-current)
source_diff_sha256=$(git -C "${source_root}" diff --binary | sha256sum | cut -d' ' -f1)
if [[ -n $(git -C "${source_root}" status --porcelain) ]]; then
  source_dirty=true
else
  source_dirty=false
fi
source_tree_sha256=$(
  cd "${source_root}"
  git ls-files --cached --others --exclude-standard -z |
    sort -z |
    while IFS= read -r -d '' path; do
      printf '%s\0' "${path}"
      if [[ -L "${path}" ]]; then
        readlink "${path}" | sha256sum
      else
        sha256sum <"${path}"
      fi
    done |
    sha256sum |
    cut -d' ' -f1
)
mounts="${host_root}:/workspace"
if [[ -d /home/scratch.jinyanc_wwfo/cargo ]]; then
  mounts+=",/home/scratch.jinyanc_wwfo/cargo:/home/scratch.jinyanc_wwfo/cargo:ro"
fi
if [[ -f /home/jinyanc/.local/bin/env ]]; then
  mounts+=",/home/jinyanc/.local/bin/env:/home/jinyanc/.local/bin/env:ro"
fi

srun \
  --ntasks=1 \
  --gpus=8 \
  --container-image="${image}" \
  --no-container-entrypoint \
  --container-mounts="${mounts}" \
  --container-workdir=/workspace/FastAFD-megamoe-multimodel \
  --export="ALL,MEASUREMENT_CONTAINER_IMAGE=${image},MEASUREMENT_SOURCE_COMMIT=${source_commit},MEASUREMENT_SOURCE_BRANCH=${source_branch},MEASUREMENT_SOURCE_DIRTY=${source_dirty},MEASUREMENT_SOURCE_DIFF_SHA256=${source_diff_sha256},MEASUREMENT_SOURCE_TREE_SHA256=${source_tree_sha256}" \
  /bin/bash "/workspace/FastAFD-megamoe-multimodel/scripts/experiments/afd/${run_script}"
