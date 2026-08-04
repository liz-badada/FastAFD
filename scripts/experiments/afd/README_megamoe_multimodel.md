# Multi-model MegaMoE measurement

## Companion branches

| Repository | Branch | Purpose |
| --- | --- | --- |
| `liz-badada/FastAFD` | `megamoe-multimodel-b200` | Measure matched MegaMoE and DeepEP+DeepGEMM stages and export exact profiles. |
| `liz-badada/aiconfigurator` | `pr1323-afd-moe-eval` | Consume exact measured profiles in matched AGG/AFD sweeps. |
| `liz-badada/dynamo` | `afd-moe-timing` | Replay selected AIC service points with Dynamo Mocker. |

Use the reproducible measurement branch:

```bash
git clone --branch megamoe-multimodel-b200 \
  git@github.com:liz-badada/FastAFD.git
cd FastAFD
git status --short
git rev-parse HEAD
export CONTAINER_IMAGE=/path/to/sglang_blackwell.sqsh
export HOST_ROOT=$(dirname "$(realpath .)")
python3 -m pip install --target "${HOST_ROOT}/python_deps" --upgrade --no-deps \
  'msgpack==1.2.1' 'nvidia-nccl-cu12==2.30.4'
```

The runners prepend `/workspace/python_deps` to `PYTHONPATH`; the submit script
maps `HOST_ROOT` to `/workspace`. This explicit dependency directory is needed
when the selected SGLang container does not already provide `msgpack`. The
split DeepEP path additionally requires the NCCL 2.30.4 Device API headers and
library; its runner selects this exact library through `LD_LIBRARY_PATH` and
`LD_PRELOAD`. The preload keeps DeepEP and ProcessGroupNCCL on one process-wide
Device API version when the base PyTorch image pins an older `libnccl.so.2`.
It fails closed when the required headers are absent instead of compiling
against an older host-only NCCL package. Match the NCCL wheel's CUDA major to
the container; the command above targets the documented CUDA 12.9 container.
Set `PYTHON_DEPS_ROOT` only when using a different mounted dependency directory.
Result metadata records PyTorch's NCCL build version separately from the
DeepEP NCCL version and library path returned by the running process.
The single-node launcher defaults `EP_DISABLE_GIN=1`, using DeepEP's official
non-GIN NVLink path; set it to `0` only on a GIN-capable deployment.

The model registry provides split-stage contracts for:

- `qwen3_235b_{fp8,fp4}`
- `minimax_m25_{fp8,fp4}`
- `minimax_m3_{fp8,fp4}`
- `deepseek_v4_flash_{fp8,fp4}`
- `deepseek_v4_pro_{fp8,fp4}`

The split AFD runner supports MegaMoE with FP8 or FP4 weights and
DeepEP+DeepGEMM with FP4 weights. The matched colocated
MegaMoE-versus-DeepEP+DeepGEMM runner uses E2M1 weights with UE8M0 block-32
scales and E4M3 activations with UE8M0 scales. Its exact AIC precision key is
`w4a8_mxfp4_mxfp8`; it is not an NVFP4 measurement.

Being listed means that the benchmark has an explicit model-shape contract. It
does not mean that every model is already qualified for profile export; each
measured point must pass the gates below.

The exact hidden size, intermediate size, routed expert count, top-k, shared
expert count, routing scale, and activation contract are defined in
`python/minisgl/moe/megamoe_model_profiles.py`.

Each profile records `model_path` for the architecture/checkpoint family and
`simulation_model_path` for the exact AIC model identity. The benchmark uses
deterministic synthetic weights with the declared shape; it does not load
checkpoint tensors. `moe_quant_contract` is therefore a separate exact profile
key, and the exporter never infers native checkpoint precision from a model
name.

## Reproduction order

1. Run the one-layer, low-batch smoke suite. This checks compilation, all-rank
   finite outputs, quantization, and the colocated numerical reference before
   allocating time to a full sweep. The matched numerical gate requires every
   output to be finite and the two BF16 backend outputs to differ by no more
   than one representable BF16 value (one ULP) elementwise. Relative L2 is
   retained in the JSON as a diagnostic and is not used to hide BF16 rounding.

   ```bash
   SOURCE_ROOT=/path/to/FastAFD \
   RUN_SCRIPT=run_megamoe_colocated_model_smoke_suite.sh \
   BACKEND=both TOKENS_PER_RANK=8 LAYERS=1 \
   sbatch scripts/experiments/afd/submit_megamoe_m2n_b200.sbatch
   ```

2. Run the exact workload grid needed by the AIC sweep. All grid variables are
   whitespace-separated lists and may be narrowed without editing Python.

   ```bash
   SOURCE_ROOT=/path/to/FastAFD \
   RUN_SCRIPT=run_megamoe_colocated_model_suite.sh \
   MODELS="qwen3_235b_fp4 minimax_m25_fp4 minimax_m3_fp4 deepseek_v4_flash_fp4 deepseek_v4_pro_fp4" \
   TOKENS_PER_RANK_GRID="8 16 32 48 64 96 128 192" \
   MTP_NEXTN_GRID="0 1 2 3" \
   sbatch scripts/experiments/afd/submit_megamoe_m2n_b200.sbatch

   SOURCE_ROOT=/path/to/FastAFD \
   RUN_SCRIPT=run_megamoe_m2n_model_suite.sh \
   BACKEND_GRID="mega deepep" \
   SEQUENCES_PER_AG_RANK_GRID="8 16 32 48 64 96 128 192" \
   MICROBATCH_GRID="1 2 4" MTP_NEXTN_GRID="0 1 2 3" \
   sbatch scripts/experiments/afd/submit_megamoe_m2n_b200.sbatch
   ```

   Unless `AFD_SPLIT_GRID` is set explicitly, the split suite selects `4A4F`
   for Qwen3-235B, MiniMax-M2.5, MiniMax-M3, and DeepSeek-V4-Flash, and
   `2A6F` for DeepSeek-V4-Pro. V4-Pro has 384 routed experts and 61 MoE
   layers; keeping all benchmark weight slots resident does not fit the
   validated `4A4F` B200 memory envelope. `2A6F` preserves an integral
   64-expert F-rank shard without streaming or reusing layer weights.

   To reproduce only the compact formal grid used by the reference data, run:

   ```bash
   SOURCE_ROOT=/path/to/FastAFD \
   RUN_SCRIPT=run_megamoe_colocated_model_suite.sh \
   MODELS="qwen3_235b_fp4 minimax_m25_fp4 minimax_m3_fp4 deepseek_v4_flash_fp4 deepseek_v4_pro_fp4" \
   TOKENS_PER_RANK_GRID="48 96" MTP_NEXTN_GRID="0 1 2 3" \
   WARMUPS=30 ITERATIONS=40 \
   sbatch scripts/experiments/afd/submit_megamoe_m2n_b200.sbatch

   SOURCE_ROOT=/path/to/FastAFD \
   RUN_SCRIPT=run_megamoe_m2n_model_suite.sh \
   MODELS="qwen3_235b_fp4 minimax_m25_fp4 minimax_m3_fp4 deepseek_v4_flash_fp4" \
   BACKEND_GRID="mega deepep" \
   AFD_SPLIT_GRID="4:4" SEQUENCES_PER_AG_RANK_GRID="48 96" \
   MICROBATCH_GRID="1 2 4" MTP_NEXTN_GRID="0 1 2 3" \
   WARMUPS=30 ITERATIONS=40 \
   sbatch scripts/experiments/afd/submit_megamoe_m2n_b200.sbatch

   SOURCE_ROOT=/path/to/FastAFD \
   RUN_SCRIPT=run_megamoe_m2n_model_suite.sh \
   MODELS="deepseek_v4_pro_fp4" AFD_SPLIT_GRID="2:6" \
   BACKEND_GRID="mega deepep" \
   SEQUENCES_PER_AG_RANK_GRID="48 96" \
   MICROBATCH_GRID="1 2 4" MTP_NEXTN_GRID="0 1 2 3" \
   WARMUPS=30 ITERATIONS=40 \
   sbatch scripts/experiments/afd/submit_megamoe_m2n_b200.sbatch
   ```

   Set `BACKEND_GRID="deepep both"` on the colocated suite to run the
   reference-only and matched paths in the same allocation when auditing
   backend-state or node-to-node stability.

3. Convert raw JSON files into the review table and exact-only AIC profile.

   ```bash
   python scripts/experiments/afd/summarize_megamoe_model_results.py \
     /path/to/colocated-results /path/to/split-results \
     --markdown /path/to/megamoe_latency_reference.md \
     --profile /path/to/afd_moe_stage_profile.json
   ```

   To append newly measured backend points without rerunning the retained
   exact measurements, use the committed profile as the validation base. A
   new split row is accepted only when the base contains a stable,
   correctness-passing colocated row for the same model, precision, system,
   backend, source-rank batch, MTP width, and layer count. Duplicate exact keys
   fail instead of silently replacing data.

   ```bash
   python scripts/experiments/afd/summarize_megamoe_model_results.py \
     /path/to/new-split-results \
     --base-profile scripts/experiments/afd/reference/b200_sxm/afd_moe_stage_profile.json \
     --markdown /path/to/new_split_latency_reference.md \
     --profile /path/to/afd_moe_stage_profile_extended.json
   ```

   | Output | Use |
   | --- | --- |
   | `megamoe_latency_reference.md` | Compact human-readable latency and qualification table. |
   | `afd_moe_stage_profile.json` | Validated exact-match timing input for AIC. |

   The selected B200 reference set is committed at
   `scripts/experiments/afd/reference/b200_sxm/` as two compact MoE artifacts:
   the generated Markdown latency table and the exact-only AIC profile. The
   profile contains 40 colocated measurements for each of MegaMoE and
   DeepEP+DeepGEMM, plus 120 split-stage MegaMoE measurements. All 200 backend-
   qualified entries pass their applicable gates. DeepEP split-stage entries
   are intentionally absent until that exact A/F path is measured. Raw per-run JSON
   is intentionally not committed and is regenerated by the commands above at
   each entry's `source.result` path. This profile is labeled `b200_sxm` and is
   intentionally rejected by a GB200 sweep.

   Add `--csv /path/to/megamoe_latency_reference.csv` only when a separate
   machine-readable table is needed.

   When a point was rerun, pass only the selected trial. The summarizer rejects
   duplicate eligible exact keys and reports both source paths instead of
   emitting an AIC profile whose result depends on directory order.

4. Consume the profile with the matching AIC branch only when the profile
   system matches the target sweep. `--require-measured-moe` prevents a
   generic MoE estimate from being mixed into either comparison arm. The
   committed reference is `b200_sxm`; AIC's current fixed-pool tool targets
   `gb200`, so it rejects that reference by design. Re-run the same measurement
   matrix on GB200 and export a profile whose entries use `system=gb200` before
   running this exact-measured command.

   ```bash
   git clone --branch pr1323-afd-moe-eval \
     git@github.com:liz-badada/aiconfigurator.git
   cd aiconfigurator
   uv sync --extra dev
   git lfs pull
   uv run python tools/afd_multimodel_mtp_experiment.py \
     --output /path/to/measured_sweep.json \
     --models qwen3_235b minimax_m25 minimax_m3 deepseek_v4_flash deepseek_v4_pro \
     --workloads 8k 16k --total-gpus 16 24 36 48 72 \
     --profile-scope primary \
     --afd-moe-profile /path/to/afd_moe_stage_profile.json \
     --require-measured-moe
   ```

   To run the current generic GB200 sweep while showing the qualified B200
   measurements only as a separate evidence table:

   ```bash
   uv run python tools/afd_multimodel_mtp_experiment.py \
     --output /path/to/generic_gb200_sweep.json \
     --models qwen3_235b minimax_m25 minimax_m3 deepseek_v4_flash deepseek_v4_pro \
     --workloads 8k 16k --total-gpus 16 24 36 48 72 \
     --profile-scope primary
   uv run python tools/render_afd_multimodel_mtp_report.py \
     --sweep /path/to/generic_gb200_sweep.json \
     --moe-reference-profile /path/to/FastAFD/scripts/experiments/afd/reference/b200_sxm/afd_moe_stage_profile.json \
     --moe-reference-url https://github.com/liz-badada/FastAFD/tree/megamoe-multimodel-b200/scripts/experiments/afd/reference/b200_sxm \
     --output-dir /path/to/report --speed-floor 30
   ```

5. Optionally replay selected service points with the matching Dynamo branch.

   ```bash
   git clone --branch afd-moe-timing \
     git@github.com:liz-badada/dynamo.git
   cd dynamo
   uv sync --extra mocker
   source .venv/bin/activate
   uv pip install 'maturin[patchelf]'
   (cd lib/bindings/python && maturin develop --uv)
   cd /path/to/aiconfigurator
   uv run python tools/afd_multimodel_mtp_mocker_replay.py \
     --sweep /path/to/measured_sweep.json \
     --dynamo /path/to/dynamo \
     --output-dir /path/to/mocker_replay \
     --models qwen3_235b minimax_m25 minimax_m3 deepseek_v4_flash deepseek_v4_pro \
     --workloads 8k 16k --total-gpus 16 24 36 48 72
   ```

   This default run measures a finite request wave. For a steady-state
   accounting check on one point, add `--output-tokens 128 --waves 64` and
   narrow the model, workload, and GPU lists to one value. MTP makes the final
   wave stochastic, so finite-wave output throughput can be below the AIC
   saturated reference even when per-request TPOT agrees.

The generated raw JSON is the source of truth for a new measurement run. The
branch keeps only the compact Markdown view and
`afd_moe_stage_profile.json`, which contains validated exact MoE points. Do not
hand-enter one constant `afd_moe_time_ms` for an entire sweep.

The reported MegaMoE speedup is a ratio of the complete colocated MoE-stage
backend paths, not a GEMM-only ratio and not an end-to-end serving speedup. It
includes each backend's production activation quantization and the reference
path's 128-row per-expert alignment, DeepEP dispatch/combine, SGLang
scatter/gather, and two DeepGEMM calls. Low expert occupancy can therefore make
the stage ratio much larger than the final system gain. AIC consumes the
absolute latency for the selected backend; the reference ratio is validation
evidence.

## Measurement paths

Use the split path for an AFD F pool:

```bash
MODEL=qwen3_235b_fp8 BACKEND=mega \
AG_SIZE=4 EG_SIZE=4 \
SEQUENCES_PER_AG_RANK=96 \
MTP_NEXTN=0 MICROBATCHES=2 \
LAYERS=94 ROUTING=balanced \
WARMUPS=30 ITERATIONS=30 \
RESULTS_DIR=/workspace/results/megamoe-split \
bash scripts/experiments/afd/run_megamoe_m2n_model_benchmark.sh
```

Use the same split boundary with DeepEP union dispatch/combine and psum-layout
FP8xFP4 DeepGEMM by selecting an FP4 profile. The default `WEIGHT_SLOTS=0`
keeps one deterministic weight set per measured layer; reduced weight-slot runs
are diagnostic and are not exported to the AIC profile.

```bash
MODEL=qwen3_235b_fp4 BACKEND=deepep \
AG_SIZE=4 EG_SIZE=4 \
SEQUENCES_PER_AG_RANK=96 \
MTP_NEXTN=0 MICROBATCHES=2 \
LAYERS=94 ROUTING=balanced \
WARMUPS=30 ITERATIONS=30 \
RESULTS_DIR=/workspace/results/deepep-split \
bash scripts/experiments/afd/run_megamoe_m2n_model_benchmark.sh
```

Use the colocated path for an AGG worker. `BACKEND=both` measures MegaMoE and
the official DeepEP normal path with SGLang scatter/gather and two contiguous
DeepGEMM GEMMs. The paths share BF16 inputs, routes, the same synthetic
MXFP4-format weights, activation, and output scaling. Each retains its
production input-activation quantization granularity: block-32 for MegaMoE and
block-128 for the SGLang DeepEP path.

```bash
MODEL=qwen3_235b_fp4 \
EP_SIZE=8 TOKENS_PER_RANK=96 \
MTP_NEXTN=0 LAYERS=94 \
ROUTING=balanced BACKEND=both WEIGHT_SLOTS=2 \
WARMUPS=30 ITERATIONS=30 \
RESULTS_DIR=/workspace/results/megamoe-colocated \
bash scripts/experiments/afd/run_megamoe_colocated_model_benchmark.sh
```

`TOKENS_PER_RANK` and `SEQUENCES_PER_AG_RANK` are logical request counts.
The scripts execute `logical_count * (MTP_NEXTN + 1)` physical verification
tokens. MTP acceptance is intentionally not applied to kernel latency.

For an AIC AGG candidate, set `TOKENS_PER_RANK` to
`agg_local_batch / attention_tp`, not to the cluster concurrency or the
per-replica batch directly. The division must be integral for an exact measured
profile. For AFD, `SEQUENCES_PER_AG_RANK` maps directly to AIC's
`batch_per_a_gpu`.

On the configured ComputeLab B200 partition, submit either runner through:

```bash
SOURCE_ROOT=/path/to/FastAFD \
RUN_SCRIPT=run_megamoe_colocated_model_benchmark.sh \
MODEL=qwen3_235b_fp4 EP_SIZE=8 TOKENS_PER_RANK=96 \
BACKEND=both LAYERS=94 WARMUPS=30 ITERATIONS=30 \
RESULTS_DIR=/workspace/results/megamoe-colocated \
sbatch scripts/experiments/afd/submit_megamoe_m2n_b200.sbatch
```

`SOURCE_ROOT` defaults to the Slurm submission directory, and `HOST_ROOT`
defaults to its parent. `CONTAINER_IMAGE` is required and must name the
site-local SGLang Blackwell image. The parent mount makes
`/workspace/results/...` persistent beside the checkout. If the site requires
additional mounts, pass their comma-separated Enroot specifications through
`EXTRA_CONTAINER_MOUNTS`; the script has no user-specific mount dependency.

Set `MEASUREMENT_SYSTEM` to the AIC system key represented by the allocation.
The submit script defaults to an 8-GPU B200 allocation and labels it
`b200_sxm`. It requests the whole node exclusively so CPU or memory co-tenants
cannot add distributed synchronization tails. For a single-node GB200 NVL72
split measurement, override the Slurm resources, set the exact A:F topology,
and label the result `gb200`:

```bash
SOURCE_ROOT=/path/to/FastAFD \
RUN_SCRIPT=run_megamoe_m2n_model_benchmark.sh \
MEASUREMENT_SYSTEM=gb200 TOTAL_GPUS=72 \
MODEL=qwen3_235b_fp4 BACKEND=mega AG_SIZE=56 EG_SIZE=16 \
SEQUENCES_PER_AG_RANK=96 MTP_NEXTN=3 MICROBATCHES=2 \
sbatch --partition=GB200_NVL72_PARTITION --gpus=72 \
  scripts/experiments/afd/submit_megamoe_m2n_b200.sbatch
```

`TOTAL_GPUS` controls the nested `srun`; it must match the Slurm allocation and
`AG_SIZE + EG_SIZE`. Do not label an 8-GPU B200 result as `gb200`.

## Acceptance gates

A result is eligible for a calibrated profile only when all applicable gates
pass:

1. The CUDA input and activation quantizers match their Torch references with
   zero FP8-byte and packed-scale mismatches on every rank.
2. MegaMoE and the official DeepEP reference produce finite BF16 outputs that
   differ by at most one BF16 ULP elementwise on every rank under the common
   numerical-check contract. Relative L2 remains recorded as a diagnostic.
   The check uses the same block-128 FP8 input and pre-L2 top-k weighting on
   both paths; timed samples retain each backend's production quantization
   path.
3. MegaMoE and DeepEP+DeepGEMM both have stage CUDA latency CV at most 3% and
   finite output on every rank.
4. The conservative same-sample speedup bound is recorded as minimum observed
   DeepEP latency divided by maximum observed MegaMoE latency. It is not an
   admission filter: keeping valid measurements below one prevents a biased
   backend comparison. `megamoe_outperforms_reference` records whether this
   conservative bound is greater than one.
5. A split AFD point has a qualified colocated comparison at the same model,
   precision, system, logical source-rank batch, MTP `nextN`, and layer count.
   The exporter copies that comparison's correctness and conservative speedup
   bound into the split point's validation evidence; it does not accept a
   different workload merely because the model name matches.
6. A split DeepEP+DeepGEMM point keeps one resident weight set per measured
   layer. A reduced `WEIGHT_SLOTS` run is retained as a diagnostic JSON but is
   not eligible for profile export.

The JSON contains every latency sample, per-rank provenance, the source commit
and tree hash, model contract, physical token count, routing load, correctness,
stability, and the final `eligible_for_profile` decision.

## AIC timing boundary

These JSON stage measurements are not automatically interchangeable with the
AIC `afd_moe_time_ms` option. That option expects one full-resident-batch F-stage
wall time across all MoE layers and microbatches, including F-stage
communication and excluding router. A single-layer MegaMoE module latency must
first be aggregated with the exact layer, microbatch, EP, physical-token, and
communication contract. For a reusable AIC backend, publish a keyed latency
table rather than one constant.

Do not label B200 measurements as GB200 calibration. Run the same matrix on
GB200 when the target AIC system is `gb200`.
