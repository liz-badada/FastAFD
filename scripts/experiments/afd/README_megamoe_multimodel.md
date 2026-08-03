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
```

The model registry provides split-stage contracts for:

- `qwen3_235b_{fp8,fp4}`
- `minimax_m25_{fp8,fp4}`
- `minimax_m3_{fp8,fp4}`
- `deepseek_v4_flash_{fp8,fp4}`
- `deepseek_v4_pro_{fp8,fp4}`

The split AFD runner supports both listed storage widths. The matched colocated
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
   allocating time to a full sweep.

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
   AFD_SPLIT_GRID="4:4" \
   SEQUENCES_PER_AG_RANK_GRID="8 16 32 48 64 96 128 192" \
   MICROBATCH_GRID="1 2 4" MTP_NEXTN_GRID="0 1 2 3" \
   sbatch scripts/experiments/afd/submit_megamoe_m2n_b200.sbatch
   ```

   Set `BACKEND_GRID="deepep both"` on the colocated suite to run the
   reference-only and matched paths in the same allocation when auditing
   backend-state or node-to-node stability.

3. Convert raw JSON files into the review table and exact-only AIC profile.

   ```bash
   python scripts/experiments/afd/summarize_megamoe_model_results.py \
     /path/to/colocated-results /path/to/split-results \
     --csv /path/to/megamoe_latency_reference.csv \
     --markdown /path/to/megamoe_latency_reference.md \
     --profile /path/to/afd_moe_stage_profile.json
   ```

   | Output | Use |
   | --- | --- |
   | `megamoe_latency_reference.md` | Compact human-readable latency and qualification table. |
   | `megamoe_latency_reference.csv` | Machine-readable view of the same exact points. |
   | `afd_moe_stage_profile.json` | Validated exact-match timing input for AIC. |

   When a point was rerun, pass only the selected trial. The summarizer rejects
   duplicate eligible exact keys and reports both source paths instead of
   emitting an AIC profile whose result depends on directory order.

4. Consume the profile with the matching AIC branch. `--require-measured-moe`
   prevents a generic MoE estimate from being mixed into either comparison
   arm.

   ```bash
   git clone --branch pr1323-afd-moe-eval \
     git@github.com:liz-badada/aiconfigurator.git
   cd aiconfigurator
   uv sync --extra dev
   git lfs pull
   uv run python tools/afd_multimodel_mtp_experiment.py \
     --output /path/to/measured_sweep.json \
     --models qwen3_235b minimax_m25 minimax_m3 deepseek_v4_flash \
     --workloads 8k 16k --total-gpus 16 24 36 48 72 \
     --profile-scope primary \
     --afd-moe-profile /path/to/afd_moe_stage_profile.json \
     --require-measured-moe
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
     --models qwen3_235b minimax_m25 minimax_m3 deepseek_v4_flash \
     --workloads 8k 16k --total-gpus 16 24 36 48 72
   ```

   This default run measures a finite request wave. For a steady-state
   accounting check on one point, add `--output-tokens 128 --waves 64` and
   narrow the model, workload, and GPU lists to one value. MTP makes the final
   wave stochastic, so finite-wave output throughput can be below the AIC
   saturated reference even when per-request TPOT agrees.

The raw JSON is the source of truth. The CSV and Markdown files are compact
views, and `afd_moe_stage_profile.json` contains only validated exact points.
Do not hand-enter one constant `afd_moe_time_ms` for an entire sweep.

## Measurement paths

Use the split path for an AFD F pool:

```bash
MODEL=qwen3_235b_fp8 \
AG_SIZE=4 EG_SIZE=4 \
SEQUENCES_PER_AG_RANK=96 \
MTP_NEXTN=0 MICROBATCHES=2 \
LAYERS=94 ROUTING=balanced \
WARMUPS=30 ITERATIONS=30 \
RESULTS_DIR=/workspace/results/megamoe-split \
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

Set `MEASUREMENT_SYSTEM` to the AIC system key represented by the allocation.
The submit script defaults to an 8-GPU B200 allocation and labels it
`b200_sxm`. For a single-node GB200 NVL72 split measurement, override the
Slurm resources, set the exact A:F topology, and label the result `gb200`:

```bash
SOURCE_ROOT=/path/to/FastAFD \
RUN_SCRIPT=run_megamoe_m2n_model_benchmark.sh \
MEASUREMENT_SYSTEM=gb200 TOTAL_GPUS=72 \
MODEL=qwen3_235b_fp4 AG_SIZE=56 EG_SIZE=16 \
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
2. MegaMoE and the official DeepEP reference have relative L2 output error at
   most `1e-3` on every rank under the common numerical-check contract. That
   check uses the same block-128 FP8 input and pre-L2 top-k weighting on both
   paths; timed samples retain each backend's production quantization path.
3. MegaMoE has stage CUDA latency CV at most 3% and finite output on every
   rank. The reference CV is still reported, but it does not gate the measured
   MegaMoE latency profile.
4. The conservative same-sample speedup bound is greater than one: minimum
   observed DeepEP latency divided by maximum observed MegaMoE latency. This
   proves the gain without relying on a noisy reference mean or median.

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
