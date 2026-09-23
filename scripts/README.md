# FastAFD Scripts

This directory is organized by workflow.

## Public Entry Points

Quickstart:

```bash
./scripts/quickstart/qwen3_30b_a3b_sample.sh
```

Correctness alignment:

```bash
./scripts/validate/qwen3_30b_a3b_alignment.sh
./scripts/validate/qwen3_30b_a3b_fastafd_alignment.sh
```

Large-scale AFD preset:

```bash
./scripts/experiments/afd/qwen3_30b/run_afd_qwen3_30b_a3b_fp8_3node_mb2_nsys_alignment.sh
```

Published Qwen3-235B and MiniMax M2.5 result presets live under
`scripts/experiments/afd/qwen3_235b/` and `scripts/experiments/afd/minimax_m25/`;
they dispatch through `scripts/run_afd_qwen3_30b_a3b_fp8_3node_mb2_nsys_alignment.sh`.

## Decode-step metrics

`scripts/serve/fastafd_server.sh` can write versioned decode-step measurements:

```bash
./scripts/serve/fastafd_server.sh \
  --model Qwen/Qwen3-235B-A22B-FP8 \
  --afd-metrics-output reports/afd_steps.jsonl \
  --afd-metrics-manifest reports/afd_manifest.json
```

The manifest uses `fastafd.afd-run-manifest.v1`:

```json
{
  "schema": "fastafd.afd-run-manifest.v1",
  "source_revision": "0123456789abcdef0123456789abcdef01234567",
  "model_id": "Qwen/Qwen3-235B-A22B-FP8",
  "model_revision": "89abcdef0123456789abcdef0123456789abcdef",
  "system": "b200_sxm",
  "hardware": {
    "gpu_models": {"0": "NVIDIA B200"},
    "gpu_to_hca": {"0": ["mlx5_0"]},
    "backend_hcas": {"0": ["mlx5_0"]},
    "gpu_clocks_mhz": {"0": 1830},
    "nic_counter_deltas": {"mlx5_0": {"rx_bytes": 0, "tx_bytes": 0}},
    "per_rank_bandwidth_ceiling_gbps": {"0": 400}
  }
}
```

Hardware map keys are FastAFD AG/EG union-world ranks and must cover every
worker rank. Ranks without a selected HCA use an empty list and a NIC bandwidth
ceiling of zero.

The output uses `fastafd.afd-step-metrics.v1`. Each decode record measures the
coordinator interval from the first worker command until all AG and EG replies
for that step arrive. It includes worker queueing and GPU completion under the
configured overlap policy; it is not an isolated MoE-kernel duration. Output
creation fails if the destination already exists.

## CUPTI step trace

`--ray-nsys` enables Nsight Systems CUDA activity tracing through CUPTI. FastAFD
adds step IDs to AG and EG NVTX ranges only while this option is enabled. A
bounded capture can use `MINISGL_RAY_NSYS_START_STEP` and
`MINISGL_RAY_NSYS_STOP_STEP`; keep the default graph-level CUDA trace.

Export each worker's `.nsys-rep` with `nsys export --type sqlite`, then run:

```bash
python -m minisgl.afd_cupti \
  --manifest manifest.json \
  --attn-workers 1 \
  --rank-report 0=ag.sqlite \
  --rank-report 1=eg.sqlite \
  --rank-nsys 0=ag.nsys-rep \
  --rank-nsys 1=eg.nsys-rep \
  --output afd_cupti_steps.jsonl
```

This example uses one AG and one EG worker. Supply every worker report for the
actual topology; `--attn-workers` is attention DP multiplied by attention TP.
Nsight's worker filename rank is the union-world rank plus one. The exporter
requires matching captured decode steps on every rank and records the SHA-256
of the manifest, raw Nsight reports, and SQLite inputs. `gpu_span_ms` is the GPU
activity span for one rank and step. Cross-node critical paths and isolated MoE
stages require separate analysis. CUPTI collection adds profiling overhead;
measure normal serving performance in a separate run without `--ray-nsys`.

## Layout

- `quickstart/`: mini-sgl sampling workflow.
- `serve/`: mini-sgl, FastAFD, and vLLM server launchers.
- `validate/`: mini-sgl/vLLM and FastAFD/vLLM alignment pipelines.
- `experiments/afd/qwen3_30b/`: the FastAFD launcher the presets dispatch to.
- `experiments/afd/qwen3_235b/`: Qwen3-235B published-result presets.
- `experiments/afd/minimax_m25/`: MiniMax M2.5 published-result presets.
- `experiments/vllm/`: the sharded vLLM alignment scorer.
- `data_gen/`: prompt generation helper.
- `lib/`: shared shell helpers.
