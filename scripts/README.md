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
worker rank.

The output uses `fastafd.afd-step-metrics.v1`. Each decode record measures the
coordinator interval from the first worker command until all AG and EG replies
for that step arrive. It includes worker queueing and GPU completion under the
configured overlap policy; it is not an isolated MoE-kernel duration. Output
creation fails if the destination already exists.

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
