#!/usr/bin/env python3
"""Measure the FastAFD split MegaMoE MoE-stage boundary on one NVLink domain.

The command is intentionally workload-driven. ``sequences-per-ag-rank`` is the
decode concurrency owned by one attention rank. With MTP next-N, each sequence
contributes ``next_n + 1`` physical verify tokens. Those tokens are divided
evenly over the requested microbatch lanes before entering MegaMoE.

This is not a full-serving benchmark. It measures the fused A-side
quant/dispatch/wait/combine kernels together with the persistent F-side expert
kernel for all MoE layers. Router, attention, dense-only layers, residuals,
sampling, and coordinator overhead are outside the boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
from minisgl.moe.megamoe_model_profiles import (
    MEGAMOE_MODEL_PROFILES,
    MegaMoEModelProfile,
    get_megamoe_model_profile,
)

INPUT_SEED_BASE = 0xAFD80000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - position) + ordered[hi] * (position - lo)


def summarize(values: list[float]) -> dict[str, float | int]:
    mean = statistics.mean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "samples": len(values),
        "min_ms": min(values),
        "p50_ms": statistics.median(values),
        "p90_ms": percentile(values, 0.90),
        "p95_ms": percentile(values, 0.95),
        "max_ms": max(values),
        "mean_ms": mean,
        "stdev_ms": stdev,
        "cv_percent": 100.0 * stdev / mean if mean else 0.0,
    }


def command_output(command: list[str], *, cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(
            command,
            cwd=cwd,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception as exc:
        return f"ERROR {type(exc).__name__}: {exc}"


def environment_bool(name: str, fallback: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return fallback
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise RuntimeError(f"{name} must be a boolean, got {value!r}")


def source_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    diff = command_output(["git", "diff", "--binary"], cwd=root)
    local_dirty = bool(command_output(["git", "status", "--porcelain"], cwd=root))
    return {
        "root": str(root),
        "commit": os.environ.get("MEASUREMENT_SOURCE_COMMIT")
        or command_output(["git", "rev-parse", "HEAD"], cwd=root),
        "branch": os.environ.get("MEASUREMENT_SOURCE_BRANCH")
        or command_output(["git", "branch", "--show-current"], cwd=root),
        "dirty": environment_bool("MEASUREMENT_SOURCE_DIRTY", local_dirty),
        "tracked_diff_sha256": os.environ.get("MEASUREMENT_SOURCE_DIFF_SHA256")
        or hashlib.sha256(diff.encode()).hexdigest(),
        "source_tree_sha256": os.environ.get("MEASUREMENT_SOURCE_TREE_SHA256", ""),
    }


def gpu_provenance(local_rank: int) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(local_rank)
    return {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": props.name,
        "gpu_total_memory_bytes": props.total_memory,
        "compute_capability": f"{props.major}.{props.minor}",
        "multiprocessor_count": props.multi_processor_count,
        "driver_power_clocks_memory": command_output(
            [
                "nvidia-smi",
                f"--id={local_rank}",
                "--query-gpu=driver_version,power.limit,clocks.max.sm,clocks.max.memory,memory.total",
                "--format=csv,noheader,nounits",
            ]
        ),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION", ""),
        "slurm_node": os.environ.get("SLURMD_NODENAME", ""),
        "container_image": os.environ.get("MEASUREMENT_CONTAINER_IMAGE", ""),
    }


def _fp8_scales(experts: int, n: int, k: int) -> torch.Tensor:
    return torch.ones(
        (experts, math.ceil(n / 128), math.ceil(k / 128)),
        dtype=torch.float32,
        device="cuda",
    )


def make_fp8_experts(
    profile: MegaMoEModelProfile,
    local_experts: int,
    *,
    layer_id: int,
) -> Any:
    hidden = profile.hidden_size
    intermediate = profile.intermediate_size
    value = 0.0625 if layer_id % 2 == 0 else 0.125
    gate_up = torch.full(
        (local_experts, 2 * intermediate, hidden),
        value,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    down = torch.full(
        (local_experts, hidden, intermediate),
        value,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    return SimpleNamespace(
        gate_up_proj=gate_up,
        gate_up_proj_scale=_fp8_scales(local_experts, 2 * intermediate, hidden),
        down_proj=down,
        down_proj_scale=_fp8_scales(local_experts, hidden, intermediate),
        quant=SimpleNamespace(method="fp8", weight_block_size=(128, 128)),
        _fp8_scale_format="block",
    )


def _shared_fp8_linear(weight: torch.Tensor, scale: torch.Tensor) -> Any:
    return SimpleNamespace(
        weight=weight,
        weight_scale=scale,
        _fp8_scale_format="block",
        _fp8_block_size=(128, 128),
    )


def make_fp8_shared_expert(profile: MegaMoEModelProfile, *, layer_id: int) -> Any:
    hidden = profile.hidden_size
    intermediate = profile.intermediate_size
    value = 0.0625 if layer_id % 2 == 0 else 0.125
    gate_up = torch.full(
        (2 * intermediate, hidden),
        value,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    down = torch.full(
        (hidden, intermediate),
        value,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    return SimpleNamespace(
        gate_up_proj=_shared_fp8_linear(
            gate_up,
            _fp8_scales(1, 2 * intermediate, hidden)[0],
        ),
        down_proj=_shared_fp8_linear(
            down,
            _fp8_scales(1, hidden, intermediate)[0],
        ),
    )


def make_fp4_source_weights(
    profile: MegaMoEModelProfile,
    local_experts: int,
    *,
    layer_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    value = 0.0625 if layer_id % 2 == 0 else 0.125
    l1 = torch.full(
        (local_experts, 2 * profile.intermediate_size, profile.hidden_size),
        value,
        dtype=torch.bfloat16,
        device="cuda",
    )
    l2 = torch.full(
        (local_experts, profile.hidden_size, profile.intermediate_size),
        value,
        dtype=torch.bfloat16,
        device="cuda",
    )
    return l1, l2


def make_routes(
    profile: MegaMoEModelProfile,
    tokens: int,
    *,
    rank: int,
    lane: int,
    routing: str,
    hot_expert_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    top_k = profile.routed_top_k
    num_experts = profile.num_routed_experts
    active_experts = num_experts
    if routing == "hotset":
        active_experts = max(top_k, int(round(num_experts * hot_expert_fraction)))
    global_tokens = torch.arange(tokens, dtype=torch.int64, device="cuda")
    global_tokens += (rank * 4 + lane) * tokens
    offsets = torch.arange(top_k, dtype=torch.int64, device="cuda")
    ids = (global_tokens[:, None] * top_k + offsets[None, :]) % active_experts
    raw_weights = torch.arange(top_k, 0, -1, dtype=torch.float32, device="cuda")
    weights = raw_weights.expand(tokens, -1).clone()
    if profile.gate_renormalize:
        weights /= weights.sum(dim=1, keepdim=True)
    counts = torch.bincount(ids.flatten(), minlength=num_experts)
    return ids.contiguous(), weights.contiguous(), counts


def route_summary(counts: torch.Tensor) -> dict[str, float | int]:
    counts_f = counts.float()
    mean = float(counts_f.mean().item())
    std = float(counts_f.std(unbiased=False).item())
    sorted_counts = counts.sort().values
    p95_index = min(len(sorted_counts) - 1, math.ceil(0.95 * len(sorted_counts)) - 1)
    return {
        "assignments": int(counts.sum().item()),
        "active_experts": int((counts > 0).sum().item()),
        "min_assignments": int(counts.min().item()),
        "mean_assignments": mean,
        "p95_assignments": int(sorted_counts[p95_index].item()),
        "max_assignments": int(counts.max().item()),
        "load_cv_percent": 100.0 * std / mean if mean else 0.0,
    }


def register_layers(
    adapter: Any,
    profile: MegaMoEModelProfile,
    *,
    layers: int,
) -> None:
    local_experts = profile.num_routed_experts // adapter.eg_size
    for layer_id in range(layers):
        if profile.weight_precision == "fp8":
            experts = make_fp8_experts(profile, local_experts, layer_id=layer_id)
            shared = (
                make_fp8_shared_expert(profile, layer_id=layer_id)
                if profile.num_shared_experts
                else None
            )
            adapter.register_eg_layer_weights(layer_id, experts, shared_experts=shared)
            del experts, shared
        else:
            l1, l2 = make_fp4_source_weights(profile, local_experts, layer_id=layer_id)
            shared_l1 = shared_l2 = None
            if profile.num_shared_experts:
                shared_l1, shared_l2 = make_fp4_source_weights(profile, 1, layer_id=layer_id)
            adapter.register_eg_layer_fp4_weights(
                layer_id,
                l1,
                l2,
                shared_l1_weight=shared_l1,
                shared_l2_weight=shared_l2,
            )
            del l1, l2, shared_l1, shared_l2
        if layer_id % 4 == 3:
            torch.cuda.empty_cache()


def run_stage(
    *,
    adapter: Any,
    is_ag: bool,
    rank: int,
    layers: int,
    expected_tokens_per_lane: int,
    lane_streams: tuple[torch.cuda.Stream, ...],
    inputs: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
) -> tuple[float, float]:
    del rank
    current = torch.cuda.current_stream()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    lane_done = [torch.cuda.Event() for _ in lane_streams]
    start.record(current)
    wall_start = time.perf_counter()
    if is_ag:
        for stream in lane_streams:
            stream.wait_event(start)
        for _layer in range(layers):
            for lane, stream in enumerate(lane_streams):
                hidden, ids, weights = inputs[lane]
                with torch.cuda.stream(stream):
                    output = adapter.ag_moe(
                        hidden,
                        ids,
                        weights,
                        lane=lane,
                        external_quant=False,
                    )
                    if adapter.routed_scaling_factor != 1.0:
                        output.mul_(adapter.routed_scaling_factor)
        for done, stream in zip(lane_done, lane_streams, strict=True):
            done.record(stream)
            current.wait_event(done)
    else:
        adapter.eg_moe_persistent(
            layers,
            num_mb=len(lane_streams),
            expected_tokens_per_rank=expected_tokens_per_lane,
        )
    end.record(current)
    end.synchronize()
    return float(start.elapsed_time(end)), (time.perf_counter() - wall_start) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MEGAMOE_MODEL_PROFILES))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ag-size", type=int, required=True)
    parser.add_argument("--eg-size", type=int, required=True)
    parser.add_argument("--sequences-per-ag-rank", type=int, required=True)
    parser.add_argument("--mtp-nextn", type=int, default=0)
    parser.add_argument("--microbatches", type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument("--layers", type=int, default=0, help="0 means all model MoE layers")
    parser.add_argument("--routing", choices=("balanced", "hotset"), default="balanced")
    parser.add_argument("--hot-expert-fraction", type=float, default=0.25)
    parser.add_argument("--expected-tokens-per-lane", type=int)
    parser.add_argument("--prefetch-mib", type=int, default=0)
    parser.add_argument("--ag-sms", type=int, default=24)
    parser.add_argument("--warmups", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()

    profile = get_megamoe_model_profile(args.model)
    profile.validate(eg_size=args.eg_size)
    layers = profile.num_moe_layers if args.layers == 0 else args.layers
    if not 1 <= layers <= profile.num_moe_layers:
        raise SystemExit(f"layers must be in [1, {profile.num_moe_layers}]")
    if args.sequences_per_ag_rank <= 0:
        raise SystemExit("sequences-per-ag-rank must be positive")
    if args.sequences_per_ag_rank % args.microbatches:
        raise SystemExit("sequences-per-ag-rank must divide evenly across microbatches")
    if args.mtp_nextn < 0:
        raise SystemExit("mtp-nextn must be non-negative")
    if not 0 < args.hot_expert_fraction <= 1:
        raise SystemExit("hot-expert-fraction must be in (0, 1]")
    if args.warmups < 1 or args.iterations < 3:
        raise SystemExit("warmups must be >=1 and iterations must be >=3")

    verify_width = args.mtp_nextn + 1
    sequences_per_lane = args.sequences_per_ag_rank // args.microbatches
    physical_tokens_per_lane = sequences_per_lane * verify_width
    expected_tokens = args.expected_tokens_per_lane or physical_tokens_per_lane
    if expected_tokens <= 0:
        raise SystemExit("expected-tokens-per-lane must be positive")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != args.ag_size + args.eg_size:
        raise SystemExit(
            f"WORLD_SIZE={world_size}, expected ag_size+eg_size={args.ag_size + args.eg_size}"
        )
    os.environ["MINISGL_MEGAMOE_AG_SMS"] = str(args.ag_sms)
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))

    from minisgl.moe.megamoe_m2n_afd import MegaMoEM2NAfdAdapter

    adapter = MegaMoEM2NAfdAdapter(
        group=dist.group.WORLD,
        ag_size=args.ag_size,
        eg_size=args.eg_size,
        real_num_experts=profile.num_routed_experts,
        hidden_size=profile.hidden_size,
        intermediate_size=profile.intermediate_size,
        top_k=profile.routed_top_k,
        num_max_dispatch_tokens_per_rank=physical_tokens_per_lane,
        num_lanes=args.microbatches,
        gate_renormalize=profile.gate_renormalize,
        shared_experts_per_rank=profile.num_shared_experts,
        routed_scaling_factor=profile.routed_scaling_factor,
        weight_precision=profile.weight_precision,
        activation=profile.activation,
        activation_alpha=profile.activation_alpha,
        activation_clamp=profile.activation_clamp,
        activation_up_bias=profile.activation_up_bias,
        eg_expected_tokens_override=expected_tokens,
        eg_prefetch_bytes=args.prefetch_mib << 20,
    )
    is_ag = rank < args.ag_size

    torch.cuda.reset_peak_memory_stats()
    init_start = time.perf_counter()
    if not is_ag:
        register_layers(adapter, profile, layers=layers)
    dist.barrier()
    torch.cuda.synchronize()
    initialization_seconds = time.perf_counter() - init_start
    peak_memory_bytes = torch.cuda.max_memory_allocated()

    lane_streams = tuple(torch.cuda.Stream() for _ in range(args.microbatches))
    local_route_counts = torch.zeros(
        profile.num_routed_experts,
        dtype=torch.int64,
        device="cuda",
    )
    inputs: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...] = ()
    if is_ag:
        materialized = []
        for lane in range(args.microbatches):
            ids, weights, counts = make_routes(
                profile,
                physical_tokens_per_lane,
                rank=rank,
                lane=lane,
                routing=args.routing,
                hot_expert_fraction=args.hot_expert_fraction,
            )
            local_route_counts += counts
            input_seed = INPUT_SEED_BASE + rank * 4 + lane
            input_generator = torch.Generator(device="cuda")
            input_generator.manual_seed(input_seed)
            hidden = (
                torch.randn(
                    (physical_tokens_per_lane, profile.hidden_size),
                    dtype=torch.float32,
                    device="cuda",
                    generator=input_generator,
                )
                .mul_(0.375)
                .to(torch.bfloat16)
            )
            materialized.append((hidden, ids, weights))
        inputs = tuple(materialized)
    dist.all_reduce(local_route_counts, op=dist.ReduceOp.SUM)

    for _ in range(args.warmups):
        dist.barrier()
        run_stage(
            adapter=adapter,
            is_ag=is_ag,
            rank=rank,
            layers=layers,
            expected_tokens_per_lane=expected_tokens,
            lane_streams=lane_streams,
            inputs=inputs,
        )
    dist.barrier()

    local_cuda: list[float] = []
    local_wall: list[float] = []
    gathered_iterations: list[list[dict[str, float]]] = []
    for _ in range(args.iterations):
        dist.barrier()
        cuda_ms, wall_ms = run_stage(
            adapter=adapter,
            is_ag=is_ag,
            rank=rank,
            layers=layers,
            expected_tokens_per_lane=expected_tokens,
            lane_streams=lane_streams,
            inputs=inputs,
        )
        local_cuda.append(cuda_ms)
        local_wall.append(wall_ms)
        gathered = [None] * world_size if rank == 0 else None
        dist.gather_object(
            {"cuda_ms": cuda_ms, "wall_ms": wall_ms},
            object_gather_list=gathered,
            dst=0,
        )
        if rank == 0:
            assert gathered is not None
            gathered_iterations.append(gathered)

    finite = True
    output_abs_mean = None
    if is_ag:
        outputs = tuple(adapter._y_cache.values())
        finite = all(bool(torch.isfinite(output).all().item()) for output in outputs)
        output_abs_mean = statistics.mean(
            float(output.float().abs().mean().item()) for output in outputs
        )
    finite_all = [None] * world_size if rank == 0 else None
    output_means = [None] * world_size if rank == 0 else None
    dist.gather_object(finite, object_gather_list=finite_all, dst=0)
    dist.gather_object(output_abs_mean, object_gather_list=output_means, dst=0)

    init_times = [None] * world_size if rank == 0 else None
    peak_memories = [None] * world_size if rank == 0 else None
    provenance = [None] * world_size if rank == 0 else None
    dist.gather_object(initialization_seconds, object_gather_list=init_times, dst=0)
    dist.gather_object(peak_memory_bytes, object_gather_list=peak_memories, dst=0)
    dist.gather_object(gpu_provenance(local_rank), object_gather_list=provenance, dst=0)

    if rank == 0:
        stage_cuda = [max(sample["cuda_ms"] for sample in rows) for rows in gathered_iterations]
        stage_wall = [max(sample["wall_ms"] for sample in rows) for rows in gathered_iterations]
        stage_summary = summarize(stage_cuda)
        payload = {
            "schema": "fastafd.megamoe-m2n-model-benchmark.v1",
            "generated_at": utc_now(),
            "system_label": os.environ.get("MEASUREMENT_SYSTEM", "unspecified"),
            "measurement_boundary": (
                "split AFD MoE stage: A-side FP8 quant, route dispatch, wait, top-k combine, "
                "and routed-output scaling; "
                "F-side persistent expert L1, model-specific activation, requant, L2, and return; "
                "includes fused replicated shared expert when the profile declares one; excludes "
                "router, attention, dense-only layers, residual, sampling, and coordinator"
            ),
            "model_profile": asdict(profile),
            "topology": {
                "world_size": world_size,
                "ag_size": args.ag_size,
                "eg_size": args.eg_size,
                "local_routed_experts_per_f_rank": profile.num_routed_experts // args.eg_size,
                "protocol_experts_per_f_rank": profile.local_experts(args.eg_size),
            },
            "workload": {
                "sequences_per_ag_rank": args.sequences_per_ag_rank,
                "mtp_nextn": args.mtp_nextn,
                "verify_width_q": verify_width,
                "microbatches": args.microbatches,
                "sequences_per_lane": sequences_per_lane,
                "physical_tokens_per_lane": physical_tokens_per_lane,
                "physical_tokens_per_step": (
                    args.ag_size * args.sequences_per_ag_rank * verify_width
                ),
                "physical_routed_assignments_per_f_rank_per_lane": (
                    args.ag_size * physical_tokens_per_lane * profile.routed_top_k / args.eg_size
                ),
                "accepted_tokens": None,
                "accepted_tokens_note": (
                    "acceptance is intentionally not applied to kernel latency; it converts "
                    "candidate output to useful tokens at the system-model layer"
                ),
                "layers": layers,
                "routing": args.routing,
                "hot_expert_fraction": args.hot_expert_fraction,
                "routing_load": route_summary(local_route_counts.cpu()),
                "input_policy": (
                    "fixed deterministic representative hidden state per A-rank and lane; "
                    "the same input is reused independently for every measured MoE layer"
                ),
                "input_seed_base": INPUT_SEED_BASE,
                "input_seed_formula": "input_seed_base + ag_rank * 4 + lane",
            },
            "kernel_tuning": {
                "expected_tokens_per_lane": expected_tokens,
                "expected_tokens_source": adapter.eg_expected_tokens_source,
                "prefetch_mib": args.prefetch_mib,
                "ag_sms": args.ag_sms,
            },
            "warmups": args.warmups,
            "iterations": args.iterations,
            "stage_cuda": stage_summary,
            "stage_wall": summarize(stage_wall),
            "stage_cuda_samples_ms": stage_cuda,
            "stage_wall_samples_ms": stage_wall,
            "rank_cuda_samples_ms": [
                [sample["cuda_ms"] for sample in rows] for rows in gathered_iterations
            ],
            "rank_wall_samples_ms": [
                [sample["wall_ms"] for sample in rows] for rows in gathered_iterations
            ],
            "all_outputs_finite_by_rank": finite_all,
            "ag_output_abs_mean_by_rank": output_means,
            "stable": bool(stage_summary["cv_percent"] <= 3.0 and all(finite_all)),
            "stability_contract": "stage CUDA CV <= 3% and finite output on every rank",
            "initialization_seconds_by_rank": init_times,
            "peak_cuda_memory_bytes_by_rank": peak_memories,
            "source": source_provenance(),
            "provenance_by_rank": provenance,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(args.output), "stage_cuda": stage_summary}, indent=2))

    dist.barrier()
    adapter.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
