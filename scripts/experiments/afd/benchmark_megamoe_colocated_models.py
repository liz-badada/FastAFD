#!/usr/bin/env python3
"""Measure colocated MegaMoE for model-exact FP4 MoE profiles on B200.

Every rank owns both source tokens and an EP shard of the experts.  The timed
boundary is per-token FP8 quantization and symmetric-buffer staging, fused
MegaMoE dispatch/FP8xFP4 compute/combine, and routed-output scaling over all
MoE layers.  Router logits/top-k selection and non-MoE blocks are excluded.
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
from typing import Any

import torch
import torch.distributed as dist
from minisgl.kernel import deepgemm
from minisgl.kernel.fp8_quant import (
    per_token_cast_to_fp8_cuda,
    per_token_cast_to_fp8_torch,
)
from minisgl.kernel.megamoe_m2n_mega import cast_weights_to_fp4, transform_weights_for_mega_moe
from minisgl.kernel.megamoe_mega import MegaMoESymmBuffer, fp8_fp4_mega_moe
from minisgl.moe.megamoe_model_profiles import (
    MEGAMOE_MODEL_PROFILES,
    MegaMoEModelProfile,
    get_megamoe_model_profile,
)

FP4_PROFILES = {
    key: profile
    for key, profile in MEGAMOE_MODEL_PROFILES.items()
    if profile.weight_precision == "fp4"
}
INPUT_SEED_BASE = 0xAFD40000


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


def protocol_shape(profile: MegaMoEModelProfile, ep_size: int) -> tuple[int, int, int]:
    profile.validate(eg_size=ep_size)
    local_routed = profile.num_routed_experts // ep_size
    local_protocol = local_routed + profile.num_shared_experts
    return local_routed, local_protocol * ep_size, profile.protocol_top_k


def make_protocol_routes(
    profile: MegaMoEModelProfile,
    tokens: int,
    *,
    rank: int,
    ep_size: int,
    routing: str,
    hot_expert_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    local_routed, protocol_experts, protocol_topk = protocol_shape(profile, ep_size)
    active_experts = profile.num_routed_experts
    if routing == "hotset":
        active_experts = max(
            profile.routed_top_k,
            int(round(profile.num_routed_experts * hot_expert_fraction)),
        )
    token_ids = torch.arange(tokens, dtype=torch.int64, device="cuda") + rank * tokens
    offsets = torch.arange(profile.routed_top_k, dtype=torch.int64, device="cuda")
    real_ids = (token_ids[:, None] * profile.routed_top_k + offsets[None, :]) % active_experts
    owner = torch.div(real_ids, local_routed, rounding_mode="floor")
    local_id = real_ids.remainder(local_routed)
    protocol_ids = owner * (local_routed + profile.num_shared_experts) + local_id

    raw_weights = torch.arange(
        profile.routed_top_k,
        0,
        -1,
        dtype=torch.float32,
        device="cuda",
    )
    weights = raw_weights.expand(tokens, -1).clone()
    if profile.gate_renormalize:
        weights /= weights.sum(dim=1, keepdim=True)
    if profile.num_shared_experts:
        shared_id = rank * (local_routed + 1) + local_routed
        shared_ids = torch.full((tokens, 1), shared_id, dtype=torch.int64, device="cuda")
        shared_weights = torch.full(
            (tokens, 1),
            1.0 / profile.routed_scaling_factor,
            dtype=torch.float32,
            device="cuda",
        )
        protocol_ids = torch.cat((protocol_ids, shared_ids), dim=1)
        weights = torch.cat((weights, shared_weights), dim=1)
    counts = torch.bincount(protocol_ids.flatten(), minlength=protocol_experts)
    assert protocol_ids.shape == (tokens, protocol_topk)
    return protocol_ids.contiguous(), weights.contiguous(), counts


def route_summary(counts: torch.Tensor) -> dict[str, float | int]:
    counts_f = counts.float()
    mean = float(counts_f.mean().item())
    std = float(counts_f.std(unbiased=False).item())
    sorted_counts = counts.sort().values
    p95_index = min(len(sorted_counts) - 1, math.ceil(0.95 * len(sorted_counts)) - 1)
    return {
        "assignments": int(counts.sum().item()),
        "active_protocol_experts": int((counts > 0).sum().item()),
        "min_assignments": int(counts.min().item()),
        "mean_assignments": mean,
        "p95_assignments": int(sorted_counts[p95_index].item()),
        "max_assignments": int(counts.max().item()),
        "load_cv_percent": 100.0 * std / mean if mean else 0.0,
    }


def make_layer_weights(
    profile: MegaMoEModelProfile,
    *,
    local_protocol_experts: int,
    layer_id: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    value = 0.0625 if layer_id % 2 == 0 else 0.125
    l1 = torch.full(
        (local_protocol_experts, 2 * profile.intermediate_size, profile.hidden_size),
        value,
        dtype=torch.bfloat16,
        device="cuda",
    )
    l2 = torch.full(
        (local_protocol_experts, profile.hidden_size, profile.intermediate_size),
        value,
        dtype=torch.bfloat16,
        device="cuda",
    )
    l1_fp4 = cast_weights_to_fp4(l1)
    l2_fp4 = cast_weights_to_fp4(l2)
    del l1, l2
    return transform_weights_for_mega_moe(l1_fp4, l2_fp4)


def validate_gran32_packed_quant(hidden_size: int, rank: int) -> dict[str, Any]:
    """Require the production CUDA quantizer to match the Torch contract exactly."""
    generator = torch.Generator(device="cuda")
    generator.manual_seed(0xA1C000 + rank)
    values = torch.randn(
        (7, hidden_size),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    ).mul_(0.375)
    values[:, ::127].mul_(3.0)
    x = values.to(torch.bfloat16).contiguous()
    cuda_fp8, cuda_scales = per_token_cast_to_fp8_cuda(
        x,
        use_ue8m0=True,
        gran_k=32,
        use_packed_ue8m0=True,
    )
    reference_fp8, reference_scales = per_token_cast_to_fp8_torch(
        x,
        use_ue8m0=True,
        gran_k=32,
        use_packed_ue8m0=True,
    )
    fp8_mismatches = int(
        (cuda_fp8.view(torch.uint8) != reference_fp8.view(torch.uint8)).sum().item()
    )
    scale_mismatches = int((cuda_scales != reference_scales).sum().item())
    result = {
        "contract": "CUDA packed UE8M0 gran_k=32 equals Torch reference",
        "input_shape": list(x.shape),
        "input_dtype": str(x.dtype),
        "fp8_elements": cuda_fp8.numel(),
        "packed_scale_elements": cuda_scales.numel(),
        "fp8_byte_mismatches": fp8_mismatches,
        "packed_scale_mismatches": scale_mismatches,
        "passed": fp8_mismatches == 0 and scale_mismatches == 0,
    }
    del values, x, cuda_fp8, cuda_scales, reference_fp8, reference_scales
    return result


def stage_once(
    *,
    profile: MegaMoEModelProfile,
    buffer: MegaMoESymmBuffer,
    weights: list[tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]],
    hidden: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    output: torch.Tensor,
) -> None:
    for l1_weights, l2_weights in weights:
        x_fp8, x_sf = deepgemm.per_token_cast_to_fp8(
            hidden,
            use_ue8m0=True,
            gran_k=32,
            use_packed_ue8m0=True,
        )
        buffer.x[: hidden.shape[0]].copy_(x_fp8)
        buffer.x_sf[: hidden.shape[0]].copy_(x_sf)
        buffer.topk_idx[: hidden.shape[0]].copy_(topk_ids)
        buffer.topk_weights[: hidden.shape[0]].copy_(topk_weights)
        fp8_fp4_mega_moe(
            output,
            l1_weights,
            l2_weights,
            buffer,
            activation_clamp=profile.activation_clamp,
            activation_alpha=profile.activation_alpha,
            activation_up_bias=profile.activation_up_bias,
            fast_math=True,
        )
        if profile.routed_scaling_factor != 1.0:
            output.mul_(profile.routed_scaling_factor)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(FP4_PROFILES))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ep-size", type=int, required=True)
    parser.add_argument("--tokens-per-rank", type=int, required=True)
    parser.add_argument("--mtp-nextn", type=int, default=0)
    parser.add_argument("--layers", type=int, default=0, help="0 means all model MoE layers")
    parser.add_argument("--routing", choices=("balanced", "hotset"), default="balanced")
    parser.add_argument("--hot-expert-fraction", type=float, default=0.25)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()

    profile = get_megamoe_model_profile(args.model)
    profile.validate(eg_size=args.ep_size)
    if profile.weight_precision != "fp4":
        raise SystemExit("colocated MegaMoE currently supports FP4 weights only")
    layers = profile.num_moe_layers if args.layers == 0 else args.layers
    if not 1 <= layers <= profile.num_moe_layers:
        raise SystemExit(f"layers must be in [1, {profile.num_moe_layers}]")
    if args.tokens_per_rank <= 0:
        raise SystemExit("tokens-per-rank must be positive")
    if args.mtp_nextn < 0:
        raise SystemExit("mtp-nextn must be non-negative")
    if not 0 < args.hot_expert_fraction <= 1:
        raise SystemExit("hot-expert-fraction must be in (0, 1]")
    if args.warmups < 1 or args.iterations < 3:
        raise SystemExit("warmups must be >=1 and iterations must be >=3")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != args.ep_size:
        raise SystemExit(f"WORLD_SIZE={world_size}, expected ep_size={args.ep_size}")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))

    quant_validation = validate_gran32_packed_quant(profile.hidden_size, rank)
    quant_passed = torch.tensor(int(quant_validation["passed"]), dtype=torch.int32, device="cuda")
    dist.all_reduce(quant_passed, op=dist.ReduceOp.MIN)
    if not bool(quant_passed.item()):
        raise RuntimeError(
            "packed gran_k=32 FP8 quantization does not match the Torch reference; "
            f"rank {rank}: {quant_validation}"
        )

    verify_width = args.mtp_nextn + 1
    physical_tokens = args.tokens_per_rank * verify_width
    local_routed, protocol_experts, protocol_topk = protocol_shape(profile, args.ep_size)
    local_protocol = protocol_experts // args.ep_size
    buffer = MegaMoESymmBuffer(
        dist.group.WORLD,
        num_experts=protocol_experts,
        num_max_tokens_per_rank=physical_tokens,
        num_topk=protocol_topk,
        hidden=profile.hidden_size,
        intermediate_hidden=profile.intermediate_size,
    )
    topk_ids, topk_weights, route_counts = make_protocol_routes(
        profile,
        physical_tokens,
        rank=rank,
        ep_size=args.ep_size,
        routing=args.routing,
        hot_expert_fraction=args.hot_expert_fraction,
    )
    dist.all_reduce(route_counts, op=dist.ReduceOp.SUM)
    input_seed = INPUT_SEED_BASE + rank
    input_generator = torch.Generator(device="cuda")
    input_generator.manual_seed(input_seed)
    hidden = (
        torch.randn(
            (physical_tokens, profile.hidden_size),
            dtype=torch.float32,
            device="cuda",
            generator=input_generator,
        )
        .mul_(0.375)
        .to(torch.bfloat16)
    )
    output = torch.empty_like(hidden)

    torch.cuda.reset_peak_memory_stats()
    init_start = time.perf_counter()
    weights = []
    for layer_id in range(layers):
        weights.append(
            make_layer_weights(
                profile,
                local_protocol_experts=local_protocol,
                layer_id=layer_id,
            )
        )
        if layer_id % 4 == 3:
            torch.cuda.empty_cache()
    dist.barrier()
    torch.cuda.synchronize()
    initialization_seconds = time.perf_counter() - init_start
    peak_memory_bytes = torch.cuda.max_memory_allocated()

    for _ in range(args.warmups):
        dist.barrier()
        stage_once(
            profile=profile,
            buffer=buffer,
            weights=weights,
            hidden=hidden,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            output=output,
        )
        torch.cuda.synchronize()

    local_cuda: list[float] = []
    local_wall: list[float] = []
    gathered_iterations: list[list[dict[str, float]]] = []
    for _ in range(args.iterations):
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        wall_start = time.perf_counter()
        stage_once(
            profile=profile,
            buffer=buffer,
            weights=weights,
            hidden=hidden,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            output=output,
        )
        end.record()
        end.synchronize()
        cuda_ms = float(start.elapsed_time(end))
        wall_ms = (time.perf_counter() - wall_start) * 1000.0
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

    finite = bool(torch.isfinite(output).all().item())
    output_abs_mean = float(output.float().abs().mean().item())
    finite_all = [None] * world_size if rank == 0 else None
    output_means = [None] * world_size if rank == 0 else None
    init_times = [None] * world_size if rank == 0 else None
    peak_memories = [None] * world_size if rank == 0 else None
    provenance = [None] * world_size if rank == 0 else None
    quant_validations = [None] * world_size if rank == 0 else None
    dist.gather_object(finite, object_gather_list=finite_all, dst=0)
    dist.gather_object(output_abs_mean, object_gather_list=output_means, dst=0)
    dist.gather_object(initialization_seconds, object_gather_list=init_times, dst=0)
    dist.gather_object(peak_memory_bytes, object_gather_list=peak_memories, dst=0)
    dist.gather_object(gpu_provenance(local_rank), object_gather_list=provenance, dst=0)
    dist.gather_object(
        quant_validation,
        object_gather_list=quant_validations,
        dst=0,
    )

    if rank == 0:
        stage_cuda = [max(sample["cuda_ms"] for sample in rows) for rows in gathered_iterations]
        stage_wall = [max(sample["wall_ms"] for sample in rows) for rows in gathered_iterations]
        stage_summary = summarize(stage_cuda)
        payload = {
            "schema": "fastafd.megamoe-colocated-model-benchmark.v1",
            "generated_at": utc_now(),
            "measurement_boundary": (
                "colocated MoE stage: per-token FP8 quantization and symmetric-buffer staging, "
                "fused expert-parallel dispatch, FP8xFP4 L1, model-specific activation, "
                "requant, L2 and combine, plus routed-output scaling; includes the replicated "
                "shared expert through an explicit per-source protocol route when declared; "
                "excludes router, attention, dense-only layers, residual, sampling and scheduler"
            ),
            "model_profile": asdict(profile),
            "topology": {
                "world_size": world_size,
                "ep_size": args.ep_size,
                "local_routed_experts": local_routed,
                "local_protocol_experts": local_protocol,
                "protocol_experts": protocol_experts,
            },
            "workload": {
                "logical_tokens_per_rank": args.tokens_per_rank,
                "mtp_nextn": args.mtp_nextn,
                "verify_width_q": verify_width,
                "physical_tokens_per_rank": physical_tokens,
                "physical_tokens_per_step": physical_tokens * args.ep_size,
                "physical_routed_assignments_per_expert_rank": (
                    physical_tokens * args.ep_size * profile.routed_top_k / args.ep_size
                ),
                "accepted_tokens": None,
                "accepted_tokens_note": (
                    "acceptance is not applied to kernel latency; the system model converts "
                    "verified candidates into useful output tokens"
                ),
                "layers": layers,
                "routing": args.routing,
                "hot_expert_fraction": args.hot_expert_fraction,
                "routing_load": route_summary(route_counts.cpu()),
                "input_policy": (
                    "fixed deterministic representative hidden state reused independently "
                    "for every measured MoE layer; pure MoE outputs are not chained because "
                    "attention, normalization and residual paths are outside this boundary"
                ),
                "input_seed_base": INPUT_SEED_BASE,
                "input_seed_by_rank": [INPUT_SEED_BASE + item for item in range(world_size)],
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
            "output_abs_mean_by_rank": output_means,
            "stable": bool(stage_summary["cv_percent"] <= 3.0 and all(finite_all)),
            "stability_contract": "stage CUDA CV <= 3% and finite output on every rank",
            "initialization_seconds_by_rank": init_times,
            "peak_cuda_memory_bytes_by_rank": peak_memories,
            "quantization_validation_by_rank": quant_validations,
            "source": source_provenance(),
            "provenance_by_rank": provenance,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(args.output), "stage_cuda": stage_summary}, indent=2))

    dist.barrier()
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
