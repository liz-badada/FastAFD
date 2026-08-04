#!/usr/bin/env python3
"""Build a compact, auditable table from multi-model MegaMoE JSON results."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

COLOCATED_SCHEMA = "fastafd.megamoe-colocated-model-benchmark.v2"
LEGACY_SPLIT_SCHEMA = "fastafd.megamoe-m2n-model-benchmark.v1"
SPLIT_SCHEMA = "fastafd.moe-m2n-model-benchmark.v2"
MEGAMOE_BACKEND = "megamoe"
DEEPEP_BACKEND = "deepep_deepgemm"
PROFILE_SCHEMA = "aic.afd-moe-stage-profile.v2"


@dataclass(frozen=True)
class ResultRow:
    path: str
    stage: str
    moe_backend: str
    backend_stack: str
    model: str
    model_path: str
    hidden_size: int
    intermediate_size: int
    routed_experts: int
    routed_top_k: int
    shared_experts: int
    activation: str
    precision: str
    system: str
    topology: str
    logical_batch: int
    mtp_nextn: int
    microbatches: int
    layers: int
    latency_p50_ms: float
    speedup_deepep_over_megamoe: float | None
    speedup_lower_bound_deepep_over_megamoe: float | None
    cv_percent: float
    correctness: bool | None
    stable: bool
    eligible: bool
    source_commit: str
    source_tree_sha256: str
    gpu_name: str
    torch_version: str
    cuda_runtime: str
    driver_power_clocks_memory: str
    container_image: str
    slurm_partition: str
    slurm_job_id: str
    slurm_node: str


def result_files(inputs: Iterable[Path]) -> list[Path]:
    files: set[Path] = set()
    for value in inputs:
        if value.is_dir():
            files.update(path for path in value.rglob("*.json") if path.is_file())
        elif value.is_file():
            files.add(value)
        else:
            raise FileNotFoundError(value)
    return sorted(files)


def _float(value: Any) -> float | None:
    return None if value is None else float(value)


def parse_results(path: Path) -> list[ResultRow]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema = payload.get("schema")
    if schema not in {COLOCATED_SCHEMA, LEGACY_SPLIT_SCHEMA, SPLIT_SCHEMA}:
        return []
    profile = payload["model_profile"]
    workload = payload["workload"]
    topology = payload["topology"]
    source = payload.get("source", {})
    provenance_rows = payload.get("provenance_by_rank") or []
    provenance = provenance_rows[0] if provenance_rows else {}
    environment = {
        "gpu_name": str(provenance.get("gpu_name", "unknown")),
        "torch_version": str(provenance.get("torch", "unknown")),
        "cuda_runtime": str(provenance.get("cuda_runtime", "unknown")),
        "driver_power_clocks_memory": str(provenance.get("driver_power_clocks_memory", "unknown")),
        "container_image": str(provenance.get("container_image", "unknown")),
        "slurm_partition": str(provenance.get("slurm_partition", "unknown")),
        "slurm_job_id": str(provenance.get("slurm_job_id", "unknown")),
        "slurm_node": str(provenance.get("slurm_node", "unknown")),
    }
    if schema == COLOCATED_SCHEMA:
        backends = payload["backend_results"]
        if "mega" not in backends:
            return []
        mega = backends["mega"]
        reference = backends.get("deepep")
        overall_eligible = bool(payload.get("eligible_for_profile", False))
        reference_metadata = payload.get("deepep_backend") or {}
        ep_size = int(topology["ep_size"])
        shared = {
            "path": str(path),
            "stage": "agg",
            "model": str(profile["key"]),
            "model_path": str(profile["simulation_model_path"]),
            "hidden_size": int(profile["hidden_size"]),
            "intermediate_size": int(profile["intermediate_size"]),
            "routed_experts": int(profile["num_routed_experts"]),
            "routed_top_k": int(profile["routed_top_k"]),
            "shared_experts": int(profile["num_shared_experts"]),
            "activation": str(profile["activation"]),
            "precision": str(profile.get("moe_quant_contract", profile["weight_precision"])),
            "system": str(payload.get("system_label", "unspecified")),
            "topology": f"ep{ep_size}",
            "logical_batch": int(workload["logical_tokens_per_rank"]),
            "mtp_nextn": int(workload["mtp_nextn"]),
            "microbatches": 1,
            "layers": int(workload["layers"]),
            "speedup_deepep_over_megamoe": _float(payload.get("speedup_deepep_over_megamoe")),
            "speedup_lower_bound_deepep_over_megamoe": _float(
                payload.get("speedup_lower_bound_deepep_over_megamoe")
            ),
            "correctness": payload.get("correctness_passed"),
            "source_commit": str(source.get("commit", "")),
            "source_tree_sha256": str(source.get("source_tree_sha256", "")),
            **environment,
        }
        rows = [
            ResultRow(
                moe_backend=MEGAMOE_BACKEND,
                backend_stack="FastAFD MegaMoE",
                latency_p50_ms=float(mega["stage_cuda"]["p50_ms"]),
                cv_percent=float(mega["stage_cuda"]["cv_percent"]),
                stable=bool(mega["stable"]),
                eligible=overall_eligible and bool(mega["stable"]),
                **shared,
            )
        ]
        if reference is not None:
            rows.append(
                ResultRow(
                    moe_backend=DEEPEP_BACKEND,
                    backend_stack=(
                        f"DeepEP {reference_metadata.get('deep_ep_version', 'unknown')} / "
                        f"SGLang {reference_metadata.get('sglang_version', 'unknown')}"
                    ),
                    latency_p50_ms=float(reference["stage_cuda"]["p50_ms"]),
                    cv_percent=float(reference["stage_cuda"]["cv_percent"]),
                    stable=bool(reference["stable"]),
                    eligible=overall_eligible and bool(reference["stable"]),
                    **shared,
                )
            )
        return rows

    stage = payload["stage_cuda"]
    ag_size = int(topology["ag_size"])
    eg_size = int(topology["eg_size"])
    backend = (
        MEGAMOE_BACKEND if schema == LEGACY_SPLIT_SCHEMA else str(payload.get("moe_backend", ""))
    )
    if backend not in {MEGAMOE_BACKEND, DEEPEP_BACKEND}:
        raise ValueError(f"unsupported split MoE backend {backend!r} in {path}")
    return [
        ResultRow(
            path=str(path),
            stage="afd",
            moe_backend=backend,
            backend_stack=str(
                payload.get(
                    "backend_implementation",
                    "FastAFD MegaMoE M2N" if backend == MEGAMOE_BACKEND else "unknown",
                )
            ),
            model=str(profile["key"]),
            model_path=str(profile["simulation_model_path"]),
            hidden_size=int(profile["hidden_size"]),
            intermediate_size=int(profile["intermediate_size"]),
            routed_experts=int(profile["num_routed_experts"]),
            routed_top_k=int(profile["routed_top_k"]),
            shared_experts=int(profile["num_shared_experts"]),
            activation=str(profile["activation"]),
            precision=str(profile.get("moe_quant_contract", profile["weight_precision"])),
            system=str(payload.get("system_label", "unspecified")),
            topology=f"{ag_size}A{eg_size}F",
            logical_batch=int(workload["sequences_per_ag_rank"]),
            mtp_nextn=int(workload["mtp_nextn"]),
            microbatches=int(workload["microbatches"]),
            layers=int(workload["layers"]),
            latency_p50_ms=float(stage["p50_ms"]),
            speedup_deepep_over_megamoe=None,
            speedup_lower_bound_deepep_over_megamoe=None,
            cv_percent=float(stage["cv_percent"]),
            correctness=None,
            stable=bool(payload["stable"]),
            eligible=bool(
                payload.get(
                    "eligible_for_profile",
                    schema == LEGACY_SPLIT_SCHEMA,
                )
            ),
            source_commit=str(source.get("commit", "")),
            source_tree_sha256=str(source.get("source_tree_sha256", "")),
            **environment,
        )
    ]


def profile_entry_key(entry: dict[str, Any]) -> tuple[Any, ...]:
    return (
        entry["model_path"],
        entry["system"],
        entry["stage"],
        entry["moe_backend"],
        entry["topology"],
        entry["logical_batch_per_source_rank"],
        entry["mtp_nextn"],
        entry["microbatches"],
        entry["moe_layers"],
        entry["moe_precision"],
    )


def profile_validation_key(entry: dict[str, Any]) -> tuple[Any, ...]:
    return (
        entry["model_profile"],
        entry["moe_precision"],
        entry["system"],
        entry["moe_backend"],
        entry["logical_batch_per_source_rank"],
        entry["mtp_nextn"],
        entry["moe_layers"],
    )


def load_base_profile(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != PROFILE_SCHEMA:
        raise ValueError(f"base profile must use schema {PROFILE_SCHEMA!r}")
    if payload.get("lookup_policy") != "exact-only":
        raise ValueError("base profile lookup_policy must be 'exact-only'")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
        raise TypeError("base profile entries must be a list of objects")
    seen: set[tuple[Any, ...]] = set()
    for entry in entries:
        key = profile_entry_key(entry)
        if key in seen:
            raise ValueError(f"duplicate exact key in base profile: {key}")
        seen.add(key)
    return copy.deepcopy(entries)


def paired_split_eligibility(
    rows: list[ResultRow],
    *,
    base_entries: list[dict[str, Any]] | None = None,
) -> list[ResultRow]:
    def validation_key(row: ResultRow) -> tuple[Any, ...]:
        return (
            row.model,
            row.precision,
            row.system,
            row.moe_backend,
            row.logical_batch,
            row.mtp_nextn,
            row.layers,
        )

    validated: dict[tuple[Any, ...], ResultRow] = {}
    for row in rows:
        if row.stage != "agg" or not row.eligible:
            continue
        key = validation_key(row)
        if previous := validated.get(key):
            raise ValueError(
                "multiple eligible colocated results match one split validation key; "
                f"select one reproducible trial: {previous.path}, {row.path}"
            )
        validated[key] = row

    base_validated = {
        profile_validation_key(entry)
        for entry in base_entries or []
        if entry.get("stage") == "agg"
        and entry.get("validation", {}).get("stable") is True
        and entry.get("validation", {}).get("correctness") is True
    }

    paired = []
    for row in rows:
        if row.stage != "afd":
            paired.append(row)
            continue
        key = validation_key(row)
        reference = validated.get(key)
        externally_validated = key in base_validated
        paired.append(
            ResultRow(
                **(
                    row.__dict__
                    | {
                        "correctness": None if reference is None else reference.correctness,
                        "speedup_deepep_over_megamoe": (
                            None if reference is None else reference.speedup_deepep_over_megamoe
                        ),
                        "speedup_lower_bound_deepep_over_megamoe": (
                            None
                            if reference is None
                            else reference.speedup_lower_bound_deepep_over_megamoe
                        ),
                        "eligible": bool(
                            row.stable
                            and row.eligible
                            and (reference is not None or externally_validated)
                        ),
                    }
                )
            )
        )
    return paired


def validate_unique_profile_keys(rows: list[ResultRow]) -> None:
    seen: dict[tuple[Any, ...], str] = {}
    for row in rows:
        if not row.eligible:
            continue
        key = (
            row.model_path,
            row.system,
            row.stage,
            row.moe_backend,
            row.topology,
            row.logical_batch,
            row.mtp_nextn,
            row.microbatches,
            row.layers,
            row.precision,
        )
        if previous := seen.get(key):
            raise ValueError(
                "multiple eligible results have the same exact profile key; "
                f"select one reproducible trial: {previous}, {row.path}"
            )
        seen[key] = row.path


def write_csv(path: Path, rows: list[ResultRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ResultRow.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(row.__dict__ for row in rows)


def format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def format_range(values: Iterable[float]) -> str:
    ordered = sorted(values)
    if not ordered:
        return "-"
    if ordered[0] == ordered[-1]:
        return f"{ordered[0]:.4f}"
    return f"{ordered[0]:.4f}-{ordered[-1]:.4f}"


def write_markdown(path: Path, rows: list[ResultRow]) -> None:
    columns = (
        "model",
        "stage",
        "moe_backend",
        "precision",
        "system",
        "topology",
        "logical_batch",
        "physical_batch",
        "mtp_nextn",
        "microbatches",
        "layers",
        "latency_p50_ms",
        "backend_stack",
        "speedup_deepep_over_megamoe",
        "speedup_lower_bound_deepep_over_megamoe",
        "cv_percent",
        "correctness",
        "eligible",
        "source_commit",
    )
    lines = [
        "# Measured AFD MoE backend latency reference",
        "",
        "Only exact measured points are listed. An AFD row is eligible only when it is stable and has a correctness-passing colocated MegaMoE-versus-DeepEP result at the same model, precision, system, logical source-rank batch, MTP nextN, and layer count.",
        "",
        "The speedup columns compare complete colocated MoE-stage backend paths, including production quantization, dispatch/combine, expert alignment, scatter/gather, and GEMMs. They are not GEMM-only or end-to-end serving speedups. The AIC profile consumes each backend's absolute latency.",
        "",
        "## Model contracts",
        "",
        "| model | hidden | expert intermediate | routed experts | routed top-k | shared experts | activation | precision |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    contracts = {
        (
            row.model,
            row.hidden_size,
            row.intermediate_size,
            row.routed_experts,
            row.routed_top_k,
            row.shared_experts,
            row.activation,
            row.precision,
        )
        for row in rows
    }
    for contract in sorted(contracts):
        lines.append("| " + " | ".join(format_value(value) for value in contract) + " |")

    lines.extend(
        [
            "",
            "## Backend summary",
            "",
            "| model | backend | precision | system | colocated points | split points | colocated p50 ms | split p50 ms | DeepEP / MegaMoE median | conservative lower bound | max CV % | backend stack |",
            "| --- | --- | --- | --- | ---: | ---: | --- | --- | --- | --- | ---: | --- |",
        ]
    )
    for model in sorted({row.model for row in rows}):
        for backend in sorted({row.moe_backend for row in rows if row.model == model}):
            model_rows = [
                row
                for row in rows
                if row.model == model and row.moe_backend == backend and row.eligible
            ]
            colocated = [row for row in model_rows if row.stage == "agg"]
            split = [row for row in model_rows if row.stage == "afd"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        model,
                        backend,
                        ", ".join(sorted({row.precision for row in model_rows})),
                        ", ".join(sorted({row.system for row in model_rows})),
                        str(len(colocated)),
                        str(len(split)),
                        format_range(row.latency_p50_ms for row in colocated),
                        format_range(row.latency_p50_ms for row in split),
                        format_range(
                            row.speedup_deepep_over_megamoe
                            for row in colocated
                            if row.speedup_deepep_over_megamoe is not None
                        ),
                        format_range(
                            row.speedup_lower_bound_deepep_over_megamoe
                            for row in colocated
                            if row.speedup_lower_bound_deepep_over_megamoe is not None
                        ),
                        format_value(max((row.cv_percent for row in model_rows), default=None)),
                        ", ".join(sorted({row.backend_stack for row in model_rows})),
                    ]
                )
                + " |"
            )
    environments: dict[tuple[str, ...], dict[str, set[str]]] = defaultdict(
        lambda: {"jobs": set(), "nodes": set(), "commits": set(), "trees": set()}
    )
    for row in rows:
        key = (
            row.system,
            row.gpu_name,
            row.driver_power_clocks_memory,
            row.torch_version,
            row.cuda_runtime,
            row.container_image,
            row.slurm_partition,
        )
        environments[key]["jobs"].add(row.slurm_job_id)
        environments[key]["nodes"].add(row.slurm_node)
        environments[key]["commits"].add(row.source_commit)
        environments[key]["trees"].add(row.source_tree_sha256)
    lines.extend(
        [
            "",
            "## Measurement environments",
            "",
            "| system | GPU | driver / power / clocks / memory | Torch | CUDA | container | partition | jobs | nodes | source commits | source tree SHA-256 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for key, values in sorted(environments.items()):
        lines.append(
            "| "
            + " | ".join(
                [
                    *key,
                    ", ".join(sorted(values["jobs"])),
                    ", ".join(sorted(values["nodes"])),
                    ", ".join(sorted(values["commits"])),
                    ", ".join(sorted(values["trees"])),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Exact measured points",
            "",
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
    )
    for row in rows:
        values = {
            key: row.logical_batch * (row.mtp_nextn + 1)
            if key == "physical_batch"
            else getattr(row, key)
            for key in columns
        }
        lines.append("| " + " | ".join(format_value(values[key]) for key in columns) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_profile(
    path: Path,
    rows: list[ResultRow],
    *,
    base_entries: list[dict[str, Any]] | None = None,
) -> None:
    entries: list[dict[str, Any]] = copy.deepcopy(base_entries or [])
    seen = {profile_entry_key(entry) for entry in entries}

    def add_entry(
        row: ResultRow,
        *,
        moe_backend: str,
        latency_ms: float,
        matched_speedup: float | None,
        matched_speedup_lower_bound: float | None,
        evidence: str,
    ) -> None:
        entry = {
            "model_path": row.model_path,
            "model_profile": row.model,
            "system": row.system,
            "stage": row.stage,
            "topology": row.topology,
            "logical_batch_per_source_rank": row.logical_batch,
            "mtp_nextn": row.mtp_nextn,
            "microbatches": row.microbatches,
            "moe_layers": row.layers,
            "moe_precision": row.precision,
            "moe_backend": moe_backend,
            "latency_ms": latency_ms,
            "validation": {
                "stable": row.stable,
                "correctness": row.correctness,
                "matched_speedup": matched_speedup,
                "matched_speedup_lower_bound": matched_speedup_lower_bound,
                "evidence": evidence,
            },
            "source": {
                "commit": row.source_commit,
                "source_tree_sha256": row.source_tree_sha256,
                "result": row.path,
            },
        }
        key = profile_entry_key(entry)
        if key in seen:
            raise ValueError(f"new result duplicates an exact key in the base profile: {key}")
        seen.add(key)
        entries.append(entry)

    for row in rows:
        if not row.eligible:
            continue
        add_entry(
            row,
            moe_backend=row.moe_backend,
            latency_ms=row.latency_p50_ms,
            matched_speedup=(
                row.speedup_deepep_over_megamoe if row.moe_backend == MEGAMOE_BACKEND else None
            ),
            matched_speedup_lower_bound=(
                row.speedup_lower_bound_deepep_over_megamoe
                if row.moe_backend == MEGAMOE_BACKEND
                else None
            ),
            evidence=(
                "same-point-colocated"
                if row.stage == "agg"
                else "same-backend-model-system-precision-colocated-plus-stable-split"
            ),
        )
    payload = {
        "schema": PROFILE_SCHEMA,
        "lookup_policy": "exact-only",
        "entries": entries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument(
        "--base-profile",
        type=Path,
        help="Validated exact profile to retain and use for colocated correctness gates",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_entries = load_base_profile(args.base_profile)
    rows = [row for path in result_files(args.inputs) for row in parse_results(path)]
    rows = paired_split_eligibility(rows, base_entries=base_entries)
    validate_unique_profile_keys(rows)
    if args.csv is not None:
        write_csv(args.csv, rows)
    write_markdown(args.markdown, rows)
    write_profile(args.profile, rows, base_entries=base_entries)
    print(json.dumps({"rows": len(rows), "eligible": sum(row.eligible for row in rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
