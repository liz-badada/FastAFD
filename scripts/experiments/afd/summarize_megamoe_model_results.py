#!/usr/bin/env python3
"""Build a compact, auditable table from multi-model MegaMoE JSON results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

COLOCATED_SCHEMA = "fastafd.megamoe-colocated-model-benchmark.v2"
SPLIT_SCHEMA = "fastafd.megamoe-m2n-model-benchmark.v1"


@dataclass(frozen=True)
class ResultRow:
    path: str
    stage: str
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
    mega_p50_ms: float
    reference_p50_ms: float | None
    reference_stack: str | None
    speedup: float | None
    speedup_lower_bound: float | None
    mega_cv_percent: float
    reference_cv_percent: float | None
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


def parse_result(path: Path) -> ResultRow | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema = payload.get("schema")
    if schema not in {COLOCATED_SCHEMA, SPLIT_SCHEMA}:
        return None
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
        "driver_power_clocks_memory": str(
            provenance.get("driver_power_clocks_memory", "unknown")
        ),
        "container_image": str(provenance.get("container_image", "unknown")),
        "slurm_partition": str(provenance.get("slurm_partition", "unknown")),
        "slurm_job_id": str(provenance.get("slurm_job_id", "unknown")),
        "slurm_node": str(provenance.get("slurm_node", "unknown")),
    }
    if schema == COLOCATED_SCHEMA:
        backends = payload["backend_results"]
        if "mega" not in backends:
            return None
        mega = backends["mega"]
        reference = backends.get("deepep")
        reference_metadata = payload.get("deepep_backend") or {}
        ep_size = int(topology["ep_size"])
        return ResultRow(
            path=str(path),
            stage="agg",
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
            topology=f"ep{ep_size}",
            logical_batch=int(workload["logical_tokens_per_rank"]),
            mtp_nextn=int(workload["mtp_nextn"]),
            microbatches=1,
            layers=int(workload["layers"]),
            mega_p50_ms=float(mega["stage_cuda"]["p50_ms"]),
            reference_p50_ms=(
                None if reference is None else float(reference["stage_cuda"]["p50_ms"])
            ),
            reference_stack=(
                None
                if reference is None
                else (
                    f"DeepEP {reference_metadata.get('deep_ep_version', 'unknown')} / "
                    f"SGLang {reference_metadata.get('sglang_version', 'unknown')}"
                )
            ),
            speedup=_float(payload.get("speedup_deepep_over_megamoe")),
            speedup_lower_bound=_float(payload.get("speedup_lower_bound_deepep_over_megamoe")),
            mega_cv_percent=float(mega["stage_cuda"]["cv_percent"]),
            reference_cv_percent=(
                None if reference is None else float(reference["stage_cuda"]["cv_percent"])
            ),
            correctness=payload.get("correctness_passed"),
            stable=bool(mega["stable"]),
            eligible=bool(payload.get("eligible_for_profile", False)),
            source_commit=str(source.get("commit", "")),
            source_tree_sha256=str(source.get("source_tree_sha256", "")),
            **environment,
        )

    stage = payload["stage_cuda"]
    ag_size = int(topology["ag_size"])
    eg_size = int(topology["eg_size"])
    return ResultRow(
        path=str(path),
        stage="afd",
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
        mega_p50_ms=float(stage["p50_ms"]),
        reference_p50_ms=None,
        reference_stack=None,
        speedup=None,
        speedup_lower_bound=None,
        mega_cv_percent=float(stage["cv_percent"]),
        reference_cv_percent=None,
        correctness=None,
        stable=bool(payload["stable"]),
        eligible=False,
        source_commit=str(source.get("commit", "")),
        source_tree_sha256=str(source.get("source_tree_sha256", "")),
        **environment,
    )


def paired_split_eligibility(rows: list[ResultRow]) -> list[ResultRow]:
    def validation_key(row: ResultRow) -> tuple[Any, ...]:
        return (
            row.model,
            row.precision,
            row.system,
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

    paired = []
    for row in rows:
        if row.stage != "afd":
            paired.append(row)
            continue
        reference = validated.get(validation_key(row))
        paired.append(
            ResultRow(
                **(
                    row.__dict__
                    | {
                        "correctness": None if reference is None else reference.correctness,
                        "speedup": None if reference is None else reference.speedup,
                        "speedup_lower_bound": (
                            None if reference is None else reference.speedup_lower_bound
                        ),
                        "eligible": bool(row.stable and reference is not None),
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
        "precision",
        "system",
        "topology",
        "logical_batch",
        "physical_batch",
        "mtp_nextn",
        "microbatches",
        "layers",
        "mega_p50_ms",
        "reference_p50_ms",
        "reference_stack",
        "speedup",
        "speedup_lower_bound",
        "mega_cv_percent",
        "reference_cv_percent",
        "correctness",
        "eligible",
        "source_commit",
    )
    lines = [
        "# MegaMoE measured latency reference",
        "",
        "Only exact measured points are listed. An AFD row is eligible only when it is stable and has a correctness-passing colocated MegaMoE-versus-DeepEP result at the same model, precision, system, logical source-rank batch, MTP nextN, and layer count.",
        "",
        "The speedup columns compare complete colocated MoE-stage backend paths, including production quantization, dispatch/combine, expert alignment, scatter/gather, and GEMMs. They are not GEMM-only or end-to-end serving speedups. The AIC profile consumes the absolute MegaMoE latency.",
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
            "| model | precision | system | colocated points | split points | colocated MegaMoE p50 ms | split MegaMoE p50 ms | DeepEP / MegaMoE median | conservative lower bound | max MegaMoE CV % | reference stack |",
            "| --- | --- | --- | ---: | ---: | --- | --- | --- | --- | ---: | --- |",
        ]
    )
    for model in sorted({row.model for row in rows}):
        model_rows = [row for row in rows if row.model == model and row.eligible]
        colocated = [row for row in model_rows if row.stage == "agg"]
        split = [row for row in model_rows if row.stage == "afd"]
        lines.append(
            "| "
            + " | ".join(
                [
                    model,
                    ", ".join(sorted({row.precision for row in model_rows})),
                    ", ".join(sorted({row.system for row in model_rows})),
                    str(len(colocated)),
                    str(len(split)),
                    format_range(row.mega_p50_ms for row in colocated),
                    format_range(row.mega_p50_ms for row in split),
                    format_range(row.speedup for row in colocated if row.speedup is not None),
                    format_range(
                        row.speedup_lower_bound
                        for row in colocated
                        if row.speedup_lower_bound is not None
                    ),
                    format_value(max((row.mega_cv_percent for row in model_rows), default=None)),
                    ", ".join(
                        sorted(
                            {
                                row.reference_stack
                                for row in colocated
                                if row.reference_stack is not None
                            }
                        )
                    ),
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


def write_profile(path: Path, rows: list[ResultRow]) -> None:
    entries = []
    for row in rows:
        if not row.eligible:
            continue
        entries.append(
            {
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
                "latency_ms": row.mega_p50_ms,
                "validation": {
                    "stable": row.stable,
                    "correctness": row.correctness,
                    "matched_speedup": row.speedup,
                    "matched_speedup_lower_bound": row.speedup_lower_bound,
                    "evidence": (
                        "same-point-colocated"
                        if row.stage == "agg"
                        else "same-model-system-precision-colocated-plus-stable-split"
                    ),
                },
                "source": {
                    "commit": row.source_commit,
                    "source_tree_sha256": row.source_tree_sha256,
                    "result": row.path,
                },
            }
        )
    payload = {
        "schema": "aic.afd-moe-stage-profile.v1",
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = [row for path in result_files(args.inputs) if (row := parse_result(path)) is not None]
    rows = paired_split_eligibility(rows)
    validate_unique_profile_keys(rows)
    if args.csv is not None:
        write_csv(args.csv, rows)
    write_markdown(args.markdown, rows)
    write_profile(args.profile, rows)
    print(json.dumps({"rows": len(rows), "eligible": sum(row.eligible for row in rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
