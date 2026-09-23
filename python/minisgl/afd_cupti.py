from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import Any, Mapping

from .afd_metrics import AfdMetricsManifest

AFD_CUPTI_SCHEMA = "fastafd.afd-cupti-steps.v1"
_STEP_LABEL = re.compile(r"^AFD_(AG|EG)_Step\[step=([1-9][0-9]*)\]\[phase=decode\]$")
_GPU_TABLES = {
    "CUPTI_ACTIVITY_KIND_KERNEL": "kernel",
    "CUPTI_ACTIVITY_KIND_GRAPH_TRACE": "graph",
    "CUPTI_ACTIVITY_KIND_MEMCPY": "memcpy",
    "CUPTI_ACTIVITY_KIND_MEMSET": "memset",
}
_API_TABLES = ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER")


def _columns(db: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in db.execute(f"PRAGMA table_info({table})")}


def _require_columns(db: sqlite3.Connection, table: str, fields: set[str]) -> set[str]:
    columns = _columns(db, table)
    missing = fields - columns
    if missing:
        raise ValueError(f"{table} is missing columns: {sorted(missing)}")
    return columns


def _report_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _step_ranges(db: sqlite3.Connection, role: str) -> dict[int, tuple[int, int, int]]:
    columns = _require_columns(
        db, "NVTX_EVENTS", {"eventType", "start", "end", "globalTid", "text"}
    )
    strings: dict[int, str] = {}
    if "textId" in columns and _columns(db, "StringIds"):
        strings = dict(db.execute("SELECT id, value FROM StringIds"))
    fields = "start, end, globalTid, text"
    if "textId" in columns:
        fields += ", textId"
    ranges: dict[int, tuple[int, int, int]] = {}
    for row in db.execute(f"SELECT {fields} FROM NVTX_EVENTS WHERE eventType IN (59, 60)"):
        name = row[3] or (strings.get(row[4]) if len(row) > 4 else None)
        match = _STEP_LABEL.fullmatch(name or "")
        if match is None or match[1] != role:
            continue
        step_id = int(match[2])
        start, end, global_tid = row[:3]
        if any(value is None for value in (start, end, global_tid)) or end <= start:
            raise ValueError(f"invalid NVTX range for step {step_id}")
        if step_id in ranges:
            raise ValueError(f"duplicate NVTX range for step {step_id}")
        ranges[step_id] = (int(start), int(end), int(global_tid))
    if not ranges:
        raise ValueError(f"no decode-step NVTX ranges found for {role}")
    return ranges


def _api_correlations(
    db: sqlite3.Connection, start: int, end: int, global_tid: int
) -> tuple[set[int], set[int]]:
    correlation_ids: set[int] = set()
    graph_launch_ids: set[int] = set()
    strings = (
        dict(db.execute("SELECT id, value FROM StringIds")) if _columns(db, "StringIds") else {}
    )
    for table in _API_TABLES:
        columns = _columns(db, table)
        if not columns:
            continue
        _require_columns(db, table, {"start", "end", "globalTid", "correlationId"})
        name_field = ", nameId" if "nameId" in columns else ""
        for row in db.execute(
            f"SELECT correlationId{name_field} FROM {table} "
            "WHERE globalTid = ? AND start >= ? AND end <= ? AND correlationId IS NOT NULL",
            (global_tid, start, end),
        ):
            correlation_id = row[0]
            correlation_ids.add(int(correlation_id))
            if len(row) > 1 and strings.get(row[1], "").startswith(
                ("cudaGraphLaunch", "cuGraphLaunch")
            ):
                graph_launch_ids.add(int(correlation_id))
    if not correlation_ids:
        raise ValueError("step NVTX range has no correlated CUDA API calls")
    return correlation_ids, graph_launch_ids


def _gpu_activities(
    db: sqlite3.Connection, correlation_ids: set[int], global_pid: int
) -> list[tuple[str, int, int, int]]:
    activities: list[tuple[str, int, int, int]] = []
    ids = sorted(correlation_ids)
    for table, kind in _GPU_TABLES.items():
        if not _columns(db, table):
            continue
        _require_columns(db, table, {"start", "end", "correlationId", "globalPid"})
        for offset in range(0, len(ids), 500):
            batch = ids[offset : offset + 500]
            placeholders = ",".join("?" for _ in batch)
            for start, end, correlation_id in db.execute(
                f"SELECT start, end, correlationId FROM {table} WHERE globalPid = ? "
                f"AND correlationId IN ({placeholders})",
                (global_pid, *batch),
            ):
                if start is None or end is None or start <= 0 or end <= start:
                    raise ValueError(f"invalid {kind} activity timestamp")
                activities.append((kind, int(start), int(end), int(correlation_id)))
    if not activities:
        raise ValueError("step has no correlated CUPTI GPU activities")
    return activities


def read_rank_report(path: str | Path, *, rank: int, role: str) -> list[dict[str, Any]]:
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise ValueError("rank must be a non-negative integer")
    if role not in {"AG", "EG"}:
        raise ValueError("role must be AG or EG")
    source = Path(path).resolve(strict=True)
    with closing(sqlite3.connect(f"{source.as_uri()}?mode=ro", uri=True)) as db:
        ranges = _step_ranges(db, role)
        records = []
        for step_id, (host_start, host_end, global_tid) in sorted(ranges.items()):
            correlation_ids, graph_launch_ids = _api_correlations(
                db, host_start, host_end, global_tid
            )
            # Nsight stores the thread ID in the low 24 bits of globalTid.
            activities = _gpu_activities(db, correlation_ids, global_tid & ~0xFFFFFF)
            traced_graph_ids = {
                correlation_id for kind, _, _, correlation_id in activities if kind == "graph"
            }
            if graph_launch_ids - traced_graph_ids:
                raise ValueError(f"step {step_id} is missing CUDA graph activity records")
            counts = Counter(kind for kind, _, _, _ in activities)
            gpu_start = min(start for _, start, _, _ in activities)
            gpu_end = max(end for _, _, end, _ in activities)
            records.append(
                {
                    "schema": AFD_CUPTI_SCHEMA,
                    "record_type": "rank_step",
                    "rank": rank,
                    "role": role,
                    "step_id": step_id,
                    "gpu_span_ms": (gpu_end - gpu_start) / 1_000_000.0,
                    "activity_counts": dict(sorted(counts.items())),
                }
            )
    return records


def export_cupti_steps(
    *,
    manifest_path: str | Path,
    rank_reports: Mapping[int, str | Path],
    raw_reports: Mapping[int, str | Path],
    attn_workers: int,
    output_path: str | Path,
) -> None:
    manifest = AfdMetricsManifest.load(manifest_path)
    manifest.validate_worker_ranks(len(rank_reports))
    if set(raw_reports) != set(rank_reports):
        raise ValueError("Nsight and SQLite reports must cover the same ranks")
    if isinstance(attn_workers, bool) or not isinstance(attn_workers, int):
        raise TypeError("attn_workers must be an integer")
    if not 0 < attn_workers < len(rank_reports):
        raise ValueError("attn_workers must be between zero and the worker count")
    all_records: list[dict[str, Any]] = []
    sqlite_hashes: dict[str, str] = {}
    raw_hashes: dict[str, str] = {}
    step_sets: list[set[int]] = []
    for rank, path in sorted(rank_reports.items()):
        role = "AG" if rank < attn_workers else "EG"
        records = read_rank_report(path, rank=rank, role=role)
        all_records.extend(records)
        step_sets.append({record["step_id"] for record in records})
        sqlite_hashes[str(rank)] = _report_digest(Path(path))
        raw_hashes[str(rank)] = _report_digest(Path(raw_reports[rank]))
    if any(steps != step_sets[0] for steps in step_sets[1:]):
        raise ValueError("captured decode-step IDs differ across ranks")
    header = {
        "schema": AFD_CUPTI_SCHEMA,
        "record_type": "run",
        "source_revision": manifest.source_revision,
        "model_id": manifest.model_id,
        "model_revision": manifest.model_revision,
        "system": manifest.system,
        "hardware": manifest.hardware,
        "attn_workers": attn_workers,
        "manifest_sha256": _report_digest(Path(manifest_path)),
        "sqlite_sha256": sqlite_hashes,
        "nsys_rep_sha256": raw_hashes,
        "measurement": "per_rank_correlated_gpu_activity_span",
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as output:
        for record in (header, *all_records):
            output.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export AFD CUPTI step spans from Nsight SQLite")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--attn-workers", type=int, required=True)
    parser.add_argument("--rank-report", action="append", required=True, metavar="RANK=SQLITE")
    parser.add_argument("--rank-nsys", action="append", required=True, metavar="RANK=NSYS_REP")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    def parse_rank_paths(items: list[str], flag: str) -> dict[int, str]:
        paths: dict[int, str] = {}
        for item in items:
            rank_text, separator, path = item.partition("=")
            if not separator or not rank_text.isdecimal() or not path:
                parser.error(f"{flag} must be RANK=FILE")
            rank = int(rank_text)
            if rank in paths:
                parser.error(f"duplicate rank in {flag}: {rank}")
            paths[rank] = path
        return paths

    export_cupti_steps(
        manifest_path=args.manifest,
        rank_reports=parse_rank_paths(args.rank_report, "--rank-report"),
        raw_reports=parse_rank_paths(args.rank_nsys, "--rank-nsys"),
        attn_workers=args.attn_workers,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
