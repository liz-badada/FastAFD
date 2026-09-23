from __future__ import annotations

import json
import math
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, TextIO

AFD_METRICS_SCHEMA = "fastafd.afd-step-metrics.v1"
AFD_MANIFEST_SCHEMA = "fastafd.afd-run-manifest.v1"
AFD_LATENCY_SCOPE = "first_worker_command_to_all_worker_replies"
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_MANIFEST_FIELDS = {
    "schema",
    "source_revision",
    "model_id",
    "model_revision",
    "system",
    "hardware",
}
_HARDWARE_FIELDS = {
    "gpu_model",
    "gpu_to_hca",
    "backend_hcas",
    "gpu_clocks_mhz",
    "nic_counter_deltas",
    "per_rank_bandwidth_ceiling_gbps",
}


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key!r}")
        result[key] = value
    return result


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _require_revision(value: Any, field: str) -> str:
    revision = _require_text(value, field)
    if not _REVISION_RE.fullmatch(revision):
        raise ValueError(f"{field} must be a lowercase 40-character Git SHA")
    return revision


def _require_rank_map(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"hardware.{field} must be a non-empty object")
    if any(not isinstance(rank, str) or not rank.isdecimal() for rank in value):
        raise ValueError(f"hardware.{field} keys must be decimal rank strings")
    return value


def _validate_hca_map(value: Any, field: str, *, allow_empty: bool) -> set[str]:
    rank_map = _require_rank_map(value, field)
    for rank, hcas in rank_map.items():
        if not isinstance(hcas, list) or (not allow_empty and not hcas):
            suffix = "a list" if allow_empty else "a non-empty list"
            raise ValueError(f"hardware.{field}.{rank} must be {suffix}")
        if any(not isinstance(hca, str) or not hca.strip() for hca in hcas):
            raise ValueError(f"hardware.{field}.{rank} must contain non-empty HCA names")
    return set(rank_map)


def _validate_positive_rank_map(value: Any, field: str) -> set[str]:
    rank_map = _require_rank_map(value, field)
    for rank, number in rank_map.items():
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(number)
            or number <= 0
        ):
            raise ValueError(f"hardware.{field}.{rank} must be finite and positive")
    return set(rank_map)


def _validate_hardware(hardware: Any) -> dict[str, Any]:
    if not isinstance(hardware, dict) or not hardware:
        raise ValueError("hardware must be a non-empty object")
    missing = _HARDWARE_FIELDS - set(hardware)
    if missing:
        raise ValueError(f"hardware fields differ: missing={sorted(missing)}")
    _require_text(hardware["gpu_model"], "hardware.gpu_model")
    rank_sets = [
        _validate_hca_map(hardware["gpu_to_hca"], "gpu_to_hca", allow_empty=False),
        _validate_hca_map(hardware["backend_hcas"], "backend_hcas", allow_empty=True),
        _validate_positive_rank_map(hardware["gpu_clocks_mhz"], "gpu_clocks_mhz"),
        _validate_positive_rank_map(
            hardware["per_rank_bandwidth_ceiling_gbps"],
            "per_rank_bandwidth_ceiling_gbps",
        ),
    ]
    if any(ranks != rank_sets[0] for ranks in rank_sets[1:]):
        raise ValueError("hardware rank maps must contain the same ranks")
    counters = hardware["nic_counter_deltas"]
    if not isinstance(counters, dict):
        raise ValueError("hardware.nic_counter_deltas must be an object")
    for hca, values in counters.items():
        if not isinstance(hca, str) or not hca.strip():
            raise ValueError("hardware.nic_counter_deltas keys must be non-empty HCA names")
        if not isinstance(values, dict) or set(values) != {"rx_bytes", "tx_bytes"}:
            raise ValueError(
                f"hardware.nic_counter_deltas.{hca} must contain rx_bytes and tx_bytes"
            )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values.values()
        ):
            raise ValueError(
                f"hardware.nic_counter_deltas.{hca} values must be non-negative integers"
            )
    selected_hcas = {hca for hcas in hardware["backend_hcas"].values() for hca in hcas}
    missing_counters = selected_hcas - set(counters)
    if missing_counters:
        raise ValueError(
            f"hardware.nic_counter_deltas is missing selected HCAs: {sorted(missing_counters)}"
        )
    try:
        json.dumps(hardware, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"hardware must contain finite JSON values: {exc}") from exc
    return hardware


@dataclass(frozen=True)
class AfdMetricsManifest:
    source_revision: str
    model_id: str
    model_revision: str
    system: str
    hardware: Mapping[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> AfdMetricsManifest:
        source = Path(path)
        try:
            payload = json.loads(
                source.read_text(encoding="utf-8"),
                object_pairs_hook=_strict_object,
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"invalid AFD metrics manifest: {source}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("AFD metrics manifest must be an object")
        unknown = set(payload) - _MANIFEST_FIELDS
        missing = _MANIFEST_FIELDS - set(payload)
        if unknown or missing:
            raise ValueError(
                f"AFD metrics manifest fields differ: missing={sorted(missing)} "
                f"unknown={sorted(unknown)}"
            )
        if payload["schema"] != AFD_MANIFEST_SCHEMA:
            raise ValueError(f"unsupported AFD metrics manifest schema: {payload['schema']!r}")
        hardware = _validate_hardware(payload["hardware"])
        return cls(
            source_revision=_require_revision(payload["source_revision"], "source_revision"),
            model_id=_require_text(payload["model_id"], "model_id"),
            model_revision=_require_revision(payload["model_revision"], "model_revision"),
            system=_require_text(payload["system"], "system"),
            hardware=hardware,
        )


@dataclass(frozen=True)
class AfdStepMeasurement:
    step_id: int
    batch_size: int
    padded_batch_size: int
    total_kv_tokens: int
    dispatch_bucket: int
    num_microbatches: int
    cuda_graph: bool
    latency_ms: float

    def __post_init__(self) -> None:
        for field in (
            "step_id",
            "batch_size",
            "padded_batch_size",
            "total_kv_tokens",
            "dispatch_bucket",
            "num_microbatches",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field} must be an integer")
            if value < 0:
                raise ValueError(f"{field} must be non-negative")
        if self.step_id == 0:
            raise ValueError("step_id must be positive")
        if self.batch_size == 0:
            raise ValueError("batch_size must be positive")
        if self.padded_batch_size < self.batch_size:
            raise ValueError("padded_batch_size must be at least batch_size")
        if self.dispatch_bucket == 0:
            raise ValueError("dispatch_bucket must be positive")
        if self.num_microbatches == 0:
            raise ValueError("num_microbatches must be positive")
        if not isinstance(self.cuda_graph, bool):
            raise TypeError("cuda_graph must be a boolean")
        if (
            isinstance(self.latency_ms, bool)
            or not isinstance(self.latency_ms, (int, float))
            or not math.isfinite(self.latency_ms)
            or self.latency_ms <= 0
        ):
            raise ValueError("latency_ms must be finite and positive")


class AfdMetricsWriter:
    def __init__(
        self,
        path: str | Path,
        *,
        manifest: AfdMetricsManifest,
        runtime: Mapping[str, Any],
    ) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        header = {
            "schema": AFD_METRICS_SCHEMA,
            "record_type": "run",
            "source_revision": manifest.source_revision,
            "model_id": manifest.model_id,
            "model_revision": manifest.model_revision,
            "system": manifest.system,
            "hardware": dict(manifest.hardware),
            "runtime": dict(runtime),
            "latency_scope": AFD_LATENCY_SCOPE,
        }
        header_line = self._encode(header)
        self._handle: TextIO = output.open("x", encoding="utf-8")
        self._lock = threading.Lock()
        self._seen_steps: set[int] = set()
        self._closed = False
        try:
            self._handle.write(header_line)
            self._handle.write("\n")
            self._handle.flush()
        except Exception:
            self._handle.close()
            self._closed = True
            raise

    def write_decode_step(self, measurement: AfdStepMeasurement) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("AFD metrics writer is closed")
            if measurement.step_id in self._seen_steps:
                raise ValueError(f"duplicate AFD metrics step_id: {measurement.step_id}")
            self._seen_steps.add(measurement.step_id)
            self._write(
                {
                    "schema": AFD_METRICS_SCHEMA,
                    "record_type": "decode_step",
                    "step_id": measurement.step_id,
                    "batch_size": measurement.batch_size,
                    "padded_batch_size": measurement.padded_batch_size,
                    "total_kv_tokens": measurement.total_kv_tokens,
                    "dispatch_bucket": measurement.dispatch_bucket,
                    "num_microbatches": measurement.num_microbatches,
                    "cuda_graph": measurement.cuda_graph,
                    "latency_ms": measurement.latency_ms,
                }
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._handle.close()
            self._closed = True

    def _write(self, record: Mapping[str, Any]) -> None:
        self._handle.write(self._encode(record))
        self._handle.write("\n")
        self._handle.flush()

    @staticmethod
    def _encode(record: Mapping[str, Any]) -> str:
        try:
            return json.dumps(
                record,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"AFD metrics record is not valid JSON: {exc}") from exc


__all__ = [
    "AFD_LATENCY_SCOPE",
    "AFD_MANIFEST_SCHEMA",
    "AFD_METRICS_SCHEMA",
    "AfdMetricsManifest",
    "AfdMetricsWriter",
    "AfdStepMeasurement",
]
