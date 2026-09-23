from __future__ import annotations

import json

import pytest
from minisgl.afd_metrics import (
    AFD_LATENCY_SCOPE,
    AFD_MANIFEST_SCHEMA,
    AFD_METRICS_SCHEMA,
    AfdMetricsManifest,
    AfdMetricsWriter,
    AfdStepMeasurement,
)


def _manifest_payload() -> dict:
    return {
        "schema": AFD_MANIFEST_SCHEMA,
        "source_revision": "0123456789abcdef0123456789abcdef01234567",
        "model_id": "Qwen/Qwen3-235B-A22B-FP8",
        "model_revision": "89abcdef0123456789abcdef0123456789abcdef",
        "system": "b200_sxm",
        "hardware": {
            "gpu_models": {"0": "NVIDIA B200"},
            "gpu_to_hca": {"0": ["mlx5_0"]},
            "backend_hcas": {"0": ["mlx5_0"]},
            "gpu_clocks_mhz": {"0": 1830},
            "nic_counter_deltas": {"mlx5_0": {"rx_bytes": 1, "tx_bytes": 2}},
            "per_rank_bandwidth_ceiling_gbps": {"0": 400},
        },
    }


def _write_manifest(tmp_path, payload: dict | None = None):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload or _manifest_payload()), encoding="utf-8")
    return path


def test_manifest_loads_strict_schema(tmp_path):
    manifest = AfdMetricsManifest.load(_write_manifest(tmp_path))

    assert manifest.system == "b200_sxm"
    assert manifest.hardware["gpu_models"]["0"] == "NVIDIA B200"
    manifest.validate_worker_ranks(1)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("source_revision", "main", "40-character Git SHA"),
        ("model_revision", "main", "40-character Git SHA"),
        ("hardware", {}, "non-empty object"),
    ],
)
def test_manifest_rejects_invalid_fields(tmp_path, field, value, match):
    payload = _manifest_payload()
    payload[field] = value

    with pytest.raises(ValueError, match=match):
        AfdMetricsManifest.load(_write_manifest(tmp_path, payload))


def test_manifest_rejects_duplicate_json_field(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(
        '{"schema":"fastafd.afd-run-manifest.v1","schema":"fastafd.afd-run-manifest.v1"}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON field"):
        AfdMetricsManifest.load(path)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("backend_hcas", {}, "non-empty object"),
        ("gpu_clocks_mhz", {"1": 1830}, "same ranks"),
        (
            "nic_counter_deltas",
            {"mlx5_0": {"rx_bytes": 1}},
            "rx_bytes and tx_bytes",
        ),
        ("nic_counter_deltas", {}, "missing selected HCAs"),
        ("per_rank_bandwidth_ceiling_gbps", {"0": 0}, "selected HCAs"),
    ],
)
def test_manifest_rejects_incomplete_hardware(tmp_path, field, value, match):
    payload = _manifest_payload()
    payload["hardware"][field] = value

    with pytest.raises(ValueError, match=match):
        AfdMetricsManifest.load(_write_manifest(tmp_path, payload))


def test_manifest_rejects_worker_rank_mismatch(tmp_path):
    manifest = AfdMetricsManifest.load(_write_manifest(tmp_path))

    with pytest.raises(ValueError, match=r"missing=\[1\]"):
        manifest.validate_worker_ranks(2)


def test_manifest_accepts_no_nic_path(tmp_path):
    payload = _manifest_payload()
    payload["hardware"]["gpu_to_hca"] = {"0": []}
    payload["hardware"]["backend_hcas"] = {"0": []}
    payload["hardware"]["nic_counter_deltas"] = {}
    payload["hardware"]["per_rank_bandwidth_ceiling_gbps"] = {"0": 0}

    manifest = AfdMetricsManifest.load(_write_manifest(tmp_path, payload))

    assert manifest.hardware["backend_hcas"]["0"] == []


def test_writer_emits_run_and_decode_records(tmp_path):
    manifest = AfdMetricsManifest.load(_write_manifest(tmp_path))
    output = tmp_path / "metrics.jsonl"
    writer = AfdMetricsWriter(
        output,
        manifest=manifest,
        runtime={"topology": {"attn_dp": 8, "mlp_ep": 8}},
    )
    writer.write_decode_step(
        AfdStepMeasurement(
            step_id=7,
            batch_size=64,
            padded_batch_size=64,
            total_kv_tokens=65536,
            dispatch_bucket=64,
            num_microbatches=2,
            cuda_graph=True,
            latency_ms=5.25,
        )
    )
    writer.close()

    run, step = [json.loads(line) for line in output.read_text().splitlines()]
    assert run["schema"] == AFD_METRICS_SCHEMA
    assert run["latency_scope"] == AFD_LATENCY_SCOPE
    assert step == {
        "schema": AFD_METRICS_SCHEMA,
        "record_type": "decode_step",
        "step_id": 7,
        "batch_size": 64,
        "padded_batch_size": 64,
        "total_kv_tokens": 65536,
        "dispatch_bucket": 64,
        "num_microbatches": 2,
        "cuda_graph": True,
        "latency_ms": 5.25,
    }


def test_writer_rejects_duplicate_step(tmp_path):
    writer = AfdMetricsWriter(
        tmp_path / "metrics.jsonl",
        manifest=AfdMetricsManifest.load(_write_manifest(tmp_path)),
        runtime={"topology": {}},
    )
    measurement = AfdStepMeasurement(1, 1, 1, 128, 1, 1, False, 1.0)
    writer.write_decode_step(measurement)

    with pytest.raises(ValueError, match="duplicate AFD metrics step_id"):
        writer.write_decode_step(measurement)

    writer.close()


def test_writer_does_not_overwrite_existing_output(tmp_path):
    output = tmp_path / "metrics.jsonl"
    output.write_text("existing\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        AfdMetricsWriter(
            output,
            manifest=AfdMetricsManifest.load(_write_manifest(tmp_path)),
            runtime={"topology": {}},
        )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("cuda_graph", 1, TypeError),
        ("latency_ms", True, ValueError),
        ("latency_ms", float("nan"), ValueError),
    ],
)
def test_measurement_rejects_invalid_types(field, value, error):
    values = {
        "step_id": 1,
        "batch_size": 1,
        "padded_batch_size": 1,
        "total_kv_tokens": 128,
        "dispatch_bucket": 1,
        "num_microbatches": 1,
        "cuda_graph": False,
        "latency_ms": 1.0,
    }
    values[field] = value

    with pytest.raises(error):
        AfdStepMeasurement(**values)
