from __future__ import annotations

import json
import sqlite3

import pytest
from minisgl.afd_cupti import AFD_CUPTI_SCHEMA, export_cupti_steps, read_rank_report
from minisgl.afd_metrics import AFD_MANIFEST_SCHEMA


def _report(path, *, role: str, step_id: int = 7, gpu: bool = True):
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE NVTX_EVENTS "
            "(eventType INTEGER, start INTEGER, end INTEGER, globalTid INTEGER, "
            "text TEXT, textId INTEGER)"
        )
        db.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")
        db.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME "
            "(start INTEGER, end INTEGER, globalTid INTEGER, correlationId INTEGER, nameId INTEGER)"
        )
        db.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE "
            "(start INTEGER, end INTEGER, globalPid INTEGER, correlationId INTEGER)"
        )
        global_pid = 0x100000000
        global_tid = global_pid + 17
        db.execute(
            "INSERT INTO NVTX_EVENTS VALUES (59, 100, 200, ?, ?, NULL)",
            (global_tid, f"AFD_{role}_Step[step={step_id}][phase=decode]"),
        )
        db.execute("INSERT INTO StringIds VALUES (1, 'cudaGraphLaunch_v10000')")
        db.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (120, 125, ?, 41, 1)",
            (global_tid,),
        )
        if gpu:
            db.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES (130, 180, ?, 41)",
                (global_pid,),
            )
        db.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES (130, 180, ?, 41)",
            (global_pid + 0x1000000,),
        )
    return path


def _manifest(path):
    payload = {
        "schema": AFD_MANIFEST_SCHEMA,
        "source_revision": "1" * 40,
        "model_id": "Qwen/Qwen3-235B-A22B-FP8",
        "model_revision": "2" * 40,
        "system": "b200_sxm",
        "hardware": {
            "gpu_models": {"0": "NVIDIA B200", "1": "NVIDIA B200"},
            "gpu_to_hca": {"0": ["mlx5_0"], "1": ["mlx5_1"]},
            "backend_hcas": {"0": ["mlx5_0"], "1": ["mlx5_1"]},
            "gpu_clocks_mhz": {"0": 1830, "1": 1830},
            "nic_counter_deltas": {
                "mlx5_0": {"rx_bytes": 1, "tx_bytes": 2},
                "mlx5_1": {"rx_bytes": 3, "tx_bytes": 4},
            },
            "per_rank_bandwidth_ceiling_gbps": {"0": 400, "1": 400},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _raw_report(path):
    path.write_bytes(b"Nsight report")
    return path


def test_rank_report_correlates_graph_to_step_and_process(tmp_path):
    report = _report(tmp_path / "rank.sqlite", role="AG")

    assert read_rank_report(report, rank=0, role="AG") == [
        {
            "schema": AFD_CUPTI_SCHEMA,
            "record_type": "rank_step",
            "rank": 0,
            "role": "AG",
            "step_id": 7,
            "gpu_span_ms": 0.00005,
            "activity_counts": {"graph": 1},
        }
    ]


def test_export_requires_matching_rank_steps(tmp_path):
    ag = _report(tmp_path / "ag.sqlite", role="AG")
    eg = _report(tmp_path / "eg.sqlite", role="EG", step_id=8)

    with pytest.raises(ValueError, match="decode-step IDs differ"):
        export_cupti_steps(
            manifest_path=_manifest(tmp_path / "manifest.json"),
            rank_reports={0: ag, 1: eg},
            raw_reports={
                0: _raw_report(tmp_path / "ag.nsys-rep"),
                1: _raw_report(tmp_path / "eg.nsys-rep"),
            },
            attn_workers=1,
            output_path=tmp_path / "profile.jsonl",
        )

    assert not (tmp_path / "profile.jsonl").exists()


def test_export_records_manifest_and_sqlite_hashes(tmp_path):
    ag = _report(tmp_path / "ag.sqlite", role="AG")
    eg = _report(tmp_path / "eg.sqlite", role="EG")
    output = tmp_path / "profile.jsonl"

    export_cupti_steps(
        manifest_path=_manifest(tmp_path / "manifest.json"),
        rank_reports={0: ag, 1: eg},
        raw_reports={
            0: _raw_report(tmp_path / "ag.nsys-rep"),
            1: _raw_report(tmp_path / "eg.nsys-rep"),
        },
        attn_workers=1,
        output_path=output,
    )

    run, ag_step, eg_step = [json.loads(line) for line in output.read_text().splitlines()]
    assert run["record_type"] == "run"
    assert set(run["sqlite_sha256"]) == {"0", "1"}
    assert set(run["nsys_rep_sha256"]) == {"0", "1"}
    assert len(run["manifest_sha256"]) == 64
    assert (ag_step["role"], eg_step["role"]) == ("AG", "EG")
    assert (ag_step["step_id"], eg_step["step_id"]) == (7, 7)

    with pytest.raises(FileExistsError):
        export_cupti_steps(
            manifest_path=tmp_path / "manifest.json",
            rank_reports={0: ag, 1: eg},
            raw_reports={0: tmp_path / "ag.nsys-rep", 1: tmp_path / "eg.nsys-rep"},
            attn_workers=1,
            output_path=output,
        )


def test_rank_report_rejects_missing_gpu_activity(tmp_path):
    report = _report(tmp_path / "rank.sqlite", role="AG", gpu=False)

    with pytest.raises(ValueError, match="no correlated CUPTI GPU activities"):
        read_rank_report(report, rank=0, role="AG")


def test_rank_report_rejects_incomplete_graph_trace(tmp_path):
    report = _report(tmp_path / "rank.sqlite", role="AG", gpu=False)
    with sqlite3.connect(report) as db:
        db.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
            "(start INTEGER, end INTEGER, globalPid INTEGER, correlationId INTEGER)"
        )
        db.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (130, 180, ?, 41)",
            (0x100000000,),
        )

    with pytest.raises(ValueError, match="missing CUDA graph activity"):
        read_rank_report(report, rank=0, role="AG")
