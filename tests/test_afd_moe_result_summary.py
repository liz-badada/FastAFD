import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "experiments"
    / "afd"
    / "summarize_megamoe_model_results.py"
)
SPEC = importlib.util.spec_from_file_location("afd_moe_summary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
summary = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = summary
SPEC.loader.exec_module(summary)


def _common_payload() -> dict:
    return {
        "system_label": "b200_sxm",
        "model_profile": {
            "key": "qwen3_235b_fp4",
            "simulation_model_path": "Qwen/Qwen3-235B-A22B-FP8",
            "hidden_size": 4096,
            "intermediate_size": 1536,
            "num_routed_experts": 128,
            "routed_top_k": 8,
            "num_shared_experts": 0,
            "activation": "silu",
            "weight_precision": "fp4",
            "moe_quant_contract": "w4a8_mxfp4_mxfp8",
        },
        "source": {"commit": "abc", "source_tree_sha256": "tree"},
        "provenance_by_rank": [{"gpu_name": "NVIDIA B200"}],
    }


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_split_backend_is_not_relabelled(tmp_path: Path) -> None:
    colocated = _common_payload() | {
        "schema": summary.COLOCATED_SCHEMA,
        "topology": {"ep_size": 8},
        "workload": {"logical_tokens_per_rank": 48, "mtp_nextn": 1, "layers": 94},
        "backend_results": {
            "mega": {"stage_cuda": {"p50_ms": 7.5, "cv_percent": 1.0}, "stable": True},
            "deepep": {
                "stage_cuda": {"p50_ms": 10.5, "cv_percent": 1.5},
                "stable": True,
            },
        },
        "deepep_backend": {"deep_ep_version": "1", "sglang_version": "2"},
        "speedup_deepep_over_megamoe": 1.4,
        "speedup_lower_bound_deepep_over_megamoe": 1.3,
        "correctness_passed": True,
        "eligible_for_profile": True,
    }
    split = _common_payload() | {
        "schema": summary.SPLIT_SCHEMA,
        "moe_backend": summary.DEEPEP_BACKEND,
        "backend_implementation": "DeepEP M2N plus DeepGEMM",
        "topology": {"ag_size": 4, "eg_size": 4},
        "workload": {"sequences_per_ag_rank": 48, "mtp_nextn": 1, "microbatches": 2, "layers": 94},
        "stage_cuda": {"p50_ms": 8.0, "cv_percent": 1.2},
        "stable": True,
        "eligible_for_profile": True,
    }
    rows = [
        *summary.parse_results(_write(tmp_path / "agg.json", colocated)),
        *summary.parse_results(_write(tmp_path / "afd.json", split)),
    ]
    rows = summary.paired_split_eligibility(rows)
    summary.validate_unique_profile_keys(rows)
    profile_path = tmp_path / "profile.json"
    summary.write_profile(profile_path, rows)
    entries = json.loads(profile_path.read_text(encoding="utf-8"))["entries"]

    assert {(row.stage, row.moe_backend) for row in rows} == {
        ("agg", summary.MEGAMOE_BACKEND),
        ("agg", summary.DEEPEP_BACKEND),
        ("afd", summary.DEEPEP_BACKEND),
    }
    assert {(entry["stage"], entry["moe_backend"]) for entry in entries} == {
        ("agg", summary.MEGAMOE_BACKEND),
        ("agg", summary.DEEPEP_BACKEND),
        ("afd", summary.DEEPEP_BACKEND),
    }


def test_split_speedup_requires_the_same_routed_assignment_point(tmp_path: Path) -> None:
    def split_payload(backend: str, latency_ms: float, *, top_k: int = 8) -> dict:
        payload = _common_payload()
        payload["model_profile"] = dict(payload["model_profile"], routed_top_k=top_k)
        return payload | {
            "schema": summary.SPLIT_SCHEMA,
            "moe_backend": backend,
            "backend_implementation": backend,
            "topology": {"ag_size": 4, "eg_size": 4},
            "workload": {
                "sequences_per_ag_rank": 48,
                "mtp_nextn": 1,
                "microbatches": 2,
                "layers": 94,
            },
            "stage_cuda": {"p50_ms": latency_ms, "cv_percent": 1.0},
            "stable": True,
            "eligible_for_profile": True,
        }

    rows = [
        *summary.parse_results(
            _write(tmp_path / "mega.json", split_payload(summary.MEGAMOE_BACKEND, 5.0))
        ),
        *summary.parse_results(
            _write(tmp_path / "deep.json", split_payload(summary.DEEPEP_BACKEND, 8.0))
        ),
        # A different top-k is a different routed-assignment load and must not
        # be used as the reference for the top-8 MegaMoE point.
        *summary.parse_results(
            _write(tmp_path / "deep_top4.json", split_payload(summary.DEEPEP_BACKEND, 2.0, top_k=4))
        ),
    ]
    rows = summary.attach_same_point_split_speedups(rows)
    mega = next(row for row in rows if row.moe_backend == summary.MEGAMOE_BACKEND)

    assert mega.speedup_deepep_over_megamoe == 1.6
    assert mega.routed_top_k == 8


def test_split_load_reports_routed_assignments(tmp_path: Path) -> None:
    payload = _common_payload() | {
        "schema": summary.SPLIT_SCHEMA,
        "moe_backend": summary.MEGAMOE_BACKEND,
        "backend_implementation": "MegaMoE M2N",
        "topology": {"ag_size": 4, "eg_size": 2},
        "workload": {
            "sequences_per_ag_rank": 48,
            "mtp_nextn": 1,
            "microbatches": 2,
            "layers": 94,
        },
        "stage_cuda": {"p50_ms": 5.0, "cv_percent": 1.0},
        "stable": True,
        "eligible_for_profile": True,
    }
    row = summary.parse_results(_write(tmp_path / "split.json", payload))[0]

    logical_tokens, routed_assignments = summary.result_row_loads(row)

    assert logical_tokens == 96
    assert routed_assignments == 768


def test_unstable_colocated_backend_is_not_exported(tmp_path: Path) -> None:
    colocated = _common_payload() | {
        "schema": summary.COLOCATED_SCHEMA,
        "topology": {"ep_size": 8},
        "workload": {"logical_tokens_per_rank": 48, "mtp_nextn": 1, "layers": 94},
        "backend_results": {
            "mega": {"stage_cuda": {"p50_ms": 7.5, "cv_percent": 1.0}, "stable": True},
            "deepep": {
                "stage_cuda": {"p50_ms": 10.5, "cv_percent": 4.0},
                "stable": False,
            },
        },
        "deepep_backend": {"deep_ep_version": "1", "sglang_version": "2"},
        "speedup_deepep_over_megamoe": 1.4,
        "speedup_lower_bound_deepep_over_megamoe": 1.3,
        "correctness_passed": True,
        "eligible_for_profile": True,
    }
    rows = summary.parse_results(_write(tmp_path / "agg.json", colocated))
    profile_path = tmp_path / "profile.json"
    summary.write_profile(profile_path, rows)
    entries = json.loads(profile_path.read_text(encoding="utf-8"))["entries"]

    assert [(row.moe_backend, row.eligible) for row in rows] == [
        (summary.MEGAMOE_BACKEND, True),
        (summary.DEEPEP_BACKEND, False),
    ]
    assert [(entry["stage"], entry["moe_backend"]) for entry in entries] == [
        ("agg", summary.MEGAMOE_BACKEND)
    ]


def test_legacy_split_schema_defaults_to_megamoe(tmp_path: Path) -> None:
    payload = _common_payload() | {
        "schema": summary.LEGACY_SPLIT_SCHEMA,
        "topology": {"ag_size": 4, "eg_size": 4},
        "workload": {"sequences_per_ag_rank": 48, "mtp_nextn": 0, "microbatches": 2, "layers": 94},
        "stage_cuda": {"p50_ms": 6.0, "cv_percent": 1.0},
        "stable": True,
    }
    rows = summary.parse_results(_write(tmp_path / "legacy.json", payload))

    assert len(rows) == 1
    assert rows[0].moe_backend == summary.MEGAMOE_BACKEND


def test_base_profile_validates_and_retains_incremental_split(tmp_path: Path) -> None:
    colocated = _common_payload() | {
        "schema": summary.COLOCATED_SCHEMA,
        "topology": {"ep_size": 8},
        "workload": {"logical_tokens_per_rank": 48, "mtp_nextn": 1, "layers": 94},
        "backend_results": {
            "mega": {"stage_cuda": {"p50_ms": 7.5, "cv_percent": 1.0}, "stable": True},
            "deepep": {
                "stage_cuda": {"p50_ms": 10.5, "cv_percent": 1.5},
                "stable": True,
            },
        },
        "deepep_backend": {"deep_ep_version": "1", "sglang_version": "2"},
        "speedup_deepep_over_megamoe": 1.4,
        "speedup_lower_bound_deepep_over_megamoe": 1.3,
        "correctness_passed": True,
        "eligible_for_profile": True,
    }
    base_rows = summary.parse_results(_write(tmp_path / "agg.json", colocated))
    base_path = tmp_path / "base.json"
    summary.write_profile(base_path, base_rows)
    base_entries = summary.load_base_profile(base_path)

    split = _common_payload() | {
        "schema": summary.SPLIT_SCHEMA,
        "moe_backend": summary.DEEPEP_BACKEND,
        "backend_implementation": "DeepEP M2N plus DeepGEMM",
        "topology": {"ag_size": 4, "eg_size": 4},
        "workload": {
            "sequences_per_ag_rank": 48,
            "mtp_nextn": 1,
            "microbatches": 2,
            "layers": 94,
        },
        "stage_cuda": {"p50_ms": 8.0, "cv_percent": 1.2},
        "stable": True,
        "eligible_for_profile": True,
    }
    split_rows = summary.parse_results(_write(tmp_path / "afd.json", split))
    paired = summary.paired_split_eligibility(split_rows, base_entries=base_entries)
    merged_path = tmp_path / "merged.json"
    summary.write_profile(merged_path, paired, base_entries=base_entries)
    entries = json.loads(merged_path.read_text(encoding="utf-8"))["entries"]

    assert paired[0].eligible is True
    assert len(entries) == 3
    assert (entries[-1]["stage"], entries[-1]["moe_backend"]) == (
        "afd",
        summary.DEEPEP_BACKEND,
    )


def test_base_profile_rejects_duplicate_incremental_key(tmp_path: Path) -> None:
    payload = _common_payload() | {
        "schema": summary.LEGACY_SPLIT_SCHEMA,
        "topology": {"ag_size": 4, "eg_size": 4},
        "workload": {
            "sequences_per_ag_rank": 48,
            "mtp_nextn": 0,
            "microbatches": 2,
            "layers": 94,
        },
        "stage_cuda": {"p50_ms": 6.0, "cv_percent": 1.0},
        "stable": True,
    }
    rows = summary.parse_results(_write(tmp_path / "split.json", payload))
    rows[0] = summary.ResultRow(**(rows[0].__dict__ | {"eligible": True}))
    base_path = tmp_path / "base.json"
    summary.write_profile(base_path, rows)

    try:
        summary.write_profile(
            tmp_path / "merged.json",
            rows,
            base_entries=summary.load_base_profile(base_path),
        )
    except ValueError as error:
        assert "duplicates an exact key" in str(error)
    else:
        raise AssertionError("duplicate incremental exact key was accepted")


def test_main_can_replace_base_stage(tmp_path: Path) -> None:
    old = _common_payload() | {
        "schema": summary.COLOCATED_SCHEMA,
        "topology": {"ep_size": 8},
        "workload": {"logical_tokens_per_rank": 48, "mtp_nextn": 1, "layers": 94},
        "backend_results": {
            "mega": {"stage_cuda": {"p50_ms": 7.5, "cv_percent": 1.0}, "stable": True},
            "deepep": {
                "stage_cuda": {"p50_ms": 10.5, "cv_percent": 1.5},
                "stable": True,
            },
        },
        "deepep_backend": {"deep_ep_version": "1", "sglang_version": "2"},
        "correctness_passed": True,
        "eligible_for_profile": True,
    }
    retained_split = _common_payload() | {
        "schema": summary.LEGACY_SPLIT_SCHEMA,
        "topology": {"ag_size": 4, "eg_size": 4},
        "workload": {
            "sequences_per_ag_rank": 48,
            "mtp_nextn": 0,
            "microbatches": 2,
            "layers": 94,
        },
        "stage_cuda": {"p50_ms": 6.0, "cv_percent": 1.0},
        "stable": True,
    }
    old_rows = [
        *summary.parse_results(_write(tmp_path / "old.json", old)),
        *summary.parse_results(_write(tmp_path / "retained_split.json", retained_split)),
    ]
    base_path = tmp_path / "base.json"
    summary.write_profile(base_path, old_rows)

    new = json.loads(json.dumps(old))
    new["backend_results"]["mega"]["stage_cuda"]["p50_ms"] = 6.5
    new_path = _write(tmp_path / "new.json", new)
    profile_path = tmp_path / "refreshed.json"
    markdown_path = tmp_path / "refreshed.md"
    argv = [
        str(SCRIPT),
        str(new_path),
        "--base-profile",
        str(base_path),
        "--replace-base-stage",
        "agg",
        "--markdown",
        str(markdown_path),
        "--profile",
        str(profile_path),
    ]
    with patch.object(sys, "argv", argv):
        assert summary.main() == 0

    entries = json.loads(profile_path.read_text(encoding="utf-8"))["entries"]
    assert len(entries) == 3
    mega = next(
        entry
        for entry in entries
        if entry["stage"] == "agg" and entry["moe_backend"] == summary.MEGAMOE_BACKEND
    )
    assert mega["latency_ms"] == 6.5
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "| qwen3_235b_fp4 | afd | megamoe | 1 | 4A4F |" in markdown
