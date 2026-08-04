import importlib.util
import json
import sys
from pathlib import Path

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
