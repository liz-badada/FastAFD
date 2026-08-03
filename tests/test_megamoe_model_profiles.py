from dataclasses import replace

import pytest
from minisgl.moe.megamoe_model_profiles import (
    MEGAMOE_MODEL_PROFILES,
    get_megamoe_model_profile,
)


@pytest.mark.parametrize("key", sorted(MEGAMOE_MODEL_PROFILES))
def test_profiles_validate_on_a_supported_ep(key: str) -> None:
    profile = get_megamoe_model_profile(key)
    supported_ep = 6 if "v4_pro" in key else 4
    profile.validate(eg_size=supported_ep)


def test_remote_shared_expert_adds_one_protocol_slot_per_eg_rank() -> None:
    profile = get_megamoe_model_profile("minimax_m3_fp8")
    assert profile.protocol_top_k == 5
    assert profile.protocol_num_experts(4) == 132
    assert profile.local_experts(4) == 33


def test_m3_contract_is_not_plain_silu() -> None:
    profile = get_megamoe_model_profile("minimax_m3_fp8")
    assert profile.activation == "swigluoai"
    assert profile.activation_alpha == pytest.approx(1.702)
    assert profile.activation_clamp == pytest.approx(7.0)
    assert profile.activation_up_bias == pytest.approx(1.0)
    assert profile.routed_scaling_factor == pytest.approx(2.0)
    assert profile.num_hidden_layers == 60
    assert profile.num_moe_layers == 57


@pytest.mark.parametrize(
    ("fp8_key", "fp4_key"),
    (
        ("qwen3_235b_fp8", "qwen3_235b_fp4"),
        ("minimax_m25_fp8", "minimax_m25_fp4"),
        ("minimax_m3_fp8", "minimax_m3_fp4"),
    ),
)
def test_fp4_profile_preserves_the_model_contract(fp8_key: str, fp4_key: str) -> None:
    fp8 = get_megamoe_model_profile(fp8_key)
    fp4 = get_megamoe_model_profile(fp4_key)
    assert fp4.weight_precision == "fp4"
    assert fp4.moe_quant_contract == "w4a8_mxfp4_mxfp8"
    assert fp4.hidden_size == fp8.hidden_size
    assert fp4.intermediate_size == fp8.intermediate_size
    assert fp4.protocol_top_k == fp8.protocol_top_k
    assert fp4.activation == fp8.activation


def test_fp4_benchmark_does_not_claim_native_nvfp4_checkpoint_weights() -> None:
    profile = get_megamoe_model_profile("qwen3_235b_fp4")
    assert profile.moe_quant_contract == "w4a8_mxfp4_mxfp8"
    assert "no checkpoint weights are loaded" in profile.checkpoint_precision
    assert "NVFP4" not in profile.checkpoint_precision


def test_invalid_expert_parallel_size_fails_closed() -> None:
    profile = get_megamoe_model_profile("deepseek_v4_pro_fp4")
    with pytest.raises(ValueError, match="must be divisible"):
        profile.validate(eg_size=5)


def test_plain_silu_cannot_silently_receive_oai_parameters() -> None:
    profile = replace(
        get_megamoe_model_profile("qwen3_235b_fp8"),
        activation_up_bias=1.0,
    )
    with pytest.raises(ValueError, match="plain SiLU"):
        profile.validate(eg_size=4)
