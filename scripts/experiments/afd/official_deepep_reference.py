"""Official DeepEP normal-mode adapter for the colocated MegaMoE benchmark.

The adapter follows SGLang's production normal-mode data path:

1. quantize source activations to FP8 with packed UE8M0 block-128 scales;
2. run official DeepEP layout, dispatch, and combine kernels;
3. expand rank-contiguous receives into expert-contiguous rows with SGLang's
   ``ep_scatter`` kernel and reduce them with ``ep_gather``.

The caller owns the two FP8xFP4 DeepGEMM operations between scatter and gather.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import torch
import torch.distributed as dist


def _distribution_version(*names: str) -> str:
    for name in names:
        try:
            return version(name)
        except PackageNotFoundError:
            continue
    return "unknown"


@dataclass
class OfficialDeepEPDispatch:
    hidden_states: torch.Tensor
    hidden_states_scale: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    num_recv_tokens_per_expert: list[int]
    psum_tokens_per_expert: torch.Tensor
    m_indices: torch.Tensor
    output_index: torch.Tensor
    handle: Any
    recv_shape: tuple[int, int]


class OfficialDeepEPReference:
    """Official DeepEP plus SGLang scatter/gather for one intranode EP group."""

    def __init__(
        self,
        *,
        group: dist.ProcessGroup,
        hidden_size: int,
        num_experts: int,
        expert_alignment: int,
    ) -> None:
        try:
            import deep_ep
            from sglang.srt.layers.moe.ep_moe.kernels import ep_gather, ep_scatter
        except ImportError as exc:
            raise RuntimeError(
                "The official DeepEP reference requires the deep_ep and sglang "
                "packages from the supported benchmark container"
            ) from exc

        self._deep_ep = deep_ep
        self._ep_scatter = ep_scatter
        self._ep_gather = ep_gather
        self.group = group
        self.world_size = dist.get_world_size(group=group)
        self.hidden_size = int(hidden_size)
        self.num_experts = int(num_experts)
        self.expert_alignment = int(expert_alignment)
        if self.hidden_size % 512:
            raise ValueError("packed block-128 scales require hidden_size divisible by 512")
        if self.num_experts % self.world_size:
            raise ValueError("num_experts must be divisible by the DeepEP group size")
        if self.expert_alignment != 128:
            raise ValueError(
                "SGLang ep_scatter requires the DeepEP receive count of every expert "
                "to be aligned to 128"
            )

        self.dispatch_config = deep_ep.Buffer.get_dispatch_config(self.world_size)
        self.combine_config = deep_ep.Buffer.get_combine_config(self.world_size)
        self.dispatch_hidden_bytes = self.hidden_size * torch.empty(
            (), dtype=torch.float8_e4m3fn
        ).element_size()
        self.combine_hidden_bytes = self.hidden_size * torch.empty(
            (), dtype=torch.bfloat16
        ).element_size()
        num_nvl_bytes = max(
            self.dispatch_config.get_nvl_buffer_size_hint(
                self.dispatch_hidden_bytes,
                self.world_size,
            ),
            self.combine_config.get_nvl_buffer_size_hint(
                self.combine_hidden_bytes,
                self.world_size,
            ),
        )
        self.buffer = deep_ep.Buffer(
            group,
            num_nvl_bytes=int(num_nvl_bytes),
            num_rdma_bytes=0,
            low_latency_mode=False,
            num_qps_per_rank=1,
            explicitly_destroy=True,
        )
        self._destroyed = False

    def metadata(self) -> dict[str, Any]:
        return {
            "implementation": "official deep_ep.Buffer normal mode",
            "deep_ep_version": _distribution_version("deep-ep", "deep_ep"),
            "deep_ep_module": str(self._deep_ep.__file__),
            "sglang_version": _distribution_version("sglang"),
            "world_size": self.world_size,
            "num_sms": int(self._deep_ep.Buffer.num_sms),
            "dispatch_config": repr(self.dispatch_config),
            "combine_config": repr(self.combine_config),
            "num_nvl_bytes": int(self.buffer.num_nvl_bytes),
            "dispatch_hidden_bytes": self.dispatch_hidden_bytes,
            "combine_hidden_bytes": self.combine_hidden_bytes,
            "input_transport": "FP8 E4M3 with packed UE8M0 block-128 scales",
            "expert_alignment": self.expert_alignment,
        }

    def dispatch_and_scatter(
        self,
        x_fp8: torch.Tensor,
        x_scale: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> OfficialDeepEPDispatch:
        if x_fp8.dtype != torch.float8_e4m3fn:
            raise ValueError(f"expected FP8 input, got {x_fp8.dtype}")
        if x_scale.dtype != torch.int32:
            raise ValueError(f"expected packed UE8M0 int32 scales, got {x_scale.dtype}")
        topk_ids = topk_ids.to(dtype=torch.int64).contiguous()
        topk_weights = topk_weights.to(dtype=torch.float32).contiguous()
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            _layout_event,
        ) = self.buffer.get_dispatch_layout(topk_ids, self.num_experts)
        (
            recv_pair,
            recv_topk_ids,
            recv_topk_weights,
            recv_counts,
            handle,
            _dispatch_event,
        ) = self.buffer.dispatch(
            (x_fp8.contiguous(), x_scale),
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            topk_idx=topk_ids,
            topk_weights=topk_weights,
            expert_alignment=self.expert_alignment,
            config=self.dispatch_config,
        )
        if not isinstance(recv_pair, tuple) or len(recv_pair) != 2:
            raise RuntimeError("official DeepEP did not preserve the FP8 activation tuple")
        recv_x, recv_x_scale = recv_pair
        if recv_topk_ids is None or recv_topk_weights is None:
            raise RuntimeError("official DeepEP dispatch did not return routing metadata")
        recv_counts = [int(item) for item in recv_counts]
        local_experts = self.num_experts // self.world_size
        if len(recv_counts) != local_experts:
            raise RuntimeError(
                f"DeepEP returned {len(recv_counts)} expert counts, expected {local_experts}"
            )
        if any(item < 0 or item % self.expert_alignment for item in recv_counts):
            raise RuntimeError(
                "DeepEP receive counts do not satisfy the requested expert alignment: "
                f"{recv_counts}"
            )

        counts_gpu = torch.tensor(
            recv_counts,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        ).cuda(non_blocking=True)
        all_tokens = sum(recv_counts)
        packed_scale_groups = self.hidden_size // 512
        expert_input = torch.empty(
            (all_tokens, self.hidden_size),
            dtype=recv_x.dtype,
            device=recv_x.device,
        )
        expert_input_scale = torch.zeros(
            (packed_scale_groups, all_tokens),
            dtype=recv_x_scale.dtype,
            device=recv_x.device,
        ).transpose(0, 1)
        m_indices = torch.empty(all_tokens, dtype=torch.int32, device=recv_x.device)
        expert_start_loc = torch.empty_like(counts_gpu)
        output_index = torch.empty_like(recv_topk_ids)
        self._ep_scatter(
            recv_x,
            recv_x_scale,
            recv_topk_ids,
            counts_gpu,
            expert_start_loc,
            expert_input,
            expert_input_scale,
            m_indices,
            output_index,
            scale_ue8m0=True,
        )
        psum = torch.cumsum(counts_gpu, dim=0, dtype=torch.int32)
        return OfficialDeepEPDispatch(
            hidden_states=expert_input,
            hidden_states_scale=expert_input_scale,
            topk_ids=recv_topk_ids,
            topk_weights=recv_topk_weights,
            num_recv_tokens_per_expert=recv_counts,
            psum_tokens_per_expert=psum,
            m_indices=m_indices,
            output_index=output_index,
            handle=handle,
            recv_shape=(int(recv_x.shape[0]), int(recv_x.shape[1])),
        )

    def gather_and_combine(
        self,
        expert_output: torch.Tensor,
        dispatch: OfficialDeepEPDispatch,
        *,
        topk_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        rank_output = torch.empty(
            dispatch.recv_shape,
            dtype=expert_output.dtype,
            device=expert_output.device,
        )
        self._ep_gather(
            expert_output,
            dispatch.topk_ids,
            dispatch.topk_weights if topk_weights is None else topk_weights,
            dispatch.output_index,
            rank_output,
        )
        combined, _combined_topk_weights, _combine_event = self.buffer.combine(
            rank_output,
            dispatch.handle,
            config=self.combine_config,
        )
        return combined

    def destroy(self) -> None:
        if self._destroyed:
            return
        self.buffer.destroy()
        self._destroyed = True


__all__ = ["OfficialDeepEPDispatch", "OfficialDeepEPReference"]
