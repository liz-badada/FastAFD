"""Python wrapper for the colocated DeepGEMM MegaMoE kernel.

Unlike the split M2N kernel, every rank is both a token source and an expert
owner.  The wrapper intentionally exposes the same steady-state boundary used
by SGLang: callers prepare/copy the routed inputs, then one MegaMoE kernel
overlaps expert-parallel dispatch, FP8xFP4 expert compute, and combine.
"""

from __future__ import annotations

import types
from typing import Optional, Tuple

import torch
import torch.distributed as dist

from . import deepgemm as _dg


def _ext():
    return _dg._load_extension()


def align(value: int, alignment: int) -> int:
    return (int(value) + int(alignment) - 1) // int(alignment) * int(alignment)


class MegaMoESymmBuffer:
    """Cached symmetric buffer for colocated MegaMoE."""

    def __init__(
        self,
        group: dist.ProcessGroup,
        *,
        num_experts: int,
        num_max_tokens_per_rank: int,
        num_topk: int,
        hidden: int,
        intermediate_hidden: int,
    ) -> None:
        import torch.distributed._symmetric_memory as symm_mem

        self.group = group
        self.num_experts = int(num_experts)
        self.num_topk = int(num_topk)
        self.hidden = int(hidden)
        self.intermediate_hidden = int(intermediate_hidden)
        self.num_max_tokens_per_rank = align(
            num_max_tokens_per_rank,
            int(_ext().get_token_alignment_for_mega_moe()),
        )
        num_bytes, slice_input_buffers = _ext().get_symm_buffer_size_for_mega_moe(
            group.size(),
            self.num_experts,
            self.num_max_tokens_per_rank,
            self.num_topk,
            self.hidden,
            self.intermediate_hidden,
            True,
            "swiglu",
        )
        allocator = torch if group.size() == 1 else symm_mem
        self.buffer = allocator.empty(int(num_bytes), dtype=torch.int8, device="cuda")
        self.handle = (
            types.SimpleNamespace(buffer_ptrs=[self.buffer.data_ptr()])
            if group.size() == 1
            else symm_mem.rendezvous(self.buffer, group=group)
        )
        self.buffer.zero_()
        dist.barrier(group=group)
        torch.cuda.synchronize()
        (
            self.x,
            self.x_sf,
            self.topk_idx,
            self.topk_weights,
            self.l1_acts,
            self.l1_acts_sf,
            self.l2_acts,
            self.l2_acts_sf,
        ) = slice_input_buffers(self.buffer)

    @property
    def buffer_ptrs(self) -> list[int]:
        return list(self.handle.buffer_ptrs)

    def destroy(self) -> None:
        self.handle = None
        self.buffer = None
        self.group = None
        self.x = None
        self.x_sf = None
        self.topk_idx = None
        self.topk_weights = None
        self.l1_acts = None
        self.l1_acts_sf = None
        self.l2_acts = None
        self.l2_acts_sf = None


def fp8_fp4_mega_moe(
    y: torch.Tensor,
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
    sym_buffer: MegaMoESymmBuffer,
    *,
    cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
    activation_clamp: Optional[float] = None,
    activation_alpha: float = 1.0,
    activation_up_bias: float = 0.0,
    fast_math: bool = True,
) -> None:
    """Launch colocated FP8-activation/FP4-weight MegaMoE."""

    _ext().fp8_fp4_mega_moe(
        y,
        l1_weights,
        l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.buffer_ptrs,
        sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts,
        sym_buffer.num_topk,
        (1, 1, 32),
        "swiglu",
        activation_clamp,
        float(activation_alpha),
        float(activation_up_bias),
        bool(fast_math),
    )


__all__ = ["MegaMoESymmBuffer", "fp8_fp4_mega_moe"]
