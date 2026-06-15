import re
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Literal
from termcolor import colored
from functools import partial
from einops import rearrange
from statistics import median, mean
import torch
from torch import nn
import torch.nn.functional as F
from timm.layers import use_fused_attn
from torch.utils.checkpoint import checkpoint
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from halo.models.backbones.gating_functions import *
from halo.models.policy.transformer_shared import *

def flex_causal_mask(b, h, q_idx, kv_idx, block_len, block_pattern_start_offset: int = 0):
    """FlexAttention causal mask function."""
    q_block = (q_idx - block_pattern_start_offset) // block_len
    kv_block = (kv_idx - block_pattern_start_offset) // block_len
    same_block = q_block == kv_block
    causal_constraint = q_idx >= kv_idx
    return same_block & causal_constraint

# FlexAttention with compiled function
def flex_attention_wrapper(query, key, value, head_dim, seq_len, block_len, device, block_pattern_start_offset: int = 0):
    mask_mod = partial(
        flex_causal_mask,
        block_len=block_len,
        block_pattern_start_offset=block_pattern_start_offset,
    )

    block_mask = create_block_mask(
        mask_mod=mask_mod,
        B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device
    )

    return flex_attention(
        query=query, key=key, value=value,
        block_mask=block_mask, scale=head_dim ** -0.5
    )

def flex_local_causal_mask(b, h, q_idx, kv_idx, block_len, block_pattern_start_offset: int = 0):
    q_idx = (q_idx - block_pattern_start_offset)
    kv_idx = (kv_idx - block_pattern_start_offset)
    within_distance = abs(q_idx - kv_idx) < block_len
    causal_constraint = q_idx >= kv_idx
    return within_distance & causal_constraint

# FlexAttention with compiled function
def flex_local_attention_wrapper(query, key, value, head_dim, seq_len, block_len, device, block_pattern_start_offset: int = 0):
    mask_mod = partial(
        flex_local_causal_mask,
        block_len=block_len,
        block_pattern_start_offset=block_pattern_start_offset,
    )

    block_mask = create_block_mask(
        mask_mod=mask_mod,
        B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device
    )

    return flex_attention(
        query=query, key=key, value=value,
        block_mask=block_mask, scale=head_dim ** -0.5
    )

def create_block_wise_causal_mask_optimized(seq_len, block_pattern, device='cpu'):
    """
    Create a block-wise causal mask with optimized memory usage.

    Args:
        seq_len: Length of the sequence
        block_pattern: List of integers indicating block boundaries
                              e.g., [1, 0, 1, 0, 1, 0, 0, 1, 0, 0] creates 4 blocks: [0,1] [2,3] [4,5,6] [7, 8, 9]
        device: Device to create the mask on

    Returns:
        mask: Boolean mask of shape (seq_len, seq_len)
    """
    if len(block_pattern) != seq_len:
        raise ValueError(f"Block pattern length {len(block_pattern)} must match sequence length {seq_len}")

    # Convert to tensor and compute cumulative sum
    block_pattern = torch.tensor(block_pattern, device=device, dtype=torch.bool, requires_grad=False)
    cumsum = torch.cumsum(block_pattern, dim=0)

    # Create mask directly without large intermediate tensors
    mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)

    # Process each row efficiently
    for i in range(seq_len):
        # Find all positions j where cumsum[j] == cumsum[i] (same block) and i >= j (causal)
        same_block = cumsum == cumsum[i]
        causal = torch.arange(seq_len, device=device) <= i
        mask[i] = same_block & causal

    return mask

class MaskedAttentionBase(Attention):
    """Shared base for pattern-masked attention variants (block / local / strided).

    Subclasses implement two hooks used by the common cached-eval path:
      _get_cached_kv  – cache management, returns (keys, values, mask_kwargs)
      _build_attn_mask – mask construction, returns (attn_mask, is_causal)
    """

    def __init__(self, args: ModelArgs):
        super().__init__(args)
        self.max_seq_len = args.max_seq_len
        self.lang_offset_in_max_seq = args.lang_offset_in_max_seq
        self.scale_factor = None

    # ---- shared training-path helpers ----------------------------------------

    def _get_attn_mask_causal_bookkeep(self, mask: Optional[torch.Tensor] = None,
                                        query_len: int = 0, key_len: int = 0,
                                        device: str = 'cpu'):
        assert mask is not None, f"Mask is required for {type(self).__name__}"
        assert mask.shape[0] >= query_len and mask.shape[1] >= key_len, \
            f"Mask too small: {mask.shape} vs {query_len}x{key_len}"
        return mask[:query_len, :key_len].unsqueeze(0).unsqueeze(0), False

    @property
    def _attn_type(self):
        return [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]

    # ---- cached eval path ----------------------------------------------------

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis,
                mask: Optional[torch.Tensor] = None):
        if (not self.training) and self.cache:
            return self._forward_cached(x, start_pos, freqs_cis)
        return super().forward(x, start_pos, freqs_cis, mask)

    def _forward_cached(self, x: torch.Tensor, start_pos: int, freqs_cis):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis, k_freqs_cis=freqs_cis)
        keys, values, mask_kwargs = self._get_cached_kv(xk, xv, start_pos, seqlen, bsz, xq.device)
        with sdpa_kernel(self._attn_type):
            output = self._apply_attn_cached(xq, keys, values, seqlen, bsz, **mask_kwargs)
        return self.wo(output)

    def _apply_attn_cached(self, xq, keys, values, seqlen, bsz, **mask_kwargs):
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)
        xq, keys, values = xq.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)
        attn_mask, is_causal = self._build_attn_mask(
            seqlen, keys.shape[2], xq.device, xq.dtype, **mask_kwargs
        )
        output = F.scaled_dot_product_attention(
            xq, keys, values, dropout_p=0., attn_mask=attn_mask, is_causal=is_causal
        )
        return output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

    # ---- hooks (must be overridden) ------------------------------------------

    def _get_cached_kv(self, xk, xv, start_pos, seqlen, bsz, device):
        """Return (keys, values, mask_kwargs) for the current chunk."""
        raise NotImplementedError

    def _build_attn_mask(self, seqlen, k_len, device, dtype, **mask_kwargs):
        """Return (attn_mask_or_None, is_causal_bool)."""
        raise NotImplementedError


class BlockAttention(MaskedAttentionBase):

    def __init__(self, args: ModelArgs):
        super().__init__(args)
        self.block_len = args.block_len
        self.block_pattern_start_offset = args.block_pattern_start_offset

    @staticmethod
    def _generate_pattern(block_len: int, max_len: int, start_offset: int = 0):
        '''
        create a pattern: [1, 0, 0, ..., 0, 1, 0, 0, ..., 0]
                          |--block_len--|, |--block_len--|, ...
        '''
        pattern_mask = torch.zeros(max_len, dtype=torch.bool)
        pattern_mask[start_offset%block_len::block_len] = 1
        pattern_mask[0] = 1
        return pattern_mask

    def _get_cached_kv(self, xk, xv, start_pos, seqlen, bsz, device):
        # The first partial block [0, offset) uses start_pos directly;
        # full blocks afterwards reset every block_len steps.
        offset = self.block_pattern_start_offset
        block_local_pos = start_pos if start_pos < offset else (start_pos - offset) % self.block_len
        keys, values = self._handle_kv_cache(xk, xv, block_local_pos, seqlen, bsz, device)
        return keys, values, {'block_local_pos': block_local_pos}

    def _build_attn_mask(self, seqlen, k_len, device, dtype, block_local_pos, **_):
        if block_local_pos > 0:
            mask = torch.triu(
                torch.full((seqlen, k_len), float('-inf'), device=device, dtype=dtype),
                diagonal=block_local_pos + 1,
            )
            return mask, False
        return None, True


class BlockAttentionFlex(BlockAttention):

    def __init__(self, args: ModelArgs):
        super().__init__(args)
        self._attn_fn = None

    def _set_flex_attn(self, attn_fn):
        self._attn_fn = attn_fn

    def _apply_attn(self, xq: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask: Optional[torch.Tensor] = None, start_pos: int = 0, seqlen: int = 0, bsz: int = 0):
        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)      # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        ############################# MAINTAIN DETERMINISM #########################################
        # avoids recompilation of the flex attention kernel
        pad_len = start_pos+self.max_seq_len+self.lang_offset_in_max_seq-xq.shape[1]
        xq = torch.cat([xq, torch.zeros((bsz, pad_len, self.n_local_heads, self.head_dim), device=xq.device, dtype=xq.dtype)], dim=1)
        keys = torch.cat([keys, torch.zeros((bsz, pad_len, self.n_local_heads, self.head_dim), device=keys.device, dtype=keys.dtype)], dim=1)
        values = torch.cat([values, torch.zeros((bsz, pad_len, self.n_local_heads, self.head_dim), device=values.device, dtype=values.dtype)], dim=1)
        #################################################################

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        output = self._attn_fn(xq, keys, values, head_dim=self.head_dim, seq_len=self.max_seq_len+self.lang_offset_in_max_seq, block_len=self.block_len, device=xq.device, block_pattern_start_offset=self.block_pattern_start_offset)

        ############################# MAINTAIN DETERMINISM #########################################
        if pad_len > 0:
            output = output[:, :, :-pad_len, :]
        ####################################################################

        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)
        return output

class LocalAttention(MaskedAttentionBase):

    def __init__(self, args: ModelArgs):
        super().__init__(args)
        self.block_len = args.block_len

    @staticmethod
    def _generate_mask(block_len: int, max_len: int, start_offset: int = 0, device: str = 'cuda'):
        '''
        The pattern is causal and local, i.e. only the previous block_len tokens are attended to
        - Start the pattern from the start_offset token
        '''
        pattern_mask = torch.zeros(max_len, max_len, dtype=torch.bool, device=device)
        for i in range(start_offset, max_len):
            for j in range(max(0, i - block_len + 1), i+1): # i,i should be included
                pattern_mask[i, j] = 1
        return pattern_mask

    def _get_cached_kv(self, xk, xv, start_pos, seqlen, bsz, device):
        keys, values, oldest_abs = self._handle_kv_cache_local(xk, xv, start_pos, seqlen, bsz, device)
        return keys, values, {'start_pos': start_pos, 'oldest_abs': oldest_abs}

    def _handle_kv_cache_local(self, xk, xv, start_pos, seqlen, bsz, device):
        """Ring buffer cache for local attention.

        Reads old keys BEFORE writing the current chunk, so entries within
        the sliding window but before start_pos are not overwritten prematurely.
        Returns combined (old || current-chunk) keys in chronological order.
        """
        self.cache_k = self.cache_k.to(device)
        self.cache_v = self.cache_v.to(device)
        if start_pos == 0:
            self.reset_cache()

        # 1. Read old keys BEFORE writing (up to block_len-1 previous tokens)
        n_old = min(start_pos, self.block_len - 1)
        oldest_abs = start_pos - n_old  # = max(0, start_pos - block_len + 1)
        if n_old > 0:
            slots_read = torch.arange(oldest_abs, start_pos, device=device) % self.block_len
            old_keys = self.cache_k[:bsz, slots_read]
            old_values = self.cache_v[:bsz, slots_read]
        else:
            old_keys = xk[:bsz, :0]    # empty (bsz, 0, heads, head_dim)
            old_values = xv[:bsz, :0]

        # 2. Write current chunk into ring buffer
        slots_write = torch.arange(start_pos, start_pos + seqlen, device=device) % self.block_len
        self.cache_k[:bsz, slots_write] = xk[:bsz].clone().detach().to(self.cache_k.dtype)
        self.cache_v[:bsz, slots_write] = xv[:bsz].clone().detach().to(self.cache_v.dtype)

        # 3. Combine: old (positions oldest_abs..start_pos-1) + current chunk
        # Combined key j has absolute position oldest_abs + j (contiguous)
        keys = torch.cat([old_keys, xk[:bsz]], dim=1)
        values = torch.cat([old_values, xv[:bsz]], dim=1)
        return keys, values, oldest_abs

    def _build_attn_mask(self, seqlen, k_len, device, dtype, start_pos, oldest_abs, **_):
        # query l (abs pos start_pos+l) attends to cache entry j (abs pos oldest_abs+j)
        # if causal (k <= q) and within window (q - k < block_len)
        l_idx = torch.arange(seqlen, device=device, dtype=torch.long).unsqueeze(1)
        j_idx = torch.arange(k_len, device=device, dtype=torch.long).unsqueeze(0)
        q_abs = start_pos + l_idx
        k_abs = oldest_abs + j_idx
        valid = (k_abs <= q_abs) & ((q_abs - k_abs) < self.block_len)
        attn_mask = torch.where(
            valid,
            torch.zeros(1, device=device, dtype=dtype),
            torch.full((1,), float('-inf'), device=device, dtype=dtype),
        )
        return attn_mask, False


class LocalAttentionFlex(LocalAttention):
    def __init__(self, args: ModelArgs):
        super().__init__(args)
        self._attn_fn = None
        self.block_pattern_start_offset = args.block_pattern_start_offset

    def _set_flex_attn(self, attn_fn):
        self._attn_fn = attn_fn

    def _apply_attn(self, xq: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask: Optional[torch.Tensor] = None, start_pos: int = 0, seqlen: int = 0, bsz: int = 0):
        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)      # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        ############################# MAINTAIN DETERMINISM #########################################
        # avoids recompilation of the flex attention kernel
        pad_len = start_pos+self.max_seq_len+self.lang_offset_in_max_seq-xq.shape[1]
        xq = torch.cat([xq, torch.zeros((bsz, pad_len, self.n_local_heads, self.head_dim), device=xq.device, dtype=xq.dtype)], dim=1)
        keys = torch.cat([keys, torch.zeros((bsz, pad_len, self.n_local_heads, self.head_dim), device=keys.device, dtype=keys.dtype)], dim=1)
        values = torch.cat([values, torch.zeros((bsz, pad_len, self.n_local_heads, self.head_dim), device=values.device, dtype=values.dtype)], dim=1)
        #################################################################

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        output = self._attn_fn(xq, keys, values, head_dim=self.head_dim, seq_len=self.max_seq_len+self.lang_offset_in_max_seq, block_len=self.block_len, device=xq.device, block_pattern_start_offset=self.block_pattern_start_offset)

        ############################# MAINTAIN DETERMINISM #########################################
        if pad_len > 0:
            output = output[:, :, :-pad_len, :]
        ####################################################################

        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)
        return output

def flex_strided_causal_mask(b, h, q_idx, kv_idx, stride_len, block_pattern_start_offset: int = 0):
    """FlexAttention strided causal mask function.
    Attends to positions at stride intervals: current, current-stride, current-2*stride, etc.
    """
    q_idx_adjusted = q_idx - block_pattern_start_offset
    kv_idx_adjusted = kv_idx - block_pattern_start_offset

    # Check if kv_idx is at a stride interval from q_idx
    distance = q_idx_adjusted - kv_idx_adjusted
    is_strided = (distance % stride_len) == 0
    causal_constraint = q_idx >= kv_idx

    return is_strided & causal_constraint

def flex_strided_attention_wrapper(query, key, value, head_dim, seq_len, stride_len, device, block_pattern_start_offset: int = 0):
    """FlexAttention wrapper for strided attention."""
    mask_mod = partial(
        flex_strided_causal_mask,
        stride_len=stride_len,
        block_pattern_start_offset=block_pattern_start_offset,
    )

    block_mask = create_block_mask(
        mask_mod=mask_mod,
        B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device
    )

    return flex_attention(
        query=query, key=key, value=value,
        block_mask=block_mask, scale=head_dim ** -0.5
    )

class StridedAttention(MaskedAttentionBase):
    """Strided attention that attends to every k-th token in the past."""

    def __init__(self, args: ModelArgs):
        super().__init__(args)
        self.stride_len = args.strided_block_len

    @staticmethod
    def _generate_mask(stride_len: int, max_len: int, start_offset: int = 0, device: str = 'cuda'):
        """
        Generate strided attention mask.
        At position i, attend to positions: i, i-stride, i-2*stride, i-3*stride, etc.
        """
        pattern_mask = torch.zeros(max_len, max_len, dtype=torch.bool, device=device)
        for i in range(start_offset, max_len):
            j = i
            while j >= 0:
                pattern_mask[i, j] = 1
                j -= stride_len
        return pattern_mask

    def _get_cached_kv(self, xk, xv, start_pos, seqlen, bsz, device):
        # Standard linear cache, reset at start_pos==0
        keys, values = self._handle_kv_cache(xk, xv, start_pos, seqlen, bsz, device)
        return keys, values, {'start_pos': start_pos}

    def _build_attn_mask(self, seqlen, k_len, device, dtype, start_pos, **_):
        # query l (abs pos start_pos+l) attends to cache entry j (abs pos j)
        # if causal (j <= start_pos+l) and strided ((start_pos+l - j) % stride == 0)
        l_idx = torch.arange(seqlen, device=device, dtype=torch.long).unsqueeze(1)
        j_idx = torch.arange(k_len, device=device, dtype=torch.long).unsqueeze(0)
        q_abs = start_pos + l_idx
        valid = (j_idx <= q_abs) & ((q_abs - j_idx) % self.stride_len == 0)
        attn_mask = torch.where(
            valid,
            torch.zeros(1, device=device, dtype=dtype),
            torch.full((1,), float('-inf'), device=device, dtype=dtype),
        )
        return attn_mask, False

class StridedAttentionFlex(StridedAttention):
    """Strided attention using FlexAttention for better performance."""

    def __init__(self, args: ModelArgs):
        super().__init__(args)
        self._attn_fn = None
        self.block_pattern_start_offset = args.block_pattern_start_offset
        self.scale_factor = None

    def _set_flex_attn(self, attn_fn):
        self._attn_fn = attn_fn

    def _apply_attn(self, xq: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask: Optional[torch.Tensor] = None, start_pos: int = 0, seqlen: int = 0, bsz: int = 0):
        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)      # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        ############################# MAINTAIN DETERMINISM #########################################
        # avoids recompilation of the flex attention kernel
        pad_len = start_pos+self.max_seq_len+self.lang_offset_in_max_seq-xq.shape[1]
        xq = torch.cat([xq, torch.zeros((bsz, pad_len, self.n_local_heads, self.head_dim), device=xq.device, dtype=xq.dtype)], dim=1)
        keys = torch.cat([keys, torch.zeros((bsz, pad_len, self.n_local_heads, self.head_dim), device=keys.device, dtype=keys.dtype)], dim=1)
        values = torch.cat([values, torch.zeros((bsz, pad_len, self.n_local_heads, self.head_dim), device=values.device, dtype=values.dtype)], dim=1)
        #################################################################

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        output = self._attn_fn(xq, keys, values, head_dim=self.head_dim, seq_len=self.max_seq_len+self.lang_offset_in_max_seq, stride_len=self.stride_len, device=xq.device, block_pattern_start_offset=self.block_pattern_start_offset)

        ############################# MAINTAIN DETERMINISM #########################################
        if pad_len > 0:
            output = output[:, :, :-pad_len, :]
        ####################################################################

        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)
        return output
