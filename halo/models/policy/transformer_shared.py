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

from halo.models.backbones.gating_functions import *
from halo.models.backbones.encoders import AttentionPool, MultiKVAttentionPool, NoPoolConcat

@dataclass
class ModelArgs:
    dim: int = 512
    # this is only the number of self-attention layers.
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    text_mode: bool = False

    # allows to gate the full attention layers
    gate_full_attn_layers: bool = False
    # downsample_full_attn_tokens: bool = False

    max_batch_size: int = 32
    max_seq_len: int = 2048
    enable_gradient_checkpointing: bool = False

    w_bias: bool = False # use bias tuning

    is_causal: bool = True
    cache: bool = False
    is_train: bool = True
    latent_len: int = 1

    ### cross-attention related parameters
    kv_dim: Optional[int] = None

    ##################### version
    lang_offset_in_max_seq: int = 0
    # memory related parameters
    mem_len: int = -1 # should be same as the model_args.max_seq_len
    mem_store_pre_rope: bool = True
    mem_chunk_len: int = -1
    mem_alpha: float = 0.9
    mem_threshold: float = 0.95

    # block attention related parameters
    block_len: int = -1
    block_pattern_start_offset: int = 0

    # strided attention related parameters
    strided_block_len: int = -1

    # layer indices for each attention type
    mem_attn_layer_ind: list[int] = field(default_factory=list)
    block_attn_layer_ind: list[int] = field(default_factory=list)
    full_attn_layer_ind: list[int] = field(default_factory=list)
    ret_topk_attn_idx: list[int] = field(default_factory=list)
    local_attn_layer_ind: list[int] = field(default_factory=list)
    strided_attn_layer_ind: list[int] = field(default_factory=list)
    gated_attn_idx: list[int] = field(default_factory=list)
    tokme_attn_layer_ind: list[int] = field(default_factory=list)

    # topk attention related parameters
    # the number of topk values to retrieve
    ret_topk: int = 8
    # the chunk length to retrieve the topk values
    ret_chunk_len: int = 8
    ret_tau_init: float = 1.0
    ret_straight_through: bool = False
    ret_recursions: int = 1
    ret_multikv: bool = False
    ret_add_time_aware: bool = False
    # If True and ret_add_time_aware, sinusoidal arg is (bank_index - query_timestep); else absolute bank index.
    ret_relative_time: bool = False
    ret_bank_causal: bool = False

    # compute dtype
    compute_dtype: torch.dtype = torch.float32

    def __post_init__(self):
        # n_layers = len(self.mem_attn_layer_ind) + len(self.block_attn_layer_ind) + len(self.full_attn_layer_ind)
        # if an index is not present in mem_attn_layer_ind or block_attn_layer_ind, then it is a full attention layer
        other_layer_ind = self.mem_attn_layer_ind + self.block_attn_layer_ind + self.ret_topk_attn_idx + self.local_attn_layer_ind + self.strided_attn_layer_ind + self.gated_attn_idx + self.tokme_attn_layer_ind
        full_attn_layer_ind = [i for i in range(self.n_layers) if i not in other_layer_ind]
        assert self.mem_len == -1, "mem_len should be -1 for now"

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_freqs_sin_cos(dim: int, end: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute the sinusoidal frequencies for rotary embeddings.

    Args:
        dim (int): The head dimension.
        end (int): The sequence length.
        theta (float): The base of the exponential (default 10000.0).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Cosine and sine tensors of shape (end, dim // 2).
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, device=freqs.device)
    angles = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(angles)
    freqs_sin = torch.sin(angles)
    return freqs_cos, freqs_sin

def apply_rotary_emb(
    xq: Optional[torch.Tensor] = None,
    xk: Optional[torch.Tensor] = None,
    freqs_cis: Optional[torch.Tensor] = None,
    k_freqs_cis: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to query and key tensors using complex sinusoidal frequencies.

    Args:
        xq (torch.Tensor): Query tensor of shape (batch, seq_len, num_heads, head_dim).
        xk (torch.Tensor): Key tensor of shape (batch, seq_len, num_heads, head_dim).
        freqs_cis (torch.Tensor): Complex sinusoidal frequencies of shape:
            (A) (seq_len, head_dim // 2) OR
            (B) (batch, seq_len, head_dim // 2) OR
            (C) (batch, seq_len, num_heads, head_dim // 2)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Transformed query and key tensors.
    """
    # Derive freqs_cos and freqs_sin from freqs_cis
    # freqs_cos = freqs_cis.real
    # freqs_sin = freqs_cis.imag
    xq_out, xk_out = None, None
    if xq is not None:
        freqs_cos = freqs_cis[0]
        freqs_sin = freqs_cis[1]

        # Split xq into real and imaginary parts
        xq_r, xq_i = xq[..., ::2], xq[..., 1::2]

        # Expand freqs_cos and freqs_sin to match xq_r and xq_i
        if freqs_cos.ndim == 2:
            freqs_cos = freqs_cos.unsqueeze(1).unsqueeze(0)
            freqs_sin = freqs_sin.unsqueeze(1).unsqueeze(0)
        elif freqs_cos.ndim == 3:
            freqs_cos = freqs_cos.unsqueeze(2)
            freqs_sin = freqs_sin.unsqueeze(2)

        # Apply rotary embeddings
        xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
        xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos

        # Combine real and imaginary parts into the output tensors
        xq_out = torch.stack((xq_out_r, xq_out_i), dim=-1).flatten(-2)
        xq_out = xq_out.type_as(xq)

    if xk is not None:
        k_freqs_cos = k_freqs_cis[0]
        k_freqs_sin = k_freqs_cis[1]

        # Split xk into real and imaginary parts
        xk_r, xk_i = xk[..., ::2], xk[..., 1::2]

        # Expand k_freqs_cos and k_freqs_sin to match xk_r and xk_i
        if k_freqs_cos.ndim == 2:
            k_freqs_cos = k_freqs_cos.unsqueeze(1).unsqueeze(0)
            k_freqs_sin = k_freqs_sin.unsqueeze(1).unsqueeze(0)
        elif k_freqs_cos.ndim == 3:
            k_freqs_cos = k_freqs_cos.unsqueeze(2)
            k_freqs_sin = k_freqs_sin.unsqueeze(2)

        # Apply rotary embeddings
        xk_out_r = xk_r * k_freqs_cos - xk_i * k_freqs_sin
        xk_out_i = xk_r * k_freqs_sin + xk_i * k_freqs_cos

        # Combine real and imaginary parts into the output tensors
        xk_out = torch.stack((xk_out_r, xk_out_i), dim=-1).flatten(-2)
        xk_out = xk_out.type_as(xk)

    return xq_out, xk_out


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.fused_attn = use_fused_attn()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = 1
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.lang_offset_in_max_seq = args.lang_offset_in_max_seq
        # use compressor block is true then we need to store tokens passed only after the compressor block
        self.true_cache_len = args.max_seq_len + self.lang_offset_in_max_seq

        self._create_wkv_layers(args)
        if args.w_bias:
            nn.init.constant_(self.wq.bias.data, 0)
            nn.init.constant_(self.wo.bias.data, 0)

        self.is_causal = args.is_causal # this overwirtes all the mask passed. use carefully. currently only used in decoder
        self.cache = args.cache

        self.cache_k = None
        self.cache_v = None

    def _create_wkv_layers(self, args):
        self.wq = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=args.w_bias
        )
        self.wk = nn.Linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False
        )
        self.wv = nn.Linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False
        )
        self.wo = nn.Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=args.w_bias
        )
        return

    def _create_cache(self):
        if self.cache:
            self.cache_k = torch.zeros(
                (self.args.max_batch_size, self.true_cache_len, self.n_local_kv_heads, self.head_dim)
            )
            self.cache_v = torch.zeros(
                (self.args.max_batch_size, self.true_cache_len, self.n_local_kv_heads, self.head_dim)
            )
        else:
            self.cache_k = None
            self.cache_v = None
        return

    def train(self, mode: bool = True):
        return super().train(mode)

    def reset_cache(self):
        """Zero out KV cache buffers. Call when starting a new sequence (start_pos == 0)."""
        if self.cache_k is not None:
            self.cache_k.zero_()
            self.cache_v.zero_()

    def _handle_kv_cache(self, xk, xv, start_pos, seqlen, bsz, device):
        if (not self.training) and (self.cache):
            self.cache_k = self.cache_k.to(device)
            self.cache_v = self.cache_v.to(device)

            if start_pos == 0:
                self.reset_cache()

            if start_pos + seqlen > self.cache_k.shape[1]:
                # we linearly increase the size of the kv cache by self.args.max_seq_len + self.lang_offset_in_max_seq
                # we don't double it because it might OOM
                print(f"Updating kv cache from length {self.cache_k.shape[1]} to length {self.cache_k.shape[1] * 2}")

                self.cache_k = self.cache_k.repeat(1, 2, 1, 1)
                self.cache_v = self.cache_v.repeat(1, 2, 1, 1)

            self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk.clone().detach()
            self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv.clone().detach()

            keys = self.cache_k[:bsz, : start_pos + seqlen]
            values = self.cache_v[:bsz, : start_pos + seqlen]
        else:
            assert start_pos==0, "start_pos should be 0 for never cache attention"
            keys = xk
            values = xv
        return keys, values

    def _prepare_qkv(self, x: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int = 0):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis, k_freqs_cis=freqs_cis)
        keys, values = self._handle_kv_cache(xk, xv, start_pos, seqlen, bsz, xq.device)

        return xq, keys, values

    def _get_attn_mask_causal_bookkeep(self, mask: Optional[torch.Tensor] = None, *args, **kwargs):
        # if mask is not None, then is_causal is False
        is_causal = False if mask is not None else self.is_causal # mask overrides is_causal unless self.is_causal is set to True
        if self.is_causal:
            assert mask is None, "mask should be None if is_causal is True"
        return mask, is_causal

    def _pad_queries(self, xq: torch.Tensor, pad_len: int, bsz: int):
        # we only pad the queries so left padding is appropriate
        if self.training: assert pad_len == 0, "pad_len should be 0 in training"
        if not self.cache: assert pad_len == 0, "pad_len should be 0 if we are not caching things"
        xq = torch.cat([torch.zeros((bsz, pad_len, self.n_local_heads, self.head_dim), device=xq.device, dtype=xq.dtype), xq], dim=1)
        return xq

    def _unpad_queries(self, xq: torch.Tensor, pad_len: int, bsz: int):
        xq = xq[:, :, pad_len:, :]
        return xq

    def _apply_attn(self, xq: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask: Optional[torch.Tensor] = None, start_pos: int = 0, seqlen: int = 0, bsz: int = 0):
        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        # Ensure mask is on the same device as input tensors
        if mask is not None and mask.device != xq.device:
            mask = mask.to(xq.device)

        # PyTorch SDPA is_causal=True uses upper-left masking: query i attends to keys 0..i.
        # For cached incremental eval (start_pos > 0), we need query i (at absolute position
        # start_pos+i) to attend to keys 0..start_pos+i, so we build an explicit mask instead.
        if self.is_causal and start_pos > 0:
            k_len = keys.shape[2]
            attn_mask = torch.full((seqlen, k_len), float('-inf'), device=xq.device, dtype=xq.dtype)
            # triu(diagonal=d) sets 1 where col >= row+d; here we want -inf where j > start_pos+i
            attn_mask = torch.triu(attn_mask, diagonal=start_pos + 1)
            is_causal = False
        else:
            attn_mask, is_causal = self._get_attn_mask_causal_bookkeep(mask, query_len=xq.shape[2], key_len=keys.shape[2], device=xq.device)

        output = F.scaled_dot_product_attention(
            xq, keys, values,
            dropout_p=0., attn_mask=attn_mask, is_causal=is_causal,
        )

        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)
        return output

    @property
    def _attn_type(self):
        return [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.shape
        xq, keys, values = self._prepare_qkv(x, freqs_cis, start_pos)
        with sdpa_kernel(self._attn_type):
            output = self._apply_attn(xq, keys, values, mask, start_pos, seqlen, bsz)
        output = self.wo(output)
        return output

    def _get_cached_memory(self):
        # assume bfloat16 and cache_k and cache_v shapes as given as measured
        # the return value is in bytes
        if self.cache_k is not None:
            return self.cache_k.numel() * 2 + self.cache_v.numel() * 2
        else:
            return 0

    def get_logging_values(self):
        return {}

    def use_cache(self, use_cache: bool):
        self.cache = use_cache
        self._create_cache() # create the cache if use_cache is True
        return

class GatedAttention(Attention):
    """
    Attention with sigmoid gating applied feature-wise before wo projection.

    The gating mechanism uses an MLP to predict gate values from the input,
    applies sigmoid activation, and performs element-wise multiplication
    with the attention output before the final wo projection.
    """
    def __init__(self, args: ModelArgs):
        super().__init__(args)
        # MLP for gating - takes input and predicts features of same dimension as attention output (before wo)
        # Attention output dimension before wo is args.n_heads * head_dim
        attn_output_dim = args.n_heads * (args.dim // args.n_heads)
        self.gate_mlp = nn.Linear(args.dim, attn_output_dim, bias=args.w_bias)
        if args.w_bias:
            nn.init.constant_(self.gate_mlp.bias.data, 0)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.shape
        xq, keys, values = self._prepare_qkv(x, freqs_cis, start_pos)
        with sdpa_kernel(self._attn_type):
            output = self._apply_attn(xq, keys, values, mask, start_pos, seqlen, bsz)

        # Compute gate from input and apply sigmoid
        gate = torch.sigmoid(self.gate_mlp(x))

        # Apply feature-wise gating (element-wise multiplication) before wo projection
        output = output * gate

        output = self.wo(output)
        return output

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        args: ModelArgs
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(
            dim, hidden_dim, bias=args.w_bias
        )
        self.w2 = nn.Linear(
            hidden_dim, dim, bias=args.w_bias
        )
        self.w3 = nn.Linear(
            dim, hidden_dim, bias=args.w_bias
        )
        if args.w_bias:
            nn.init.constant_(self.w1.bias.data, 0)
            nn.init.constant_(self.w2.bias.data, 0)
            nn.init.constant_(self.w3.bias.data, 0)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def get_logging_values(self):
        return {}
