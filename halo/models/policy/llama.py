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
from halo.models.backbones.encoders import AttentionPool, MultiKVAttentionPool, NoPoolConcat
from halo.models.policy.transformer_shared import *
from halo.models.policy.retriever_topk import TopKTransformerBlock
from halo.models.policy.block_attn_variants import BlockAttention, BlockAttentionFlex, LocalAttention, LocalAttentionFlex, StridedAttention, StridedAttentionFlex, flex_attention_wrapper, flex_local_attention_wrapper, flex_strided_attention_wrapper, create_block_wise_causal_mask_optimized
import torch._dynamo
# Disable DDP graph-bucketing optimizer (incompatible with higher-order ops like flex_attention)
torch._dynamo.config.optimize_ddp = False


class TransformerBlock(nn.Module):
    def __init__(
            self,
            layer_id: int,
            args: ModelArgs,
            block_attention: bool = False,
            local_attention: bool = False,
            strided_attention: bool = False,
            gated_attention: bool = False,
        ):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self._use_block_attention = (block_attention and True) if not args.is_train else (block_attention and False)
        self._use_flex_block_attention = (block_attention and True) if args.is_train else (block_attention and False)
        self._use_local_attention = (local_attention and True) if not args.is_train else (local_attention and False)
        self._use_local_flex_attention = (local_attention and True) if args.is_train else (local_attention and False)
        self._use_strided_attention = (strided_attention and True) if not args.is_train else (strided_attention and False)
        self._use_flex_strided_attention = (strided_attention and True) if args.is_train else (strided_attention and False)
        self._use_gated_attention = gated_attention
        if self._use_flex_block_attention:
            self.attention = BlockAttentionFlex(args)
        elif self._use_block_attention:
            self.attention = BlockAttention(args)
        elif self._use_local_flex_attention:
            self.attention = LocalAttentionFlex(args)
        elif self._use_local_attention:
            self.attention = LocalAttention(args)
        elif self._use_flex_strided_attention:
            self.attention = StridedAttentionFlex(args)
        elif self._use_strided_attention:
            self.attention = StridedAttention(args)
        elif self._use_gated_attention:
            self.attention = GatedAttention(args)
        else:
            self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4*args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            args=args
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            freqs_cis: Tuple[torch.Tensor, torch.Tensor],
            mask: Optional[torch.Tensor],
            enable_checkpointing: bool = False,
        ):
        # Define the attention forward function
        def attention_forward(x):
            return self.attention.forward(x, start_pos, freqs_cis, mask)

        # Apply checkpointing if enabled, otherwise use standard forward pass
        h = checkpoint(attention_forward, self.attention_norm(x)) if enable_checkpointing else attention_forward(self.attention_norm(x))
        h = x + h
        out = h + self.feed_forward.forward(self.ffn_norm(h))

        return out, None

    def _get_cached_memory(self):
        return self.attention._get_cached_memory()

    def use_cache(self, use_cache: bool):
        self.attention.use_cache(use_cache)
        return

    def get_logging_values(self):
        return {
            "attn": self.attention.get_logging_values(),
            "ffw": self.feed_forward.get_logging_values()
        }

    @property
    def use_block_attention(self):
        return self._use_block_attention

    @property
    def use_flex_block_attention(self):
        return self._use_flex_block_attention

    @property
    def use_local_attention(self):
        return self._use_local_attention

    @property
    def use_local_flex_attention(self):
        return self._use_local_flex_attention

    @property
    def use_strided_attention(self):
        return self._use_strided_attention

    @property
    def use_flex_strided_attention(self):
        return self._use_flex_strided_attention

    @property
    def use_gated_attention(self):
        return self._use_gated_attention

class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self._params = params
        self.n_layers = params.n_layers

        self.layers = torch.nn.ModuleList()
        self.is_causal = params.is_causal
        for layer_id in range(params.n_layers):
            use_block_attn = layer_id in params.block_attn_layer_ind
            use_ret_topk_attn = layer_id in params.ret_topk_attn_idx
            use_local_attn = layer_id in params.local_attn_layer_ind
            use_strided_attn = layer_id in params.strided_attn_layer_ind
            use_gated_attn = layer_id in params.gated_attn_idx
            if use_ret_topk_attn:
                topk_transformer_block = TopKTransformerBlock(args=params, layer_id=layer_id)
                self.layers.append(topk_transformer_block)
            else:
                self.layers.append(TransformerBlock(layer_id, params, block_attention=use_block_attn, local_attention=use_local_attn, strided_attention=use_strided_attn, gated_attention=use_gated_attn))

        # block attention needs extra preprocessing
        if len(params.block_attn_layer_ind) > 0:
            total_seq_len = params.max_seq_len + params.lang_offset_in_max_seq
            block_len = params.block_len
            print(f"generating block mask for {total_seq_len=} with {block_len=}")
            use_flex_attn = params.is_train
            if use_flex_attn:
                self.compiled_flex_attn = torch.compile(flex_attention_wrapper, mode="reduce-overhead", dynamic=False)
                for layer in self.layers:
                    if isinstance(layer.attention, BlockAttentionFlex):
                        layer.attention._set_flex_attn(self.compiled_flex_attn)
            else:
                pattern_mask = BlockAttention._generate_pattern(block_len, total_seq_len, start_offset=params.block_pattern_start_offset)
                self.block_mask = create_block_wise_causal_mask_optimized(total_seq_len, pattern_mask, device='cuda')
        if len(params.local_attn_layer_ind) > 0:
            total_seq_len = params.max_seq_len + params.lang_offset_in_max_seq
            block_len = params.block_len
            print(f"generating local mask for {total_seq_len=} with {block_len=}")
            use_flex_attn = params.is_train
            if use_flex_attn:
                self.compiled_local_flex_attn = torch.compile(flex_local_attention_wrapper, mode="reduce-overhead", dynamic=False)
                for layer in self.layers:
                    if isinstance(layer.attention, LocalAttentionFlex):
                        layer.attention._set_flex_attn(self.compiled_local_flex_attn)
            else:
                self.local_mask = LocalAttention._generate_mask(block_len, total_seq_len, start_offset=params.block_pattern_start_offset, device='cuda')
        if len(params.strided_attn_layer_ind) > 0:
            total_seq_len = params.max_seq_len + params.lang_offset_in_max_seq
            stride_block_len = params.strided_block_len
            print(f"generating strided mask for {total_seq_len=} with {stride_block_len=}")
            use_flex_attn = params.is_train
            if use_flex_attn:
                self.compiled_strided_flex_attn = torch.compile(flex_strided_attention_wrapper, mode="reduce-overhead", dynamic=False)
                for layer in self.layers:
                    if isinstance(layer.attention, StridedAttentionFlex):
                        layer.attention._set_flex_attn(self.compiled_strided_flex_attn)
            else:
                self.strided_mask = StridedAttention._generate_mask(stride_block_len, total_seq_len, start_offset=0, device='cuda')

        assert len(self.layers) == params.n_layers, "Number of layers should be equal to the sum of self-attention"
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        # only needed for text generation: how do we we make this smaller?
        if params.vocab_size > 0:
            self.tok_embeddings = nn.Embedding(
                params.vocab_size, params.dim
            )
        if params.text_mode:
            self.output = nn.Linear(
                params.dim, params.vocab_size, bias=False
            )
        self.freqs_cos, self.freqs_sin = self.compute_freqs_cis(self.params.max_seq_len + self.params.lang_offset_in_max_seq)

    def get_token_embeds(self, input_ids: torch.Tensor):
        return self.tok_embeddings(input_ids)

    def compute_freqs_cis(self, seq_len: int):
        freqs_cos, freqs_sin = precompute_freqs_sin_cos(self.params.dim // self.params.n_heads, seq_len, theta=self.params.rope_theta)
        return freqs_cos, freqs_sin

    def forward(
            self,
            seq : torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            start_pos: int = 0,
        ):
        """
        seq: B, 2T-1, C
        """
        _, seqlen, _ = seq.shape
        self.freqs_cos = self.freqs_cos.to(seq.device)
        self.freqs_sin = self.freqs_sin.to(seq.device)
        freqs_cis = self.freqs_cos[start_pos:start_pos + seqlen], self.freqs_sin[start_pos:start_pos + seqlen]

        for layer in self.layers:
            mask_to_pass = mask
            if layer.use_local_attention:
                mask_to_pass = self.local_mask
            elif layer.use_block_attention:
                mask_to_pass = self.block_mask
            out = layer(
                x=seq,
                start_pos=start_pos,
                freqs_cis=freqs_cis,
                mask=mask_to_pass,
                enable_checkpointing=(self.training and self.params.enable_gradient_checkpointing),
            )
            seq = out[0] if isinstance(out, (tuple, list)) else out

        seq = self.norm(seq)
        return seq, None

    def use_cache(self, use_cache: bool):
        for layer in self.layers:
            layer.use_cache(use_cache)
        return

    @torch.inference_mode()
    def forward_text(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cos = self.freqs_cos.to(h.device)
        self.freqs_sin = self.freqs_sin.to(h.device)
        freqs_cis = self.freqs_cos[start_pos: start_pos + seqlen], self.freqs_sin[start_pos: start_pos + seqlen]
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask=None)
        h = self.norm(h)
        output = self.output(h).float()
        return output

    def get_logging_values(self):
        # for each layer, get the gating function scale
        logging_values = {}
        for layer_id, layer in enumerate(self.layers):
            logging_values[f"layer_{layer_id}"] = layer.get_logging_values()
        return logging_values

    def rename_state_dict_keys(self, state_dict, phase):
        return state_dict

    def keep_first_n_layers(self, n: int):
        self.layers = self.layers[:n]

    @property
    def vocab_size(self) -> int:
        return self.params.vocab_size

    @property
    def params(self):
        return self._params
