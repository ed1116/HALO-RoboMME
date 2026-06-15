import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint
import math
from halo.models.policy.transformer_shared import ModelArgs
from halo.models.policy.transformer_shared import apply_rotary_emb
from halo.models.policy.transformer_shared import Attention, FeedForward, RMSNorm

class SinusoidalTimeEncoding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base

        half = dim // 2
        # div_term: (half,)
        div_term = torch.exp(
            torch.arange(half).float() * (-math.log(base) / max(half - 1, 1))
        )
        self.register_buffer("div_term", div_term)  # moves w/ device, not trained

    def forward(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        x:        (B, T, K, D)
        indices:  (B, T, K)  time indices or relative lags (e.g. bank_pos - query_pos)
        returns:  x + sinusoidal_encoding(indices)  with shape (B, T, K, D)
        """
        # (B,T,K,1) * (1,1,1,half) -> (B,T,K,half)
        angles = indices.to(dtype=x.dtype).unsqueeze(-1) * self.div_term.to(x.dtype).view(
            1, 1, 1, -1
        )
        pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B,T,K,2*half)
        return x + pe

class LongTermBank(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.model_args = args  # live ModelArgs; cache + latent_len affect store_kv
        # maintain a bank of k and v tensors with zeros using max
        self.register_buffer("cache_k", torch.zeros(args.max_batch_size, args.max_seq_len, args.dim, dtype=torch.float32))
        self.register_buffer("cache_v", torch.zeros(args.max_batch_size, args.max_seq_len, args.dim, dtype=torch.float32))
        self.register_buffer("mask", torch.zeros(args.max_batch_size, args.max_seq_len, dtype=torch.bool))
        self.max_n_valid_entries = 0
        self.register_buffer("n_valid_entries", torch.zeros(args.max_batch_size, dtype=torch.int32))
        self.ret_tau_init = args.ret_tau_init
        self.ret_add_time_aware = args.ret_add_time_aware
        self.ret_relative_time = args.ret_relative_time
        self.chunk_len = args.ret_chunk_len
        if args.ret_add_time_aware:
            self.time_encoding = SinusoidalTimeEncoding(args.dim)

    def _retrieval_time_indices(
        self, indices: torch.Tensor, query_timesteps: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Sinusoidal phase input: relative lag if ret_relative_time and query_timesteps set, else absolute bank index."""
        if query_timesteps is None or not self.ret_relative_time:
            return indices
        assert indices.ndim == 3, f"indices should be of shape (bsz, seqlen, topk), but got {indices.shape=}"
        assert query_timesteps.shape == indices.shape[:2], f"query_timesteps should be of shape (bsz, seqlen), but got {query_timesteps.shape=}"
        assert indices.max() <= query_timesteps.min(), f"indices should be less than or equal to query_timesteps: {indices.max()=} {query_timesteps.min()=}"
        return query_timesteps.unsqueeze(-1) - indices

    def use_cache(self, use_cache: bool) -> None:
        """Keep ModelArgs.cache in sync for store_kv trailing-token policy (inference KV caching)."""
        self.model_args.cache = bool(use_cache)

    def _store_seq_skip(self) -> int:
        ma = self.model_args
        if ma.cache and (not self.training):
            return int(ma.latent_len)
        return 0

    def store_kv(self, k: torch.Tensor, v: torch.Tensor):
        # k,v: (B, T, H, Hd) -> (B, T, D)
        B, T = k.shape[0], k.shape[1]
        store_seq_skip = self._store_seq_skip()
        skip = min(store_seq_skip, T) if store_seq_skip > 0 else 0
        T_store = T - skip
        if T_store <= 0:
            return

        k = k[:, :T_store].reshape(B, T_store, -1).contiguous().to(self.cache_k.dtype)
        v = v[:, :T_store].reshape(B, T_store, -1).contiguous().to(self.cache_v.dtype)
        D = k.shape[-1]
        device = k.device

        # start positions per batch: shape (B, 1)
        if self.max_n_valid_entries == 0:
            start = torch.zeros((B, 1), device=device, dtype=torch.long)
        else:
            start = self.n_valid_entries[:B].view(B, 1)  # (B,1)


        # seq indices: (B, T_store)
        seq_idx = start + torch.arange(T_store, device=device).view(1, T_store)

        # scatter expects index shape same as src (broadcast along last dim)
        idx3 = seq_idx.unsqueeze(-1).expand(B, T_store, D)  # (B,T_store,D)

        # write into caches (only first B rows)
        self.cache_k[:B].scatter_(1, idx3, k)
        self.cache_v[:B].scatter_(1, idx3, v)

        # update mask (bool scatter)
        self.mask[:B].scatter_(1, seq_idx, torch.ones_like(seq_idx, dtype=torch.bool))

        # update counters (no Python loop if T is uniform)
        if self.max_n_valid_entries == 0:
            self.n_valid_entries[:B].fill_(T_store)
            self.max_n_valid_entries = T_store
        else:
            self.n_valid_entries[:B] += T_store
            # keep as python int, but compute cheaply
            self.max_n_valid_entries = min(
                self.max_n_valid_entries + T_store,
                self.cache_k.shape[1],   # args.max_seq_len
            )

    def retrieve_from_empty_bank(self, q: torch.Tensor, topk: int):
        scores = torch.zeros((q.shape[0], q.shape[1], topk), device=q.device, dtype=q.dtype)
        indices = torch.zeros((q.shape[0], q.shape[1], topk), device=q.device, dtype=torch.long)
        retrieved_v = torch.zeros((q.shape[0], q.shape[1], topk, q.shape[2], q.shape[3]), device=q.device, dtype=q.dtype)
        return scores, indices, retrieved_v

    def retrieve_kv(
        self, q: torch.Tensor, topk: int, query_timesteps: Optional[torch.Tensor] = None
    ):
        '''
        Args:
            q: the query tensor of shape (bsz, seqlen, n_heads, head_dim)
            topk: the number of topk values to retrieve
            query_timesteps: optional (bsz, seqlen) global index of each query token; when
                ret_add_time_aware and ret_relative_time, encoding uses (bank_index - query_timestep).
        '''
        max_topk = min(topk, self.max_n_valid_entries)
        # assert q.ndim == 4, f"q should be of shape (bsz, seqlen, n_heads, head_dim), but got {q.shape}"
        bsz, seqlen, n_heads, head_dim = q.shape
        q = q.view(bsz, seqlen, -1)
        # q: b, s, d and k: b, n, d: find top k values for i,j of q
        cache_k = self.cache_k[:bsz, :self.max_n_valid_entries]
        mask = self.mask[:bsz, :self.max_n_valid_entries]

        sim = torch.einsum('bsd,bnd->bsn', q, cache_k) / (q.norm(dim=-1, keepdim=True) * cache_k.norm(dim=-1, keepdim=True).transpose(-1, -2) + 1e-8)
        sim = sim.masked_fill(~mask[:, None, :], -torch.inf)
        # non-differentiable topk
        scores, indices = sim.topk(max_topk, dim=-1) # (bsz, seqlen, topk)
        if max_topk < topk:
            scores = torch.cat([scores, torch.zeros((bsz, seqlen, topk - max_topk), device=q.device, dtype=q.dtype)], dim=-1)
            indices = torch.cat([indices, torch.zeros(bsz, seqlen, topk - max_topk, device=q.device, dtype=torch.long)], dim=-1)
        # use straight-through estimator to make the query in the computation graph
        if self.training:
            # idx = indices.detach()  # treat membership as constant
            # soft_selected_scores = sim.gather(dim=-1, index=idx)  # (b,s,k), differentiable w.r.t sim
            # scores = scores.detach() + soft_selected_scores - soft_selected_scores.detach()
            # # recalculate the scores to maintain the gradient flow
            q_vec = q.unsqueeze(-2) # (bsz, seqlen, 1, dim)
            batch = torch.arange(cache_k.size(0), device=cache_k.device)[:, None, None] # (bsz, 1, 1)
            keys = cache_k[batch, indices] # (bsz, seqlen, topk, dim)
            scores = torch.einsum('btsd,btnd->btsn', q_vec, keys) / (q_vec.norm(dim=-1, keepdim=True) * keys.norm(dim=-1, keepdim=True).transpose(-1, -2) + 1e-8)
            scores = scores.squeeze(-2)
            # assert the new_scores and scores are the same

        # retrieve the values from the cache
        cache_v = self.cache_v[:bsz, :self.max_n_valid_entries] # (bsz, n, d)
        batch = torch.arange(cache_v.size(0), device=cache_v.device)[:, None, None]  # [b,1,1]
        retrieved_v = cache_v[batch, indices] # (bsz, seqlen, topk, d)
        if self.ret_add_time_aware:
            te_idx = self._retrieval_time_indices(indices, query_timesteps)
            retrieved_v = self.time_encoding(retrieved_v, te_idx)
        retrieved_v = retrieved_v.view(bsz, seqlen, topk, n_heads, head_dim)
        # Convert retrieved values and scores to match query dtype (important for autocast compatibility)
        retrieved_v = retrieved_v.to(q.dtype)
        scores = scores.to(q.dtype)
        return scores, indices, retrieved_v

    def is_empty(self):
        return self.max_n_valid_entries == 0

    def reset_to_zero(self):
        """Reset the bank to empty state"""
        # Use in-place operations to preserve buffer registration
        self.cache_k.zero_()
        self.cache_v.zero_()
        self.mask.zero_()
        self.n_valid_entries.zero_()
        self.max_n_valid_entries = 0
        return

class LongTermBankCausal(LongTermBank):
    def __init__(self, args: ModelArgs):
        super().__init__(args)
        # create the mask of the seqlen, seqlen
        offset = -(args.ret_chunk_len - 1)
        seqlen = args.max_seq_len
        causal_mask = torch.triu(
            torch.ones(seqlen, seqlen, dtype=torch.bool), diagonal=offset
        )
        self.register_buffer("causal_mask", causal_mask)

    def retrieve_kv(
        self,
        q: torch.Tensor,
        topk: int,
        start_pos: int = 0,
        query_timesteps: Optional[torch.Tensor] = None,
    ):
        max_topk = min(topk, self.max_n_valid_entries)
        # assert q.ndim == 4, f"q should be of shape (bsz, seqlen, n_heads, head_dim), but got {q.shape}"
        bsz, seqlen, n_heads, head_dim = q.shape
        q = q.view(bsz, seqlen, -1)
        # q: b, s, d and k: b, n, d: find top k values for i,j of q
        cache_k = self.cache_k[:bsz, :self.max_n_valid_entries]
        mask = self.mask[:bsz, :self.max_n_valid_entries]
        # Slice causal mask with start_pos offset so query position s in local coords
        # corresponds to absolute position start_pos+s, retrieving from keys 0..max_n_valid_entries-1
        causal = self.causal_mask[start_pos:start_pos + seqlen, :self.max_n_valid_entries]
        mask = (~mask[:, None, :]) | causal[None, :, :]

        sim = torch.einsum('bsd,bnd->bsn', q, cache_k) / (q.norm(dim=-1, keepdim=True) * cache_k.norm(dim=-1, keepdim=True).transpose(-1, -2) + 1e-8)
        sim = sim.masked_fill(mask, -torch.inf)
        # non-differentiable topk
        scores, indices = sim.topk(max_topk, dim=-1) # (bsz, seqlen, topk)
        if max_topk < topk:
            scores = torch.cat([scores, torch.zeros((bsz, seqlen, topk - max_topk), device=q.device, dtype=q.dtype)], dim=-1)
            indices = torch.cat([indices, torch.zeros(bsz, seqlen, topk - max_topk, device=q.device, dtype=torch.long)], dim=-1)
        # change the scores to be 0 for positions that are not valid; correspondingly change the indices to be 0
        gathered_mask = torch.gather(mask, dim=-1, index=indices)
        scores.masked_fill_(gathered_mask, 0.0)
        indices.masked_fill_(gathered_mask, 0)

        # use straight-through estimator to make the query in the computation graph
        if self.training:
            # # recalculate the scores to maintain the gradient flow
            q_vec = q.unsqueeze(-2) # (bsz, seqlen, 1, dim)
            batch = torch.arange(cache_k.size(0), device=cache_k.device)[:, None, None] # (bsz, 1, 1)
            keys = cache_k[batch, indices] # (bsz, seqlen, topk, dim)
            scores = torch.einsum('btsd,btnd->btsn', q_vec, keys) / (q_vec.norm(dim=-1, keepdim=True) * keys.norm(dim=-1, keepdim=True).transpose(-1, -2) + 1e-8)
            scores = scores.squeeze(-2)

        # retrieve the values from the cache
        cache_v = self.cache_v[:bsz, :self.max_n_valid_entries] # (bsz, n, d)
        batch = torch.arange(cache_v.size(0), device=cache_v.device)[:, None, None]  # [b,1,1]
        retrieved_v = cache_v[batch, indices] # (bsz, seqlen, topk, d)
        if self.ret_add_time_aware:
            te_idx = self._retrieval_time_indices(indices, query_timesteps)
            retrieved_v = self.time_encoding(retrieved_v, te_idx)
        retrieved_v = retrieved_v.view(bsz, seqlen, topk, n_heads, head_dim)
        # Convert retrieved values and scores to match query dtype (important for autocast compatibility)
        retrieved_v = retrieved_v.to(q.dtype)
        scores = scores.to(q.dtype)
        return scores, indices, retrieved_v

class LongTermBankMultiKV(LongTermBank):
    def retrieve_kv(self, q: torch.Tensor, topk: int):
        '''
        Args:
            q: the query tensor of shape (bsz, seqlen, n_heads, head_dim)
            topk: the number of topk values to retrieve
        '''
        max_topk = min(topk, self.max_n_valid_entries)
        assert q.ndim == 4, f"q should be of shape (bsz, seqlen, n_heads, head_dim), but got {q.shape}"
        bsz, seqlen, n_heads, head_dim = q.shape
        # q = q.view(bsz, seqlen, -1)
        # q: b, s, d and k: b, n, d: find top k values for i,j of q
        cache_k = self.cache_k[:bsz, :self.max_n_valid_entries]
        mask = self.mask[:bsz, :self.max_n_valid_entries]

        # bsz, n_heads, seqlen, head_dim
        q = q.transpose(1, 2).contiguous()
        cache_k = cache_k.view(bsz, self.max_n_valid_entries, n_heads, head_dim).transpose(1, 2).contiguous()

        sim = torch.einsum('bhsd,bhnd->bhsn', q, cache_k) / (q.norm(dim=-1, keepdim=True) * cache_k.norm(dim=-1, keepdim=True).transpose(-1, -2) + 1e-8)
        # bsz, 1, 1, topk
        sim = sim.masked_fill(~mask[:, None, None, :], -torch.inf)
        # non-differentiable topk
        scores, indices = sim.topk(max_topk, dim=-1) # (bsz, heads, seqlen, topk)
        if max_topk < topk:
            scores = torch.cat([scores, torch.zeros((bsz, n_heads, seqlen, topk - max_topk), device=q.device, dtype=q.dtype)], dim=-1)
            indices = torch.cat([indices, torch.zeros(bsz, n_heads, seqlen, topk - max_topk, device=q.device, dtype=torch.long)], dim=-1)
        # use straight-through estimator to make the query in the computation graph
        if self.training:
            # recalculate the scores to maintain the gradient flow
            q_vec = q.unsqueeze(-2) # (bsz, n_heads, seqlen, 1, dim)
            batch = torch.arange(cache_k.size(0), device=cache_k.device)[:, None, None, None] # (bsz, 1, 1, 1)
            heads = torch.arange(n_heads, device=cache_k.device)[None, :, None, None] # (1, n_heads, 1, 1)
            keys = cache_k[batch, heads, indices] # (bsz, n_heads, seqlen, topk, head_dim)
            scores = torch.einsum('bhtsd,bhtnd->bhtsn', q_vec, keys) / (q_vec.norm(dim=-1, keepdim=True) * keys.norm(dim=-1, keepdim=True).transpose(-1, -2) + 1e-8)
            scores = scores.squeeze(-2) # (bsz, n_heads, seqlen, topk)
            # assert the new_scores and scores are the same

        # retrieve the values from the cache
        cache_v = self.cache_v[:bsz, :self.max_n_valid_entries] # (bsz, n, d)
        cache_v = cache_v.view(bsz, self.max_n_valid_entries, n_heads, head_dim).transpose(1, 2).contiguous()
        batch = torch.arange(cache_v.size(0), device=cache_v.device)[:, None, None, None]  # [b,1,1,1]
        heads = torch.arange(n_heads, device=cache_v.device)[None, :, None, None] # (1, n_heads, 1, 1)
        retrieved_v = cache_v[batch, heads, indices] # (bsz, n_heads, seqlen, topk, head_dim)
        retrieved_v = retrieved_v

        indices = indices.permute(0, 2, 3, 1) # (bsz, seqlen, topk, n_heads)
        scores = scores.permute(0, 2, 3, 1) # (bsz, seqlen, topk, n_heads)
        retrieved_v = retrieved_v.permute(0, 2, 3, 1, 4) # (bsz, seqlen, topk, n_heads, head_dim)
        return scores, indices, retrieved_v

    def retrieve_from_empty_bank(self, q: torch.Tensor, topk: int):
        bsz, seqlen, n_heads, head_dim = q.shape
        scores = torch.zeros((bsz, seqlen, topk, n_heads), device=q.device, dtype=q.dtype)
        indices = torch.zeros((bsz, seqlen, topk, n_heads), device=q.device, dtype=torch.long)
        retrieved_v = torch.zeros((bsz, seqlen, topk, n_heads, head_dim), device=q.device, dtype=q.dtype)
        return scores, indices, retrieved_v


class TopKAttention(Attention):
    def __init__(self, args: ModelArgs):
        super().__init__(args)
        self.ret_topk = args.ret_topk
        self.chunk_len = args.ret_chunk_len
        if args.ret_multikv:
            self.long_term_bank = LongTermBankMultiKV(args)
        elif self.is_bank_causal:
            self.long_term_bank = LongTermBankCausal(args)
        else:
            self.long_term_bank = LongTermBank(args)
        self.straight_through_estimator = args.ret_straight_through
        self.tau = nn.Parameter(torch.tensor(args.ret_tau_init, dtype=args.compute_dtype))
        # assert self.ret_topk <= self.chunk_len, f"ret_topk must be less than or equal to chunk_len: {self.ret_topk=} {self.chunk_len=}"
        self.logging_values = {}

    def use_cache(self, use_cache: bool) -> None:
        super().use_cache(use_cache)
        self.long_term_bank.use_cache(use_cache)

    def _create_wkv_layers(self, args):
        '''
            Create the wq_bank layer for the topk attention.
            Removes the wk and wv layers and replaces them with identity layers.
        '''
        super()._create_wkv_layers(args)
        # vq_bank will be trained separately using the retriever loss
        self.wq_bank = nn.Linear(
            args.dim,
            self.n_local_heads * self.head_dim,
            bias=False
        )
        # make self.wk, self.wv, self.wq = Identity
        self.wk = nn.Identity()
        self.wv = nn.Identity()
        self.wq = nn.Identity()
        return

    # set the attn type to sdpa
    @property
    def _attn_type(self):
        return [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]

    def _prepare_qkv(self, x: torch.Tensor, freqs_cis: Tuple[torch.Tensor, torch.Tensor], start_pos: int = 0, timesteps: Optional[torch.Tensor] = None):
        if timesteps is not None:
            # Ensure timesteps is on the same device as freqs_cis
            if timesteps.device != freqs_cis[0].device:
                timesteps = timesteps.to(freqs_cis[0].device)
            freqs_cis = freqs_cis[0][timesteps], freqs_cis[1][timesteps]
        # Always pass start_pos=0 to the parent: TopKAttention does not use the
        # parent KV cache (wk/wv are Identity), so the cache assertion is irrelevant.
        # RoPE for xq_bank is applied below with the correct timestep indices.
        xq, keys, values = super()._prepare_qkv(x, freqs_cis, start_pos=0)
        bsz, seqlen, _ = x.shape
        xq_bank = self.wq_bank(x)
        xq_bank = xq_bank.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xq_bank, _ = apply_rotary_emb(xq_bank, xk=None, freqs_cis=freqs_cis, k_freqs_cis=None)
        return xq_bank, xq, keys, values

    def _retrieve_values(
        self,
        xq_bank: torch.Tensor,
        topk: int,
        query_timesteps: Optional[torch.Tensor] = None,
    ):
        '''
        Args:
            topk: the number of topk values to retrieve
            query_timesteps: (bsz, seqlen) global query positions for relative retrieval-time encoding.
        '''
        # retrieve keys and values from long-term bank: make it all zeros if bank is empty
        scores, indices, retrieved_v = None, None, None
        if not self.long_term_bank.is_empty():
            # retrieve the topk values from the long-term bank
            scores, indices, retrieved_v = self.long_term_bank.retrieve_kv(
                xq_bank, topk, query_timesteps=query_timesteps
            )
        else:
            scores, indices, retrieved_v = self.long_term_bank.retrieve_from_empty_bank(xq_bank, topk)
        scores = scores / self.tau
        return scores, indices, retrieved_v

    def _requires_individual_values(self):
        return self.training and (not self.straight_through_estimator)

    def _get_output_from_retrieved_values(self, scores: torch.Tensor, values: torch.Tensor, retrieved_v: torch.Tensor):
        '''
        Args:
            scores: the scores tensor of shape (bsz, seqlen, topk)
            values: the values tensor of shape (bsz, seqlen, n_heads, head_dim)
            retrieved_v: the retrieved values tensor of shape (bsz, seqlen, topk, n_heads, head_dim)
            mode: the mode to combine the values, can be "weighted_sum" or "individual"
                - weighted_sum: use the score to weight the retrieved values
                - individual: use the retrieved values into batch size for calculating the log_py_xzk (later used for the retriever loss)
        '''

        # use score for the weighted sum of retrieved values
        log_pz = F.log_softmax(scores, dim=2) # (bsz, seqlen, topk) OR (bsz, seqlen, topk, n_heads)
        pz = log_pz.exp().unsqueeze(dim=-1) # (bsz, seqlen, topk, 1) OR (bsz, seqlen, topk, n_heads, 1)
        if pz.ndim == 4:
            pz = pz.unsqueeze(dim=-1) # (bsz, seqlen, topk, 1, 1)
        assert pz.ndim == retrieved_v.ndim, f"scores and retrieved_v should have the same number of dimensions, but got {scores.ndim=}, {retrieved_v.ndim=}"
        if self.straight_through_estimator:
            weighted_retrieved_v = (pz * retrieved_v).sum(dim=2)
        else:
            weighted_retrieved_v = (pz.detach() * retrieved_v).sum(dim=2) # (bsz, seqlen, n_heads, head_dim)
        assert values.shape == weighted_retrieved_v.shape, f"values and retrieved_v should have the same shape, but got {values.shape=}, {weighted_retrieved_v.shape=}"

        # use self.wo to combine the values
        bsz, seqlen, n_heads, head_dim = values.shape
        combined_values = self.wo(weighted_retrieved_v.view(bsz, seqlen, -1))

        retriever_info = {}
        if self._requires_individual_values():
            bsz, seqlen, topk, n_heads, head_dim = retrieved_v.shape
            retrieved_v = retrieved_v.transpose(1, 2).contiguous()
            retrieved_v = retrieved_v.view(bsz * topk, seqlen, n_heads, head_dim).view(bsz * topk, seqlen, -1)
            output_individual = self.wo(retrieved_v)
            retriever_info["output_individual"] = output_individual
            retriever_info["scores"] = scores
            retriever_info["log_pz"] = log_pz
        return combined_values, retriever_info

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: Tuple[torch.Tensor, torch.Tensor], mask: Optional[torch.Tensor] = None, topk: Optional[int] = None):
        if topk is None:
            topk = self.ret_topk
        assert mask is None, "mask should be None for topk attention"

        if start_pos == 0:
            self.long_term_bank.reset_to_zero()
            return self._forward_full(x, freqs_cis, topk)
        else:
            return self._forward_incremental(x, freqs_cis, topk, start_pos)

    def _forward_full(self, x: torch.Tensor, freqs_cis: Tuple[torch.Tensor, torch.Tensor], topk: int):
        """
        Full-sequence forward (start_pos=0, training / offline eval).

        Processes the sequence chunk-by-chunk with an internal loop so that
        chunk i can only retrieve from chunks 0..i-1 (chunk-causal ordering).
        freqs_cis must cover the full sequence (not pre-sliced).
        """
        bsz, seqlen, _ = x.shape
        # Global absolute timesteps [0, seqlen-1]; used to index into full freqs_cis.
        all_timesteps = torch.arange(0, seqlen, device=x.device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
        output = torch.zeros((bsz, seqlen, x.shape[2]), device=x.device, dtype=x.dtype).contiguous()

        retriever_info_chunks = {
            "output_individual": [],
            "scores": [],
            "log_pz": [],
            "indices": [],
        }
        all_scores = []

        for i in range(0, seqlen + self.chunk_len - 1, self.chunk_len):
            seqlen_i = min(self.chunk_len, seqlen - i)
            if seqlen_i <= 0:
                break
            chunk_x = x[:, i:i+seqlen_i, :]
            chunk_timesteps = all_timesteps[:, i:i+seqlen_i]

            xq_bank, xq, keys, values = self._prepare_qkv(chunk_x, freqs_cis, start_pos=0, timesteps=chunk_timesteps)

            scores, indices, retrieved_v = self._retrieve_values(
                xq_bank, topk, query_timesteps=chunk_timesteps
            )
            all_scores.append(scores)
            self.long_term_bank.store_kv(keys.detach(), values.detach())

            retriever_info_chunks["indices"].append(indices)
            chunk_output, chunk_retriever_info = self._get_output_from_retrieved_values(scores, values, retrieved_v)

            if self._requires_individual_values():
                assert all(k in chunk_retriever_info for k in ["output_individual", "scores", "log_pz"]), \
                    f"chunk_retriever_info should contain the keys 'output_individual', 'scores', and 'log_pz', but got {chunk_retriever_info.keys()=}"
                retriever_info_chunks["output_individual"].append(chunk_retriever_info["output_individual"])
                retriever_info_chunks["scores"].append(chunk_retriever_info["scores"])
                retriever_info_chunks["log_pz"].append(chunk_retriever_info["log_pz"])

            batch_idx = torch.arange(bsz, device=x.device).unsqueeze(1).expand(-1, seqlen_i)
            seq_idx = torch.arange(i, i+seqlen_i, device=x.device).unsqueeze(0).expand(bsz, -1)
            put_indices = (batch_idx.flatten(), seq_idx.flatten())
            chunk_output = chunk_output.to(x.dtype)
            with torch.autocast(device_type=output.device.type, enabled=False):
                output = output.index_put(put_indices, chunk_output.view(-1, chunk_output.shape[-1]))

        retriever_info = {}
        if self.training:
            if len(retriever_info_chunks["output_individual"]) > 0:
                retriever_info["scores"] = torch.cat(retriever_info_chunks["scores"], dim=1)
                retriever_info["log_pz"] = torch.cat(retriever_info_chunks["log_pz"], dim=1)
                retriever_info["output_individual"] = torch.cat(retriever_info_chunks["output_individual"], dim=1)
                retriever_info["indices"] = torch.cat(retriever_info_chunks["indices"], dim=1)
        self.logging_values["indices"] = torch.cat(retriever_info_chunks["indices"], dim=1)
        self.logging_values["topk"] = topk
        self.logging_values["all_scores"] = torch.cat(all_scores, dim=1)
        return output, retriever_info

    def _forward_incremental(self, x: torch.Tensor, freqs_cis: Tuple[torch.Tensor, torch.Tensor], topk: int, start_pos: int):
        """
        Incremental (cached) forward (start_pos > 0, eval-time KV caching).

        Skips the inner chunk loop entirely: retrieves from the already-accumulated
        bank, then stores the new tokens.  freqs_cis must be pre-sliced to the
        current chunk's position range (same convention as TopKAttentionCausal).
        Uses local timesteps [0, seqlen-1] to index into the pre-sliced freqs_cis.

        Block-causal enforcement: retrieval is restricted to bank entries from
        completed blocks only (positions [0, block_start)), matching the full-pass
        semantics where chunk i retrieves before storing, so no token can see a
        within-block sibling.  block_start = (start_pos // chunk_len) * chunk_len.
        """
        bsz, seqlen, _ = x.shape
        # Local timesteps [0, seqlen-1]: freqs_cis is pre-sliced by the caller.
        local_timesteps = torch.arange(0, seqlen, device=x.device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)

        xq_bank, xq, keys, values = self._prepare_qkv(x, freqs_cis, start_pos=0, timesteps=local_timesteps)

        # Enforce block-causal ordering: only retrieve from completed blocks.
        # Tokens in the current block that were stored by earlier incremental calls
        # must not be visible — temporarily cap the bank view to block_start.
        block_start = (start_pos // self.chunk_len) * self.chunk_len
        saved_max = self.long_term_bank.max_n_valid_entries
        self.long_term_bank.max_n_valid_entries = min(block_start, saved_max)

        query_timesteps = start_pos + local_timesteps
        scores, indices, retrieved_v = self._retrieve_values(
            xq_bank, topk, query_timesteps=query_timesteps
        )

        self.long_term_bank.max_n_valid_entries = saved_max  # restore before storing
        self.long_term_bank.store_kv(keys.detach(), values.detach())
        output, retriever_info = self._get_output_from_retrieved_values(scores, values, retrieved_v)

        self.logging_values["indices"] = indices
        self.logging_values["topk"] = topk
        self.logging_values["all_scores"] = scores
        return output, retriever_info

    def get_logging_values(self):
        # return the indices vector as (bsz * seqlen, topk)
        return {
            "ret_indices": self.logging_values["indices"][:, self.chunk_len:].reshape(-1, self.ret_topk).detach().cpu().numpy(),
            "topk": self.logging_values["topk"],
            "tau": self.tau.detach().cpu().numpy(),
            "all_scores": self.logging_values["all_scores"][:, self.chunk_len:].reshape(-1, self.ret_topk).detach().cpu().float().numpy()
        }

    @property
    def is_bank_causal(self):
        return False

class TopKAttentionCausal(TopKAttention):

    @property
    def is_bank_causal(self):
        return True

    def _retrieve_values(self, xq_bank: torch.Tensor, topk: int, start_pos: int = 0):
        bsz, seqlen = xq_bank.shape[0], xq_bank.shape[1]
        query_timesteps = start_pos + torch.arange(
            seqlen, device=xq_bank.device, dtype=torch.long
        ).unsqueeze(0).expand(bsz, -1)
        scores, indices, retrieved_v = None, None, None
        if not self.long_term_bank.is_empty():
            scores, indices, retrieved_v = self.long_term_bank.retrieve_kv(
                xq_bank, topk, start_pos=start_pos, query_timesteps=query_timesteps
            )
        else:
            scores, indices, retrieved_v = self.long_term_bank.retrieve_from_empty_bank(xq_bank, topk)
        scores = scores / self.tau
        return scores, indices, retrieved_v

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: Tuple[torch.Tensor, torch.Tensor], mask: Optional[torch.Tensor] = None, topk: Optional[int] = None):
        if topk is None:
            topk = self.ret_topk
        assert mask is None, "mask should be None for topk attention"

        # Reset bank at the start of a new sequence; preserve it for incremental eval steps
        if start_pos == 0:
            self.long_term_bank.reset_to_zero()

        bsz, seqlen, _ = x.shape
        all_timesteps = torch.arange(0, seqlen, device=x.device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)

        xq_bank, xq, keys, values = self._prepare_qkv(x, freqs_cis, start_pos=start_pos, timesteps=all_timesteps)
        # Store new tokens first, then retrieve with causal mask (position-aware)
        self.long_term_bank.store_kv(keys.detach(), values.detach())
        scores, indices, retrieved_v = self._retrieve_values(xq_bank, topk, start_pos=start_pos)
        output, _ = self._get_output_from_retrieved_values(scores, values, retrieved_v)
        assert output.shape == (bsz, seqlen, x.shape[2]), f"output should have the shape (bsz, seqlen, {x.shape[2]}), but got {output.shape=}"

        retriever_info = {}
        self.logging_values["topk"] = topk
        self.logging_values["indices"] = indices
        self.logging_values["all_scores"] = scores
        return output, retriever_info

class TopKTransformerBlock(nn.Module):
    def __init__(
            self,
            layer_id: int,
            args: ModelArgs,
        ):
        super().__init__()
        self.attention = TopKAttention(args) if not args.ret_bank_causal else TopKAttentionCausal(args)
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
        self.n_recursions = args.ret_recursions

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            freqs_cis: Tuple[torch.Tensor, torch.Tensor],
            mask: Optional[torch.Tensor],
            enable_checkpointing: bool = False,
            topk: Optional[int] = None,
            mem_state: Optional[int] = None,
        ):
        assert mem_state is None
        # Define the attention forward function
        def attention_forward(x):
            return self.attention.forward(x, start_pos, freqs_cis, mask, topk)

        for _ in range(self.n_recursions):
            # Apply checkpointing if enabled, otherwise use standard forward pass
            h, retriever_info = checkpoint(attention_forward, self.attention_norm(x)) if enable_checkpointing else attention_forward(self.attention_norm(x))
            h = x + h
            out = h + self.feed_forward.forward(self.ffn_norm(h))

            # here, repeat the output_individual for the topk
            if 'output_individual' in retriever_info and len(retriever_info["output_individual"]) > 0:
                bsz = x.shape[0]
                # Use passed topk if available, otherwise fall back to default
                if topk is None:
                    topk = self.attention.ret_topk
                bsztopk, seqlen, dim = retriever_info['output_individual'].shape
                assert bsz * topk == bsztopk, f"bsz * topk should be equal to bsztopk, but got {bsz=}, {topk=}, {bsztopk=}"
                x_individual = x.unsqueeze(1).repeat(1, topk, 1, 1).view(bsz * topk, seqlen, dim)
                out_individual = retriever_info["output_individual"]
                assert x_individual.shape == out_individual.shape, f"{x_individual.shape=}, {out_individual.shape=}"
                # skip connection for the output_individual
                h_ind = x_individual + out_individual
                # feed forward & skip connection for the output_individual
                out_individual = h_ind + self.feed_forward.forward(self.ffn_norm(h_ind))
                retriever_info["output_individual"] = out_individual

            x = out

        return out, retriever_info

    def _get_cached_memory(self):
        return self.attention._get_cached_memory()

    def use_cache(self, use_cache: bool) -> None:
        """Forwards to attention (standard KV cache + LongTermBank via TopKAttention.use_cache)."""
        self.attention.use_cache(use_cache)

    def get_logging_values(self):
        return {
            "attn": self.attention.get_logging_values(),
            "ffw": self.feed_forward.get_logging_values()
        }

    @property
    def use_block_attention(self):
        return False

    @property
    def use_flex_block_attention(self):
        return False

    @property
    def use_memory_attention(self):
        return False

    @property
    def use_local_attention(self):
        return False

    @property
    def use_local_flex_attention(self):
        return False

    @property
    def use_strided_attention(self):
        return False

    @property
    def use_flex_strided_attention(self):
        return False

    @property
    def use_gated_attention(self):
        return False

    def convert_to_emdr2_batch(self, x: torch.Tensor):
        '''
        Args:
            x: the tensor of shape (bsz, seqlen, ...)
        Returns:
            x_ind: the tensor of shape (bsz + (bsz * topk), seqlen, dim)
        '''
        if x is None:
            return None
        ndim = x.ndim
        repeat_tuple = (1 for _ in range(ndim - 1)) # repeat along the sequence dimension
        x_ind = x[:, None].repeat(1, self.attention.ret_topk, *repeat_tuple).view(-1, *x.shape[1:])
        return torch.cat([x, x_ind], dim=0)
