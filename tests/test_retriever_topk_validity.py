import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn


def _load_retriever_module():
    """Load the bank without importing HALO's unrelated vision dependencies."""
    shared_name = "halo.models.policy.transformer_shared"
    original_shared = sys.modules.get(shared_name)
    shared = types.ModuleType(shared_name)
    shared.ModelArgs = object
    shared.apply_rotary_emb = lambda *args, **kwargs: (args[0], None)
    shared.Attention = nn.Module
    shared.FeedForward = nn.Module
    shared.RMSNorm = nn.Module

    module_name = "_halo_retriever_topk_under_test"
    path = (
        Path(__file__).parents[1]
        / "halo"
        / "models"
        / "policy"
        / "retriever_topk.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[shared_name] = shared
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
        if original_shared is None:
            sys.modules.pop(shared_name, None)
        else:
            sys.modules[shared_name] = original_shared
    return module


RETRIEVER = _load_retriever_module()
LongTermBank = RETRIEVER.LongTermBank
LongTermBankCausal = RETRIEVER.LongTermBankCausal
TopKAttention = RETRIEVER.TopKAttention


def _args(*, max_batch_size=2, max_seq_len=8, time_aware=False):
    return SimpleNamespace(
        dim=4,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        ret_tau_init=1.0,
        ret_add_time_aware=time_aware,
        ret_relative_time=False,
        ret_chunk_len=2,
        cache=False,
        latent_len=1,
    )


def _bank_tensors(keys, values):
    return (
        torch.tensor(keys, dtype=torch.float32).view(1, -1, 1, 4),
        torch.tensor(values, dtype=torch.float32).view(1, -1, 1, 4),
    )


def _attention_combiner(straight_through=True):
    attention = TopKAttention.__new__(TopKAttention)
    nn.Module.__init__(attention)
    attention.straight_through_estimator = straight_through
    attention.wo = nn.Identity()
    return attention


def test_fewer_than_topk_never_uses_padded_slot_zero():
    bank = LongTermBank(_args(time_aware=True))
    keys, values = _bank_tensors(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        [[10.0, 0.0, 0.0, 0.0], [0.0, 20.0, 0.0, 0.0]],
    )
    bank.store_kv(keys, values)
    query = torch.tensor(
        [[[[0.4, 0.7, 0.2, 0.1]]]], dtype=torch.float32, requires_grad=True
    )

    scores, indices, retrieved, validity = bank.retrieve_kv(
        query, topk=4, return_validity=True
    )

    assert len(bank.retrieve_kv(query.detach(), topk=4)) == 3
    assert validity.tolist() == [[[True, True, False, False]]]
    assert indices[0, 0, :2].tolist() == [1, 0]
    assert torch.equal(indices[~validity], torch.zeros(2, dtype=torch.long))
    assert torch.equal(scores[~validity], torch.zeros(2))
    assert torch.equal(
        retrieved.masked_select(~validity[..., None, None]), torch.zeros(8)
    )

    # Even deliberately poisoned invalid values must receive exactly zero weight.
    poisoned = torch.where(
        validity[..., None, None], retrieved, torch.full_like(retrieved, 1e6)
    )
    current_values = torch.zeros((1, 1, 1, 4), dtype=torch.float32)
    output, _ = _attention_combiner()._get_output_from_retrieved_values(
        scores, current_values, poisoned, validity
    )
    expected_weights = torch.softmax(scores[..., :2], dim=2)
    expected = (
        expected_weights[..., None, None] * retrieved[..., :2, :, :]
    ).sum(dim=2).reshape(1, 1, 4)
    torch.testing.assert_close(output, expected)

    output.square().sum().backward()
    assert query.grad is not None
    assert torch.isfinite(query.grad).all()
    assert query.grad.abs().sum() > 0


def test_causal_all_masked_path_is_zero_and_has_finite_gradients():
    bank = LongTermBankCausal(_args())
    keys, values = _bank_tensors(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        [[100.0, 0.0, 0.0, 0.0], [0.0, 200.0, 0.0, 0.0]],
    )
    bank.store_kv(keys, values)
    query = torch.tensor(
        [[[[0.3, 0.8, 0.1, 0.2]], [[0.6, 0.2, 0.4, 0.1]]]],
        dtype=torch.float32,
        requires_grad=True,
    )

    scores, indices, retrieved, validity = bank.retrieve_kv(
        query, topk=4, start_pos=0, return_validity=True
    )

    assert not validity.any()
    assert not scores.any()
    assert not indices.any()
    assert not retrieved.any()

    poisoned = torch.full_like(retrieved, 1e6)
    current_values = torch.zeros((1, 2, 1, 4), dtype=torch.float32)
    output, _ = _attention_combiner()._get_output_from_retrieved_values(
        scores, current_values, poisoned, validity
    )
    assert not output.any()
    output.sum().backward()
    assert query.grad is not None
    assert torch.isfinite(query.grad).all()


def test_capacity_failure_happens_before_any_bank_mutation():
    bank = LongTermBank(_args(max_batch_size=1, max_seq_len=2))
    first = _bank_tensors(
        [[1.0, 0.0, 0.0, 0.0]], [[1.0, 2.0, 3.0, 4.0]]
    )
    overflow = _bank_tensors(
        [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        [[5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
    )
    bank.store_kv(*first)
    snapshot = (
        bank.cache_k.clone(),
        bank.cache_v.clone(),
        bank.mask.clone(),
        bank.n_valid_entries.clone(),
        bank.max_n_valid_entries,
    )

    with pytest.raises(ValueError, match="capacity 2 would be exceeded"):
        bank.store_kv(*overflow)

    assert torch.equal(bank.cache_k, snapshot[0])
    assert torch.equal(bank.cache_v, snapshot[1])
    assert torch.equal(bank.mask, snapshot[2])
    assert torch.equal(bank.n_valid_entries, snapshot[3])
    assert bank.max_n_valid_entries == snapshot[4]


def test_reset_removes_values_and_validity_from_the_previous_episode():
    bank = LongTermBank(_args(max_batch_size=1))
    old_episode = _bank_tensors(
        [[1.0, 0.0, 0.0, 0.0]], [[999.0, 999.0, 999.0, 999.0]]
    )
    new_episode = _bank_tensors(
        [[0.0, 1.0, 0.0, 0.0]], [[1.0, 2.0, 3.0, 4.0]]
    )
    bank.store_kv(*old_episode)

    bank.reset_to_zero()

    assert bank.is_empty()
    assert not bank.cache_k.any()
    assert not bank.cache_v.any()
    assert not bank.mask.any()
    assert not bank.n_valid_entries.any()

    bank.store_kv(*new_episode)
    query = torch.tensor([[[[0.0, 1.0, 0.0, 0.0]]]])
    _, _, retrieved, validity = bank.retrieve_kv(
        query, topk=2, return_validity=True
    )
    assert validity.tolist() == [[[True, False]]]
    torch.testing.assert_close(
        retrieved[0, 0, 0, 0], torch.tensor([1.0, 2.0, 3.0, 4.0])
    )
    assert not retrieved[0, 0, 1].any()
