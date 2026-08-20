import sys
import types

import pytest
import torch
from torch import nn

if "termcolor" not in sys.modules:
    termcolor = types.ModuleType("termcolor")
    termcolor.colored = lambda text, *_args, **_kwargs: text
    sys.modules["termcolor"] = termcolor

from halo.models.policy.action_head import (
    MLPHead,
    resolve_physical_action_dim,
    select_action_targets,
)
from halo.robomme import (
    RELEASED_ADAPTER_HEADS,
    RELEASED_BLOCK_ATTN_INDICES,
    RELEASED_BLOCK_CHUNK_TS_LEN,
    RELEASED_MEMORY_CADENCE,
    RELEASED_POLICY_TOKENIZER,
    RELEASED_RETRIEVAL_ATTN_INDICES,
    RELEASED_RET_CHUNK_LEN,
    RELEASED_RET_TOPK,
    ROBO_MME_ACTION_HORIZON,
    ROBO_MME_PHYSICAL_ACTION_DIM,
    ROBO_MME_SERIALIZED_ACTION_DIM,
    build_robomme_policy_config,
    build_robomme_shared_config,
    robomme_policy_constructor,
    serialize_training_actions,
)


def test_training_serialization_keeps_gripper_before_eos():
    actions = torch.zeros(2, 3, ROBO_MME_PHYSICAL_ACTION_DIM)
    actions[..., 7] = torch.tensor([[-1.0, 1.0, -1.0], [1.0, -1.0, 1.0]])
    eos = torch.tensor(
        [[[0.0], [0.0], [1.0]], [[0.0], [0.0], [1.0]]]
    )

    serialized = serialize_training_actions(actions, eos)

    assert serialized.shape == (2, 3, ROBO_MME_SERIALIZED_ACTION_DIM)
    torch.testing.assert_close(serialized[..., :8], actions)
    torch.testing.assert_close(serialized[..., 7], actions[..., 7])
    torch.testing.assert_close(serialized[..., 8:], eos)


def test_physical_width_preserves_native_and_robomme_contracts():
    assert resolve_physical_action_dim(8) == 7
    assert resolve_physical_action_dim(9, expected_action_dim=8) == 8
    with pytest.raises(ValueError, match="configured physical action width"):
        resolve_physical_action_dim(8, expected_action_dim=8)


def test_robomme_configs_fix_released_topology_and_action_horizon():
    shared = build_robomme_shared_config(seq_length=64)
    policy = build_robomme_policy_config(decoder_hidden_features=32)

    assert shared.num_pred_steps == ROBO_MME_ACTION_HORIZON
    assert shared.compute_dtype == "float16"
    assert shared.downsample_obs == RELEASED_MEMORY_CADENCE
    assert shared.tokenizer_name == RELEASED_POLICY_TOKENIZER
    assert policy.adapter_num_heads == RELEASED_ADAPTER_HEADS
    assert policy.block_attn_idx == list(RELEASED_BLOCK_ATTN_INDICES)
    assert policy.block_chunk_ts_len == RELEASED_BLOCK_CHUNK_TS_LEN
    assert policy.ret_bank_causal is True
    assert policy.ret_chunk_len == RELEASED_RET_CHUNK_LEN
    assert policy.ret_straight_through is True
    assert policy.ret_topk == RELEASED_RET_TOPK
    assert policy.ret_topk_attn_idx == list(RELEASED_RETRIEVAL_ATTN_INDICES)
    assert policy.use_topk_attention is False

    with pytest.raises(ValueError, match="num_pred_steps"):
        build_robomme_shared_config(num_pred_steps=16)
    with pytest.raises(ValueError, match="block_attn_idx"):
        build_robomme_policy_config(block_attn_idx=[0, 1, 2, 3])
    with pytest.raises(ValueError, match="use_topk_attention"):
        build_robomme_policy_config(use_topk_attention=True)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("downsample_obs", 4),
        ("tokenizer_name", "Qwen/Qwen3-8B"),
    ],
)
def test_robomme_shared_config_rejects_cadence_or_tokenizer_drift(field, invalid):
    with pytest.raises(ValueError, match=field):
        build_robomme_shared_config(**{field: invalid})


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("block_chunk_ts_len", 4),
        ("ret_bank_causal", False),
        ("ret_chunk_len", 4),
        ("ret_straight_through", False),
        ("ret_topk", 4),
    ],
)
def test_robomme_policy_config_rejects_retrieval_drift(field, invalid):
    with pytest.raises(ValueError, match=field):
        build_robomme_policy_config(**{field: invalid})


def test_robomme_constructor_uses_nine_serialized_values_without_mutating_caller(
    monkeypatch,
):
    class StubMemModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_mem_model_module = types.ModuleType("halo.models.policy.mem_model")
    fake_mem_model_module.MemModel = StubMemModel
    monkeypatch.setitem(
        sys.modules, "halo.models.policy.mem_model", fake_mem_model_module
    )
    shared = build_robomme_shared_config(seq_length=64)
    policy = build_robomme_policy_config()
    caller_kwargs = {"image_keys": ["front", "wrist"], "sentinel": object()}

    model = robomme_policy_constructor(
        policy,
        shared,
        vision_encoder=object(),
        extra_kwargs=caller_kwargs,
    )

    kwargs = model.kwargs["extra_kwargs"]
    assert kwargs["action_dim"] == ROBO_MME_SERIALIZED_ACTION_DIM
    assert kwargs["action_eos_dim"] == 1
    assert kwargs["physical_action_dim"] == ROBO_MME_PHYSICAL_ACTION_DIM
    assert kwargs["proprio_dim"] == 8
    assert "action_dim" not in caller_kwargs

    shared.compute_dtype = "bfloat16"
    with pytest.raises(ValueError, match="compute_dtype"):
        robomme_policy_constructor(
            policy,
            shared,
            vision_encoder=object(),
            extra_kwargs=caller_kwargs,
        )


class _FixedChunk(nn.Module):
    def __init__(self, action_horizon: int, action_dim: int):
        super().__init__()
        chunk = torch.arange(action_horizon * action_dim, dtype=torch.float32)
        chunk = chunk.view(action_horizon, action_dim)
        chunk[:, -1] = 0.25
        self.register_buffer("chunk", chunk)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.chunk.flatten().expand(x.shape[0], -1)


def test_raw_chunk_prediction_returns_all_actions_and_leaves_queue_untouched():
    head = MLPHead(
        input_dim=4,
        hidden_features=8,
        output_dim=ROBO_MME_ACTION_HORIZON * ROBO_MME_PHYSICAL_ACTION_DIM,
        loss_fn=nn.L1Loss(reduction="none"),
        action_chunk_len=ROBO_MME_ACTION_HORIZON // 2,
    )
    head.mlp = _FixedChunk(
        ROBO_MME_ACTION_HORIZON,
        ROBO_MME_PHYSICAL_ACTION_DIM,
    )
    queued = torch.full((2, ROBO_MME_PHYSICAL_ACTION_DIM), -3.0)
    head.action_queue.put(queued)
    chunk = head.predict_chunk(
        torch.ones(2, 4),
        num_pred_steps=ROBO_MME_ACTION_HORIZON,
        action_dim=ROBO_MME_PHYSICAL_ACTION_DIM,
    )

    assert chunk.shape == (2, ROBO_MME_ACTION_HORIZON, ROBO_MME_PHYSICAL_ACTION_DIM)
    torch.testing.assert_close(chunk[..., -1], torch.full((2, 20), 0.25))
    assert head.action_queue.qsize() == 1
    torch.testing.assert_close(head.action_queue.queue[0], queued)


def test_action_loss_mask_reshape_keeps_gripper_and_excludes_eos():
    batch_size, timesteps = 2, 2
    actions = torch.zeros(batch_size, timesteps, ROBO_MME_ACTION_HORIZON, 9)
    values = torch.arange(actions.numel(), dtype=torch.float32).view_as(actions)
    actions[..., :8] = values[..., :8]
    actions[..., 8] = -9999.0
    output_positions = torch.tensor([[1, -1], [0, 3]])
    selected = select_action_targets(
        actions,
        valid_position_mask=output_positions != -1,
        num_pred_steps=ROBO_MME_ACTION_HORIZON,
        action_dim=ROBO_MME_PHYSICAL_ACTION_DIM,
    )

    flattened = actions[..., :8].reshape(batch_size, timesteps, -1)
    expected = flattened.reshape(-1, flattened.shape[-1])[
        output_positions.reshape(-1) != -1
    ]
    torch.testing.assert_close(selected, expected)
    torch.testing.assert_close(selected[:, 7::8], expected[:, 7::8])
    assert not torch.any(selected == -9999.0)
