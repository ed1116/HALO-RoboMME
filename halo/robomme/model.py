"""RoboMME-specific HALO model and action contract."""

from __future__ import annotations

from typing import Any, Optional

import torch


ROBO_MME_PHYSICAL_ACTION_DIM = 8
ROBO_MME_ACTION_EOS_DIM = 1
ROBO_MME_SERIALIZED_ACTION_DIM = 9
ROBO_MME_ACTION_HORIZON = 20
ROBO_MME_PROPRIO_DIM = 8

RELEASED_BLOCK_ATTN_INDICES = (1, 3, 4, 5)
RELEASED_RETRIEVAL_ATTN_INDICES = (0, 2)
RELEASED_ADAPTER_HEADS = 8
RELEASED_MODEL_CONFIG = "config/model/libero_1_5x_small.json"
RELEASED_BLOCK_CHUNK_TS_LEN = 8
RELEASED_RET_TOPK = 8
RELEASED_RET_CHUNK_LEN = 8
RELEASED_MEMORY_CADENCE = 8
RELEASED_POLICY_TOKENIZER = "Qwen/Qwen2-7B-Instruct"


def serialize_training_actions(
    physical_actions: torch.Tensor,
    eos: torch.Tensor,
) -> torch.Tensor:
    """Append HALO's training-only EOS channel to 8-D RoboMME actions."""
    if physical_actions.shape[-1] != ROBO_MME_PHYSICAL_ACTION_DIM:
        raise ValueError(
            f"Expected {ROBO_MME_PHYSICAL_ACTION_DIM} physical action values, "
            f"got {physical_actions.shape[-1]}"
        )
    if eos.shape != physical_actions.shape[:-1] + (ROBO_MME_ACTION_EOS_DIM,):
        raise ValueError(
            "EOS must match the action leading dimensions and have one channel; "
            f"got action {tuple(physical_actions.shape)} and EOS {tuple(eos.shape)}"
        )
    return torch.cat((physical_actions, eos.to(physical_actions)), dim=-1)


def build_robomme_shared_config(**overrides: Any):
    """Build the fixed RoboMME portions of HALO's shared configuration."""
    from halo.util.args import SharedConfig

    fixed = {
        "compute_dtype": "float16",
        "downsample_obs": RELEASED_MEMORY_CADENCE,
        "has_base_action": False,
        "is_bimanual": False,
        "num_cameras": 2,
        "num_pred_steps": ROBO_MME_ACTION_HORIZON,
        "remove_action": False,
        "rot_6d": False,
        "rot_euler": False,
        "tokenizer_name": RELEASED_POLICY_TOKENIZER,
        "use_lerobot": False,
        "use_tokenizer_dataset": True,
    }
    _reject_fixed_overrides(fixed, overrides)
    return SharedConfig(**fixed, **overrides)


def build_robomme_policy_config(**overrides: Any):
    """Build a policy config with the released interleaved HALO topology."""
    from halo.util.args import PolicyConfig

    fixed = {
        "adapter_num_heads": RELEASED_ADAPTER_HEADS,
        "block_attn_idx": list(RELEASED_BLOCK_ATTN_INDICES),
        "block_chunk_ts_len": RELEASED_BLOCK_CHUNK_TS_LEN,
        "ret_bank_causal": True,
        "ret_chunk_len": RELEASED_RET_CHUNK_LEN,
        "ret_straight_through": True,
        "ret_topk": RELEASED_RET_TOPK,
        "ret_topk_attn_idx": list(RELEASED_RETRIEVAL_ATTN_INDICES),
        "scratch_llama_config": RELEASED_MODEL_CONFIG,
        "use_topk_attention": False,
    }
    _reject_fixed_overrides(fixed, overrides)
    return PolicyConfig(**fixed, **overrides)


def robomme_policy_constructor(
    policy_cfg,
    shared_config,
    vision_encoder,
    train: bool = True,
    extra_kwargs: Optional[dict[str, Any]] = None,
):
    """Construct HALO with RoboMME dimensions while leaving native construction unchanged."""
    from halo.models.policy.mem_model import MemModel

    _validate_robomme_config(policy_cfg, shared_config)
    kwargs = dict(extra_kwargs or {})
    if "image_keys" not in kwargs:
        raise ValueError("image_keys are required for the RoboMME tokenizer dataset")
    kwargs.update(
        {
            "action_dim": ROBO_MME_SERIALIZED_ACTION_DIM,
            "action_eos_dim": ROBO_MME_ACTION_EOS_DIM,
            "physical_action_dim": ROBO_MME_PHYSICAL_ACTION_DIM,
            "proprio_dim": ROBO_MME_PROPRIO_DIM,
        }
    )
    return MemModel(
        shared_config=shared_config,
        policy_config=policy_cfg,
        vision_encoder=vision_encoder,
        train=train,
        extra_kwargs=kwargs,
        output_modes=["action", "text"],
    )


def robomme_model_constructor(
    model_config,
    shared_config,
    train: bool = True,
    extra_kwargs: Optional[dict[str, Any]] = None,
):
    """Construct the complete two-camera RoboMME HALO policy."""
    from halo.util.model_constructor import vision_encoder_constructor

    kwargs = dict(extra_kwargs or {})
    kwargs.update(
        {
            "mini_batch_size": 1,
            "img_seq_len": shared_config.seq_length // shared_config.downsample_obs,
            "num_cameras": shared_config.num_cameras,
        }
    )
    vision_encoder = vision_encoder_constructor(
        model_config.vision_encoder_cfg,
        extra_kwargs=kwargs,
    )
    return robomme_policy_constructor(
        model_config.policy_cfg,
        shared_config,
        vision_encoder,
        train=train,
        extra_kwargs=kwargs,
    )


def _reject_fixed_overrides(fixed: dict[str, Any], overrides: dict[str, Any]) -> None:
    conflicts = {
        key: value
        for key, value in overrides.items()
        if key in fixed and value != fixed[key]
    }
    if conflicts:
        names = ", ".join(sorted(conflicts))
        raise ValueError(f"RoboMME fixes these configuration fields: {names}")


def _validate_robomme_config(policy_cfg, shared_config) -> None:
    expected_shared = {
        "compute_dtype": "float16",
        "downsample_obs": RELEASED_MEMORY_CADENCE,
        "has_base_action": False,
        "is_bimanual": False,
        "num_cameras": 2,
        "num_pred_steps": ROBO_MME_ACTION_HORIZON,
        "remove_action": False,
        "rot_6d": False,
        "rot_euler": False,
        "tokenizer_name": RELEASED_POLICY_TOKENIZER,
        "use_lerobot": False,
        "use_tokenizer_dataset": True,
    }
    for name, expected in expected_shared.items():
        value = getattr(shared_config, name)
        if value != expected:
            raise ValueError(f"RoboMME requires {name}={expected!r}, got {value!r}")

    expected_policy = {
        "adapter_num_heads": RELEASED_ADAPTER_HEADS,
        "block_attn_idx": list(RELEASED_BLOCK_ATTN_INDICES),
        "block_chunk_ts_len": RELEASED_BLOCK_CHUNK_TS_LEN,
        "ret_bank_causal": True,
        "ret_chunk_len": RELEASED_RET_CHUNK_LEN,
        "ret_straight_through": True,
        "ret_topk": RELEASED_RET_TOPK,
        "ret_topk_attn_idx": list(RELEASED_RETRIEVAL_ATTN_INDICES),
        "scratch_llama_config": RELEASED_MODEL_CONFIG,
        "use_topk_attention": False,
    }
    for name, expected in expected_policy.items():
        value = getattr(policy_cfg, name)
        if value != expected:
            raise ValueError(
                f"Released HALO topology requires {name}={expected!r}, got {value!r}"
            )
