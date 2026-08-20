"""RoboMME-specific data and serving adapters."""

from importlib import import_module

from .dataset import (
    MANIFEST_SCHEMA_VERSION,
    TASK_SUITES,
    TASKS,
    DatasetStatistics,
    RoboMMEHDF5Dataset,
    RoboMMEPolicyInput,
    RoboMMESample,
    build_manifest,
    compute_statistics,
    write_manifest,
)

_MODEL_EXPORTS = (
    "RELEASED_ADAPTER_HEADS",
    "RELEASED_BLOCK_ATTN_INDICES",
    "RELEASED_BLOCK_CHUNK_TS_LEN",
    "RELEASED_MEMORY_CADENCE",
    "RELEASED_MODEL_CONFIG",
    "RELEASED_POLICY_TOKENIZER",
    "RELEASED_RETRIEVAL_ATTN_INDICES",
    "RELEASED_RET_CHUNK_LEN",
    "RELEASED_RET_TOPK",
    "ROBO_MME_ACTION_EOS_DIM",
    "ROBO_MME_ACTION_HORIZON",
    "ROBO_MME_PHYSICAL_ACTION_DIM",
    "ROBO_MME_PROPRIO_DIM",
    "ROBO_MME_SERIALIZED_ACTION_DIM",
    "build_robomme_policy_config",
    "build_robomme_shared_config",
    "robomme_model_constructor",
    "robomme_policy_constructor",
    "serialize_training_actions",
)


def __getattr__(name):
    if name not in _MODEL_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".model", __name__), name)
    globals()[name] = value
    return value

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "TASK_SUITES",
    "TASKS",
    "DatasetStatistics",
    "RoboMMEHDF5Dataset",
    "RoboMMEPolicyInput",
    "RoboMMESample",
    "build_manifest",
    "compute_statistics",
    "write_manifest",
    *_MODEL_EXPORTS,
]
