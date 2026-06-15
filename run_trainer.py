#!/usr/bin/env python3
"""
Pythonic version of the train.sh script launcher for long-term-mem.

This script provides a clean, object-oriented interface for launching training experiments
with proper parameter validation, environment setup, and command generation.
"""

import os
import sys
import random
import subprocess
import argparse
from pathlib import Path
from typing import Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum
from termcolor import colored

def required_env_var(env_var_name: str) -> str:
    """Validate that an environment variable is set."""
    if env_var_name not in os.environ:
        raise ValueError(f"Error: {env_var_name} is required and cannot be empty")
    return os.environ[env_var_name]

class LaunchLocation(Enum):
    """Supported launch locations."""
    LOCAL = "local"

class VisionEncoder(Enum):
    """Supported vision encoders."""
    CLIP = "clip"
    DINOV3 = "dinov3"
    CROSSMAE = "crossmae"
    RESNET18 = "resnet18"
    RESNET18_NOPOOL = "resnet18_nopool"


class GatingFlag(Enum):
    """Supported gating flags."""
    TRUE = "True"
    FALSE = "False"
    BLOCK_NOGATE = "block_nogate"
    BLOCK_SINK = "block_sink"


class BlockFinetune(Enum):
    """Supported block finetune options."""
    EVERY_STEP = "every_step"
    START = "start"
    EVERY_STEP_OP = "every_step_op"
    START_OP = "start_op"
    BLOCKFT = "blockft"


# Architecture defaults shared by all task presets (HALO Top-K sparse attention).
_PRESET_BLOCK_ATTN_IND = [1, 3, 4, 5]
_PRESET_RET_TOPK_ATTN_IDX = [0, 2]

# Per-task presets selected via --task. Each bundles the sequence length, the
# task + qa data configs (passed together to -dc), and the state-supervision loss
# coefficient. GPT state supervision (bbox_str mode), block-attn indices, and
# TopK-attn indices are auto-set for all of them, since the qa configs are the
# *_relevant_objs_gptstate variants and the architecture is shared across tasks.
TASK_PRESETS = {
    "washandreturn": {
        "seq_length": 512,
        "data_config": ["task_robocasa_mem_washandreturn.json",
                        "qa_robocasa_mem_washandreturn_relevant_objs_gptstate.json"],
        "coeff_state_supervision_loss": 0.1,
        "block_attn_ind": _PRESET_BLOCK_ATTN_IND,
        "ret_topk_attn_idx": _PRESET_RET_TOPK_ATTN_IDX,
    },
    "retrieve_oil": {
        "seq_length": 512,
        "data_config": ["task_robocasa_mem_retrieve_oil.json",
                        "qa_robocasa_mem_retrieve_oil_relevant_objs_gptstate.json"],
        "coeff_state_supervision_loss": 1.0,
        "block_attn_ind": _PRESET_BLOCK_ATTN_IND,
        "ret_topk_attn_idx": _PRESET_RET_TOPK_ATTN_IDX,
    },
    "heatpot": {
        "seq_length": 2048,
        "data_config": ["task_robocasa_mem_heatpot.json",
                        "qa_robocasa_mem_heatpot_relevant_objs_gptstate.json"],
        "coeff_state_supervision_loss": 1.0,
        "block_attn_ind": _PRESET_BLOCK_ATTN_IND,
        "ret_topk_attn_idx": _PRESET_RET_TOPK_ATTN_IDX,
    },
    "kbreads": {
        "seq_length": 2048,
        "data_config": ["task_robocasa_mem_kbreads.json",
                        "qa_robocasa_mem_kbreads_relevant_objs_gptstate.json"],
        "coeff_state_supervision_loss": 1.0,
        "block_attn_ind": _PRESET_BLOCK_ATTN_IND,
        "ret_topk_attn_idx": _PRESET_RET_TOPK_ATTN_IDX,
    },
}


@dataclass
class TrainingConfig:
    """Configuration for training experiments."""
    # Required parameters
    downsample_obs: int
    batch_size: int
    num_gpus: int
    num_workers: int
    epochs: int
    warmup_epochs: int
    model_config: str
    data_config: List[str]
    task_val_config: Optional[List[str]] = None
    weight_by_dataset: Union[int, list[int]] = field(default_factory=list)
    seed: int = 1
    attn_latent_len: int = 1
    seq_length: int = 4
    phase: str = "pretrain"
    vision_encoder: VisionEncoder = VisionEncoder.CLIP
    launch_location: LaunchLocation = LaunchLocation.LOCAL
    basic_run: bool = False  # Add basic_run parameter for single-GPU training
    load_in_mem: bool = False  # Add load_in_mem parameter for loading dataset in memory
    compile_model: bool = False  # Add compile_model parameter for compiling the model

    # Action head
    action_head: str = "mlp"
    # TopK attention specific parameters
    use_topk_attention: bool = False
    ret_topk: int = 8
    ret_topk_max: Optional[int] = None  # Maximum topk for curriculum learning (None = no curriculum)
    ret_chunk_len: int = 8
    ret_n_topk_blocks: int = 1
    ret_straight_through: bool = True
    ret_recursions: int = 1
    ret_multikv: bool = False
    ret_add_time_aware: bool = False
    ret_relative_time: bool = False
    # Toy dataset specific parameters
    use_toy_dataset: bool = False
    task_id: Optional[int] = 3
    n_distractors: Optional[int] = 1
    train_dataset_size: str = "1_000_000"

    # k_ptp parameter for multi-step prediction
    k_ptp: int = 0  # Add k_ptp parameter

    # num_pred_steps parameter
    num_pred_steps: int = 32  # Number of prediction steps (default: 32)

    # Learning rate
    lr: Optional[float] = None  # Learning rate (if None, uses default from optimizer config)

    # Hyperparameter tuning flag
    hyper_tune: bool = False  # Enable hyperparameter tuning mode

    # Wandb configuration
    wandb_project_name: Optional[str] = None  # Will be set based on experiment type if None
    wandb_group: Optional[str] = None  # Wandb group name for grouping runs

    # Computed fields
    extra_flags: List[str] = field(default_factory=list)
    exp_base_dir: str = "memory_exps"
    exp_name: str = ""
    task_config: str = ""
    gbs: int = 0
    gbs_factor: float = 1.0
    repeat_traj_factor: float = 1.0
    accum_iter: int = 0
    num_repeat_traj: int = 0
    port_num: int = 0

    # resume from checkpoint
    resume: Optional[str] = None
    resume_new_exp: bool = False
    break_after_n_epochs: Optional[int] = None

    # evaluation related parameters
    eval_every: int = -2
    eval_at_zero: bool = False
    n_eval: int = 24

    # to be calculed in post init params
    use_lora: bool = False
    # add state supervision
    add_state_supervision: bool = False
    state_supervision_mode: str = "bbox_str"  # "bbox" or "bbox_str"
    add_gpt_state_supervision: bool = False
    add_fake_state_supervision: bool = False
    ss_create_mode: str = "inst_generic"  # "inst_generic", "inst_specific", or "time"
    coeff_state_supervision_loss: float = 1.0

    # block-attn specific parameters
    block_attn_ind: Optional[list[int]] = None
    block_chunk_ts_len: int = 8

    # local-attn specific parameters
    local_attn_ind: Optional[list[int]] = None

    # strided-attn specific parameters
    strided_attn_ind: Optional[list[int]] = None
    strided_len: int = -1

    # ret_topk_attn_idx parameter
    ret_topk_attn_idx: Optional[list[int]] = None

    # gated_attn_idx parameter (gai for short)
    gated_attn_idx: Optional[list[int]] = None

    # tokme_attn_idx parameter (tmai for short)
    tokme_attn_idx: Optional[list[int]] = None

    # QA dataset specific parameters
    qa_remove_query: bool = False
    max_ss_size: int = -1

    # use original (non-generated) instructions
    use_og_inst: bool = False

    # number of gradient steps (overrides epochs if > 0)
    n_gradients_steps: int = -1

    # bit-exact regression mode: none | generate | verify
    test_runs: str = "none"

    def _set_any_params_specific_to_phase(self):
        """Set any parameters specific to the phase."""
        if self.phase == "pretrain_lora":
            self.use_lora = True
        elif self.phase == "pretrain":
            pass
        else:
            raise ValueError(f"Unknown phase: {self.phase}")
    def __post_init__(self):
        """Validate and compute derived fields after initialization."""
        self._set_any_params_specific_to_phase()
        self._validate_required_params()
        self._validate_environment()
        self._compute_derived_fields()
        self._generate_experiment_name()
        self._generate_extra_flags()
        self._set_wandb_project_name()

    def _validate_required_params(self):
        """Validate that all required parameters are provided."""
        required_params = {
            'downsample_obs': self.downsample_obs,
            'batch_size': self.batch_size,
            'num_gpus': self.num_gpus,
            'model_config': self.model_config,
            'data_config': self.data_config,
        }

        # For toy dataset, validate toy-specific parameters
        if self.use_toy_dataset:
            toy_required_params = {
                'task_id': self.task_id,
                'n_distractors': self.n_distractors,
            }
            for param_name, param_value in toy_required_params.items():
                if param_value is None:
                    raise ValueError(f"Error: {param_name} is required when using toy dataset")

        for param_name, param_value in required_params.items():
            if param_value is None or param_value == "":
                raise ValueError(f"Error: {param_name} is required and cannot be empty")

    def _validate_environment(self):
        """Validate that all environment variables are set."""
        required_env_vars = [
            'CASAPLAY_DATAROOT',
            'WANDB_API_KEY',
            'EXP_STORAGE_BASE_DIR'
        ]
        for env_var_name in required_env_vars:
            required_env_var(env_var_name)
        return

    def _compute_derived_fields(self):
        """Compute derived fields based on input parameters."""
        # Global batch size calculation - different for toy dataset
        if self.use_toy_dataset:
            token_batch_size = 2048  # Toy dataset uses smaller token batch size
            self.gbs = token_batch_size
        else:
            token_batch_size = 8192 # with sl = 256, bs = 32
            self.gbs = max(32, token_batch_size // self.seq_length)
        self.gbs = int(self.gbs * self.gbs_factor)
        # make it closes to the nearest multiple of num_gpus (upper bound) * batch_size
        # self.gbs = ((self.gbs + self.num_gpus - 1) // self.num_gpus) * self.num_gpus
        print(f"token_batch_size: {token_batch_size}, gbs: {self.gbs}, num_gpus: {self.num_gpus}, batch_size: {self.batch_size}")
        self.gbs = ((self.gbs + self.num_gpus * self.batch_size - 1) // (self.num_gpus * self.batch_size)) * (self.num_gpus * self.batch_size)

        # Accumulation iterations
        self.accum_iter = max(1, self.gbs // (self.batch_size * self.num_gpus))

        # Number of repeat trajectories
        self.num_repeat_traj = max(32, int(32 * (256 / self.seq_length)))

        # Adjust for specific datasets
        if any(str(data_config) in ["task_robocasa_atomic.json", "task_robocasa_atomic_all.json", "task_robocasa_mem_mix.json"] for data_config in self.data_config):
            self.num_repeat_traj = max(1, self.num_repeat_traj // 2)
        if any(str(data_config) in ["task_robocasa_mem_washandreturn.json"] for data_config in self.data_config):
            self.num_repeat_traj = self.num_repeat_traj * 2
        if os.path.basename(self.data_config[0]).startswith("qa_"):
            self.num_repeat_traj = 1

        self.num_repeat_traj = int(self.num_repeat_traj * self.repeat_traj_factor)

        def _resolve_data_config(name: str) -> str:
            base = os.path.basename(name)
            subdir = "qa" if base.startswith("qa_") else "task"
            return f"config/{subdir}/{base}"

        # Task config path
        if self.use_toy_dataset:
            # For toy dataset, task config doesn't matter, use atomic as placeholder
            self.task_config = "config/task/task_robocasa_atomic.json"
        else:
            self.task_config = [_resolve_data_config(data_config) for data_config in self.data_config]

        # Task validation config path
        if self.task_val_config is None:
            self.task_val_config = []
        else:
            self.task_val_config = [_resolve_data_config(task_config) for task_config in self.task_val_config]

        # Random port number
        self.port_num = 2452 + random.randint(0, 99)

        # Compute strided_len if strided_attn_ind is set
        if self.strided_attn_ind is not None and len(self.strided_attn_ind) > 0:
            self.strided_len = self.seq_length // (self.block_chunk_ts_len * self.downsample_obs)

        self._weight_by_dataset = [str(w) for w in self.weight_by_dataset]

    def _generate_extra_flags(self):
        """Generate extra command line flags based on configuration."""
        self.extra_flags = []

        # Toy dataset specific flags
        if self.use_toy_dataset:
            self.extra_flags.extend([
                "--shared-cfg.use-toy-vision-dataset",
                "--dataset-cfg.toy-task-id", str(self.task_id),
                "--dataset-cfg.toy-max-distractors", str(self.n_distractors),
                "--dataset-cfg.toy-train-dataset-size", self.train_dataset_size,
                "--dataset-cfg.toy-val-dataset-size", "100_000"
            ])

        # Vision encoder flags
        if self.vision_encoder == VisionEncoder.CLIP:
            self.extra_flags.extend([
                "--model-cfg.vision-encoder-cfg.vision-encoder", "vit_base_patch32_clip_224.openai"
            ])
            if self.exp_base_dir is None:
                self.exp_base_dir = "clip_exps"
        elif self.vision_encoder == VisionEncoder.DINOV3:
            self.extra_flags.extend([
                "--model-cfg.vision-encoder-cfg.vision-encoder", "facebook/dinov3-vits16plus-pretrain-lvd1689m"
            ])
        elif self.vision_encoder == VisionEncoder.CROSSMAE:
            # get the CASAPLAY_DATAROOT and if the file is not there, install it using: https://huggingface.co/mlfu7/ICRT/blob/main/crossmae_rtx/cross-mae-rtx-vitb.pth
            CASAPLAY_DATAROOT = os.environ.get("CASAPLAY_DATAROOT")
            if CASAPLAY_DATAROOT is None:
                raise ValueError("CASAPLAY_DATAROOT is not set")
            VISION_ENCODER_PATH = os.path.join(CASAPLAY_DATAROOT, "crossmae_rtx", "cross-mae-rtx-vitb.pth")
            if not os.path.exists(VISION_ENCODER_PATH):
                os.makedirs(os.path.dirname(VISION_ENCODER_PATH), exist_ok=True)
                subprocess.run(["wget", "https://huggingface.co/mlfu7/ICRT/resolve/main/crossmae_rtx/cross-mae-rtx-vitb.pth", "-O", VISION_ENCODER_PATH])
            self.extra_flags.extend([
                "--model-cfg.vision-encoder-cfg.vision-encoder", VISION_ENCODER_PATH
            ])
        elif self.vision_encoder == VisionEncoder.RESNET18 or self.vision_encoder == VisionEncoder.RESNET18_NOPOOL:
            self.extra_flags.extend([
                "--model-cfg.vision-encoder-cfg.vision-encoder", self.vision_encoder.value
            ])
            self.extra_flags.extend([
                "--model-cfg.vision-encoder-cfg.vision-unfreeze-all"
            ])

        # Dataset-specific flags
        if any("robocasa" in data_config for data_config in self.data_config):
            self.extra_flags.extend([
                "--shared-cfg.has_base_action"
            ])

        self.extra_flags.extend(["--shared-cfg.attn_latent_len", str(self.attn_latent_len)])

        # TODO: Load in memory is not working with run_trainer.py but works fine separately from shell.
        # TODO: Load in memory is also not working with multi-node training because we use world_rank instead of local_rank (easy fx after the previous TODO)
        if self.load_in_mem:
            self.extra_flags.append("--dataset-cfg.load-in-mem")

        if self.compile_model:
            self.extra_flags.append("--trainer-cfg.compile_model")

        if self.action_head != "mlp":
            self.extra_flags.append(f"--model-cfg.policy-cfg.decoder_pred_head {self.action_head}")

        if self.use_topk_attention or self.ret_topk_attn_idx is not None:
            # both cannot be True
            assert not (self.use_topk_attention and self.ret_topk_attn_idx is not None), "use_topk_attention and ret_topk_attn_idx cannot both be set."
            if self.exp_base_dir is None:
                self.exp_base_dir = "topk_exps"
            if self.use_topk_attention:
                self.extra_flags.append("--model-cfg.policy-cfg.use_topk_attention")
                self.extra_flags.append("--model-cfg.policy-cfg.ret_bank_causal")
            if self.ret_topk_attn_idx is not None and len(self.ret_topk_attn_idx) > 0:
                self.extra_flags.extend(["--model-cfg.policy-cfg.ret_topk_attn_idx", " ".join(map(str, self.ret_topk_attn_idx))])
                self.extra_flags.append("--model-cfg.policy-cfg.ret_bank_causal")
            self.extra_flags.extend(["--model-cfg.policy-cfg.ret_topk", str(self.ret_topk)])
            if self.ret_topk_max is not None:
                self.extra_flags.extend(["--model-cfg.policy-cfg.ret_topk_max", str(self.ret_topk_max)])
            self.extra_flags.extend(["--model-cfg.policy-cfg.ret_chunk_len", str(self.ret_chunk_len)])
            if self.use_topk_attention:
                self.extra_flags.extend(["--model-cfg.policy-cfg.ret_n_topk_blocks", str(self.ret_n_topk_blocks)])
            self.extra_flags.append("--model-cfg.policy-cfg.ret_straight_through")
            if self.ret_recursions > 1:
                self.extra_flags.append("--model-cfg.policy-cfg.ret_recursions")
                self.extra_flags.append(str(self.ret_recursions))
            if self.ret_multikv:
                self.extra_flags.append("--model-cfg.policy-cfg.ret_multikv")
            if self.ret_add_time_aware:
                self.extra_flags.append("--model-cfg.policy-cfg.ret_add_time_aware")

        # use original instructions flag
        if self.use_og_inst:
            self.extra_flags.append("--dataset-cfg.use-og-inst")

        # QA dataset specific flags
        if self.qa_remove_query:
            self.extra_flags.append("--dataset-cfg.qa-remove-query")
        if self.max_ss_size != -1:  # Only add if different from default
            self.extra_flags.append("--dataset-cfg.max-ss-size")
            self.extra_flags.append(str(self.max_ss_size))

        if self.exp_base_dir is None:
            self.exp_base_dir = "memory_exps"

        # Debug flags (if not in debug mode)
        if os.environ.get("DEBUG", "").lower() not in ["true", "1"]:
            print(colored("Adding logging flags", "green"))
            self.extra_flags.extend([
                "--logging-cfg.log-name", self.exp_name,
            ])
            # Add wandb group if specified
            if self.wandb_group is not None:
                self.extra_flags.extend([
                    "--logging-cfg.wandb-group", self.wandb_group,
                ])

        if self.break_after_n_epochs is not None:
            self.extra_flags.extend([
                "--trainer-cfg.break_after_n_epochs", str(self.break_after_n_epochs)
            ])

        self.extra_flags.extend([
            "--eval-cfg.eval_every", str(self.eval_every),
            "--eval-cfg.n_eval", str(self.n_eval),
        ])
        if self.eval_at_zero:
            self.extra_flags.extend([
                "--eval-cfg.eval_at_zero"
            ])

    def _generate_experiment_name(self):
        """Generate the experiment name based on configuration."""
        if self.use_toy_dataset:
            # Toy dataset experiment naming
            self.exp_name = (
                f"toy_taskid{self.task_id}_{Path(self.model_config).stem}_"
                f"GBS{self.gbs}_"
                f"sl{self.seq_length}_s{self.seed}"
            )

            # Add gating suffix
            # Add k_ptp suffix if > 0
            if self.k_ptp > 0:
                self.exp_name += f"_kptp{self.k_ptp}"

            # Add num_pred_steps suffix if not 32
            if self.num_pred_steps != 32:
                self.exp_name += f"_ps{self.num_pred_steps}"

            # Add distractors and dataset size
            self.exp_name += f"_distractors{self.n_distractors}_{self.train_dataset_size}"
        else:
            # Regular experiment naming
            # task_config_name = Path(self.task_config).stem
            task_config_name = "_".join([Path(task_config).stem for task_config in self.task_config])

            data_config_name = "_".join([data_config.split("/")[-1].split(".")[0] for data_config in self.data_config])
            # Base experiment name
            self.exp_name = (
                f"exp_ds{self.downsample_obs}_{Path(self.model_config).stem}_"
                f"{data_config_name}_GBS{self.gbs}_"
                f"sl{self.seq_length}_s{self.seed}_{self.vision_encoder.value}"
            )

            # Add k_ptp suffix if > 0
            if self.k_ptp > 0:
                self.exp_name += f"_kptp{self.k_ptp}"

            # Add num_pred_steps suffix if not 32
            if self.num_pred_steps != 32:
                self.exp_name += f"_ps{self.num_pred_steps}"

            if self.attn_latent_len != 1:
                self.exp_name += f"_attn_len{self.attn_latent_len}"

        if self.action_head != "mlp":
            self.exp_name += f"_{self.action_head}"

        if self.use_topk_attention or self.ret_topk_attn_idx is not None:
            if self.ret_topk_attn_idx is not None and len(self.ret_topk_attn_idx) > 0:
                self.exp_name += f"_retattnidx{'_'.join(map(str, self.ret_topk_attn_idx))}"
            self.exp_name += f"_topk{self.ret_topk}_retchunk{self.ret_chunk_len}"
            # Add curriculum max topk to exp name if curriculum is enabled
            if self.ret_topk_max is not None:
                self.exp_name += f"_mx{self.ret_topk_max}"
            if self.use_topk_attention and self.ret_n_topk_blocks > 1:
                self.exp_name += f"_retblks{self.ret_n_topk_blocks}"
            self.exp_name += "_str_th"
            if self.ret_recursions > 1:
                self.exp_name += f"_retrecur{self.ret_recursions}"
            if self.ret_multikv:
                self.exp_name += "_multikv"
            if self.ret_add_time_aware:
                self.exp_name += "_time"
                if self.ret_relative_time:
                    self.exp_name += "rel"

        if self._weight_by_dataset != "1":
            self.exp_name += f"_wd{'_'.join(self._weight_by_dataset)}"

        if self.use_lora:
            self.exp_name += "_lora"

        if self.phase != "pretrain":
            self.exp_name += f"_ph{self.phase}"

        def add_state_supervision_to_exp_name(prefix):
            if self.state_supervision_mode == "bbox":
                self.exp_name += f"{prefix}"
            elif self.state_supervision_mode == "bbox_str":
                self.exp_name += f"{prefix}str"
            elif self.state_supervision_mode == "bbox_inst_str":
                self.exp_name += f"{prefix}inststr"
            elif self.state_supervision_mode == "time":
                self.exp_name += f"{prefix}time"
            else:
                raise ValueError(f"Invalid state supervision mode: {self.state_supervision_mode}")
        if self.add_state_supervision:
            add_state_supervision_to_exp_name("_ss")

        if self.add_gpt_state_supervision:
            add_state_supervision_to_exp_name("_gptss")
            assert (self.state_supervision_mode == "bbox_str"), "GPT state supervision only supports bbox_str or bbox_inst_str mode"

        if self.add_fake_state_supervision:
            add_state_supervision_to_exp_name("_fakess")

        if self.qa_remove_query:
            self.exp_name += "_noq"

        if self.max_ss_size != -1:
            self.exp_name += f"_maxss{self.max_ss_size}"

        if self.coeff_state_supervision_loss != 1.0:
            self.exp_name += f"_ss{self.coeff_state_supervision_loss}"

        # Add n_gradients_steps to experiment name if specified
        if self.n_gradients_steps != -1:
            self.exp_name += f"_ngs{self.n_gradients_steps}"

        if self.block_attn_ind is not None and len(self.block_attn_ind) > 0:
            self.exp_name += f"_block_attn_ind{'_'.join(map(str, self.block_attn_ind))}"
            self.exp_name += f"_blk{self.block_chunk_ts_len}"
        if self.local_attn_ind is not None and len(self.local_attn_ind) > 0:
            self.exp_name += f"_local_attn_ind{'_'.join(map(str, self.local_attn_ind))}"
        if self.strided_attn_ind is not None and len(self.strided_attn_ind) > 0:
            self.exp_name += f"_strd_attn_ind{'_'.join(map(str, self.strided_attn_ind))}_blk{self.strided_len}"
        if self.gated_attn_idx is not None and len(self.gated_attn_idx) > 0:
            self.exp_name += f"_gai{'_'.join(map(str, self.gated_attn_idx))}"
        if self.tokme_attn_idx is not None and len(self.tokme_attn_idx) > 0:
            self.exp_name += f"_tmai{'_'.join(map(str, self.tokme_attn_idx))}"
        # Add lr to experiment name if hyper_tune is enabled
        if self.hyper_tune and self.lr is not None:
            # Format lr to use readable format (e.g., 5e-4 -> lr5en4, 0.001 -> lr1ep3)
            # Convert to scientific notation and replace characters that might cause issues in filenames
            lr_str = f"{self.lr:.0e}".replace("e-0", "en").replace("e+0", "ep").replace("e-", "en").replace("e+", "ep")
            self.exp_name += f"_lr{lr_str}"

        print(colored(self.exp_name, "green"))

    def _set_wandb_project_name(self):
        """Set wandb project name based on experiment type if not provided."""
        if self.wandb_project_name is None:
            if self.use_toy_dataset:
                self.wandb_project_name = "toy_vision"
            else:
                self.wandb_project_name = "halo-run"

    def get_training_command(self) -> List[str]:
        """Generate the complete training command with all arguments."""
        exp_storage_base_dir = os.environ.get("EXP_STORAGE_BASE_DIR", "/tmp/experiments")

        # Set num_workers based on basic_run flag
        num_workers = 0 if self.basic_run else self.num_workers

        # Set validation and save frequencies based on dataset type
        if self.use_toy_dataset:
            val_every = 1
            save_every = 25
        else:
            val_every = 10
            # divide the numer of epochs by 4
            save_every = max(1, self.epochs // 4)

        num_cameras = 2
        # if tiago in the data_config name, then set num_cameras to 1
        if 'tiago' in self.task_config[0]:
            num_cameras = 1
        # Core arguments - each argument and value as separate list elements
        args = [
            "--dataset-cfg.dataset-json", " ".join(self.task_config),
            "--shared-cfg.num_cameras", str(num_cameras),
            "--dataset-cfg.num_repeat_traj", str(self.num_repeat_traj),
            "--model-cfg.policy-cfg.scratch-llama-config", f"config/model/{self.model_config}",
            "--shared-cfg.seq_length", str(self.seq_length),
            "--shared-cfg.seed", str(self.seed),
            "--shared-cfg.batch-size", str(self.batch_size),
            "--trainer-cfg.accum-iter", str(self.accum_iter),
            "--shared-cfg.num-pred-steps", str(self.num_pred_steps),
            "--model-cfg.policy-cfg.phase", self.phase,
            "--trainer-cfg.epochs", str(self.epochs),
            "--optimizer-cfg.warmup-epochs", str(self.warmup_epochs),
            "--trainer-cfg.num-workers", str(num_workers),
            "--trainer-cfg.val-every", str(val_every),
            "--shared-cfg.save-every", str(save_every),
            "--shared-cfg.downsample_obs", str(self.downsample_obs),
            "--logging-cfg.output-dir", f"{exp_storage_base_dir}/{self.exp_base_dir}/{self.exp_name}",
            "--trainer-cfg.wandb-project", self.wandb_project_name,  # Add wandb project name
            "--shared-cfg.resume", str(self.resume),
            "--dataset-cfg.weight_by_dataset", " ".join(self._weight_by_dataset),
            "--model-cfg.policy-cfg.decoder_hidden_features", "128",
            "--model-cfg.policy-cfg.ce_hidden_features", "64",
        ]

        # Add validation dataset config if provided
        if self.task_val_config is not None and len(self.task_val_config) > 0:
            args.extend([
                "--dataset-cfg.dataset-val-json", " ".join(self.task_val_config),
            ])

        # Add k_ptp argument if > 0
        if self.k_ptp > 0:
            args.extend([
                "--shared-cfg.k_ptp", str(self.k_ptp)
            ])

        if 'lerobot' in self.task_config:
            args.extend([
                "--shared-cfg.use-lerobot",
                "--dataset-cfg.no-use-dali",
            ])

        if self.use_lora:
            args.extend([
                "--model-cfg.policy-cfg.use_lora",
            ])

        if self.add_state_supervision:
            args.extend([
                "--shared-cfg.add-state-supervision",
                "--shared-cfg.state-supervision-mode", self.state_supervision_mode,
                "--shared-cfg.ss-create-mode", self.ss_create_mode,
            ])

        if self.add_gpt_state_supervision:
            args.extend([
                "--shared-cfg.add-gpt-state-supervision",
                "--shared-cfg.state-supervision-mode", self.state_supervision_mode,
            ])

        if self.add_fake_state_supervision:
            args.extend([
                "--shared-cfg.add-fake-state-supervision",
                "--shared-cfg.state-supervision-mode", self.state_supervision_mode,
            ])

        if self.coeff_state_supervision_loss != 1.0:
            args.extend([
                "--shared-cfg.coeff-state-supervision-loss", str(self.coeff_state_supervision_loss),
            ])

        if self.block_attn_ind is not None:
            args.extend([
                "--model-cfg.policy-cfg.block-attn-idx", " ".join(map(str, self.block_attn_ind)),
                "--model-cfg.policy-cfg.block-chunk-ts-len", str(self.block_chunk_ts_len),
            ])
        if self.local_attn_ind is not None:
            args.extend([
                "--model-cfg.policy-cfg.local-attn-idx", " ".join(map(str, self.local_attn_ind)),
            ])
        if self.strided_attn_ind is not None:
            args.extend([
                "--model-cfg.policy-cfg.strided-attn-idx", " ".join(map(str, self.strided_attn_ind)),
                "--model-cfg.policy-cfg.strided-len", str(self.strided_len),
            ])
        if self.gated_attn_idx is not None:
            args.extend([
                "--model-cfg.policy-cfg.gated-attn-idx", " ".join(map(str, self.gated_attn_idx)),
            ])
        if self.tokme_attn_idx is not None:
            args.extend([
                "--model-cfg.policy-cfg.tokme-attn-idx", " ".join(map(str, self.tokme_attn_idx)),
            ])
        if self.resume_new_exp:
            args.extend([
                "--shared-cfg.resume-new-exp",
            ])

        # Add learning rate if specified
        if self.lr is not None:
            args.extend([
                "--optimizer-cfg.lr", str(self.lr),
            ])

        # Add n_gradients_steps if specified
        if self.n_gradients_steps != -1:
            args.extend([
                "--trainer-cfg.n_gradients_steps", str(self.n_gradients_steps),
            ])

        # Forward test-runs mode
        if self.test_runs != "none":
            args.extend([
                "--trainer-cfg.test_runs", self.test_runs,
            ])

        # Add hyper_tune flag if enabled
        if self.hyper_tune:
            args.extend([
                "--hyper-tune",
            ])

        # Add extra flags
        args.extend(self.extra_flags)

        return args

    def get_torchrun_command(self) -> List[str]:
        """Generate the torchrun command for training."""
        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
        exp_storage_base_dir = os.environ.get("EXP_STORAGE_BASE_DIR")

        # Base command
        cmd = [
            "MUJOCO_GL=egl OMP_NUM_THREADS=1 MPI_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CC=$(command -v gcc) CXX=$(command -v g++) torchrun",
            f"--nproc_per_node={self.num_gpus}",
            f"--master_port={self.port_num}",
            "scripts/train.py"
        ]

        # Add training arguments
        cmd.extend(self.get_training_command())

        return cmd


    def print_config(self):
        """Print the current configuration."""
        print("=" * 50)
        print("TRAINING CONFIGURATION")
        print("=" * 50)
        print(f"DOWNSAMPLE_OBS: {self.downsample_obs}")
        print(f"BATCH_SIZE: {self.batch_size}")
        print(f"NUM_GPUS: {self.num_gpus}")
        print(f"MODEL_CONFIG: {self.model_config}")
        print(f"DATA_CONFIG: {self.data_config}")
        print(f"SEED: {self.seed}")
        print(f"ATTN_LATENT_LEN: {self.attn_latent_len}")
        print(f"SEQ_LENGTH: {self.seq_length}")
        print(f"VISION_ENCODER: {self.vision_encoder.value}")
        print(f"LAUNCH_LOCATION: {self.launch_location.value}")
        print(f"USE_TOY_DATASET: {self.use_toy_dataset}")
        if self.use_toy_dataset:
            print(f"TASK_ID: {self.task_id}")
            print(f"N_DISTRACTORS: {self.n_distractors}")
            print(f"TRAIN_DATASET_SIZE: {self.train_dataset_size}")
        print(f"K_PTP: {self.k_ptp}")  # Add k_ptp to config print
        print(f"WANDB_PROJECT_NAME: {self.wandb_project_name}")  # Add wandb project name to config print
        print(f"ACCUM_ITER: {self.accum_iter}")
        print(f"NUM_REPEAT_TRAJ: {self.num_repeat_traj}")
        print(f"TASK_CONFIG: {self.task_config}")
        print(f"GBS: {self.gbs}")
        print(f"PORT_NUM: {self.port_num}")
        print(f"N_GRADIENTS_STEPS: {self.n_gradients_steps}")
        print(f"exp_name: {self.exp_name}")
        print(f"exp_base_dir: {self.exp_base_dir}")
        print(f"EXTRA_FLAGS: {' '.join(self.extra_flags)}")
        print("=" * 50)

    def run_training(self, dry_run: bool = False):
        """Run the training command."""
        self.print_config()

        if self.launch_location != LaunchLocation.LOCAL:
            raise ValueError(f"Unknown launch location: {self.launch_location}")
        self._run_local(dry_run)

    def _run_local(self, dry_run: bool = False):
        """Run training locally."""
        # Set CUDA_VISIBLE_DEVICES
        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

        # Get the command
        cmd = self.get_torchrun_command()

        print(f"\nExecuting command locally:")
        print(" ".join(cmd))
        print()

        if dry_run:
            print("DRY RUN - Command not executed")
            return

        # Execute the command with proper environment inheritance
        try:
            # Use shell=True to ensure proper environment inheritance
            subprocess.run(" ".join(cmd), shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Training failed with exit code {e.returncode}")
            sys.exit(1)
        except KeyboardInterrupt:
            print("Training interrupted by user")
            sys.exit(1)

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Pythonic training script launcher for long-term-mem",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples (see RUNS_HALO.md for full HALO finals commands):

    # Dry run (washandreturn)
    python run_trainer.py -ds 8 -bs 8 -ng 1 -mc libero_1_5x_small.json -dc task_robocasa_mem_washandreturn.json -s 1 -sl 512 -ll local --dry-run
    """
    )

    # Required positional arguments
    parser.add_argument("-ds", "--downsample_obs", type=int, help="Downsample observation factor")
    parser.add_argument("-bs", "--batch_size", type=int, help="Batch size")
    parser.add_argument("-ng", "--num_gpus", type=int, help="Number of GPUs")
    parser.add_argument("-nw", "--num_workers", type=int, help="Number of workers per gpu", default=64)
    parser.add_argument("-mc", "--model_config", type=str, default="libero_1_5x_small.json", help="Model configuration file (default: libero_1_5x_small.json)")
    parser.add_argument("-dc", "--data_config", type=str, nargs="+", help="Data configuration file")
    parser.add_argument("-t", "--task", type=str, default=None, choices=sorted(TASK_PRESETS.keys()),
                        help="Task preset: sets sequence length, task+qa data configs, and state-supervision "
                             "loss coefficient (and enables GPT state supervision). Explicitly-passed "
                             "-sl/-dc/-ss-coeff flags override the preset.")
    parser.add_argument("-vdc", "--val_data_config", type=str, nargs="+", help="Validation data configuration file", default=None)
    parser.add_argument("-wep", "--warmup-epochs", type=int, help="Number of epochs for warmup", default=2)
    parser.add_argument("-ep", "--epochs", type=int, help="Number of epochs to train", default=200)
    parser.add_argument("-br", "--break-after-n-epochs", type=int, help="Stop training after N epochs", default=None)
    parser.add_argument("-wd", "--weight-by-dataset", type=int, nargs="+", help="Data configuration file", default=[1])
    parser.add_argument("-ss", "--state-supervision", action="store_true", help="Add state supervision to the dataset class that samples actions")
    # bbox: bbox as float will be returned (with ss set to True), bbox_str: bbox as string with query, bbox_inst_str: bbox as string with instruction
    parser.add_argument("-ss-mode", "--ss-mode", type=str, default="bbox_str", choices=["bbox", "bbox_str", "bbox_inst_str", "time"], help="State supervision mode (default: bbox_str)")
    parser.add_argument("--ss-create-mode", type=str, default="inst_generic", choices=["inst_generic", "inst_specific", "time"], help="State supervision creation mode (default: inst_generic)")
    parser.add_argument("-gptss", "--add-gpt-state-supervision", action="store_true", help="Add GPT state supervision")
    parser.add_argument("-fakess", "--add-fake-state-supervision", action="store_true", help="Add fake state supervision")
    parser.add_argument("-ss-coeff", "--coeff-state-supervision-loss", type=float, default=1.0, help="Coefficient for state supervision loss (default: 1.0)")
    # add a phase argument
    parser.add_argument("-ph", "--phase", type=str, help="Phase of training", default="pretrain", choices=["pretrain", "pretrain_lora", "finetune"])

    # Optional arguments
    parser.add_argument("-s", "--seed", type=int, default=1, help="Random seed (default: 1)")
    parser.add_argument("--gbs-factor", type=float, default=1.0, help="Calculated global batch size will be downsampled by this factor (default: 1)")
    parser.add_argument("--repeat-traj-factor", type=float, default=1.0, help="Number of repeat trajectories will be multiplied by this factor (default: 1)")
    parser.add_argument("--attn-latent-len", type=int, default=1, help="Number of attention latent tokens per image (default: 1)")
    parser.add_argument("-sl", "--seq-length", type=int, default=4, help="Sequence length (default: 4)")
    parser.add_argument("-ve", "--vision-encoder", type=VisionEncoder, default=VisionEncoder.CROSSMAE,
                       choices=list(VisionEncoder), help="Vision encoder to use")
    parser.add_argument("-ll", "--launch-location", type=LaunchLocation, default=LaunchLocation.LOCAL,
                       choices=list(LaunchLocation), help="Where to launch the job (only 'local' is supported)")
    parser.add_argument("--dry-run", action="store_true", help="Print command without executing")
    parser.add_argument("--basic-run", action="store_true", help="Use single-GPU training with python instead of torchrun (num_workers=0)")
    parser.add_argument("--action-head", type=str, default="mlp", choices=["mlp", "fm"], help="Action head type (default: mlp)")

    # block-attn specific parameters
    parser.add_argument('-bai', "--block-attn-ind", type=int, default=None, help="Block attention indices", nargs="+")
    parser.add_argument("--block-chunk-ts-len", type=int, default=8, help="Block chunk timesteps length")
    # local-attn specific parameters
    parser.add_argument('-lai', "--local-attn-ind", type=int, default=None, help="Local attention indices", nargs="+")
    # strided-attn specific parameters
    parser.add_argument('-sai', "--strided-attn-ind", type=int, default=None, help="Strided attention indices", nargs="+")
    # gated-attn specific parameters
    parser.add_argument('-gai', "--gated-attn-idx", type=int, default=None, help="Gated attention indices", nargs="+")
    parser.add_argument('-tmai', "--tokme-attn-idx", type=int, default=None, help="TokenMerge attention layer indices", nargs="+")
    # TopK attention specific arguments
    parser.add_argument("--use-topk-attn", action="store_true", help="Use TopK attention")
    parser.add_argument("--ret-topk", type=int, default=8, help="TopK for retrieval")
    parser.add_argument("--ret-topk-max", type=int, default=None, help="Maximum TopK for curriculum learning (None = no curriculum)")
    parser.add_argument("--ret-chunk-ts-len", type=int, default=8, help="Retrieval chunk timesteps length")
    parser.add_argument("--ret-n-topk-blocks", type=int, default=2, help="Number of TopK blocks")
    parser.add_argument("--ret-recursions", type=int, default=1, help="Number of recursions for TopK attention")
    parser.add_argument("--ret-multikv", action="store_true", help="Use multi-key-value attention for TopK")
    parser.add_argument("--add-time", action="store_true", help="Add time awareness to TopK attention")
    parser.add_argument(
        "-timerel",
        "--ret-time-relative",
        action="store_true",
        help="With --add-time: encode retrieval time as lag (bank_index - query_pos). Default: off; "
    )
    parser.add_argument("-rai", "--ret-topk-attn-idx", type=int, default=None, help="TopK attention indices", nargs="+")

    # Toy dataset specific arguments
    parser.add_argument("--use-toy-dataset", action="store_true", help="Use toy vision dataset instead of regular dataset")
    parser.add_argument("--task-id", type=int, help="Task ID for toy dataset (required when using toy dataset)")
    parser.add_argument("--n-distractors", type=int, help="Number of distractors for toy dataset (required when using toy dataset)")
    parser.add_argument("--train-dataset-size", type=str, default="10_000_000", help="Training dataset size for toy dataset (default: 10_000_000)")


    parser.add_argument("--load-in-mem", action="store_true", help="Load dataset in memory")
    parser.add_argument("--compile-model", action="store_true", help="Compile model")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint (default: last)")
    parser.add_argument("--resume-new-exp", action="store_true", help="Resume with new experiment (default: False)")

    # k_ptp parameter
    parser.add_argument("--k-ptp", type=int, default=0, help="k_ptp parameter for multi-step prediction (default: 0)")

    # num_pred_steps parameter
    parser.add_argument("-nps", "--num-pred-steps", type=int, default=32, help="Number of prediction steps (default: 32)")

    # Learning rate
    parser.add_argument("-lr", type=float, default=None, help="Learning rate (default: None, uses default from optimizer config)")

    # Hyperparameter tuning
    parser.add_argument("--hyper-tune", action="store_true", help="Enable hyperparameter tuning mode (adds lr to experiment name)")

    # Total number of gradient steps (overrides epochs if > 0)
    parser.add_argument("-ngs", "--n-gradients-steps", type=int, default=-1, help="Total number of gradient steps (default: -1, disabled)")
    parser.add_argument("--test-runs", type=str, default="none", choices=["none", "generate", "verify"], help="Bit-exact regression mode: generate saves a 10-step artifact, verify compares against it")

    # Wandb configuration
    parser.add_argument("--wandb-project-name", type=str, help="Wandb project name (defaults to experiment type: toy_vision or icrt_litev2)")
    parser.add_argument("--wandb-group", type=str, default=None, help="Wandb group name for grouping runs (default: None)")

    # add the option to specify the exp_base_dir
    parser.add_argument("--exp-base-dir", type=str, default=None, help="Base directory for experiments (default: None)")

    # evaluation related parameters
    parser.add_argument("--eval-every", type=int, default=-2, help="Evaluate every n epochs (default: 25)")
    parser.add_argument("--eval-at-zero", action="store_true", help="Evaluate at zero (default: False)")
    parser.add_argument("--n-eval", type=int, default=24, help="Number of evaluations (default: 24)")

    # QA dataset specific parameters
    parser.add_argument("--qa-remove-query", action="store_true", help="Remove query from QA dataset (default: False)")
    parser.add_argument("--max-ss-size", type=int, default=-1, help="Max number of samples for GPT state supervision (default: -1 for no maxout)")

    parser.add_argument("--og-inst", action="store_true", help="Use original (non-generated) instructions (default: False)")

    # add a flag called temp which is boolean
    parser.add_argument("--temp", action="store_true", help="Use temperature for the state supervision (default: False)")

    args = parser.parse_args()

    # Apply a task preset (if any). Explicitly-passed CLI flags override preset values.
    if args.task is not None:
        preset = TASK_PRESETS[args.task]
        passed = set(sys.argv[1:])
        def _passed(*flags):
            return any(f in passed for f in flags)
        if not _passed("-sl", "--seq-length", "--seq_length"):
            args.seq_length = preset["seq_length"]
        if not _passed("-dc", "--data_config", "--data-config"):
            args.data_config = list(preset["data_config"])
        if not _passed("-ss-coeff", "--coeff-state-supervision-loss"):
            args.coeff_state_supervision_loss = preset["coeff_state_supervision_loss"]
        if not _passed("-bai", "--block-attn-ind"):
            args.block_attn_ind = list(preset["block_attn_ind"])
        if not _passed("-rai", "--ret-topk-attn-idx"):
            args.ret_topk_attn_idx = list(preset["ret_topk_attn_idx"])
        # The qa configs require GPT state supervision (bbox_str mode); enable it
        # unless the user has already configured state supervision explicitly.
        if not _passed("-gptss", "--add-gpt-state-supervision"):
            args.add_gpt_state_supervision = True
        if not _passed("-ss-mode", "--ss-mode"):
            args.ss_mode = "bbox_str"

    if args.temp:
        # set the exp_base_dir to temp
        args.exp_base_dir = "temp"
        args.wandb_project_name = "temp"
        args.launch_location = LaunchLocation.LOCAL

    # Create training configuration
    config = TrainingConfig(
        downsample_obs=args.downsample_obs,
        batch_size=args.batch_size,
        num_gpus=args.num_gpus,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        break_after_n_epochs=args.break_after_n_epochs,
        num_workers=args.num_workers*args.num_gpus,
        model_config=args.model_config,
        data_config=args.data_config,
        task_val_config=args.val_data_config,
        weight_by_dataset=args.weight_by_dataset,
        seed=args.seed,
        attn_latent_len=args.attn_latent_len,
        gbs_factor=args.gbs_factor,
        repeat_traj_factor=args.repeat_traj_factor,
        seq_length=args.seq_length,
        vision_encoder=args.vision_encoder,
        launch_location=args.launch_location,
        basic_run=args.basic_run,  # Pass basic_run parameter
        phase=args.phase,  # Pass phase parameter
        use_toy_dataset=args.use_toy_dataset,
        task_id=args.task_id,
        n_distractors=args.n_distractors,
        train_dataset_size=args.train_dataset_size,
        k_ptp=args.k_ptp,  # Pass k_ptp parameter
        num_pred_steps=args.num_pred_steps,  # Pass num_pred_steps parameter
        wandb_project_name=args.wandb_project_name,  # Pass wandb project name
        load_in_mem=args.load_in_mem,
        compile_model=args.compile_model,  # Pass compile_model parameter
        resume=args.resume,  # Pass resume parameter
        resume_new_exp=args.resume_new_exp,  # Pass resume_new_exp parameter
        action_head=args.action_head,  # Pass action_head parameter
        exp_base_dir=args.exp_base_dir,  # Pass exp_base_dir parameter
        eval_every=args.eval_every,  # Pass eval_every parameter
        eval_at_zero=args.eval_at_zero,  # Pass eval_at_zero parameter
        n_eval=args.n_eval,  # Pass n_eval parameter
        add_state_supervision=args.state_supervision,  # Pass add_state_supervision parameter
        state_supervision_mode=args.ss_mode,  # Pass state_supervision_mode parameter
        ss_create_mode=args.ss_create_mode,  # Pass ss_create_mode parameter
        add_gpt_state_supervision=args.add_gpt_state_supervision,  # Pass add_gpt_state_supervision parameter
        add_fake_state_supervision=args.add_fake_state_supervision,  # Pass add_fake_state_supervision parameter
        coeff_state_supervision_loss=args.coeff_state_supervision_loss,  # Pass coeff_state_supervision_loss parameter
        # block-attn specific parameters
        block_attn_ind=args.block_attn_ind,  # Pass block_attn_ind parameter
        block_chunk_ts_len=args.block_chunk_ts_len,  # Pass block_chunk_ts_len parameter
        # local-attn specific parameters
        local_attn_ind=args.local_attn_ind,  # Pass local_attn_ind parameter
        # strided-attn specific parameters
        strided_attn_ind=args.strided_attn_ind,  # Pass strided_attn_ind parameter
        # gated-attn specific parameters
        gated_attn_idx=args.gated_attn_idx,  # Pass gated_attn_idx parameter
        # tokme-attn specific parameters
        tokme_attn_idx=args.tokme_attn_idx,  # Pass tokme_attn_idx parameter
        # TopK attention specific parameters
        use_topk_attention=args.use_topk_attn,  # Pass use_topk_attention parameter
        ret_topk=args.ret_topk,  # Pass ret_topk parameter
        ret_topk_max=args.ret_topk_max,  # Pass ret_topk_max parameter
        ret_chunk_len=args.ret_chunk_ts_len,  # Pass ret_chunk_len parameter
        ret_n_topk_blocks=args.ret_n_topk_blocks,  # Pass ret_n_topk_blocks parameter
        ret_recursions=args.ret_recursions,  # Pass ret_recursions parameter
        ret_multikv=args.ret_multikv,  # Pass ret_multikv parameter
        ret_add_time_aware=args.add_time,  # Pass ret_add_time_aware parameter
        ret_relative_time=args.ret_time_relative,
        ret_topk_attn_idx=args.ret_topk_attn_idx,  # Pass ret_topk_attn_idx parameter
        qa_remove_query=args.qa_remove_query,  # Pass qa_remove_query parameter
        max_ss_size=args.max_ss_size,  # Pass max_ss_size parameter
        use_og_inst=args.og_inst,  # Pass use_og_inst parameter
        lr=args.lr,  # Pass lr parameter
        hyper_tune=args.hyper_tune,  # Pass hyper_tune parameter
        wandb_group=args.wandb_group,  # Pass wandb_group parameter
        n_gradients_steps=args.n_gradients_steps,  # Pass n_gradients_steps parameter
        test_runs=args.test_runs,
    )

    config.run_training(dry_run=args.dry_run)

if __name__ == "__main__":
    main()
