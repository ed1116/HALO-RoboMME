import os
import dataclasses
from dataclasses import field
from typing import Literal, Optional, Tuple, Union, List

import tyro

@dataclasses.dataclass
class DatasetConfig:
    # Dataset config path
    dataset_json : Union[str, list[str]]

    # Dataset validation config path
    dataset_val_json : Union[str, list[str]] = ""

    # Add language tokens
    add_language_tokens : bool = True

    # variance of the noise added to proprioception
    proprio_noise : float = 0.005

    # variance of the noise added to action
    action_noise : float = 0.0

    # add vision data aug
    vision_aug : bool = True

    # each batch contains data that is non overlapping (i.e. for each epoch the same state action does not appear twice)
    non_overlapping : Union[bool, int, list[int]] = 2

    # enable repeats of trajectory
    num_repeat_traj : Union[int, list[int]] = 2

    # weight the number of samples in each batch size from each dataset_json
    weight_by_dataset : Union[int, list[int]] = 1

    # load the complete dataset in memory
    load_in_mem: bool = False

    # share the RAM-loaded HDF5 data across VL and QA datasets that reference the same files
    share_ram: bool = True

    # use DALI for data loading
    use_dali: bool = False

    random_patch_masking: bool = False

    n_examples_only: int = -1

    train_on_exploration: bool = False

    # sample each state
    sample_each_state: bool = False

    # maxout the number of samples of the GPT state supervision from rel_frame_index
    max_ss_size: int = -1 # -1 means no maxout; use whatever is the length of the answer

    # for the qa dataset config,
    qa_remove_query: bool = False
    # max_qa_size for the qa dataset
    max_qa_size_post_shuffle: int = 20000

    # use original (non-generated) instructions
    use_og_inst: bool = False

    def __post_init__(self):
        # Ensure dataset_json is always stored as a list
        if isinstance(self.dataset_json, str):
            self.dataset_json = [self.dataset_json]
        if isinstance(self.dataset_val_json, str):
            self.dataset_val_json = [self.dataset_val_json]
        if isinstance(self.num_repeat_traj, int):
            self.num_repeat_traj = [self.num_repeat_traj]
        if isinstance(self.non_overlapping, int):
            self.non_overlapping = [self.non_overlapping]
        if isinstance(self.weight_by_dataset, int):
            self.weight_by_dataset = [self.weight_by_dataset]
        # max padding length for each one of them
        max_len = max(len(self.dataset_json), len(self.dataset_val_json))
        # pad the other ones to the max length
        self.dataset_json = self.dataset_json + [self.dataset_json[0]] * (max_len - len(self.dataset_json))
        self.dataset_val_json = self.dataset_val_json + [self.dataset_val_json[0]] * (max_len - len(self.dataset_val_json))
        self.num_repeat_traj = self.num_repeat_traj + [1.0] * (max_len - len(self.num_repeat_traj))
        self.non_overlapping = self.non_overlapping + [1.0] * (max_len - len(self.non_overlapping))
        self.weight_by_dataset = self.weight_by_dataset + [self.weight_by_dataset[0]] * (max_len - len(self.weight_by_dataset))

@dataclasses.dataclass
class VisionEncoderConfig:
    # vision encoder type (or path to checkpoint)
    vision_encoder : str = "vit_base_patch16_224.mae"

    # Whether to use a randomly initialized vision encoder instead of pretrained weights
    vision_nonpretrained : bool = False

    # Whether to unfreeze the vision encoder
    vision_unfreeze_all : bool = False

    # Whether to use LoRA in the vision encoder
    vision_lora : bool = False

    # Rank of LoRA layers
    vision_lora_rank : int = 8

    # Number of blocks unfrozen in the vision encoder
    vision_unfreeze_last_n : int = 0

    # Freeze all layers in the vision encoder
    freeze_all : bool = False

@dataclasses.dataclass
class PolicyConfig:
    # path to LLaMA pretrained checkpoint (only required when loading Llama3 from a checkpoint)
    llama_ckpt_dir : str = ""

    # uses different linear layers for attention pooling
    multikv_attn_pool : bool = False

    # Number of heads for adapter
    adapter_num_heads : int = 8

    # Adapter MLP ratio
    adapter_mlp_ratio : float = 4.0

    # Separate encoder adapter for different cameras
    separate_camera_adapter : bool = True

    # Phase of training
    phase : Literal["pretrain", "finetune", "pretrain_lora"] = "pretrain"

    # path to checkpoint from pretrain stage
    pretrained_path : Optional[str] = None

    # Prediction head, can be one of "mlp", "gmm", "diffusion", "ce", "fm"
    decoder_pred_head : Literal["mlp", "gmm", "diffusion", "ce", "fm"] = "mlp"
    decoder_hidden_features : int = 128
    ce_hidden_features : int = -1

    # load llama
    load_llama : bool = True

    # train llama from scratch
    scratch_llama_config : Optional[str] = "config/model/libero_1_5x_small.json"

    # only overriten in the ICRTWrapper IF tt_window_obs passed True as an argument in the wrapper
    tt_window_obs: bool = True

    ## model version; setting default to v1 for backward compatibility
    model_version: str = "v3"

    # use_lora
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1

    # block attention indices (empty by default)
    block_attn_idx: list[int] = field(default_factory=list)
    block_pattern_start_offset: int = 0
    block_chunk_ts_len: int = 8
    # memory attention indices (empty by default)
    mem_attn_idx: list[int] = field(default_factory=list)
    mem_chunk_ts_len: int = -1 # how many timesteps to process in a chunk at a time
    mem_threshold: float = 0.95

    # local attention indices (empty by default)
    local_attn_idx: list[int] = field(default_factory=list)

    # topk attention related parameters: used in ICRT
    use_topk_attention: bool = False # used to add topk attention in the mem_model
    ret_topk: int = 8
    ret_topk_max: Optional[int] = None  # Maximum topk for curriculum learning (None = no curriculum)
    ret_chunk_len: int = 8
    ret_tau_init: float = 1.0
    ret_emdr2_loss: bool = False
    ret_straight_through: bool = True
    ret_n_topk_blocks: int = 1
    ret_recursions: int = 1
    ret_multikv: bool = False
    ret_topk_attn_idx: list[int] = field(default_factory=list) # to add topk attention in the llama model
    ret_add_time_aware: bool = False
    ret_relative_time: bool = False
    ret_bank_causal: bool = False

    # strided attention related parameters
    strided_len: int = -1
    strided_attn_idx: list[int] = field(default_factory=list)
    gated_attn_idx: list[int] = field(default_factory=list)
    tokme_attn_idx: list[int] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.scratch_llama_config, str) and self.scratch_llama_config == '':
            self.scratch_llama_config = None

        assert self.model_version in ["v2", "v3"]

@dataclasses.dataclass
class ModelConfig:
    # Policy (llama + adapter) configuration
    policy_cfg : PolicyConfig

    # vision encoder config
    vision_encoder_cfg : VisionEncoderConfig


@dataclasses.dataclass
class OptimizerConfig:
    # weight decay (default: 0.01)
    weight_decay : float = 0.01

    # learning rate (absolute lr)
    lr : Optional[float] = 5e-4

    # base learning rate: absolute_lr = base_lr * total_batch_size / 256
    blr : float = 1e-3

    # lower lr bound for cyclic schedulers that hit 0
    min_lr : float = 0.0

    # epochs to warmup LR
    warmup_epochs : float = 40

@dataclasses.dataclass
class TrainerConfig:
    # number of epochs
    epochs : int = 200

    # Accumulate gradient iterations (for increasing the effective batch size under memory constraints)
    accum_iter : int = 8

    # pin memory for dataloader
    pin_memory : bool = True

    # number of workers for dataloader
    num_workers : int = 20

    # compile the model or not
    compile_model : bool = False

    # evaluate after every n epochs
    val_every : int = 1

    # log gradients, parameters, etc.
    wandb_watch : bool = True
    wandb_project : str = "memory-visuomotor-policies"
    wandb_entity : str = ""

    # break training after n epochs. This is to keep the lr scheduler to reach zero for continued training.
    break_after_n_epochs : int = -1

    # if mentioned, it wll calculate the epochs to satisfy the total number of gradient steps
    n_gradients_steps: int = -1

    # bit-exact regression mode: none|generate|verify (see halo/util/test_runs.py)
    test_runs: str = "none"

@dataclasses.dataclass
class SharedConfig:
    use_lerobot: bool = False
    # Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus
    batch_size : int = 1

    # use tokenizer dataset
    use_tokenizer_dataset: bool = True
    tokenizer_name: str = "Qwen/Qwen2-7B-Instruct"

    # Use 6DoF Rotation
    rot_6d : bool = False

    # Use 3DoF euler rotations
    rot_euler : bool = False

    # number of frames in a sequence
    seq_length : int = 512

    # seed for random number generators
    seed : int = 0

    # start epoch
    start_epoch : int = 0

    # frequency of saving checkpoint
    save_every : int = 50

    # resume from checkpoint
    resume : Optional[str] = None

    # resume with new experiment (i.e., optimizer, epoch, etc. will be reset but not the model weights)
    resume_new_exp: bool = False

    # split epoch into k different epochs
    split_epoch : int = 1

    # Number of cameras
    num_cameras : int = 2

    # Number of predicted action steps
    num_pred_steps : int = 32

    # does it have no images:
    no_img : bool = False

    # enabling/disabling gradient checkpointing
    enable_gradient_checkpointing: bool = False

    # downsampling the observations by a factor of k
    downsample_obs: int = 1

    # flag for is bimanual
    is_bimanual: bool = False

    # flag for has base action
    has_base_action: bool = False

    # use language conditioned policy
    use_language_conditioning: bool = True

    # keep only the first observation
    only_first_obs: bool = False

    # pad to max length
    pad_to_max_length: bool = True

    # k past tokens to predict per step used in the ptp paper
    k_ptp: int = 0

    # number of tokens per image
    attn_latent_len: int = 1

    # remove the action from input
    remove_action: bool = False

    # remove proprio from input
    remove_proprio: bool = False

    # use Kornia for data augmentation
    use_kornia_augmentation: bool = True

    # add state supervision
    add_state_supervision: bool = False
    # does not instantiate the state supervision inside the Dataset action class but using json file
    add_gpt_state_supervision: bool = False
    add_fake_state_supervision: bool = False
    # mode of the state supervision: 'bbox', 'bbox_str', 'bbox_inst_str'
    state_supervision_mode: Literal["bbox", "bbox_str", "bbox_inst_str"] = "bbox_str"
    # max length of the state supervision string
    max_state_supervision_len: int = 64
    coeff_state_supervision_loss: float = 1.0

    # set the compute dtype
    compute_dtype: str = "bfloat16"

    # language tokens padding
    pad_inst_tokens: bool = True
    max_inst_tokens: int = 24

    # tokenizer dropout
    tokenizer_dropout: float = 0.1

    # embeddings noise
    embeddings_noise: float = 0.02

    # mode of the state supervision creation: 'inst_generic', 'inst_specific'
    ss_create_mode: Literal["inst_generic", "inst_specific", "time"] = "inst_generic"

@dataclasses.dataclass
class LoggingConfig:
    # path where to save, empty for no saving
    output_dir: str = "./output"

    # path where to save tensorboard logs
    log_dir : Optional[str] = None

    # log name (for wandb)
    log_name : Optional[str] = None

    # wandb group name (for grouping runs)
    wandb_group : Optional[str] = None

@dataclasses.dataclass
class EvalConfig:
    # skip_n_hdf5 files
    skip_n_hdf5: int = 0

    # should we even save in wandb?
    save_wandb: bool = True

    # eval every n epochs
    eval_every: int = -2

    # eval at zero
    eval_at_zero: bool = False # for debugging

    # number of episodes to evaluate
    n_eval: int = 24

    # save videos
    save_videos: bool = True

    # save_videos in wandb
    save_videos_in_wandb: bool = False

    # render the environment
    render: bool = False

    # renderer
    renderer: str = "mjviewer"

    # camera segmentation
    camera_segmentations: str = "instance"

    # do not save the results to the csv
    no_save_csv: bool = False

    # replay prompt trajectory
    replay_prompt_traj: bool = False

    # replay training trajectory
    replay_train_traj: bool = False

    # use ground truth actions
    use_gt: bool = False

    # eval checkpoint path
    ckpt_path: Optional[str] = None

    # or we want to store the latest epoch number we are evaluating on
    latest_epoch: Optional[int] = None

    # debug mode
    debug: bool = False

    # redo evals
    redo_evals: bool = False

    # video downsample factor
    video_downsample_factor: int = 1

    # store video from every rank
    store_video_from_every_rank: bool = False

    eval_only_directory: Optional[str] = None
    eval_only_ckpt_num: Optional[int] = None

    # save as hdf5 file
    save_as_hdf5: bool = False
    hdf5_path: Optional[str] = None

    # task name
    task_name: Optional[str] = None

    # eval caching
    eval_caching: bool = False

    def __post_init__(self):
        assert not self.replay_prompt_traj, f"Replaying prompt trajectory needs some reimplementation"

@dataclasses.dataclass
class ExperimentConfig:
    # Dataset configuration
    dataset_cfg: DatasetConfig

    # Model configuration
    model_cfg: ModelConfig

    # Optimizer configuration
    optimizer_cfg: OptimizerConfig

    # Shared configuration
    shared_cfg: SharedConfig

    # Logging configuration
    logging_cfg: LoggingConfig

    # trainer configuration
    trainer_cfg: TrainerConfig

    # evaluation config
    eval_cfg: EvalConfig

    # train or eval
    train : bool = True

    # number of distributed processes (required by torch distributed)
    world_size: int = 1

    # distributed training on the optimizer (required by torch distributed)
    dist_on_itp: bool = False

    # distributed training url (required by torch distributed)
    dist_url: str = 'env://'

    # device to use for training / testing (required by torch distributed)
    device : str = "cuda"

    # load config. instead of using command line arguments, load from a config file
    load_config: Optional[str] = None

    # hyperparameter tuning mode
    hyper_tune: bool = False


@dataclasses.dataclass
class GPTDatasetConfig:
    """
    Configuration for GPT-style dataset.
    Simple dataset that reads HDF5 files and outputs variable-length sequences.
    """
    hdf5_paths: List[str]

    # get all the hdf5 file paths from the json file
    dataset_json: Union[str, List[str]]

    # base directory where the hdf5 files are stored
    base_dir: str = os.environ.get("CASAPLAY_DATAROOT")

    # Control frequency (Hz) for calculating timestep_time = timestep_index / control_freq
    control_freq: float = 20.0

    # Keys to output from HDF5 files (modalities)
    output_keys: List[str] = field(default_factory=lambda: ["actions", "observations"])

    # Whether to output full episodes or chunks
    output_full_episode: bool = False

    # If output_full_episode is False, minimum chunk length to sample
    # Actual chunk length will be sampled between min_chunk_length and trajectory length
    min_chunk_length: int = 64
    # Whether to use the same chunk length for all chunks
    use_same_chunk_length: bool = False
    # maximum dataset size of the generated dataset per hdf5 file
    max_dataset_size: int = 1000
    # number of times to repeat the trajectory
    num_repeat_traj: int = 1

@dataclasses.dataclass
class GenerateQAGenericArgs:

    # dataset configuration
    dataset_config: GPTDatasetConfig

    # shared configuration also used for the dataset
    shared_config: SharedConfig

    # maximum number of timesteps to generate queries for
    max_timesteps: int = 8

    # number of queries to generate
    n_queries: int = 20

    # number of retries to generate queries
    n_retries: int = 3

    # save path for the dataset
    save_path: str = ""

    # model name
    model_name: str = "gpt-4o-mini"

    # seed for the random number generator
    seed: int = 0

    # mode of the data generation: generic or task-specific
    mode: Literal["generic", "ts", "ts_text_only_bbox", "ts_text_only_bbox_v2", "ts_text_only_time", "ts_video"] = "generic"

    debug: bool = False

    prefix: str = ""

    resume: bool = False
    batch_mode: bool = False

    # skip the second VLM verification/filtering call; all generated QA pairs are kept as-is
    skip_verification: bool = False

    # if True, generate an HTML visualization for each processed data point
    visualize: bool = False

    # directory to write visualizations; defaults to <save_path_dir>/viz
    visualize_dir: str = ""

    def __post_init__(self):
        if self.save_path == "":
            self.save_path = self.hdf5_path_to_qa_json_path(
                hdf5_path=self.dataset_config.hdf5_paths[0],
                mode=self.mode,
                model_name=self.model_name,
                max_dataset_size=self.dataset_config.max_dataset_size,
                prefix=self.prefix,
            )

    @staticmethod
    def hdf5_path_to_qa_json_path(
            hdf5_path: str,
            mode: str,
            model_name: str,
            max_dataset_size: int,
            prefix: str = "",
        ):
        base_dir = os.path.dirname(hdf5_path)
        base_name = os.path.basename(hdf5_path).replace('.hdf5', '')
        if (prefix != "") and (not prefix.startswith("_")):
            prefix = "_" + prefix
        if mode == "generic":
            return os.path.join(base_dir, "generated_qa", f"{base_name}_{model_name}_results_{max_dataset_size}{prefix}.json")
        elif mode == "ts" or mode == "ts_text_only_bbox" or mode == "ts_text_only_bbox_v2" or mode == "ts_text_only_time" or mode == "ts_video":
            return os.path.join(base_dir, "generated_qa", f"{base_name}_{model_name}_results_{max_dataset_size}{prefix}_{mode}.json")
        else:
            raise ValueError(f"Invalid mode: {mode}")
        return None

if __name__ == "__main__":
    args = tyro.cli(ExperimentConfig)
    dict_args = dataclasses.asdict(args)
