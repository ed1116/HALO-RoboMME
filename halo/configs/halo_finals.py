"""Per-task config classes mirroring the HALO finals commands in docs/audit/RUNS_HALO.md.

Each task config captures the exact arguments used for that task. Building a config produces a
``TrainingConfig`` (from ``run_trainer.py``) ready to be launched, and ``cli_argv()`` regenerates
the equivalent ``run_trainer.py`` command line.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import ClassVar, Dict, List, Type


@dataclass
class HaloFinalsBaseConfig:
    """Shared HALO finals settings.

    Defaults match the arguments common to all four task commands in RUNS_HALO.md:
    ``-ds 8 -bs 8 -ng 1 -mc libero_1_5x_small.json -nw 20 -s 1 -ll local -bai 1 3 4 5 -rai 0 2
    -gptss -ss-mode bbox_str --compile-model -br 50 --max-ss-size 16 --temp`` plus
    ``--exp-base-dir halo_finals --wandb-project-name tokenized_input``.
    """

    # Per-task — must be set by subclasses
    task_name: ClassVar[str] = ""
    data_config: List[str] = field(default_factory=list)
    seq_length: int = 512
    coeff_state_supervision_loss: float = 1.0

    # Shared training arguments
    downsample_obs: int = 8
    batch_size: int = 8
    num_gpus: int = 1
    num_workers: int = 20
    seed: int = 1
    model_config: str = "libero_1_5x_small.json"
    launch_location: str = "local"

    # State supervision (-gptss -ss-mode bbox_str --max-ss-size 16)
    add_gpt_state_supervision: bool = True
    state_supervision_mode: str = "bbox_str"
    max_ss_size: int = 16

    # Block / TopK attention layout (-bai 1 3 4 5 -rai 0 2)
    block_attn_ind: List[int] = field(default_factory=lambda: [1, 3, 4, 5])
    ret_topk_attn_idx: List[int] = field(default_factory=lambda: [0, 2])

    # Misc training flags
    compile_model: bool = True
    break_after_n_epochs: int = 50

    # All four RUNS_HALO commands pass --temp, which inside run_trainer.py overrides
    # exp_base_dir → "temp" and wandb_project_name → "temp". The explicit values below
    # are kept for ``cli_argv()`` parity with the original commands; set ``temp=False``
    # to actually write to ``halo_finals`` / ``tokenized_input``.
    temp: bool = True
    exp_base_dir: str = "halo_finals"
    wandb_project_name: str = "tokenized_input"

    def to_training_config(self):
        """Build a ``TrainingConfig`` from this preset."""
        # Local import: run_trainer.py lives at the repo root and imports the world.
        from run_trainer import LaunchLocation, TrainingConfig, VisionEncoder

        if self.temp:
            exp_base_dir = "temp"
            wandb_project_name = "temp"
            launch_location = LaunchLocation.LOCAL
        else:
            exp_base_dir = self.exp_base_dir
            wandb_project_name = self.wandb_project_name
            launch_location = LaunchLocation(self.launch_location)

        return TrainingConfig(
            downsample_obs=self.downsample_obs,
            batch_size=self.batch_size,
            num_gpus=self.num_gpus,
            num_workers=self.num_workers * self.num_gpus,
            epochs=200,
            warmup_epochs=2,
            model_config=self.model_config,
            data_config=list(self.data_config),
            seed=self.seed,
            seq_length=self.seq_length,
            vision_encoder=VisionEncoder.CROSSMAE,
            launch_location=launch_location,
            compile_model=self.compile_model,
            break_after_n_epochs=self.break_after_n_epochs,
            block_attn_ind=list(self.block_attn_ind),
            ret_topk_attn_idx=list(self.ret_topk_attn_idx),
            add_gpt_state_supervision=self.add_gpt_state_supervision,
            state_supervision_mode=self.state_supervision_mode,
            max_ss_size=self.max_ss_size,
            coeff_state_supervision_loss=self.coeff_state_supervision_loss,
            exp_base_dir=exp_base_dir,
            wandb_project_name=wandb_project_name,
        )

    def cli_argv(self) -> List[str]:
        """Regenerate the ``run_trainer.py`` arguments equivalent to this config."""
        argv = [
            "-ds", str(self.downsample_obs),
            "-bs", str(self.batch_size),
            "-ng", str(self.num_gpus),
            "-mc", self.model_config,
            "-dc", *self.data_config,
            "-s", str(self.seed),
            "-sl", str(self.seq_length),
            "-ll", self.launch_location,
            "--exp-base-dir", self.exp_base_dir,
            "--wandb-project-name", self.wandb_project_name,
            "-nw", str(self.num_workers),
            "-bai", *map(str, self.block_attn_ind),
            "-rai", *map(str, self.ret_topk_attn_idx),
        ]
        if self.add_gpt_state_supervision:
            argv += ["-gptss", "-ss-mode", self.state_supervision_mode]
        if self.compile_model:
            argv.append("--compile-model")
        argv += ["-br", str(self.break_after_n_epochs)]
        if self.coeff_state_supervision_loss != 1.0:
            argv += ["-ss-coeff", str(self.coeff_state_supervision_loss)]
        argv += ["--max-ss-size", str(self.max_ss_size)]
        if self.temp:
            argv.append("--temp")
        return argv


@dataclass
class RetrieveOilConfig(HaloFinalsBaseConfig):
    task_name: ClassVar[str] = "retrieve_oil"
    data_config: List[str] = field(default_factory=lambda: [
        "task_robocasa_mem_retrieve_oil.json",
        "qa_configs/qa_robocasa_mem_retrieve_oil_relevant_objs_gptstate.json",
    ])
    seq_length: int = 512
    # Note: the RUNS_HALO command for retrieve_oil omits -ss-coeff, so it uses the default 1.0.
    coeff_state_supervision_loss: float = 1.0


@dataclass
class WashAndReturnConfig(HaloFinalsBaseConfig):
    task_name: ClassVar[str] = "washandreturn"
    data_config: List[str] = field(default_factory=lambda: [
        "task_robocasa_mem_washandreturn.json",
        "qa_configs/qa_robocasa_mem_washandreturn_relevant_objs_gptstate.json",
    ])
    seq_length: int = 512
    coeff_state_supervision_loss: float = 0.5


@dataclass
class KbreadsConfig(HaloFinalsBaseConfig):
    task_name: ClassVar[str] = "kbreads"
    data_config: List[str] = field(default_factory=lambda: [
        "task_robocasa_mem_kbreads.json",
        "qa_configs/qa_robocasa_mem_kbreads_relevant_objs_gptstate.json",
    ])
    seq_length: int = 2048
    coeff_state_supervision_loss: float = 0.5


@dataclass
class HeatpotConfig(HaloFinalsBaseConfig):
    task_name: ClassVar[str] = "heatpot"
    data_config: List[str] = field(default_factory=lambda: [
        "task_robocasa_mem_heatpot.json",
        "qa_configs/qa_robocasa_mem_heatpot_relevant_objs_gptstate.json",
    ])
    seq_length: int = 2048
    coeff_state_supervision_loss: float = 0.5


HALO_FINALS_CONFIGS: Dict[str, Type[HaloFinalsBaseConfig]] = {
    RetrieveOilConfig.task_name: RetrieveOilConfig,
    WashAndReturnConfig.task_name: WashAndReturnConfig,
    KbreadsConfig.task_name: KbreadsConfig,
    HeatpotConfig.task_name: HeatpotConfig,
}


def get_config(task_name: str, **overrides) -> HaloFinalsBaseConfig:
    """Look up a HALO finals config by task name (e.g. ``"washandreturn"``).

    Extra kwargs are applied as overrides on the dataclass before returning it.
    """
    if task_name not in HALO_FINALS_CONFIGS:
        raise KeyError(
            f"Unknown HALO finals task '{task_name}'. "
            f"Available: {sorted(HALO_FINALS_CONFIGS)}"
        )
    cls = HALO_FINALS_CONFIGS[task_name]
    valid_fields = {f.name for f in fields(cls)}
    bad = set(overrides) - valid_fields
    if bad:
        raise TypeError(f"Unknown override(s) for {cls.__name__}: {sorted(bad)}")
    return cls(**overrides)
