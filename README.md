<div align="center">

# Memory Retrieval in Visuomotor Policies for Long&#8209;Horizon&nbsp;Robot&nbsp;Control

[Rutav Shah](#)<sup>†</sup>, [Yisu Li](#)<sup>\*</sup>, [Femi Bello](#)<sup>\*</sup>, [Yuke Zhu](#), [Roberto Martín-Martín](#)

The University of Texas at Austin

<sup>\*</sup> Equal contribution &nbsp;&nbsp; <sup>†</sup> Corresponding author

**Robotics: Science and Systems (RSS) 2026**

[Project Page](https://robin-lab.cs.utexas.edu/HALO) &nbsp;|&nbsp; [Dataset](https://huggingface.co/datasets/Rutav/ReMemBench-Dataset) &nbsp;|&nbsp; [BibTeX](#bibtex)

<img src="assets/github/halo.png" alt="HALO method overview" width="90%"/>

*HALO is a visuomotor policy with attention-based memory retrieval for long-horizon robot control.
It distills VLM priors via video question-answering and uses Top-K sparse attention to retrieve
task-relevant information from up to eight minutes of past experience.*

</div>

---

## Overview

General-purpose robots operating in partially observable environments (e.g., homes) must recall
diverse information from the past (where objects were placed, which subtasks a partner has
completed, when an appliance was turned on) to accomplish long-horizon tasks. HALO addresses two
challenges that arise when training long-context transformers by imitation learning:

1. **Spurious correlations from history**, mitigated by distilling VLM priors through a video
   question-answering objective trained jointly with imitation learning.
2. **Compounding errors during closed-loop control**, mitigated by Top-K sparse attention that
   restricts retrieval to the most relevant parts of the history.

## Installation

1. **Set up environment variables** (required before all subsequent steps):
```bash
export CASAPLAY_DATAROOT=/path/to/data            # Where datasets and assets live
export EXP_STORAGE_BASE_DIR=/path/to/experiments  # Where training experiments are saved
export WANDB_API_KEY=<your_wandb_api_key>         # For logging
```

2. **Clone required repositories:**
```bash
git clone --branch=latest git@github.com:ShahRutav/ReMemBench.git ReMemBench
git clone --branch=abs_robot git@github.com:ShahRutav/robosuite.git robosuite
git clone --branch=main    git@github.com:UT-Austin-RobIn/HALO.git halo
```

3. **Create the conda environment:**
```bash
conda create -n longmem python=3.10 pip -y
conda activate longmem
```

4. **Install PyTorch with CUDA support:**
```bash
pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.9.1 torchvision==0.24.1
```

5. **Install repositories in editable mode:**
```bash
pip install -e ReMemBench
pip install -e robosuite
# Pin mujoco after robosuite to satisfy robocasa's strict version requirement
pip install mujoco==3.2.6
pip install -e halo
# Fix numba/numpy conflict (robocasa pins numba==0.56.4 which doesn't support numpy>=1.24)
pip install "numba==0.57.1" "numpy==1.23.5"
```
6. **Download kitchen assets** (required for evaluation):  
```bash
cd ReMemBench && python robocasa/scripts/download_kitchen_assets.py && cd ..
```

7. **Download the pretrained vision encoder:**
```bash
mkdir -p $CASAPLAY_DATAROOT/crossmae_rtx
wget https://huggingface.co/mlfu7/ICRT/resolve/main/crossmae_rtx/cross-mae-rtx-vitb.pth \
     -O $CASAPLAY_DATAROOT/crossmae_rtx/cross-mae-rtx-vitb.pth
```

8. **Download the dataset:**
```bash
# Main dataset (demonstrations and assets)
hf download Rutav/ReMemBench-Dataset \
    --repo-type dataset \
    --local-dir $CASAPLAY_DATAROOT/memory

# Task queries and generated QA (merged into the same directory)
hf download Rutav/HALO_QAs \
    --repo-type dataset \
    --local-dir $CASAPLAY_DATAROOT/memory
```
The second download restores the `<task>/<date>/task_queries/` and
`<task>/<date>/generated_qa/qa.json` files into the dataset directory, matching the
structure expected by the configs.

## Training

Training is launched through `run_trainer.py`, which composes the model, dataset, and optimizer
configs and calls `torchrun` (or generates a SLURM script for cluster launches).

Each task has a preset (`--task`) that sets the sequence length, the task and QA data
configs, the state-supervision loss coefficient, and the Top-K sparse attention layout.
Any explicitly-passed flag (e.g. `-sl`, `-dc`) overrides the preset.

**WashAndReturn** (single GPU, local launch):
```bash
python run_trainer.py -ds 8 -bs 8 -ng 1 --task washandreturn -ll local --exp-base-dir all_rw -nw 16 -br 51 --compile-model
```

**RetrieveOil:**
```bash
python run_trainer.py -ds 8 -bs 8 -ng 1 --task retrieve_oil -ll local --exp-base-dir all_rw -nw 16 -br 51 --compile-model
```

**HeatPot:**
```bash
python run_trainer.py -ds 8 -bs 8 -ng 1 --task heatpot -ll local --exp-base-dir all_rw -nw 16 -br 51 --compile-model
```

**KBreads:**
```bash
python run_trainer.py -ds 8 -bs 8 -ng 1 --task kbreads -ll local --exp-base-dir all_rw -nw 16 -br 51 --compile-model
```

**Common flags:**

| Flag | Meaning |
| --- | --- |
| `-t` / `--task` | Task preset (`washandreturn`, `retrieve_oil`, `heatpot`, `kbreads`) |
| `-ds` | Observation downsample factor |
| `-bs` | Per-GPU batch size |
| `-ng` | Number of GPUs |
| `-mc` | Model config (under `config/model/`), defaults to `libero_1_5x_small.json` |
| `-dc` | Data config (under `config/task/`), supports multiple; overrides the preset |
| `-sl` | Sequence length (context window in tokens); overrides the preset |
| `-s`  | Random seed (default: 1) |
| `-ll` | Launch location (only `local` is supported) |
| `--compile-model` | Compile the model with `torch.compile` |
| `--use-topk-attn` `--ret-topk` `--ret-chunk-ts-len` | Top-K sparse attention |
| `--dry-run` | Print the command/SLURM script without executing |

Run `python run_trainer.py -h` for the full argument list.

## Evaluation

Evaluation is driven by `shell/eval.sh`, which dispatches `scripts/eval.py` across GPUs and tasks.

```bash
bash shell/eval.sh
```

Edit the `exp_ckpt_dirs` array at the top of `shell/eval.sh` to point at your trained
checkpoint. Each entry is `"<exp_dir> <ckpt_num>"` (use `-1` for the latest checkpoint).
The default `gpu_list=(0 1 2 3 4 5 6 7)` and `exps_in_parallel=1` settings can be tuned
for your machine.

## BibTeX

If you find this work useful, please cite:

```bibtex
@inproceedings{shah2026halo,
  title={Memory Retrieval in Visuomotor Policies for Long-Horizon Robot Control},
  author={Shah, Rutav and Li, Yisu and Bello, Femi and Zhu, Yuke and Mart{\'{i}}n-Mart{\'{i}}n, Roberto},
  booktitle={Proceedings of Robotics: Science and Systems},
  year={2026}
}
```

## License

See [LICENSE.txt](LICENSE.txt).
