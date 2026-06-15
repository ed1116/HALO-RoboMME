# Remote install: `rutavms-4XL40S-V0`

Reproduces the long-term-mem environment on the L40S SkyPilot box using **uv** (no conda).

## Box facts
- Host: `rutavms-4XL40S-V0` (SSH alias, root user)
- 4× NVIDIA L40S, driver 580.95.05, CUDA runtime 13.0
- No `nvcc` / CUDA toolkit on the host — anything that needs to compile against CUDA must use prebuilt wheels
- `uv` 0.11.8 already at `/usr/bin/uv`
- Code lives at `/root/long-term-mem`; sibling repos at `/root/robosuite`, `/root/robocasa`

## tmux layout
Session `l40-dev` has three windows:
- `0:` bash (idle, `~/lab42_vr`)
- `1:` bbox (idle, `~/video_data_annotation`)
- `2:` longmem-install (this work)

Attach with `ssh rutavms-4XL40S-V0 -t tmux attach -t l40-dev`, then `Ctrl-b 2` to reach the install window.

## Codebase sync (one-shot, from this Mac)
```bash
rsync -az --progress \
  --exclude='.git/' --exclude='.venv/' --exclude='__pycache__/' \
  --exclude='*.pyc' --exclude='*.pyo' --exclude='wandb/' \
  --exclude='outputs/' --exclude='output/' --exclude='experiments/' \
  --exclude='*.egg-info/' --exclude='build/' --exclude='dist/' \
  --exclude='.DS_Store' --exclude='*.pth' --exclude='*.ckpt' \
  --exclude='*.hdf5' --exclude='*.h5' --exclude='*.mp4' --exclude='*.npz' \
  --exclude='node_modules/' --exclude='.idea/' --exclude='.vscode/' \
  --exclude='long_term_bank/' \
  /Users/rutavms/research/long-term-mem/ rutavms-4XL40S-V0:/root/long-term-mem/
```

## Sibling repos (already cloned on remote)
```bash
git clone --branch=abs_robot git@github.com:ShahRutav/robosuite.git ~/robosuite
git clone --branch=mem      git@github.com:ShahRutav/robocasa.git  ~/robocasa
```

## `.env` (already written to `/root/long-term-mem/.env`)
```bash
export CASAPLAY_DATAROOT=/root/datasets
export EXP_STORAGE_BASE_DIR=/root/experiments/
export DATA_DIR=/root/datasets
export CODE_DIR=/root/long-term-mem
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export PYTHONNOUSERSITE=1
export PYTHONFAULTHANDLER=1
export WANDB_API_KEY=    # fill in before first wandb run
```

> Note: the codebase reads `EXP_STORAGE_BASE_DIR` (per `INSTRUCTIONS.md`), not `EXP_BASE_STORAGE_DIR`.

## Install (run inside `l40-dev:longmem-install`)
```bash
cd /root/long-term-mem

# 1. uv-managed Python 3.10 venv
uv venv --python 3.10 .venv
source .venv/bin/activate

# 2. PyTorch with CUDA 12.9 wheels (lets uv resolve a version available on cu129)
uv pip install --index-url https://download.pytorch.org/whl/cu129 \
    torch torchvision torchaudio

# 3. Project requirements — work around the numpydantic/numpy conflict.
#    requirements.txt pins numpy==1.23.5 (the version run_trainer.py:1238
#    overlay-installs at runtime), but numpydantic==1.6.7's metadata
#    requires numpy>=1.24.0. Install numpydantic without its deps so the
#    1.23.5 / numba 0.57.1 pair survives, then install everything else.
uv pip install --no-deps numpydantic==1.6.7
grep -v '^numpydantic' requirements.txt | uv pip install -r /dev/stdin

# Make sure the runtime-canonical pair is what's actually pinned
uv pip install numpy==1.23.5 numba==0.57.1

# 4. Editable installs of the three repos (deps already satisfied)
uv pip install --no-deps -e /root/robosuite
uv pip install --no-deps -e /root/robocasa
uv pip install --no-deps -e /root/long-term-mem

# 5. Re-pin mujoco AFTER robosuite/robocasa (robocasa requires this exact version)
uv pip install mujoco==3.2.6
```

### Skipped on purpose
- conda / miniconda — replaced by uv
- nvidia-dali, flash-attn — need nvcc; not requested
- kitchen assets, CrossMAE weights — no inference setup requested

## Auto-source `.env` from `~/.bashrc`
So `CASAPLAY_DATAROOT`, `EXP_STORAGE_BASE_DIR`, `MUJOCO_GL`, etc. are set in every new shell:
```bash
cat >> ~/.bashrc <<'EOF'

# long-term-mem env
[ -f /root/long-term-mem/.env ] && source /root/long-term-mem/.env
EOF
```
Verify in a fresh interactive shell: `bash -ic 'echo $CASAPLAY_DATAROOT $EXP_STORAGE_BASE_DIR'`.

## Download the inflated MemKScoopPopcorn dataset
The dataset lives at `Rutav/MemKScoopPopcorn_Inflated` on Hugging Face (~28 GB, 20 hdf5 files). It **must** be placed at `$CASAPLAY_DATAROOT/memory/mutex/MemKScoopPopcorn_Inflated` — that's the exact path `config/task_rw_mutex_mem_kscoop_popcorn.json` and `tools/inflate_popcorn_time_1belt.py` read from.

```bash
source /root/long-term-mem/.env  # for CASAPLAY_DATAROOT
mkdir -p "$CASAPLAY_DATAROOT/memory/mutex"
uvx --from "huggingface_hub[cli]" hf download Rutav/MemKScoopPopcorn_Inflated \
    --repo-type dataset \
    --local-dir "$CASAPLAY_DATAROOT/memory/mutex/MemKScoopPopcorn_Inflated"
```
Run inside the `l40-dev:longmem-install` tmux window so a dropped SSH session doesn't kill the download.

## Smoke test
```bash
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import robosuite, robocasa, icrt, mujoco; print(mujoco.__version__)"
```

## Daily use
```bash
ssh rutavms-4XL40S-V0 -t tmux attach -t l40-dev   # Ctrl-b 2 for install window
cd /root/long-term-mem
source .venv/bin/activate
source .env
```
