# RAI Instructions
### Main Repo
```
git clone --branch=docker git@github.com:rjenamani-rai/memory-visuomotor-policies.git
```
### Environment Variables 
```bash
# Dataset directory (RoboCasa)
export CASAPLAY_DATAROOT="/path/to/your/robocasa/datasets/"

# Weights & Biases API key for experiment tracking
export WANDB_API_KEY="your_wandb_api_key_here"

# Experiment storage directory
export EXP_STORAGE_BASE_DIR="/path/to/your/experiments/"

# Path to the directory where the latest code is stored.
export CODE_DIR="/path/to/latest/code"

export GHCR_PAT="your-github-pat-token"
```

### Installations to be done:

##### Three codebase cloning

Clone the three required repositories:

```bash
# Main repository
git clone --branch=docker git@github.com:rjenamani-rai/memory-visuomotor-policies.git
cd memory-visuomotor-policies

# RoboCasa repository
git clone --branch=mem git@github.com:ShahRutav/robocasa.git

# RoboSuite repository  
git clone --branch=abs_robot git@github.com:ShahRutav/robosuite.git
```

##### Conda setup: pip and extra parts to install

1. **Create conda environment:**
```bash
# Create Python 3.11 environment
conda create -n memory python=3.11 pip -y
conda activate memory
```

2. **Install PyTorch with CUDA 12.4 support:**
```bash
pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
```

3. **Install NVIDIA DALI:**
```bash
pip install --extra-index-url https://pypi.nvidia.com nvidia-dali-cuda120
```

4. **Install flash-attention:**
```bash
pip install --no-build-isolation flash-attn
```

5. **Install additional dependencies:**
```bash
# Install requirements from the main repo
pip install -r requirements.txt

# Install additional packages for graphics/rendering
pip install mink==0.0.5 psutil mambapy==1.2.0
```

6. **Install the three repositories in editable mode (no dependencies):**
```bash
# Install main repository (no deps since requirements.txt already installed them)
pip install --no-deps -e .

# Install robosuite (no deps)
pip install --no-deps -e ../robosuite

# Install robocasa (no deps)
pip install --no-deps -e ../robocasa
```

7. **Set up environment variables:**
```bash
# Add to your ~/.bashrc or run before training
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export PYTHONNOUSERSITE=1
export PYTHONFAULTHANDLER=1
```

**Note:** Make sure you have CUDA 12.2+ installed on your system and the appropriate NVIDIA drivers for GPU support.

# 🧠 Memory Docker Environment

This repository provides a reproducible Docker environment for the **Memory** project, including:
- CUDA 12.2 (development image with `nvcc` and toolchain)
- NVIDIA EGL/GL stack (for headless rendering)
- Google Cloud CLI
- Micromamba environment (`memory`) with Python 3.10
- Pre-installed repos: `icrt`, `robosuite`, `robocasa`

---

## 📦 Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (with [BuildKit](https://docs.docker.com/build/buildkit/) enabled)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)  
  Install with:
```bash
sudo apt-get install -y nvidia-container-toolkit
```
- A GitHub Personal Access Token (PAT) with repo access (needed if repos are private).
Save it to a file, e.g.:
```bash
echo "<YOUR_PAT>" > ~/.gh_pat
```

## Building the Image

```
DOCKER_BUILDKIT=1 docker build \\
  --secret id=gh_token,src=$HOME/.gh_pat \\
  -t memory:cuda12.2 .
```

## Running the container

```
docker run --gpus all -it \\
  -v $PWD:/workspace \\
  memory:cuda12.2
```
---
## Installing the latest code

This should be installed in the parent directory of this codebase (`CODE_DIR`).
```
cd ..
git clone --branch=mem git@github.com:ShahRutav/mimicdroid-robocasa.git
git clone --branch=abs_robot git@github.com:ShahRutav/robosuite.git
```

---

## 🚀 Training Setup with `run_trainer.py`

The `run_trainer.py` script provides a Pythonic interface for launching training experiments. Here's how to set it up and use it:

### Environment Variables Setup

Before running the trainer, you need to set up the following required environment variables:

```bash
# Dataset directory (RoboCasa)
export CASAPLAY_DATAROOT="/path/to/your/robocasa/datasets/"

# Weights & Biases API key for experiment tracking
export WANDB_API_KEY="your_wandb_api_key_here"

# Experiment storage directory
export EXP_STORAGE_BASE_DIR="/path/to/your/experiments/"

# Optional: CUDA devices (defaults to 0,1,2,3,4,5,6,7)
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# Path to the directory where the latest code is stored.
export CODE_DIR="/path/to/latest/code"
```

### Conda Environment Activation

The training script expects either the `icrt` or `lib_casa` conda environment to be active:

```bash
# Activate the conda environment
conda activate icrt
# OR
conda activate lib_casa
```

### Configuration Files

The trainer requires two types of configuration files:

#### Model Configuration Files
Located in `config/model/`, examples include:
- `libero_2x.json` - 2x model size for LIBERO tasks
- `gr00t_config.json` - GR00T model configuration

#### Data Configuration Files  
Located in `config/task/` (and `config/qa/` for QA-supervised configs), examples include:
- `task_libero_90.json` - LIBERO 90 task configuration
- `task_robocasa_atomic.json` - RoboCasa atomic tasks
- `task_robocasa_mem_mix.json` - RoboCasa memory tasks

### Basic Usage Examples

```bash
# Basic training command
python run_trainer.py -ds 4 -bs 8 -ng 2 -mc action_head.json -dc task_libero_90.json

# With custom parameters
python run_trainer.py -ds 4 -bs 8 -ng 2 -mc action_head.json -dc task_libero_90.json \
  --seed 42 --vision-encoder crossmae --gating-flag block_nogate

# Dry run to see the command without executing
python run_trainer.py -ds 4 -bs 8 -ng 2 -mc action_head.json -dc task_libero_90.json --dry-run

# Launch on TACC cluster
python run_trainer.py -ds 4 -bs 8 -ng 2 -mc action_head.json -dc task_libero_90.json \
  --launch-location tacc
```

### Parameter Descriptions

| Parameter | Short | Description | Example |
|-----------|-------|-------------|---------|
| `--downsample_obs` | `-ds` | Downsample observation factor | `4` |
| `--batch_size` | `-bs` | Batch size per GPU | `8` |
| `--num_gpus` | `-ng` | Number of GPUs to use | `2` |
| `--model_config` | `-mc` | Model config file name | `action_head.json` |
| `--data_config` | `-dc` | Data config file name | `task_libero_90.json` |
| `--seed` | `-s` | Random seed (default: 1) | `42` |
| `--vision-encoder` | `-ve` | Vision encoder type | `crossmae`, `clip`, `dinov3` |
| `--gating-flag` | `-gf` | Gating mechanism | `False`, `True`, `block_nogate`, `block_sink` |
| `--seq-length` | `-sl` | Sequence length (default: 4) | `8` |
| `--block-finetune` | `-bf` | Block finetune option | `start`, `every_step`, `blockft` |
| `--launch-location` | `-ll` | Where to launch | `local`, `tacc`, `rai` |

### Advanced Features

#### Vision Encoders
- `crossmae` (default) - CrossMAE vision encoder
- `clip` - CLIP vision encoder  
- `dinov3` - DINOv3 vision encoder

#### Gating Mechanisms
- `False` - No gating (standard pooling)
- `True` - Basic gating
- `block_nogate` - Block attention without gating
- `block_sink` - Block attention with sink token

#### Launch Locations
- `local` - Run locally with torchrun
- `tacc` - Generate SLURM script for TACC Lonestar6
- `rai` - RAI execution (not implemented)

### Troubleshooting

1. **Environment Variables Missing**: Ensure all required environment variables are set before running
2. **Conda Environment**: Make sure you're in the correct conda environment (`icrt` or `lib_casa`)
3. **Configuration Files**: Verify that model and data config files exist in the correct directories
4. **GPU Memory**: Adjust batch size and number of GPUs based on available GPU memory
5. **Port Conflicts**: The script automatically assigns random ports, but you can check for conflicts

### Example Training Workflow

```bash
# 1. Set up environment variables
export CASAPLAY_DATAROOT="/mnt/team-storage/memory-visuomotor-policies/datasets/robocasa/datasets/"
export WANDB_API_KEY="your_api_key_here"
export EXP_STORAGE_BASE_DIR="/mnt/team-storage/memory-visuomotor-policies/experiments"

# 2. Activate conda environment
conda activate icrt

# 3. Run training
python run_trainer.py -ds 4 -bs 8 -ng 2 -mc action_head.json -dc task_libero_90.json \
  --vision-encoder crossmae --gating-flag block_nogate --seed 1
```

This will start training with the specified configuration and save results to the experiment storage directory.
