#!/usr/bin/env bash
# Simple evaluation launcher for HALO finals checkpoints.
#
# Modes:
#   DEBUG=1            Single GPU, no CSV save, n_eval=1
#   DEBUG=single       Single GPU, no CSV save, n_eval=1
#   REDO=1             Re-run evals even if results exist
#
# Evaluates all tasks in the checkpoint's dataset by default (task name is not passed).
set -euo pipefail

gpu_list=(0 1 2 3 4 5 6 7)
exps_in_parallel=1

# Each entry: "<exp_dir> <ckpt_num>"  (use -1 for the latest checkpoint)
exp_ckpt_dirs=(
    "<path/to/checkpoint-dir> <checkpoint-number>"
)

# ----- Debug-mode overrides -----
debug="false"
VARIABLES=""
if [[ "${DEBUG:-}" == "true" || "${DEBUG:-}" == "1" ]]; then
    gpu_list=(0)
    exps_in_parallel=1
    VARIABLES="$VARIABLES --eval-cfg.debug --eval-cfg.no-save-csv"
    debug="true"
elif [[ "${DEBUG:-}" == "single" ]]; then
    gpu_list=(0)
    exps_in_parallel=1
    VARIABLES="$VARIABLES --eval-cfg.no-save-csv"
    debug="true"
fi
num_gpus=${#gpu_list[@]}

# ----- Common eval flags -----
CUDA_VISIBLE_DEVICES_VAL=$(IFS=','; echo "${gpu_list[*]}")
CMD_ENV="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES_VAL MUJOCO_GL=egl OMP_NUM_THREADS=1 MPI_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CC=$(command -v gcc) CXX=$(command -v g++)"

VARIABLES="$VARIABLES --dataset-cfg.dataset-json ''"
VARIABLES="$VARIABLES --world_size $num_gpus"
VARIABLES="$VARIABLES --eval-cfg.n_eval 48"
VARIABLES="$VARIABLES --eval-cfg.video-downsample-factor 1"
VARIABLES="$VARIABLES --eval-cfg.store-video-from-every-rank"
VARIABLES="$VARIABLES --eval-cfg.save-videos-in-wandb"
VARIABLES="$VARIABLES --eval-cfg.save-videos"
VARIABLES="$VARIABLES --eval-cfg.save-wandb"

# wandb entity / project (optional)
if [[ -n "${WANDB_ENTITY:-}" ]]; then
    VARIABLES="$VARIABLES --trainer-cfg.wandb-entity $WANDB_ENTITY"
fi
if [[ -n "${WANDB_PROJECT:-}" ]]; then
    VARIABLES="$VARIABLES --trainer-cfg.wandb-project $WANDB_PROJECT"
fi

if [[ "${REDO:-}" == "1" ]]; then
    VARIABLES="$VARIABLES --eval-cfg.redo-evals"
fi

# ----- Launch loop -----
exps_launched=0
n_exps=0
RANDOM=$$
PORT_NUM=$((2452 + RANDOM % 100))

for exp_ckpt_dir in "${exp_ckpt_dirs[@]}"; do
    exp_dir=$(echo "$exp_ckpt_dir" | cut -d' ' -f1)
    ckpt_num=$(echo "$exp_ckpt_dir" | cut -d' ' -f2)
    echo "exp_dir: $exp_dir"
    echo "ckpt_num: $ckpt_num"

    MASTER_PORT=$((PORT_NUM + n_exps))
    CMD_TO_RUN="$CMD_ENV torchrun --nproc_per_node=$num_gpus --master_port=$MASTER_PORT scripts/eval.py --eval-cfg.eval-only-directory $exp_dir --eval-cfg.eval-only-ckpt-num $ckpt_num $VARIABLES"

    if [[ "$debug" == "false" ]]; then
        CMD_TO_RUN="$CMD_TO_RUN &"
    fi
    echo "$CMD_TO_RUN"
    eval "$CMD_TO_RUN"

    if [[ "$debug" == "true" ]]; then
        break
    fi

    exps_launched=$((exps_launched + 1))
    n_exps=$((n_exps + 1))
    if [[ $exps_launched -eq $exps_in_parallel ]]; then
        wait -n
        ((exps_launched--))
    fi
done
wait
