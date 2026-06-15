# README Install/Train — Verification & Fixes

Tested the README end-to-end on **cybertron3** in a throwaway conda env
(`longmem_test`, on `/mnt/nfs_client/rutavms_n/`) against a fresh clone of the
`cleanup` branch. Training was confirmed to run (loss decreasing) **after** the
fixes below. Existing data at `/mnt/data1/rutavms/robocasa/datasets/` was reused,
so steps 7–8 (downloads) were not re-tested.

## Blockers found (all required to get training running)

### 1. PyTorch version too old  — **FIXED in README step 4**
`torch==2.5.1+cu124` crashes at the first training step:
```
NotImplementedError: Could not run 'aten::nonzero_static' with arguments from the 'CUDA' backend
  at halo/models/policy/mem_model.py:324  (torch.nonzero_static(..., size=size))
```
`aten::nonzero_static` has no CUDA kernel in torch 2.5.1. The working `longmem`
env uses torch 2.9.1. Minimal change:
```diff
-pip install --index-url https://download.pytorch.org/whl/cu124 \
-    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
+pip install --index-url https://download.pytorch.org/whl/cu128 \
+    torch==2.9.1 torchvision==0.24.1
```
(`torchaudio` dropped — not installed in the working env and not imported.)

### 2. Default training command points at a non-existent config — **FIXED in README**
The default command and the "Available data configs" list both referenced
`task_rw_mutex_mem_washandreturn.json`, which does not exist on this branch.
The real file (and the one in `run_trainer.py -h`) is
`task_robocasa_mem_washandreturn.json`. Fixed in `README.md` and `run_trainer.py`.
```diff
-    -dc task_rw_mutex_mem_washandreturn.json \
+    -dc task_robocasa_mem_washandreturn.json \
```
…and the bogus entry was removed from the Available-data-configs list.

### 3. numba / numpy conflict on a fresh install — **RECOMMENDED (not yet applied)**
A clean install resolves to `numba==0.56.4` (pinned by robocasa) + `numpy==1.24.0`
(pinned by `halo`), which crashes on import:
```
SystemError: initialization of _internal failed without raising an exception
  numba/np/ufunc/_internal  (numba 0.56.4 doesn't support numpy >= 1.24)
```
The working env uses `numba 0.57.1 + numpy 1.23.5`. Suggested fix — add after the
editable installs in step 5:
```bash
pip install "numba==0.57.1" "numpy==1.23.5"
```
(Better long-term: relax robocasa's `numba==0.56.4` / `numpy==1.23.3` pins.)

## Minor notes (no README change strictly required)
- `conda` is not on PATH in non-login shells; scripted use needs
  `source ~/miniconda3/etc/profile.d/conda.sh` before `conda activate`.
- `run_trainer.py` requires `WANDB_API_KEY` to be **non-empty** even for `--dry-run`
  (set it, or use `WANDB_MODE=offline` with any dummy key when just testing).

## Verified-working sequence (smoke test)
```bash
conda create -n longmem python=3.10 pip -y && conda activate longmem
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.9.1 torchvision==0.24.1
pip install -e ReMemBench && pip install -e robosuite && pip install mujoco==3.2.6 && pip install -e long-term-mem
pip install "numba==0.57.1" "numpy==1.23.5"     # fix #3
python run_trainer.py -ds 8 -bs 8 -ng 1 -mc libero_1_5x_small.json \
    -dc task_robocasa_mem_washandreturn.json -s 1 -sl 512 -ll local \
    --exp-base-dir all_rw --wandb-project-name wandb-exp -nw 8 -br 51 --repeat-traj-factor 2
# -> Epoch [0] runs, loss 0.34 -> 0.28, ~0.75 s/step, ~7.8 GB on one RTX A5000
```
