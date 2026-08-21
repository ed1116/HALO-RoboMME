# Provenance and scope

## Source revisions

| Component | Source | Revision | Role |
| --- | --- | --- | --- |
| HALO | `git@github.com:UT-Austin-RobIn/HALO.git` | `7778638401cc07043e59e2381e84f2383401f491` | Native policy and retrieval architecture |
| RoboMME policy learning | `git@github.com:RoboMME/robomme_policy_learning.git` | `ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b` | Clean policy/serving contract reference |
| RoboMME benchmark | `git@github.com:RoboMME/robomme_benchmark.git` | `856bc3a189d4172f3f47dbee4424d585f8d78db3` | Simulator interface reference |

## Adaptation boundary

The project will keep HALO's frozen CrossMAE encoder, local-attention layers,
differentiable top-k sparse retrieval, imitation objective, and VQA objective.
RoboMME-specific changes are limited to the two-camera observation contract,
8-D joint/gripper actions, a 20-step prediction horizon with 16 executed actions,
balanced training across 16 tasks, offline open-VLM supervision, and the official
RoboMME policy-server lifecycle.

The raw HDF5 dataset is read from
`/data/ed1116/Datasets/robomme_data_h5`. Generated VQA, processed artifacts,
checkpoints, and runs live under `/data/ed1116/robomme` and are never committed.


## Environments

The repository has no in-tree virtual environment. Interpreters live outside
Git under `/data/ed1116/robomme/envs` so that weights, caches, and packages
stay off the repository:

| Environment | Path | Role |
| --- | --- | --- |
| HALO policy | `/data/ed1116/robomme/envs/halo` | Policy code and the `tests/` suite. Pins Transformers 4.51.3. |
| HALO VQA | `/data/ed1116/robomme/envs/halo-vqa` | Pinned Qwen3-VL generation/judging runtime, isolated because it needs Transformers 5.x. |

Run the suite with the policy environment from the repository root:

```bash
/data/ed1116/robomme/envs/halo/bin/python -m pytest -q tests
```

The `robomme` micromamba environment is the RoboMME simulator/benchmark
runtime. It is deliberately separate and cannot run this suite.

## Evaluation rollout budget

Simulator evaluation uses **10 rollouts per task** (160 per method), not the
official 50. This is an evaluation-time budget only: it must never influence
training data, hyperparameters, checkpoint selection, or VQA corpus
construction, all of which use the full 100 demonstrations per task with the
canonical 90/10 episode split.
