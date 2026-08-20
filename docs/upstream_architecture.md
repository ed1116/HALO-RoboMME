# Released HALO architecture map

This audit fixes the preservation boundary for the RoboMME adaptation. It maps
the released implementation at upstream commit
`7778638401cc07043e59e2381e84f2383401f491`; it is not a proposal for a
replacement policy. The selected RoboMME baseline follows the released finals
topology: retrieval layers `0, 2`, block-local layers `1, 3, 4, 5`, and eight
attention heads. The paper and implementation differences behind that choice
are recorded below.

## End-to-end path

`halo.util.model_constructor.model_constructor` builds a `VisionEncoder` and a
single `MemModel`. A training batch then follows this path:

1. `MemModel.preprocessing` embeds language placeholders, encodes both cameras,
   pools each camera separately with proprioception, encodes previous actions,
   and writes those learned embeddings into the token sequence.
2. `halo.models.policy.llama.Transformer.forward` applies the six policy
   layers in index order. In the released finals configuration, layers 0 and 2
   are `TopKTransformerBlock`; layers 1, 3, 4, and 5 are
   `TransformerBlock(BlockAttention[Flex])`.
3. `MemModel.compute_all_losses` routes policy latents to the action, VQA/state,
   and optional generic language heads under their masks.

The construction entry points are `vision_encoder_constructor`,
`policy_constructor`, and `model_constructor` in
`halo/util/model_constructor.py`. `config/model/libero_1_5x_small.json` fixes
the released backbone at width 256, six layers, and eight heads.

## Visual encoding and learned pooling

- `halo.models.backbones.encoders.VisionEncoder.__init__` recognizes a
  `cross-mae-rtx` checkpoint path, creates Timm
  `vit_base_patch16_224.mae`, and loads that checkpoint with `strict=False`.
  `VisionEncoder.forward` calls `forward_features`; when the encoder is not
  being finetuned it does so under `torch.no_grad()`.
- `halo.util.model_constructor.vision_encoder_constructor` defaults to
  `freeze_all=True`, so the released CrossMAE weights are frozen unless an
  explicit unfreeze/LoRA option is passed.
- `MemModel.preprocessing` concatenates all camera tensors for one visual
  encoder pass, splits the features back by camera, applies
  `icrt_vision_norm`, and sends each view through its own learned
  `AttentionPool` (or `MultiKVAttentionPool`). `MemModel.__init__` requires
  `separate_camera_adapter=True`.
- `AttentionPool.combine_forward` in
  `halo/models/backbones/encoders.py` appends the encoded proprioceptive token
  to the visual patch tokens before learned-query pooling. Each camera thus
  retains a separate pooling module while sharing the frozen visual encoder.

The released launcher resolves CrossMAE in
`run_trainer.TrainingConfig._generate_extra_flags`: it downloads
`mlfu7/ICRT/crossmae_rtx/cross-mae-rtx-vitb.pth` from Hugging Face when the
expected local file is absent. The URL has no recorded model revision or
checksum; the adaptation must pin both in its artifact manifest without
changing the encoder.

## Token assembly

- `halo.models.backbones.token_sequence_gen.LangTrajSequence.__call__` builds
  `[padded instruction][camera tokens, optional previous-action token]` for
  every retained frame. It also returns exact positions for camera
  replacement, action-input replacement, and action-output prediction.
- `MemModel.preprocessing` starts from the policy token embedding table, then
  uses `index_put_` to replace camera placeholders with pooled visual features
  and action placeholders with `icrt_action_encoder` outputs. The action and
  proprio encoders are MLPs defined in `MemModel.__init__`.
- With two cameras, one pooled token per camera, and the previous-action token,
  the released default has three policy tokens per retained environment frame
  (`MemModel.latent_len`). The exact count remains configuration-derived.
- Observation cadence is supplied by the data path, not by a hidden scheduler
  in the retrieval bank: `TaskGroupDataset.__getitem__` in
  `halo/data/dataset_vl.py` slices observations, proprioception, actions, and
  masks by `downsample_obs`; the finals preset uses 8. The RoboMME adapter must
  therefore preserve original environment timestep identifiers when it builds
  the same eight-step cadence.

## Block-local and top-k layers

### Released block-local path

`halo.models.policy.llama.Transformer.__init__` selects a layer type by its
configured index. A retrieval index takes precedence; otherwise a block,
sliding-local, strided, gated, or full attention block is created.

For released finals, `run_trainer._PRESET_BLOCK_ATTN_IND` is `[1, 3, 4, 5]`
and `_PRESET_RET_TOPK_ATTN_IDX` is `[0, 2]`. The former selects
`BlockAttentionFlex` during training and `BlockAttention` during evaluation
through `TransformerBlock.__init__`. `flex_causal_mask` in
`halo/models/policy/block_attn_variants.py` permits causal attention only
inside the same non-overlapping block. This is distinct from the separately
implemented sliding-window `LocalAttention`, which is not selected by the
released finals preset. `block_chunk_ts_len=8` is converted to policy-token
length in `MemModel._get_default_model_args`.

### Sparse retrieval path

- `TopKTransformerBlock` in `halo/models/policy/retriever_topk.py` wraps
  `TopKAttention` or `TopKAttentionCausal`, RMSNorm, residual connections, and
  the standard feed-forward block. The released launcher also enables
  `ret_bank_causal` for finals retrieval indices.
- Each retrieval layer owns a `LongTermBank` of float32 K/V buffers, a validity
  mask, and valid-entry counters. `TopKAttentionCausal.forward` resets the bank
  at `start_pos == 0`, stores detached current K/V, then applies its position
  mask so a query cannot select its current retrieval chunk.
- `LongTermBank.retrieve_kv` computes cosine similarity, masks ineligible
  entries, takes `topk`, and gathers values. `TopKAttention` divides scores by
  a learned temperature `tau`, applies a softmax, and forms a weighted sum of
  the selected values. The finals defaults are `k=8`, retrieval chunk length 8
  retained timesteps (scaled by `latent_len`), and straight-through training.
- The released straight-through path fixes the discrete membership chosen by
  `topk` but recomputes selected similarities during training so gradients
  reach the query/scoring path; stored historical K/V are detached.
- `Transformer.forward` passes `start_pos` through every layer, and
  `MemModel.use_cache` forwards cache enablement to the transformer and its
  retrieval banks.

Before RoboMME training, the released bank needs narrow correctness hardening,
not a new retriever: when fewer than `k` entries are eligible,
`LongTermBank.retrieve_kv` pads indices with zero and scores with zero; the
causal variant can likewise map fully masked selections to slot zero. It then
gathers slot-zero values, and its training-time similarity recomputation can
overwrite the earlier masked score. Those padded/fully masked slots therefore
need an explicit validity mask so they contribute exactly zero. In addition,
`LongTermBank.store_kv` caps the counter only after scattering and has no
pre-scatter capacity assertion. Tests must cover fewer-than-k, all-masked,
capacity, episode reset, and finite straight-through gradients.

## VQA supervision and text heads

The VLM is an offline data generator/judge; it is not loaded by the deployed
HALO policy. The released generation entry point is
`scripts/data_gen/generate_qa.py`, which constructs prompts and uses
`halo.models.policy.vlm.GPTQueryGenerator`. The RoboMME adaptation replaces
that external generation/judging dependency with offline
`Qwen/Qwen3-VL-8B-Instruct` passes while preserving the policy-side objective.

The released finals VQA route is specifically:

1. `halo.data.dataset_qa.QADatasetGPTState.__getitem__` creates a query plus
   trajectory token sequence, stores tokenized answers as `state_supervision`,
   marks `state_supervision_out_token_pos`, and disables action and generic
   text loss masks for that item.
2. `MemModel.__init__` creates `state_supervision_output_head=CEHead` when GPT
   state supervision uses `bbox_str` mode.
3. `MemModel._compute_state_supervision_loss` selects the requested policy
   latents and calls `CEHead.loss`.
4. `halo.models.policy.action_head.CEHead` concatenates the selected policy
   latent with teacher-forced answer-token embeddings, runs one causal
   transformer layer, and predicts the next answer token with cross entropy.

`MemModel.text_output_head` is a separate linear vocabulary projection used by
`_compute_text_loss` for ordinary next-token masks. It is not the released
finals VQA/state route and must not be substituted for `CEHead` during the
RoboMME port.

## Action prediction

- `MemModel.__init__` creates an `MLPHead` by default with output width
  `num_pred_steps * action_dim` and elementwise L1 loss. The policy latent at
  each `action_out_token_pos` is selected in `MemModel._compute_action_loss`.
- Upstream trajectory data appends an EOS channel to the physical action in
  `TaskGroupDataset.__getitem__`; `MemModel` correspondingly sets its predicted
  physical dimension to `extra_kwargs["action_dim"] - 1`. The released
  RoboCasa route therefore consumes seven physical action values plus EOS and
  predicts seven values.
- `MLPHead.pred` reshapes the raw prediction horizon, but its default inference
  behavior fills an action queue of length `num_pred_steps // 2` and returns
  one action. `MemModel.forward_inference` also returns one queued action at a
  time and binarizes the gripper in `_binarize_gripper`.

For RoboMME, the data representation must be eight physical targets plus its
training-only EOS/padding convention, while the learned head must expose a raw
`[B, 20, 8]` prediction. The RoboMME serving adapter returns exactly the first
16 actions; it must not inherit the existing one-action public return shape or
silently truncate the learned horizon inside the head.

## Loss composition

`MemModel.compute_all_losses` is the single loss router:

- imitation: masked elementwise L1 from `_compute_action_loss`, coefficient 1;
- VQA/state: token cross entropy from `_compute_state_supervision_loss`,
  coefficient `coeff_state_supervision_loss`;
- ordinary text: next-token cross entropy from `_compute_text_loss`, fixed
  coefficient 0.01 when that mask is used;
- `ret_*` losses are initialized to zero; `ret_emdr2_loss` is explicitly not
  implemented for `MemModel`.

Thus the released finals joint VQA/imitation behavior comes from mixing task
and `QADatasetGPTState` examples and backpropagating both heads through the
shared transformer/retrieval layers. It does not use a standalone retriever
loss. The paper reports `L = L_IL + lambda L_VQA`, L1/CE objectives, and
`lambda` in `{0.1, 1.0}`; the source exposes the VQA coefficient per task.

## Checkpoints and resume

- `halo.util.misc.save_model` writes `model`, optimizer, epoch, AMP scaler,
  serialized arguments, and best validation loss. `scripts/train.py` creates
  `checkpoint-0.pth`, periodic checkpoints, `best_val_loss.pth`, and
  `last_epoch.pth`.
- `MemModel.state_dict` intentionally filters to the phase's trainable policy
  state plus selected buffers. A frozen CrossMAE checkpoint is not embedded and
  must be resolved from the separately pinned visual artifact on restore.
- `halo.util.misc.load_model` and `resume_from_ckpt` load policy state with
  `strict=False`; resume can also restore optimizer, epoch, scaler, and best
  validation loss. Long-term-bank cache buffers are treated as reconstructible
  state rather than required learned weights.
- The upstream `save_on_master` is a direct `torch.save`, not an atomic
  temp-file/rename operation. RoboMME packaging must add atomic writes,
  checksums, complete config/normalization manifests, and a load-and-forward
  verification while retaining the same learned parameter set.

## Released serving lifecycle

There is no network policy server in the released repository. The reference
lifecycle lives in `scripts/eval.py` and `halo/util/casa_rollouts.py`:

1. construct `MemModel(train=False)`, optionally enable caches, load the
   checkpoint, and enter eval mode;
2. call `model.reset(batch_size=1)` and `model.prompt(language)` once per
   episode;
3. preprocess the next two-camera observation and proprioception, then call
   `model.forward_inference` once per environment action;
4. execute the returned single action and retain the model's predicted action
   as future history.

`MemModel.reset_inference_state` clears positions, the action queue, buffered
observations/actions/proprioception, and rebuilds `LangTrajSequence`;
`forward_inference` updates those buffers and transformer cache positions.
The RoboMME server must wrap this stateful core rather than replace it. It must
add the benchmark's `infer/reset` transport, consume every newly supplied
history frame, retain the actually returned actions because the request omits
past actions, return exactly `(16, 8)`, and prove reset equivalence across
episodes.

## Paper/source/plan reconciliation

| Topic | HALO paper | Released source | Selected RoboMME baseline |
| --- | --- | --- | --- |
| Six policy layers | Four local layers followed by two retrieval layers | Retrieval at 0 and 2; `BlockAttention` at 1, 3, 4, 5 | Preserve the released interleaved topology and non-overlapping block-local mask for source/checkpoint fidelity. |
| Attention heads | Four per policy layer | Eight in `libero_1_5x_small.json` | Preserve eight. |
| VQA vocabulary/head | One transformer block over standard Qwen3 vocabulary | Finals use one-block `CEHead`, but `SharedConfig.tokenizer_name` defaults to `Qwen/Qwen2-7B-Instruct`; generic text projection is separate | Preserve the `QADatasetGPTState -> CEHead` route; pin the chosen policy tokenizer explicitly rather than changing it accidentally with the offline VLM. |
| Offline VLM | GPT-4o-mini generation and GPT-4o judging | `GPTQueryGenerator` external API path | Use Qwen3-VL-8B offline for generation and an independent judge pass; no VLM at policy inference. |
| Action horizon/interface | Predict 32 future actions | Default 32, queue length 16, public inference returns one action | Train a 20-by-8 head and expose exactly its first 16 through the RoboMME server. |
| Embodiment | Task/robot-specific action layout; the appendix does not define the released tensor serialization | Seven physical values plus EOS; model subtracts EOS from output width | Eight RoboMME physical values; keep EOS/padding training-only. |
| Memory cadence | Add observation/action entries every eight interactions, `k=8` | Finals downsample by 8; each retrieval layer owns a token bank, `k=8` | Preserve original timestep IDs and insert the corresponding front/wrist/prior-action tokens on the eight-step cadence. |
| Training scope | Separate model per task | Four task presets | One task-balanced model across all 16 RoboMME tasks, as required by the plan. |
| Precision | Not specified in the paper's implementation details or hyperparameter table | `SharedConfig.compute_dtype` defaults to BF16 | Use validated FP16 plus dynamic loss scaling on RTX 8000; record numerical checks. |

The topology/head-count choice is deliberate and user-approved. All other
RoboMME changes above are interface, data, artifact, or hardware adaptations;
none authorizes replacing CrossMAE, the learned per-camera pooling, the
released six-layer transformer, sparse top-k banks, joint imitation/VQA
training, or the native action head.
