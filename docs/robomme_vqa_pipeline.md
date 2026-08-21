# RoboMME offline VQA pipeline

The code in `halo/robomme_vqa` constructs VQA supervision data offline. It is
not imported by the RoboMME policy or policy server, so Qwen3-VL is never part
of deployed inference.

## Reproducibility contract

- Model and processor: `Qwen/Qwen3-VL-8B-Instruct`
- Hugging Face revision: `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`
- Dtype and attention: FP16 with PyTorch SDPA (RTX 8000 does not support BF16)
- VQA runtime: Torch `2.9.1+cu128`, Transformers `5.15.1`, and
  `qwen-vl-utils==0.0.14`
- Visual limits: at most 16 timestamped frame pairs (32 images) per call;
  pinned image processor range 65,536–16,777,216 pixels; no video payloads
- Configuration: `config/vqa/robomme_qwen3_vl_8b_v1.json`
- Schemas: `schemas/vqa/v1`
- Generator prompt: `robomme-generator-v1`
- Independent judge prompt: `robomme-judge-v1`

The generator and judge are distinct interfaces with distinct prompts and
decoding settings. A judge request is built from the parsed question/answer,
the permitted visual history, and the task goal. Generator prose or hidden
reasoning is never forwarded. The pilot CLI shares one loaded model between
the interfaces to fit GPU memory, but every generation or judgment remains a
fresh model call.

The configuration and runtime checks fail closed if the model, processor
revision, dtype, package versions, or deterministic judge settings change.
These VQA dependencies are isolated from HALO's policy environment, which
retains Transformers 4.51.3. The lazy backend defaults to cache-only loading.
`--allow-download` is required to permit Hugging Face to populate the existing
cache.

## Request preparation

Each JSONL request must satisfy `request.schema.json`. A request contains a
strictly ordered visual timeline, one query timestep, and optional known event
counts keyed by question family. Training-only event-boundary annotations and
image-change scores may guide frame selection; they are not included in the
VLM prompt. Subgoal boundaries are not task-event counts, so the v1 builder
leaves known counts empty unless a future task-specific oracle can establish
them. The selector always discards frames after the query timestep.

Build the bounded, task-balanced 100-request pilot directly from the canonical
training episode split with:

```bash
python scripts/vqa/build_robomme_vqa_requests.py \
  --raw-root /data/ed1116/Datasets/robomme_data_h5 \
  --shared-manifest /data/ed1116/robomme/manifests/robomme_hdf5_v1.json \
  --output-dir /data/ed1116/robomme/vqa/halo/requests/pilot-v1
```

The new output directory contains only selected PNGs under `images/`, strict
v1 `requests.jsonl`, and a build manifest binding the canonical train split,
shared-manifest hash, raw HDF5 size/mtime signatures, selection seed, task,
suite and eligible-family counts, plus encoded and source-RGB hashes for every
image. The v1 per-request hash covers image path strings, not image bytes or
split identity; those are bound by the companion manifest. Embedding them in
each request would require a breaking request-schema v2.

The builder writes into a sibling `.<name>.*.incomplete` staging directory and
renames it onto the final path only after the whole artifact is written, so an
interrupted build leaves nothing behind and can simply be retried. Task names
and episode keys arrive from the shared manifest and are used to construct
output paths, so each is required to be a known task and a literal
`episode_<N>`; an absolute HDF5 key such as `/episode_3` addresses the same
group but would otherwise place images outside the artifact root. Every image
path is additionally proved to resolve below `images/` before any write.

Front and wrist images are labeled independently at every selected timestamp.
The task-specific prompt allows only question families relevant to that one of
the 16 RoboMME tasks. A corpus builder should create requests only from the
training episode split.

## Pilot

Use a new directory below `/data/ed1116/robomme/vqa/halo` for every run:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/vqa/run_robomme_vqa_pilot.py \
  --requests-jsonl /data/ed1116/robomme/vqa/halo/requests/pilot-v1/requests.jsonl \
  --output-dir /data/ed1116/robomme/vqa/halo/pilot-v1 \
  --max-requests 100
```

The command refuses relative paths, output outside the approved artifact
root, the artifact root itself, and an existing run directory. It writes all
accepted and rejected candidate records plus a manifest containing exact
model, prompt, decoding, config-hash, and failure provenance. It never writes
to raw RoboMME HDF5 files. Model-load and quality gates must be completed
before full generation: one RTX 8000, both cameras, at least 90% valid JSON on
100 requests, representative identity/count/swap/spatial/order examples, and
recorded VRAM, latency, and throughput.

## Deterministic filtering and audit

Filtering validates task/suite/episode identifiers, causal evidence
timestamps, question-family membership, forbidden internal or privileged
field names, optional event counts, episode-local near duplicates, and answer
length. Acceptance additionally requires every independent-judge boolean to
pass with `future_leakage=false`. Original rejected records and their reason
codes remain in the corpus; do not hand-edit them.

Every corpus record is parsed against the full v1 value contract, not only its
field names: booleans must be real booleans (a string `"false"` is rejected
rather than silently read as truthy), suite must match task, timestamps must be
sorted non-negative integers, hashes must be lower-case SHA-256, and
`rejection_reason` must be null exactly when `accepted` is true.

Loading a request artifact reads `requests.jsonl` and `manifest.json` exactly
once and hashes the same bytes it parses, so the audit manifest's provenance
hashes cannot drift if a source file is rewritten while the packet is being
built. At that point every image referenced by every request timeline — not
only the frames a record happens to cite — is bound to the build manifest by
path, identity metadata, and encoded SHA-256.

Audit packet construction then joins each record to its source request by
`source_request_sha256`, verifies task/suite/episode/query identity, and fails
closed if any candidate, judge, or query timestamp is absent from the source
timeline. The packet references those images read-only; it never copies or
embeds them.

Create the suite-level pilot review packet with:

```bash
python scripts/vqa/audit_robomme_vqa.py \
  --records-jsonl /data/ed1116/robomme/vqa/halo/pilot-v1/records.jsonl \
  --requests-jsonl /data/ed1116/robomme/vqa/halo/requests/pilot-v1/requests.jsonl \
  --request-manifest /data/ed1116/robomme/vqa/halo/requests/pilot-v1/manifest.json \
  --output-dir /data/ed1116/robomme/vqa/halo/pilot-v1-audit \
  --stage pilot --seed 0
```

This requests 20 accepted and 10 rejected examples per suite and reports any
quota deficit. Before training, run `--stage pretrain` to stratify by every
task and question family. Reviewers record status, error classes, and notes in
the audit packet while preserving the source records.
