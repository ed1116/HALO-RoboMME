"""Deterministic RoboMME HDF5 to offline-VQA request construction."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
from PIL import Image

from halo.robomme.dataset import TASKS, RoboMMEHDF5Dataset

from . import io as vqa_io
from .io import (
    REQUEST_SCHEMA_VERSION,
    parse_request_jsonl,
    request_from_dict,
    write_json,
    write_jsonl,
)
from .pipeline import VQARequest
from .prompts import allowed_families
from .timeline import TimelineFrame, select_causal_evidence


BUILD_MANIFEST_SCHEMA_VERSION = "halo.robomme.vqa.request-build-manifest/v1"
BUILDER_REVISION = "robomme-hdf5-request-builder-v1"
CANONICAL_DEV_EPISODES_PER_TASK = 10
DEFAULT_CANDIDATE_TIMELINE_FRAMES = 64
EPISODE_KEY_PATTERN = re.compile(r"episode_[0-9]+")


@dataclass(frozen=True)
class SourceEpisode:
    task_name: str
    suite_name: str
    file_path: Path
    episode_key: str
    episode_id: int
    num_timesteps: int
    task_goal: str
    execution_start: int


@dataclass(frozen=True)
class PlannedFrame:
    timestep: int
    event_boundary: bool
    change_score: float


@dataclass(frozen=True)
class PlannedRequest:
    episode: SourceEpisode
    query_timestep: int
    frames: tuple[PlannedFrame, ...]
    known_counts_by_family: Mapping[str, int]
    annotation_source: str


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _stable_rank(seed: int, *parts: object) -> bytes:
    value = ":".join((str(seed), *(str(part) for part in parts)))
    return hashlib.sha256(value.encode("utf-8")).digest()


def _decode_text(value: object) -> str:
    array = np.asarray(value)
    if array.size:
        value = array.reshape(-1)[0]
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return str(value)


def _read_rgb(step: h5py.Group, camera: str) -> np.ndarray:
    value = np.asarray(step["obs"][camera][()])
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"{step.name}/obs/{camera} must be uint8 RGB")
    return value


def _annotation_boundaries(episode: h5py.Group, length: int) -> tuple[list[bool], str]:
    boundaries: list[bool] = []
    explicit_available = True
    for timestep in range(length):
        info = episode[f"timestep_{timestep}"]["info"]
        if "is_subgoal_boundary" not in info:
            explicit_available = False
            break
        boundaries.append(bool(info["is_subgoal_boundary"][()]))
    if explicit_available:
        return boundaries, "is_subgoal_boundary"

    # Some exports retain subgoal text but omit the boundary flag. Text is used
    # only to detect transitions and is never retained in requests or manifests.
    boundaries = [False] * length
    previous: tuple[str, ...] | None = None
    observed_fields: tuple[str, ...] = ()
    for timestep in range(length):
        info = episode[f"timestep_{timestep}"]["info"]
        fields = tuple(
            field for field in ("simple_subgoal", "grounded_subgoal") if field in info
        )
        if not fields:
            return [False] * length, "none"
        observed_fields = fields
        current = tuple(_decode_text(info[field][()]) for field in fields)
        boundaries[timestep] = previous is not None and current != previous
        previous = current
    return boundaries, "subgoal_transition:" + ",".join(observed_fields)


def _query_timestep(
    episode: SourceEpisode,
    boundaries: Sequence[bool],
    *,
    seed: int,
) -> int:
    if episode.num_timesteps < 2:
        raise ValueError(f"{episode.task_name}/{episode.episode_key} has no visual history")
    execution_start = max(1, episode.execution_start)
    last = episode.num_timesteps - 1
    if execution_start > last:
        raise ValueError(f"{episode.task_name}/{episode.episode_key} has no causal query frame")

    after_events = sorted(
        {
            max(execution_start, timestep + 1)
            for timestep, boundary in enumerate(boundaries)
            if boundary and max(execution_start, timestep + 1) <= last
        }
    )
    candidates = after_events
    if not candidates:
        midpoint = execution_start + (last - execution_start) // 2
        candidates = list(range(midpoint, last + 1))
    rank = int.from_bytes(
        _stable_rank(seed, "query", episode.task_name, episode.episode_key), "big"
    )
    return candidates[rank % len(candidates)]


def _evenly_spaced_timestamps(stop: int, count: int) -> set[int]:
    if stop == 0:
        return {0}
    count = min(count, stop + 1)
    if count == 1:
        return {stop}
    return {round(position * stop / (count - 1)) for position in range(count)}


def _frame_change_score(episode: h5py.Group, timestep: int) -> float:
    if timestep == 0:
        return 0.0
    current = episode[f"timestep_{timestep}"]
    previous = episode[f"timestep_{timestep - 1}"]
    camera_scores: list[float] = []
    for camera in ("front_rgb", "wrist_rgb"):
        current_rgb = _read_rgb(current, camera)
        previous_rgb = _read_rgb(previous, camera)
        if current_rgb.shape != previous_rgb.shape:
            raise ValueError(f"camera shape changed at {episode.name}/timestep_{timestep}")
        difference = np.abs(current_rgb.astype(np.int16) - previous_rgb.astype(np.int16))
        camera_scores.append(float(difference.mean(dtype=np.float64) / 255.0))
    return round(sum(camera_scores) / len(camera_scores), 10)


def _plan_episode(
    episode: SourceEpisode,
    *,
    seed: int,
    max_evidence_frames: int,
) -> PlannedRequest:
    with h5py.File(episode.file_path, "r") as handle:
        group = handle[episode.episode_key]
        boundaries, annotation_source = _annotation_boundaries(
            group, episode.num_timesteps
        )
        query = _query_timestep(episode, boundaries, seed=seed)
        candidates = _evenly_spaced_timestamps(
            query, DEFAULT_CANDIDATE_TIMELINE_FRAMES
        )
        candidates.update(
            timestep
            for timestep, boundary in enumerate(boundaries[: query + 1])
            if boundary
        )
        candidates.add(query)
        timeline = tuple(
            TimelineFrame(
                timestep=timestep,
                front_image="pending-front-image",
                wrist_image="pending-wrist-image",
                event_boundary=boundaries[timestep],
                change_score=_frame_change_score(group, timestep),
            )
            for timestep in sorted(candidates)
        )
    selected = select_causal_evidence(
        timeline, query_timestep=query, max_frames=max_evidence_frames
    )
    if selected[-1].timestep != query or not any(
        frame.timestep < query for frame in selected
    ):
        raise RuntimeError("planned VQA evidence is not causal")

    # Subgoal boundaries are useful evidence-selection hints, but they are not
    # task events and therefore cannot validate a generated count answer.
    return PlannedRequest(
        episode=episode,
        query_timestep=query,
        frames=tuple(
            PlannedFrame(frame.timestep, frame.event_boundary, frame.change_score)
            for frame in selected
        ),
        known_counts_by_family={},
        annotation_source=annotation_source,
    )


def _request_counts_by_task(target_requests: int) -> dict[str, int]:
    if target_requests < len(TASKS):
        raise ValueError(f"target_requests must be at least {len(TASKS)}")
    quotient, remainder = divmod(target_requests, len(TASKS))
    return {
        task: quotient + (task_index < remainder)
        for task_index, task in enumerate(TASKS)
    }


def _training_episodes(
    raw_root: Path,
    shared_manifest: Path,
    *,
    split_seed: int,
) -> tuple[dict[str, tuple[SourceEpisode, ...]], RoboMMEHDF5Dataset]:
    dataset = RoboMMEHDF5Dataset(
        raw_root,
        horizon=1,
        split="train",
        split_seed=split_seed,
        dev_episodes_per_task=CANONICAL_DEV_EPISODES_PER_TASK,
        manifest_path=shared_manifest,
    )
    result: dict[str, tuple[SourceEpisode, ...]] = {}
    # The dataset owns the canonical episode-hash split. This package-local
    # conversion deliberately reuses its selected episode index rather than
    # duplicating the split algorithm.
    for task in TASKS:
        for item in dataset._task_episodes[task]:
            # Both values reach the manifest from an external file and are used
            # to build output paths, so neither may carry separators.
            if item.task_name != task:
                raise ValueError(
                    f"{task} manifest episode declares task {item.task_name!r}"
                )
            if EPISODE_KEY_PATTERN.fullmatch(item.episode_key) is None:
                raise ValueError(
                    f"{task} manifest episode key is not episode_N: {item.episode_key!r}"
                )
        result[task] = tuple(
            SourceEpisode(
                task_name=item.task_name,
                suite_name=item.suite_name,
                file_path=item.file_path,
                episode_key=item.episode_key,
                episode_id=item.episode_id,
                num_timesteps=item.num_timesteps,
                task_goal=item.task_goal,
                execution_start=item.execution_start,
            )
            for item in dataset._task_episodes[task]
        )
    return result, dataset


def plan_pilot_requests(
    raw_root: str | Path,
    shared_manifest: str | Path,
    *,
    target_requests: int = 100,
    split_seed: int = 0,
    seed: int = 0,
    max_evidence_frames: int = 16,
) -> tuple[PlannedRequest, ...]:
    """Plan a task-balanced pilot using only canonical training episodes."""
    if max_evidence_frames < 2:
        raise ValueError("max_evidence_frames must be at least 2")
    raw_root = Path(raw_root).resolve()
    shared_manifest = Path(shared_manifest).resolve()
    counts = _request_counts_by_task(target_requests)
    episodes_by_task, _ = _training_episodes(
        raw_root, shared_manifest, split_seed=split_seed
    )

    planned: list[PlannedRequest] = []
    for task in TASKS:
        ranked = sorted(
            episodes_by_task[task],
            key=lambda episode: _stable_rank(
                seed, "request", task, episode.episode_key
            ),
        )
        if len(ranked) < counts[task]:
            raise ValueError(
                f"{task} has {len(ranked)} train episodes but needs {counts[task]} requests"
            )
        planned.extend(
            _plan_episode(
                episode,
                seed=seed,
                max_evidence_frames=max_evidence_frames,
            )
            for episode in ranked[: counts[task]]
        )
    return tuple(planned)


def _request_to_dict(request: VQARequest) -> dict[str, Any]:
    return {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "task_name": request.task_name,
        "suite_name": request.suite_name,
        "episode_id": request.episode_id,
        "task_goal": request.task_goal,
        "query_timestep": request.query_timestep,
        "timeline": [
            {
                "timestep": frame.timestep,
                "front_image": frame.front_image,
                "wrist_image": frame.wrist_image,
                "event_boundary": frame.event_boundary,
                "change_score": frame.change_score,
            }
            for frame in request.timeline
        ],
        "known_counts_by_family": dict(request.known_counts_by_family),
    }


def _save_png(path: Path, value: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_sha256 = _sha256_bytes(value.tobytes(order="C"))
    with path.open("xb") as handle:
        Image.fromarray(value, mode="RGB").save(
            handle, format="PNG", optimize=False, compress_level=9
        )
    encoded = path.read_bytes()
    return {
        "encoded_sha256": _sha256_bytes(encoded),
        "source_rgb_sha256": raw_sha256,
        "size_bytes": len(encoded),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }


def _source_file_provenance(
    raw_root: Path, manifest_payload: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    tasks = manifest_payload.get("tasks")
    if not isinstance(tasks, dict) or set(tasks) != set(TASKS):
        raise ValueError("shared manifest must contain exactly the 16 RoboMME tasks")
    result: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        entry = tasks[task]
        result[task] = {
            "path": str((raw_root / entry["file_name"]).resolve()),
            "file_name": entry["file_name"],
            "size_bytes": entry["size_bytes"],
            "mtime_ns": entry["mtime_ns"],
        }
    return result


def build_pilot_requests(
    raw_root: str | Path,
    shared_manifest: str | Path,
    output_dir: str | Path,
    *,
    target_requests: int = 100,
    split_seed: int = 0,
    seed: int = 0,
    max_evidence_frames: int = 16,
) -> Path:
    """Extract a guarded, immutable pilot request artifact and its provenance."""
    raw_root = Path(raw_root).resolve()
    shared_manifest = Path(shared_manifest).resolve()
    output_dir = vqa_io.guarded_output_directory(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    manifest_bytes = shared_manifest.read_bytes()
    try:
        manifest_payload = json.loads(manifest_bytes)
    except json.JSONDecodeError as error:
        raise ValueError("shared manifest is invalid JSON") from error
    source_files = _source_file_provenance(raw_root, manifest_payload)
    plans = plan_pilot_requests(
        raw_root,
        shared_manifest,
        target_requests=target_requests,
        split_seed=split_seed,
        seed=seed,
        max_evidence_frames=max_evidence_frames,
    )

    # Build into a sibling staging directory so an interruption never leaves a
    # partial artifact that the exclusive-create image writer could not retry.
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.", suffix=".incomplete", dir=output_dir.parent
        )
    )
    try:
        _write_pilot_artifact(
            plans,
            staging,
            output_dir,
            build_manifest_head={
                "raw_root": str(raw_root),
                "shared_manifest": {
                    "path": str(shared_manifest),
                    "sha256": _sha256_bytes(manifest_bytes),
                    "size_bytes": len(manifest_bytes),
                    "schema_version": manifest_payload.get("schema_version"),
                },
                "source_hdf5_files": source_files,
                "split": {
                    "name": "train",
                    "split_seed": split_seed,
                    "dev_episodes_per_task": CANONICAL_DEV_EPISODES_PER_TASK,
                    "unit": "whole_episode",
                    "selection": "canonical RoboMMEHDF5Dataset stable-hash split",
                },
                "selection": {
                    "seed": seed,
                    "target_requests": target_requests,
                    "max_evidence_frames": max_evidence_frames,
                    "candidate_timeline_frames": DEFAULT_CANDIDATE_TIMELINE_FRAMES,
                    "causality": "all timeline timesteps are <= query_timestep",
                    "privileged_annotation_use": (
                        "frame selection only; values are not included in visual prompts"
                    ),
                },
            },
        )
        if output_dir.exists():
            raise FileExistsError(output_dir)
        staging.rename(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_dir


def _write_pilot_artifact(
    plans: Sequence[PlannedRequest],
    staging: Path,
    output_dir: Path,
    *,
    build_manifest_head: Mapping[str, Any],
) -> None:
    """Write images, requests, and manifest into ``staging`` under final paths."""
    (staging / "images").mkdir()
    images_root = (output_dir / "images").resolve(strict=False)
    rows: list[dict[str, Any]] = []
    request_provenance: list[dict[str, Any]] = []
    image_provenance: list[dict[str, Any]] = []
    for plan in plans:
        episode = plan.episode
        timeline: list[TimelineFrame] = []
        with h5py.File(episode.file_path, "r") as handle:
            group = handle[episode.episode_key]
            for frame in plan.frames:
                image_paths: dict[str, str] = {}
                for camera, dataset_name in (
                    ("front", "front_rgb"),
                    ("wrist", "wrist_rgb"),
                ):
                    image = _read_rgb(
                        group[f"timestep_{frame.timestep}"], dataset_name
                    )
                    relative = Path("images") / episode.task_name / episode.episode_key
                    relative /= f"timestep_{frame.timestep:06d}_{camera}.png"
                    absolute = (output_dir / relative).resolve(strict=False)
                    if images_root not in absolute.parents:
                        raise ValueError(
                            f"image path escapes the artifact image root: {relative}"
                        )
                    provenance = _save_png(staging / relative, image)
                    provenance.update(
                        {
                            "relative_path": str(relative),
                            "task_name": episode.task_name,
                            "episode_id": f"{episode.task_name}/{episode.episode_key}",
                            "timestep": frame.timestep,
                            "camera": camera,
                            "source_hdf5_dataset": (
                                f"/{episode.episode_key}/timestep_{frame.timestep}/"
                                f"obs/{dataset_name}"
                            ),
                        }
                    )
                    image_provenance.append(provenance)
                    image_paths[camera] = str(absolute)
                timeline.append(
                    TimelineFrame(
                        timestep=frame.timestep,
                        front_image=image_paths["front"],
                        wrist_image=image_paths["wrist"],
                        event_boundary=frame.event_boundary,
                        change_score=frame.change_score,
                    )
                )
        request = VQARequest(
            task_name=episode.task_name,
            suite_name=episode.suite_name,
            episode_id=f"{episode.task_name}/{episode.episode_key}",
            task_goal=episode.task_goal,
            query_timestep=plan.query_timestep,
            timeline=tuple(timeline),
            known_counts_by_family=plan.known_counts_by_family,
        )
        row = _request_to_dict(request)
        parsed = request_from_dict(row)
        if parsed.source_sha256() != request.source_sha256():
            raise RuntimeError("serialized request changed its source hash")
        rows.append(row)
        request_provenance.append(
            {
                "source_request_sha256": request.source_sha256(),
                "task_name": episode.task_name,
                "suite_name": episode.suite_name,
                "episode_id": request.episode_id,
                "query_timestep": plan.query_timestep,
                "timeline_timestamps": [frame.timestep for frame in timeline],
                "eligible_question_families": list(allowed_families(episode.task_name)),
                "annotation_source": plan.annotation_source,
            }
        )

    staged_requests = staging / "requests.jsonl"
    write_jsonl(staged_requests, rows)
    requests_bytes = staged_requests.read_bytes()
    if len(tuple(parse_request_jsonl(requests_bytes.decode("utf-8")))) != len(rows):
        raise RuntimeError("request JSONL round-trip count mismatch")

    task_counts = Counter(plan.episode.task_name for plan in plans)
    suite_counts = Counter(plan.episode.suite_name for plan in plans)
    family_counts: Counter[str] = Counter()
    for plan in plans:
        family_counts.update(allowed_families(plan.episode.task_name))
    build_manifest = {
        "schema_version": BUILD_MANIFEST_SCHEMA_VERSION,
        "builder_revision": BUILDER_REVISION,
        **build_manifest_head,
        "counts": {
            "requests": len(rows),
            "images": len(image_provenance),
            "by_task": dict(sorted(task_counts.items())),
            "by_suite": dict(sorted(suite_counts.items())),
            "eligible_requests_by_question_family": dict(
                sorted(family_counts.items())
            ),
        },
        "requests_jsonl": {
            "path": str(output_dir / "requests.jsonl"),
            "sha256": _sha256_bytes(requests_bytes),
            "size_bytes": len(requests_bytes),
            "request_schema_version": REQUEST_SCHEMA_VERSION,
        },
        "requests": request_provenance,
        "images": image_provenance,
        "runtime_versions": {
            "h5py": version("h5py"),
            "numpy": version("numpy"),
            "Pillow": version("Pillow"),
        },
        "request_v1_provenance_scope": {
            "per_request_hash_covers": "request fields and image path strings",
            "per_request_hash_does_not_cover": "image bytes or train-split identity",
            "binding": (
                "this companion manifest binds the canonical train split, source HDF5 "
                "signatures, request JSONL hash, and every encoded/source-RGB image hash"
            ),
            "future_schema_note": (
                "A self-contained request contract would require a breaking v2 with split "
                "and image content hashes; v1 is kept unchanged for current io compatibility."
            ),
        },
    }
    write_json(staging / "manifest.json", build_manifest)


__all__ = [
    "BUILD_MANIFEST_SCHEMA_VERSION",
    "BUILDER_REVISION",
    "CANONICAL_DEV_EPISODES_PER_TASK",
    "PlannedFrame",
    "PlannedRequest",
    "build_pilot_requests",
    "plan_pilot_requests",
]
