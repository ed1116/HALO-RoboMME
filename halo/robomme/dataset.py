"""Read-only RoboMME HDF5 dataset contract.

The adapter deliberately has no torch dependency so the raw schema, split,
and action-alignment rules can be validated before a model environment is
installed.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Literal, Sequence

import h5py
import numpy as np


TASK_SUITES: dict[str, tuple[str, ...]] = {
    "counting": ("BinFill", "PickXtimes", "SwingXtimes", "StopCube"),
    "permanence": (
        "VideoUnmask",
        "ButtonUnmask",
        "VideoUnmaskSwap",
        "ButtonUnmaskSwap",
    ),
    "reference": (
        "PickHighlight",
        "VideoRepick",
        "VideoPlaceButton",
        "VideoPlaceOrder",
    ),
    "imitation": ("MoveCube", "InsertPeg", "PatternLock", "RouteStick"),
}
TASKS: tuple[str, ...] = tuple(task for suite in TASK_SUITES.values() for task in suite)
TASK_TO_SUITE = {task: suite for suite, tasks in TASK_SUITES.items() for task in tasks}
MANIFEST_SCHEMA_VERSION = 1
MISSING_ACTION = np.zeros(8, dtype=np.float32)


def _numbered_key(value: str) -> int:
    return int(value.rsplit("_", 1)[1])


def _decode_text(value: object) -> str:
    if isinstance(value, np.ndarray):
        value = value.reshape(-1)[0]
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return str(value)


def _file_signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _read_is_video_demo(episode_group: h5py.Group, timestep: int) -> bool:
    return bool(episode_group[f"timestep_{timestep}"]["info"]["is_video_demo"][()])


def _execution_start(
    episode_group: h5py.Group,
    num_timesteps: int,
    *,
    validate_demo_prefix: bool,
) -> int:
    """Find the first execution frame.

    Full prefix validation requires one small HDF5 read per timestep and is
    intended for explicit validation runs. Normal manifest generation uses a
    logarithmic boundary search because the raw corpus is hundreds of GB.
    """
    if validate_demo_prefix:
        execution_start = num_timesteps
        for timestep in range(num_timesteps):
            is_demo = _read_is_video_demo(episode_group, timestep)
            if is_demo and execution_start != num_timesteps:
                raise ValueError("is_video_demo must be one contiguous prefix")
            if not is_demo and execution_start == num_timesteps:
                execution_start = timestep
    elif not _read_is_video_demo(episode_group, 0):
        execution_start = 0
    elif _read_is_video_demo(episode_group, num_timesteps - 1):
        execution_start = num_timesteps
    else:
        low = 0
        high = num_timesteps - 1
        while low + 1 < high:
            middle = (low + high) // 2
            if _read_is_video_demo(episode_group, middle):
                low = middle
            else:
                high = middle
        execution_start = high

    if execution_start == num_timesteps:
        raise ValueError("episode has no execution frames")
    return execution_start


@dataclass(frozen=True)
class _EpisodeMetadata:
    episode_key: str
    episode_id: int
    num_timesteps: int
    task_goal: str
    execution_start: int


@dataclass(frozen=True)
class _Episode:
    task_name: str
    suite_name: str
    file_path: Path
    episode_key: str
    episode_id: int
    num_timesteps: int
    task_goal: str
    execution_start: int
    global_start: int


@dataclass(frozen=True)
class RoboMMEPolicyInput:
    """Only fields available to a deployed RoboMME policy."""

    timestep: int
    task_goal: str
    front_rgb: np.ndarray
    wrist_rgb: np.ndarray
    state: np.ndarray
    previous_action: np.ndarray
    previous_action_valid: bool
    is_video_demo: bool
    is_execution_start: bool


@dataclass(frozen=True)
class RoboMMESample:
    task_name: str
    suite_name: str
    episode_id: str
    timestep: int
    task_goal: str
    front_rgb: np.ndarray
    wrist_rgb: np.ndarray
    joint_state: np.ndarray
    gripper_state: float
    state: np.ndarray
    previous_action: np.ndarray
    previous_action_valid: bool
    target_actions: np.ndarray
    action_valid_mask: np.ndarray
    is_video_demo: bool
    is_execution_start: bool
    is_subgoal_boundary: bool
    simple_subgoal: str
    grounded_subgoal: str

    def to_policy_input(self) -> RoboMMEPolicyInput:
        """Drop action targets and training-only privileged annotations."""
        return RoboMMEPolicyInput(
            timestep=self.timestep,
            task_goal=self.task_goal,
            front_rgb=self.front_rgb,
            wrist_rgb=self.wrist_rgb,
            state=self.state,
            previous_action=self.previous_action,
            previous_action_valid=self.previous_action_valid,
            is_video_demo=self.is_video_demo,
            is_execution_start=self.is_execution_start,
        )


@dataclass(frozen=True)
class DatasetStatistics:
    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    sample_count: int
    action_count: int
    provenance: dict[str, object] | None = None

    def normalize_state(self, value: np.ndarray) -> np.ndarray:
        return (value - self.state_mean) / self.state_std

    def denormalize_state(self, value: np.ndarray) -> np.ndarray:
        return value * self.state_std + self.state_mean

    def normalize_action(self, value: np.ndarray) -> np.ndarray:
        return (value - self.action_mean) / self.action_std

    def denormalize_action(self, value: np.ndarray) -> np.ndarray:
        return value * self.action_std + self.action_mean

    def to_json(self) -> str:
        payload: dict[str, object] = {
            "state_mean": self.state_mean.tolist(),
            "state_std": self.state_std.tolist(),
            "action_mean": self.action_mean.tolist(),
            "action_std": self.action_std.tolist(),
            "sample_count": self.sample_count,
            "action_count": self.action_count,
        }
        if self.provenance is not None:
            payload["provenance"] = self.provenance
        return json.dumps(payload, indent=2, sort_keys=True)


def _scan_task_file(
    task: str,
    file_path: Path,
    *,
    validate_demo_prefix: bool,
) -> list[_EpisodeMetadata]:
    metadata: list[_EpisodeMetadata] = []
    with h5py.File(file_path, "r") as handle:
        episode_keys = sorted(
            (key for key in handle if key.startswith("episode_")),
            key=_numbered_key,
        )
        for episode_key in episode_keys:
            episode_group = handle[episode_key]
            if "setup" not in episode_group:
                raise ValueError(f"missing setup group in {task}/{episode_key}")
            num_timesteps = len(episode_group) - 1
            if (
                num_timesteps <= 0
                or "timestep_0" not in episode_group
                or f"timestep_{num_timesteps - 1}" not in episode_group
                or f"timestep_{num_timesteps}" in episode_group
            ):
                raise ValueError(f"non-contiguous timesteps in {task}/{episode_key}")
            try:
                execution_start = _execution_start(
                    episode_group,
                    num_timesteps,
                    validate_demo_prefix=validate_demo_prefix,
                )
            except ValueError as error:
                raise ValueError(f"{task}/{episode_key}: {error}") from error
            metadata.append(
                _EpisodeMetadata(
                    episode_key=episode_key,
                    episode_id=_numbered_key(episode_key),
                    num_timesteps=num_timesteps,
                    task_goal=_decode_text(episode_group["setup"]["task_goal"][()]),
                    execution_start=execution_start,
                )
            )
    return metadata


def build_manifest(
    root: str | Path,
    *,
    validate_demo_prefix: bool = False,
) -> dict[str, object]:
    """Scan raw files once and return portable episode metadata.

    File size and nanosecond mtime are recorded so stale manifests fail closed.
    Set ``validate_demo_prefix`` for a slower, exhaustive flag scan.
    """
    root = Path(root)
    expected_files = {f"record_dataset_{task}.h5" for task in TASKS}
    actual_files = {path.name for path in root.glob("record_dataset_*.h5")}
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        extra = sorted(actual_files - expected_files)
        raise ValueError(f"RoboMME task files differ: missing={missing}, extra={extra}")

    task_entries: dict[str, object] = {}
    for task in TASKS:
        file_path = root / f"record_dataset_{task}.h5"
        episodes = _scan_task_file(
            task,
            file_path,
            validate_demo_prefix=validate_demo_prefix,
        )
        task_entries[task] = {
            "file_name": file_path.name,
            **_file_signature(file_path),
            "episodes": [
                {
                    "episode_key": episode.episode_key,
                    "episode_id": episode.episode_id,
                    "num_timesteps": episode.num_timesteps,
                    "task_goal": episode.task_goal,
                    "execution_start": episode.execution_start,
                }
                for episode in episodes
            ],
        }
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "demo_prefix_validation": "full" if validate_demo_prefix else "boundary",
        "tasks": task_entries,
    }


def write_manifest(
    root: str | Path,
    output_path: str | Path,
    *,
    validate_demo_prefix: bool = False,
) -> Path:
    """Build a manifest without silently replacing an existing artifact."""
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    manifest = build_manifest(root, validate_demo_prefix=validate_demo_prefix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(manifest, indent=2, sort_keys=True))
        output.write("\n")
    return output_path


def _load_manifest(
    root: Path,
    manifest_path: Path,
) -> dict[str, list[_EpisodeMetadata]]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read RoboMME manifest: {manifest_path}") from error
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported RoboMME manifest schema")
    task_entries = manifest.get("tasks")
    if not isinstance(task_entries, dict) or set(task_entries) != set(TASKS):
        raise ValueError("manifest must contain exactly the 16 RoboMME tasks")

    result: dict[str, list[_EpisodeMetadata]] = {}
    for task in TASKS:
        entry = task_entries[task]
        if not isinstance(entry, dict):
            raise ValueError(f"invalid manifest entry for {task}")
        expected_name = f"record_dataset_{task}.h5"
        if entry.get("file_name") != expected_name:
            raise ValueError(f"invalid file name in manifest for {task}")
        file_path = root / expected_name
        if not file_path.is_file():
            raise FileNotFoundError(file_path)
        signature = _file_signature(file_path)
        if any(entry.get(key) != value for key, value in signature.items()):
            raise ValueError(f"stale manifest signature for {file_path}")
        raw_episodes = entry.get("episodes")
        if not isinstance(raw_episodes, list) or not raw_episodes:
            raise ValueError(f"manifest has no episodes for {task}")

        episodes: list[_EpisodeMetadata] = []
        seen_keys: set[str] = set()
        for raw in raw_episodes:
            if not isinstance(raw, dict):
                raise ValueError(f"invalid episode metadata for {task}")
            try:
                episode = _EpisodeMetadata(
                    episode_key=str(raw["episode_key"]),
                    episode_id=int(raw["episode_id"]),
                    num_timesteps=int(raw["num_timesteps"]),
                    task_goal=str(raw["task_goal"]),
                    execution_start=int(raw["execution_start"]),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid episode metadata for {task}") from error
            if episode.episode_key in seen_keys:
                raise ValueError(f"duplicate episode key for {task}: {episode.episode_key}")
            if episode.episode_id != _numbered_key(episode.episode_key):
                raise ValueError(f"episode ID mismatch for {task}/{episode.episode_key}")
            if not (0 <= episode.execution_start < episode.num_timesteps):
                raise ValueError(f"invalid execution boundary for {task}/{episode.episode_key}")
            seen_keys.add(episode.episode_key)
            episodes.append(episode)
        result[task] = sorted(episodes, key=lambda episode: episode.episode_id)
    return result


def _read_vector(value: object, *, size: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be finite with shape ({size},)")
    return array


def _read_rgb(value: object, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"{label} must be uint8 RGB")
    return array


def _read_state(step: h5py.Group) -> tuple[np.ndarray, float, np.ndarray]:
    joint_state = _read_vector(
        step["obs"]["joint_state"][()], size=7, label="joint_state"
    )
    gripper_values = np.asarray(
        step["obs"]["gripper_state"][()], dtype=np.float32
    ).reshape(-1)
    if gripper_values.shape != (2,) or not np.all(np.isfinite(gripper_values)):
        raise ValueError("gripper_state must be finite with shape (2,)")
    gripper = float(gripper_values[0])
    state = np.concatenate(
        (joint_state, np.asarray([gripper], dtype=np.float32)), dtype=np.float32
    )
    return joint_state, gripper, state


def _read_action(episode_group: h5py.Group, timestep: int) -> np.ndarray:
    value = episode_group[f"timestep_{timestep}"]["action"]["joint_action"][()]
    action = _read_vector(value, size=8, label="joint_action")
    if action[-1] not in {-1.0, 1.0}:
        raise ValueError("joint_action gripper must be -1 (close) or +1 (open)")
    return action


class RoboMMEHDF5Dataset(Sequence[RoboMMESample]):
    """Lazy, read-only view over RoboMME episodes.

    The 90/10 development split is selected by stable hashes of whole episode
    identifiers. Pass a reusable manifest for real data to avoid rescanning HDF5
    metadata; omitting it remains convenient for small fixtures.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        horizon: int,
        split: Literal["train", "dev", "all"] = "train",
        split_seed: int = 0,
        dev_episodes_per_task: int = 10,
        tasks: Sequence[str] = TASKS,
        manifest_path: str | Path | None = None,
    ) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if split not in {"train", "dev", "all"}:
            raise ValueError(f"unsupported split: {split}")
        if not isinstance(dev_episodes_per_task, int) or dev_episodes_per_task < 0:
            raise ValueError("dev_episodes_per_task must be a non-negative integer")
        if split == "dev" and dev_episodes_per_task == 0:
            raise ValueError("a development split requires at least one episode")
        if not tasks:
            raise ValueError("tasks must not be empty")
        if len(set(tasks)) != len(tasks):
            raise ValueError("tasks must not contain duplicates")

        self.root = Path(root)
        self.horizon = horizon
        self.split = split
        self.split_seed = split_seed
        self.dev_episodes_per_task = dev_episodes_per_task
        self.tasks = tuple(tasks)
        self.manifest_path = Path(manifest_path) if manifest_path is not None else None
        self.manifest_sha256: str | None = None
        self._episodes: list[_Episode] = []
        self._task_episodes: dict[str, list[_Episode]] = {}
        self._ends: list[int] = []
        self._build_index()

    def _build_index(self) -> None:
        if self.manifest_path is None:
            metadata_by_task: dict[str, list[_EpisodeMetadata]] = {}
            for task in self.tasks:
                if task not in TASK_TO_SUITE:
                    raise ValueError(f"unknown RoboMME task: {task}")
                file_path = self.root / f"record_dataset_{task}.h5"
                if not file_path.is_file():
                    raise FileNotFoundError(file_path)
                metadata_by_task[task] = _scan_task_file(
                    task, file_path, validate_demo_prefix=False
                )
        else:
            metadata_by_task = _load_manifest(self.root, self.manifest_path)
            self.manifest_sha256 = hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()

        total = 0
        for task in self.tasks:
            if task not in TASK_TO_SUITE:
                raise ValueError(f"unknown RoboMME task: {task}")
            metadata = metadata_by_task[task]
            if len(metadata) <= self.dev_episodes_per_task:
                raise ValueError(
                    f"{task} has {len(metadata)} episodes, not enough for a "
                    f"{self.dev_episodes_per_task}-episode development split"
                )
            ranked = sorted(
                metadata,
                key=lambda episode: hashlib.sha256(
                    f"{self.split_seed}:{task}:{episode.episode_key}".encode("utf-8")
                ).digest(),
            )
            dev_keys = {
                episode.episode_key for episode in ranked[: self.dev_episodes_per_task]
            }
            if self.split == "train":
                selected = [episode for episode in metadata if episode.episode_key not in dev_keys]
            elif self.split == "dev":
                selected = [episode for episode in metadata if episode.episode_key in dev_keys]
            else:
                selected = metadata

            file_path = self.root / f"record_dataset_{task}.h5"
            task_episodes: list[_Episode] = []
            for item in selected:
                episode = _Episode(
                    task_name=task,
                    suite_name=TASK_TO_SUITE[task],
                    file_path=file_path,
                    episode_key=item.episode_key,
                    episode_id=item.episode_id,
                    num_timesteps=item.num_timesteps,
                    task_goal=item.task_goal,
                    execution_start=item.execution_start,
                    global_start=total,
                )
                total += episode.num_timesteps
                self._ends.append(total)
                self._episodes.append(episode)
                task_episodes.append(episode)
            self._task_episodes[task] = task_episodes

    def __len__(self) -> int:
        return self._ends[-1] if self._ends else 0

    @property
    def episode_counts(self) -> dict[str, int]:
        return {task: len(episodes) for task, episodes in self._task_episodes.items()}

    @property
    def timestep_count(self) -> int:
        return len(self)

    def _locate(self, index: int) -> tuple[_Episode, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode_position = bisect.bisect_right(self._ends, index)
        episode = self._episodes[episode_position]
        return episode, index - episode.global_start

    def __getitem__(self, index: int) -> RoboMMESample:
        episode, timestep = self._locate(index)
        with h5py.File(episode.file_path, "r") as handle:
            group = handle[episode.episode_key]
            step = group[f"timestep_{timestep}"]
            info = step["info"]
            is_video_demo = bool(info["is_video_demo"][()])
            joint_state, gripper, state = _read_state(step)

            if timestep == 0:
                previous_was_demo = False
            else:
                previous_was_demo = _read_is_video_demo(group, timestep - 1)
            previous_action_valid = (
                timestep > 0 and not is_video_demo and not previous_was_demo
            )
            previous_action = (
                _read_action(group, timestep - 1)
                if previous_action_valid
                else MISSING_ACTION.copy()
            )

            actions: list[np.ndarray] = []
            valid = np.zeros(self.horizon, dtype=np.bool_)
            last_action: np.ndarray | None = None
            for offset in range(self.horizon):
                target_timestep = timestep + offset
                if target_timestep < episode.num_timesteps:
                    last_action = _read_action(group, target_timestep)
                    valid[offset] = True
                if last_action is None:
                    raise RuntimeError("episode contains no action at requested timestep")
                actions.append(last_action.copy())

            return RoboMMESample(
                task_name=episode.task_name,
                suite_name=episode.suite_name,
                episode_id=f"{episode.task_name}/{episode.episode_key}",
                timestep=timestep,
                task_goal=episode.task_goal,
                front_rgb=_read_rgb(step["obs"]["front_rgb"][()], label="front_rgb"),
                wrist_rgb=_read_rgb(step["obs"]["wrist_rgb"][()], label="wrist_rgb"),
                joint_state=joint_state,
                gripper_state=gripper,
                state=state,
                previous_action=previous_action,
                previous_action_valid=previous_action_valid,
                target_actions=np.stack(actions),
                action_valid_mask=valid,
                is_video_demo=is_video_demo,
                is_execution_start=(not is_video_demo) and (
                    timestep == 0 or previous_was_demo
                ),
                is_subgoal_boundary=bool(info["is_subgoal_boundary"][()]),
                simple_subgoal=_decode_text(info["simple_subgoal"][()]),
                grounded_subgoal=_decode_text(info["grounded_subgoal"][()]),
            )

    def execution_indices(self) -> Iterator[int]:
        """Yield every behavior-cloning index without video-demo frames."""
        for episode in self._episodes:
            start = episode.global_start + episode.execution_start
            stop = episode.global_start + episode.num_timesteps
            yield from range(start, stop)

    def balanced_indices(
        self,
        count: int,
        *,
        seed: int,
        execution_only: bool = True,
    ) -> Iterator[int]:
        """Yield deterministic, task-balanced indices.

        Behavior-cloning-safe execution-only sampling is the default. Callers
        building visual history or VQA data must explicitly request all frames.
        """
        if count < 0:
            raise ValueError("count must be non-negative")
        rng = random.Random(seed)
        produced = 0
        while produced < count:
            task_block = list(self.tasks)
            rng.shuffle(task_block)
            for task in task_block:
                episodes = self._task_episodes[task]
                episode = rng.choice(episodes)
                low = episode.execution_start if execution_only else 0
                timestep = rng.randrange(low, episode.num_timesteps)
                yield episode.global_start + timestep
                produced += 1
                if produced == count:
                    return

    def compute_training_statistics(self) -> DatasetStatistics:
        """Compute image-free statistics from training execution samples only."""
        if self.split != "train":
            raise ValueError("normalization statistics must use the training split")

        def state_action_pairs() -> Iterator[tuple[np.ndarray, np.ndarray]]:
            current_path: Path | None = None
            handle: h5py.File | None = None
            try:
                for episode in self._episodes:
                    if current_path != episode.file_path:
                        if handle is not None:
                            handle.close()
                        handle = h5py.File(episode.file_path, "r")
                        current_path = episode.file_path
                    assert handle is not None
                    group = handle[episode.episode_key]
                    for timestep in range(episode.execution_start, episode.num_timesteps):
                        _, _, state = _read_state(group[f"timestep_{timestep}"])
                        yield state, _read_action(group, timestep)[None, :]
            finally:
                if handle is not None:
                    handle.close()

        provenance: dict[str, object] = {
            "split": self.split,
            "split_seed": self.split_seed,
            "dev_episodes_per_task": self.dev_episodes_per_task,
            "horizon": self.horizon,
            "tasks": list(self.tasks),
            "episodes": {
                task: [episode.episode_key for episode in episodes]
                for task, episodes in self._task_episodes.items()
            },
            "manifest_sha256": self.manifest_sha256,
        }
        return _compute_statistics(state_action_pairs(), provenance=provenance)


def _compute_statistics(
    state_action_pairs: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    provenance: dict[str, object] | None = None,
) -> DatasetStatistics:
    state_sum = np.zeros(8, dtype=np.float64)
    state_square_sum = np.zeros(8, dtype=np.float64)
    action_sum = np.zeros(8, dtype=np.float64)
    action_square_sum = np.zeros(8, dtype=np.float64)
    sample_count = 0
    action_count = 0
    for state_value, action_values in state_action_pairs:
        state = _read_vector(state_value, size=8, label="state")
        actions = np.asarray(action_values, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1:] != (8,) or not np.all(np.isfinite(actions)):
            raise ValueError("actions must be finite with shape (N, 8)")
        state64 = state.astype(np.float64)
        actions64 = actions.astype(np.float64)
        state_sum += state64
        state_square_sum += state64 * state64
        action_sum += actions64.sum(axis=0)
        action_square_sum += (actions64 * actions64).sum(axis=0)
        sample_count += 1
        action_count += len(actions64)
    if sample_count == 0 or action_count == 0:
        raise ValueError("cannot compute statistics from an empty sample set")
    state_mean = state_sum / sample_count
    action_mean = action_sum / action_count
    state_std = np.sqrt(np.maximum(state_square_sum / sample_count - state_mean**2, 0.0))
    action_std = np.sqrt(
        np.maximum(action_square_sum / action_count - action_mean**2, 0.0)
    )
    return DatasetStatistics(
        state_mean=state_mean.astype(np.float32),
        state_std=np.maximum(state_std, 1e-6).astype(np.float32),
        action_mean=action_mean.astype(np.float32),
        action_std=np.maximum(action_std, 1e-6).astype(np.float32),
        sample_count=sample_count,
        action_count=action_count,
        provenance=provenance,
    )


def compute_statistics(samples: Iterable[RoboMMESample]) -> DatasetStatistics:
    """Compute statistics for explicitly supplied samples.

    Training artifacts should use ``RoboMMEHDF5Dataset.compute_training_statistics``
    so split provenance and execution-only filtering are enforced.
    """
    return _compute_statistics(
        (
            (sample.state, sample.target_actions[sample.action_valid_mask])
            for sample in samples
        )
    )
