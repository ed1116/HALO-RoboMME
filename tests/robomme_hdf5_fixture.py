from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from halo.robomme import TASKS


def write_fixture(root: Path, *, episodes: int = 4, timesteps: int = 3) -> Path:
    for task_index, task in enumerate(TASKS):
        is_stick_task = task in {"PatternLock", "RouteStick"}
        with h5py.File(root / f"record_dataset_{task}.h5", "w") as handle:
            for episode_id in range(episodes):
                episode = handle.create_group(f"episode_{episode_id}")
                setup = episode.create_group("setup")
                setup.create_dataset(
                    "task_goal", data=np.asarray([f"goal for {task}".encode()], dtype="S")
                )
                for timestep in range(timesteps):
                    step = episode.create_group(f"timestep_{timestep}")
                    obs = step.create_group("obs")
                    obs.create_dataset(
                        "front_rgb",
                        data=np.full((4, 4, 3), task_index + timestep, dtype=np.uint8),
                    )
                    obs.create_dataset(
                        "wrist_rgb",
                        data=np.full((4, 4, 3), episode_id + timestep, dtype=np.uint8),
                    )
                    obs.create_dataset(
                        "joint_state", data=np.arange(7, dtype=np.float32) + timestep
                    )
                    obs.create_dataset(
                        "gripper_state",
                        data=np.zeros(2, dtype=np.float32)
                        if is_stick_task
                        else np.full(
                            2, 0.04 if timestep % 2 == 0 else 0.0, dtype=np.float32
                        ),
                    )
                    action = step.create_group("action")
                    gripper_action = (
                        -1.0
                        if is_stick_task
                        else (1.0 if timestep % 2 == 0 else -1.0)
                    )
                    action.create_dataset(
                        "joint_action",
                        data=np.concatenate(
                            (
                                np.arange(7, dtype=np.float32) + timestep,
                                np.asarray([gripper_action], dtype=np.float32),
                            )
                        ),
                    )
                    info = step.create_group("info")
                    info.create_dataset("is_video_demo", data=timestep < 2)
                    info.create_dataset("is_subgoal_boundary", data=timestep == 2)
                    info.create_dataset("simple_subgoal", data=np.bytes_(f"simple-{timestep}"))
                    info.create_dataset(
                        "grounded_subgoal", data=np.bytes_(f"grounded-{timestep}")
                    )
    return root
