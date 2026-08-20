from __future__ import annotations

import json
import os
from dataclasses import fields

import h5py
import numpy as np
import pytest

from halo.robomme import (
    TASKS,
    RoboMMEHDF5Dataset,
    build_manifest,
    write_manifest,
)
from robomme_hdf5_fixture import write_fixture


def test_episode_split_is_disjoint_and_exact(tmp_path):
    write_fixture(tmp_path)
    train = RoboMMEHDF5Dataset(
        tmp_path, horizon=4, split="train", dev_episodes_per_task=1
    )
    dev = RoboMMEHDF5Dataset(tmp_path, horizon=4, split="dev", dev_episodes_per_task=1)

    assert train.episode_counts == {task: 3 for task in TASKS}
    assert dev.episode_counts == {task: 1 for task in TASKS}
    train_ids = {(sample.task_name, sample.episode_id) for sample in train}
    dev_ids = {(sample.task_name, sample.episode_id) for sample in dev}
    assert train_ids.isdisjoint(dev_ids)


def test_split_constraints_fail_early(tmp_path):
    write_fixture(tmp_path)
    with pytest.raises(ValueError, match="non-negative"):
        RoboMMEHDF5Dataset(
            tmp_path, horizon=4, split="train", dev_episodes_per_task=-1
        )
    with pytest.raises(ValueError, match="not enough"):
        RoboMMEHDF5Dataset(
            tmp_path, horizon=4, split="train", dev_episodes_per_task=4
        )
    with pytest.raises(ValueError, match="development split"):
        RoboMMEHDF5Dataset(
            tmp_path, horizon=4, split="dev", dev_episodes_per_task=0
        )


def test_sample_shapes_execution_boundary_and_terminal_padding(tmp_path):
    write_fixture(tmp_path)
    dataset = RoboMMEHDF5Dataset(
        tmp_path, horizon=4, split="all", dev_episodes_per_task=1
    )

    first = dataset[0]
    assert first.episode_id == "BinFill/episode_0"
    assert first.front_rgb.shape == (4, 4, 3)
    assert first.wrist_rgb.shape == (4, 4, 3)
    assert first.front_rgb.dtype == np.uint8
    assert first.wrist_rgb.dtype == np.uint8
    assert first.state.shape == (8,)
    np.testing.assert_allclose(first.state[-1], 0.04)
    np.testing.assert_array_equal(first.previous_action, np.zeros(8, dtype=np.float32))
    assert not first.previous_action_valid
    assert first.target_actions.shape == (4, 8)
    assert first.action_valid_mask.tolist() == [True, True, True, False]
    np.testing.assert_array_equal(first.target_actions[2], first.target_actions[3])
    assert set(first.target_actions[:, -1]) <= {-1.0, 1.0}
    assert first.is_video_demo
    assert not first.is_execution_start

    execution_start = dataset[2]
    assert not execution_start.is_video_demo
    assert execution_start.is_execution_start
    assert not execution_start.previous_action_valid
    np.testing.assert_array_equal(
        execution_start.previous_action, np.zeros(8, dtype=np.float32)
    )
    assert execution_start.action_valid_mask.tolist() == [True, False, False, False]
    np.testing.assert_array_equal(
        execution_start.target_actions,
        np.repeat(execution_start.target_actions[:1], 4, axis=0),
    )


def test_previous_action_is_available_only_after_execution_start(tmp_path):
    write_fixture(tmp_path, timesteps=4)
    dataset = RoboMMEHDF5Dataset(
        tmp_path, horizon=2, split="all", dev_episodes_per_task=1
    )

    after_start = dataset[3]
    assert not after_start.is_video_demo
    assert not after_start.is_execution_start
    assert after_start.previous_action_valid
    np.testing.assert_array_equal(after_start.previous_action, dataset[2].target_actions[0])


def test_stick_tasks_keep_zero_gripper_state_and_closed_action(tmp_path):
    write_fixture(tmp_path)
    dataset = RoboMMEHDF5Dataset(
        tmp_path,
        horizon=3,
        split="all",
        dev_episodes_per_task=1,
        tasks=("PatternLock", "RouteStick"),
    )

    for task in ("PatternLock", "RouteStick"):
        sample = next(sample for sample in dataset if sample.task_name == task)
        assert sample.gripper_state == 0.0
        assert sample.state[-1] == 0.0
        np.testing.assert_array_equal(sample.target_actions[:, -1], -1.0)


def test_task_balanced_sampling_is_deterministic_and_execution_only(tmp_path):
    write_fixture(tmp_path)
    dataset = RoboMMEHDF5Dataset(
        tmp_path, horizon=2, split="train", dev_episodes_per_task=1
    )

    indices_a = list(dataset.balanced_indices(len(TASKS) * 3, seed=7))
    indices_b = list(dataset.balanced_indices(len(TASKS) * 3, seed=7))
    assert indices_a == indices_b
    samples = [dataset[index] for index in indices_a]
    assert not any(sample.is_video_demo for sample in samples)
    assert {task: sum(sample.task_name == task for sample in samples) for task in TASKS} == {
        task: 3 for task in TASKS
    }


def test_policy_projection_excludes_labels_and_privileged_annotations(tmp_path):
    write_fixture(tmp_path)
    dataset = RoboMMEHDF5Dataset(
        tmp_path, horizon=2, split="all", dev_episodes_per_task=1
    )

    policy_input = dataset[2].to_policy_input()
    policy_fields = {field.name for field in fields(type(policy_input))}
    assert policy_fields.isdisjoint(
        {
            "target_actions",
            "action_valid_mask",
            "is_subgoal_boundary",
            "simple_subgoal",
            "grounded_subgoal",
        }
    )
    assert policy_input.task_goal == dataset[2].task_goal
    assert not policy_input.previous_action_valid


def test_manifest_round_trip_and_stale_signature_detection(tmp_path):
    write_fixture(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    write_manifest(tmp_path, manifest_path, validate_demo_prefix=True)

    dataset = RoboMMEHDF5Dataset(
        tmp_path,
        horizon=2,
        split="train",
        dev_episodes_per_task=1,
        manifest_path=manifest_path,
    )
    assert dataset.episode_counts == {task: 3 for task in TASKS}
    assert dataset.manifest_sha256 is not None

    source = tmp_path / "record_dataset_BinFill.h5"
    stat = source.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    with pytest.raises(ValueError, match="stale manifest signature"):
        RoboMMEHDF5Dataset(
            tmp_path,
            horizon=2,
            split="train",
            dev_episodes_per_task=1,
            manifest_path=manifest_path,
        )


def test_full_manifest_validation_rejects_nonmonotonic_demo_flags(tmp_path):
    write_fixture(tmp_path, timesteps=4)
    source = tmp_path / "record_dataset_BinFill.h5"
    with h5py.File(source, "r+") as handle:
        handle["episode_0/timestep_3/info/is_video_demo"][()] = True

    with pytest.raises(ValueError, match="contiguous prefix"):
        build_manifest(tmp_path, validate_demo_prefix=True)


def test_sample_validation_rejects_nonfinite_actions(tmp_path):
    write_fixture(tmp_path)
    source = tmp_path / "record_dataset_BinFill.h5"
    with h5py.File(source, "r+") as handle:
        action = handle["episode_0/timestep_0/action/joint_action"]
        value = action[()]
        value[0] = np.nan
        action[...] = value

    dataset = RoboMMEHDF5Dataset(
        tmp_path, horizon=2, split="all", dev_episodes_per_task=1
    )
    with pytest.raises(ValueError, match="joint_action"):
        _ = dataset[0]


def test_sample_validation_rejects_invalid_gripper_action(tmp_path):
    write_fixture(tmp_path)
    source = tmp_path / "record_dataset_BinFill.h5"
    with h5py.File(source, "r+") as handle:
        action = handle["episode_0/timestep_0/action/joint_action"]
        value = action[()]
        value[-1] = 0.0
        action[...] = value

    dataset = RoboMMEHDF5Dataset(
        tmp_path, horizon=2, split="all", dev_episodes_per_task=1
    )
    with pytest.raises(ValueError, match="gripper"):
        _ = dataset[0]


def test_training_statistics_are_execution_only_and_record_provenance(tmp_path):
    write_fixture(tmp_path)
    train = RoboMMEHDF5Dataset(
        tmp_path, horizon=4, split="train", dev_episodes_per_task=1
    )
    stats = train.compute_training_statistics()

    sample = train[next(train.execution_indices())]
    np.testing.assert_allclose(
        stats.denormalize_state(stats.normalize_state(sample.state)),
        sample.state,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        stats.denormalize_action(stats.normalize_action(sample.target_actions)),
        sample.target_actions,
        atol=1e-5,
    )
    assert stats.sample_count == 16 * 3
    assert stats.action_count == stats.sample_count
    payload = json.loads(stats.to_json())
    assert payload["provenance"]["split"] == "train"
    assert payload["provenance"]["split_seed"] == 0
    assert payload["provenance"]["dev_episodes_per_task"] == 1
    assert len(payload["provenance"]["episodes"]["BinFill"]) == 3

    dev = RoboMMEHDF5Dataset(
        tmp_path, horizon=4, split="dev", dev_episodes_per_task=1
    )
    with pytest.raises(ValueError, match="training split"):
        dev.compute_training_statistics()
