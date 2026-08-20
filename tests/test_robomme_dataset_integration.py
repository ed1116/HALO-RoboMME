from pathlib import Path

import pytest

from halo.robomme import TASKS, RoboMMEHDF5Dataset


RAW_DATA = Path("/data/ed1116/Datasets/robomme_data_h5")
SHARED_MANIFEST = Path("/data/ed1116/robomme/manifests/robomme_hdf5_v1.json")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not RAW_DATA.is_dir(), reason="raw RoboMME data is unavailable"),
    pytest.mark.skipif(
        not SHARED_MANIFEST.is_file(),
        reason="explicit RoboMME manifest has not been generated",
    ),
]


def test_real_dataset_manifest():
    dataset = RoboMMEHDF5Dataset(
        RAW_DATA, horizon=20, split="all", manifest_path=SHARED_MANIFEST
    )

    assert dataset.episode_counts == {task: 100 for task in TASKS}
    assert 760_000 <= dataset.timestep_count <= 780_000
    for task in TASKS:
        episode = dataset._task_episodes[task][0]
        first = dataset[episode.global_start]
        execution_start = dataset[episode.global_start + episode.execution_start]
        terminal = dataset[episode.global_start + episode.num_timesteps - 1]

        assert first.task_name == task
        assert first.front_rgb.shape == (256, 256, 3)
        assert first.wrist_rgb.shape == (256, 256, 3)
        assert first.state.shape == (8,)
        assert first.target_actions.shape == (20, 8)
        assert set(first.target_actions[:, -1]) <= {-1.0, 1.0}

        assert execution_start.is_execution_start
        assert not execution_start.is_video_demo
        assert not execution_start.previous_action_valid
        assert terminal.action_valid_mask.tolist() == [True] + [False] * 19
