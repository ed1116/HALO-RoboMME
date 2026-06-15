import torch
import numpy as np
import torchvision.transforms as transforms
from typing import Optional, Union

from halo.data.utils import convert_multi_step
from halo.util.args import DatasetConfig, SharedConfig
from halo.data.utils import load_json, IMAGENET_MEAN, IMAGENET_STD

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata


def make_delta_timestamps(delta_indices: list[int] | None, fps: int, reverse: bool = False) -> list[float]:
    if delta_indices is None:
        return [0]
    multiplier = -1 if reverse else 1
    return [i * multiplier / fps for i in delta_indices]

class LeRobotDatasetWrapper(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_config: DatasetConfig,
        shared_config: SharedConfig,
        vision_transform: transforms.Compose,
        no_aug_vision_transform: Optional[transforms.Compose] = None,
        split: str = "train",
    ):
        # 1. Basic configuration setup
        self.dataset_config = dataset_config
        self.shared_config = shared_config
        self.split = split
        self.vision_transform = vision_transform
        self.no_aug_vision_transform = (
            no_aug_vision_transform
            if no_aug_vision_transform is not None
            else vision_transform
        )
        self.image_mean = IMAGENET_MEAN
        self.image_std = IMAGENET_STD

        json_path = dataset_config.dataset_json if split == "train" else dataset_config.dataset_val_json
        if isinstance(json_path, list):
            assert (
                len(json_path) == 1
            ), "LeRobotDataset only supports one dataset JSON configuration file."
            json_path = json_path[0]
        dataset_metadata = load_json(json_path)

        # 3. Store data keys and direct HDF5 paths from metadata
        self.dataset_paths = dataset_metadata["dataset_path"]
        self.seq_length = shared_config.seq_length
        self.num_pred_steps = shared_config.num_pred_steps
        self.total_seq_length = self.seq_length + self.num_pred_steps
        self.downsample_obs = shared_config.downsample_obs
        self.downsample_act = self.downsample_obs
        assert self.seq_length >= self.downsample_obs, f"seq_length must be greater than downsample_obs: {self.seq_length} is not greater than {self.downsample_obs}"
        self.n_examples_only = dataset_config.n_examples_only
        self.image_keys = dataset_metadata.get("image_keys", ['observation.image.img1'])
        self.proprio_keys = dataset_metadata.get("proprio_keys", ['observation.state'])
        self.low_dim_keys = dataset_metadata.get("low_dim_keys", [])
        self.action_keys = dataset_metadata.get("action_keys", ['action'])
        self.extra_keys = ['policy_mode'] if 'memory' in self.dataset_paths[0] else []

        self.all_keys = self.image_keys + self.low_dim_keys + self.extra_keys + self.action_keys + self.proprio_keys
        self.return_keys = ['observation', 'proprio'] + self.action_keys + self.extra_keys
        self.no_pad_return_keys = ['task', 'task_clip_embedding']

        self.proprio_noise = dataset_config.proprio_noise
        self.action_noise = dataset_config.action_noise

        dataset_metadatas = [
            LeRobotDatasetMetadata(
                dataset_id,
                force_cache_sync=False,
            ) for dataset_id in self.dataset_paths
        ]
        # To perform action chunking, ACT expects a given number of actions as targets
        assert self.seq_length % self.downsample_obs == 0, f"seq_length must be divisible by downsample_obs: {self.seq_length} is not divisible by {self.downsample_obs}"
        assert self.total_seq_length % self.downsample_obs == 0, f"total_seq_length must be divisible by downsample_obs: {self.total_seq_length} is not divisible by {self.downsample_obs}"
        # assert dataset_metadatas[0].fps % self.downsample_obs == 0, f"fps must be divisible by downsample_obs: {dataset_metadatas[0].fps} is not divisible by {self.downsample_obs}"
        # sample both ways and then strip out maximum padded frames from left and the remaining from right
        delta_indices = np.arange(-self.total_seq_length, self.total_seq_length)
        delta_timestamps = {
            k: make_delta_timestamps(delta_indices, dataset_metadatas[0].fps) \
                    for k in self.all_keys
        }
        # Instantiate the dataset
        datasets = [
            LeRobotDataset(
                dataset_id,
                delta_timestamps=delta_timestamps,
                force_cache_sync=False,
                video_backend="pyav",
            ) for dataset_id in self.dataset_paths
        ]
        want = list(self.all_keys) + ["timestamp", "index", "episode_index", "task_index", "task", "frame_index", "task_clip_embedding"]
        have = set(datasets[0].hf_dataset.column_names)
        drop = [c for c in have if c not in want]
        for dataset in datasets:
            dataset.hf_dataset = dataset.hf_dataset.remove_columns(drop)
        self.concat_datasets = torch.utils.data.ConcatDataset(datasets)
        self.shuffled_index_to_index_mapping = {i: i for i in range(len(self.concat_datasets))}

    def shuffle_dataset(self, seed: int = 0) -> None:
        rng = np.random.RandomState(seed=seed)
        keys = list(self.shuffled_index_to_index_mapping.keys())
        rng.shuffle(keys)
        self.shuffled_index_to_index_mapping = {key: i for i, key in enumerate(keys)}
        return

    def __len__(self):
        return len(self.concat_datasets)
    
    def __getitem__(self, index: int) -> dict:
        '''
            observation.image.robot0_agentview_center <class 'torch.Tensor'> torch.Size([T, C, H, W])
            observation.image.robot0_eye_in_hand <class 'torch.Tensor'> torch.Size([T, C, H, W])
            observation.state <class 'torch.Tensor'> torch.Size([T, D])
            action <class 'torch.Tensor'> torch.Size([T, A])
            policy_mode <class 'torch.Tensor'> torch.Size([T])
            timestamp <class 'torch.Tensor'> torch.Size([])
            frame_index <class 'torch.Tensor'> torch.Size([])
            episode_index <class 'torch.Tensor'> torch.Size([])
            index <class 'torch.Tensor'> torch.Size([])
            task_index <class 'torch.Tensor'> torch.Size([])
            observation.image.robot0_agentview_center_is_pad <class 'torch.Tensor'> torch.Size([T])
            observation.image.robot0_eye_in_hand_is_pad <class 'torch.Tensor'> torch.Size([T])
            policy_mode_is_pad <class 'torch.Tensor'> torch.Size([T])
            observation.state_is_pad <class 'torch.Tensor'> torch.Size([T])
            action_is_pad <class 'torch.Tensor'> torch.Size([T])
            task <class 'str'> 84
        '''
        index = self.shuffled_index_to_index_mapping[index]
        data = self.concat_datasets[index]
        action_key = self.action_keys[0]
        T = data[action_key].shape[0]
        # location where first non-padded step is located.
        max_left_padding = torch.where(~data[action_key + "_is_pad"])[0][0]


        data["observation"] = torch.stack(
            [data[key] for key in self.image_keys], dim=1
        )
        data["observation_is_pad"] = torch.stack(
            [data[key + "_is_pad"] for key in self.image_keys], dim=1
        )
        data['proprio'] = data['observation.state'].unsqueeze(1)
        data['proprio_is_pad'] = data['observation.state_is_pad'].unsqueeze(1)
        data['task_embedding'] = data['task_clip_embedding']
        action = data[action_key]
        action = (
            convert_multi_step(action, self.num_pred_steps)
            if self.num_pred_steps > 1
            else action.unsqueeze(1)
        )
        data['action'] = action
        if self.proprio_noise > 0:
            data['proprio'] += torch.normal(mean=0, std=self.proprio_noise, size=data['proprio'].shape)
        if self.action_noise > 0:
            data['action'] += torch.normal(mean=0, std=self.action_noise, size=data['action'].shape)

        data_final = {
            k: data[k][max_left_padding:][::self.downsample_obs][: self.seq_length // self.downsample_obs]
            if (isinstance(data[k], torch.Tensor) or isinstance(data[k], np.ndarray)) and k not in self.no_pad_return_keys
            else data[k]
            for k in self.return_keys + [k + '_is_pad' for k in self.return_keys] + self.no_pad_return_keys
        }
        # create the prompt mask for action loss calculation. zero means not used for loss calculation.
        prompt_mask = ~(data_final['action_is_pad'].clone())
        if 'policy_mode' in data_final:
            assert prompt_mask.shape == data_final['policy_mode'].shape
            prompt_mask = prompt_mask * (data_final['policy_mode'] == 0)
        data_final['prompt_mask'] = prompt_mask.unsqueeze(-1).unsqueeze(-1)
        # remove all keys with 'is_pad' suffix
        data_final = {k: v for k, v in data_final.items() if (not k.endswith('_is_pad')) and (not k.endswith('policy_mode')) and (not k == 'task')}
        data_final['language_embedding'] = data_final['task_clip_embedding']
        return data_final

    def save_split(self, path : str):
        pass