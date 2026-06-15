import os
import h5py
import torch
import numpy as np
from typing import Optional, List, Dict, Any
from termcolor import colored
from tqdm import tqdm
import copy
from halo.util.args import GPTDatasetConfig, SharedConfig
from halo.util.misc import get_data_base_dir
import halo.data.utils as data_utils
import halo.util.misc as misc_utils

class GPTDataset(torch.utils.data.Dataset):
    """
    Simple dataset class for GPT-style training.
    Reads HDF5 files and outputs variable-length sequences with timestep information.
    """
    
    def __init__(
        self,
        action_index_mapping: Dict[str, Dict[str, int]],
        dataset_config: GPTDatasetConfig,
        shared_config: SharedConfig,
        split: str = "train",
        seed: int = 0,
    ):
        json_path = dataset_config.dataset_json
        if isinstance(json_path, list):
            assert (
                len(json_path) == 1
            ), "Dataset only supports one dataset JSON configuration file."
            json_path = json_path[0]
        dataset_metadata = data_utils.load_json(json_path)

        # 3. Store data keys and direct HDF5 paths from metadata
        # self.dataset_paths = dataset_metadata["dataset_path"]
        self.img_keys = dataset_metadata.get("image_keys", [])
        # we are assuming that the gripper is a separate key
        self.proprio_keys = dataset_metadata.get("proprio_keys", [])
        assert any("gripper" in k for k in self.proprio_keys), "Atleast one proprio key must contain 'gripper' in the key name."
        self.action_keys = dataset_metadata.get("action_keys", [])
        bbox_keys, bbox_names_keys, bbox_mask_keys, bbox_cc_keys = misc_utils.convert_image_keys_to_bbox_keys(self.img_keys)
        self.bbox_keys = bbox_keys + bbox_names_keys + bbox_mask_keys
        

        self.output_keys = self.img_keys + self.bbox_keys + self.proprio_keys + self.action_keys
        self.action_index_mapping = action_index_mapping
        self.dataset_config = dataset_config
        self.shared_config = shared_config
        self.split = split

        self.max_dataset_size = dataset_config.max_dataset_size
        
        # Get HDF5 file paths
        self.hdf5_paths = [os.path.join(dataset_config.base_dir, hdf5_path) for hdf5_path in dataset_config.hdf5_paths]
        
        # Control frequency for time calculation
        self.control_freq = dataset_config.control_freq
        
        # Whether to output full episodes or chunks
        self.output_full_episode = dataset_config.output_full_episode
        self.use_same_chunk_length = dataset_config.use_same_chunk_length
        self.min_chunk_length = dataset_config.min_chunk_length

        # Downsample observations by a factor of k
        self.downsample_obs = shared_config.downsample_obs
        
        # Scan HDF5 files and build index
        self._build_index(seed=seed)

        summary_data = {}
        for hdf5_path in dataset_config.hdf5_paths:
            full_hdf5_path = os.path.join(dataset_config.base_dir, hdf5_path)
            rel_hdf5_path = os.path.relpath(hdf5_path, dataset_config.base_dir)
            pathname = os.path.join(os.path.dirname(full_hdf5_path), "summary.json")
            if not os.path.exists(pathname):
                hdf5_basename = os.path.basename(full_hdf5_path).replace('.hdf5', '')
                pathname = os.path.join(os.path.dirname(full_hdf5_path), f"{hdf5_basename}_summary.json")
                assert os.path.exists(pathname), f"Summary file not found at {pathname}"
            summary_data[full_hdf5_path] = data_utils.load_json(pathname)[rel_hdf5_path]
        self.summary_data = summary_data
    
    def _build_index(self, seed: int):
        """
        Scan HDF5 files and build index of (hdf5_path, demo_key, start_idx, end_idx) tuples.
        """
        self._index = []
        rng = np.random.RandomState(seed=seed)  # Use fixed seed for reproducibility during indexing
        
        print(f"Scanning {len(self.hdf5_paths)} HDF5 files for split '{self.split}'...")
        for hdf5_path in tqdm(self.hdf5_paths, desc="Scanning HDF5 files"):
            data_dir = get_data_base_dir(hdf5_path, None) if hasattr(self, 'shared_config') else None
            if data_dir and os.path.exists(data_dir):
                hdf5_path = os.path.join(data_dir, hdf5_path)
            
            if not os.path.exists(hdf5_path):
                print(colored(f"Warning: HDF5 file does not exist: {hdf5_path}", "yellow"))
                continue
            
            with h5py.File(hdf5_path, "r") as f:
                if "data" not in f:
                    print(colored(f"Warning: 'data' group not found in {hdf5_path}", "yellow"))
                    continue
                
                data_group = f["data"]
                demo_keys = list(data_group.keys())
                demo_keys = sorted(demo_keys, key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 0)
                
                for demo_key in demo_keys:
                    demo_group = data_group[demo_key]
                    
                    # Get trajectory length from first available key
                    traj_len = None
                    for key in self.output_keys:
                        if key in demo_group:
                            if isinstance(demo_group[key], h5py.Dataset):
                                traj_len = demo_group[key].shape[0]
                                break
                    
                    if traj_len is None or traj_len == 0:
                        continue
                    
                    # Add indices based on output mode
                    if self.output_full_episode:
                        # Add full episode
                        self._index.append((hdf5_path, demo_key, 0, traj_len))
                    elif self.use_same_chunk_length:
                        chunk_length = self.min_chunk_length // self.downsample_obs
                        max_start = max(0, traj_len - chunk_length * self.downsample_obs)
                        for st in range(0, max_start + self.downsample_obs, 1):
                            start_idx = st
                            end_idx = min(traj_len, start_idx + chunk_length * self.downsample_obs)
                            self._index.append((hdf5_path, demo_key, start_idx, end_idx))
                    else:
                        assert self.min_chunk_length <= traj_len, f"{self.min_chunk_length=} <= {traj_len=}"
                        # Sample chunk length (between min_chunk_length and traj_len)
                        for chunk_length in range(self.min_chunk_length//self.downsample_obs, traj_len//self.downsample_obs + 1):
                            # chunk_length = (rng.randint(self.min_chunk_length, traj_len + 1) // self.downsample_obs) + 1
                            # Sample starting position such that chunk fits
                            max_start = max(0, traj_len - chunk_length * self.downsample_obs)
                            for st in range(0, max_start + self.downsample_obs, 1):
                                start_idx = st
                                end_idx = min(traj_len, start_idx + chunk_length * self.downsample_obs)
                                self._index.append((hdf5_path, demo_key, start_idx, end_idx))
            
        # if the repeat trajectory is  > 1, then we need to repeat the index
        if self.dataset_config.num_repeat_traj > 1:
            self._index = self._index * self.dataset_config.num_repeat_traj
        # shuffle the index
        rng.shuffle(self._index)
        # truncate the index to the max dataset size
        if self.max_dataset_size != -1:
            self._index = self._index[:self.max_dataset_size]
        print(colored(f"Total samples: {len(self._index)}", "green"))

    def get_dataset_config(self):
        return {
            'image_keys': self.img_keys,
            'proprio_keys': self.proprio_keys,
            'action_keys': self.action_keys,
            'bbox_keys': self.bbox_keys,
            'action_index_mapping': self.action_index_mapping,
            'dataset_config': self.dataset_config,
            'shared_config': self.shared_config,
        }

    def __repr__(self) -> str:
        return f"GPTDataset(image_keys={self.img_keys}, proprio_keys={self.proprio_keys}, action_keys={self.action_keys}, bbox_keys={self.bbox_keys}, action_index_mapping={self.action_index_mapping}, dataset_config={self.dataset_config}, shared_config={self.shared_config})"
    def __len__(self):
        return len(self._index)

    def __getitem__(self, index):
        hdf5_path, demo_key, start_idx, end_idx = self._index[index]
        summary_list = self.summary_data[hdf5_path][demo_key]
        summary_list = data_utils.get_relevant_summary_data(start_idx, end_idx, summary_list, self.downsample_obs)
        # Load data from HDF5
        data = {}
        with h5py.File(hdf5_path, "r") as f:
            demo_group = f["data"][demo_key]
            
            for key in self.output_keys:
                if key in demo_group:
                    dataset = demo_group[key]
                    if isinstance(dataset, h5py.Dataset):
                        # Load the chunk
                        data[key] = np.array(dataset[start_idx:end_idx:self.downsample_obs])
                    elif isinstance(dataset, h5py.Group):
                        # Handle nested groups (e.g., observations/images)
                        data[key] = {}
                        for subkey in dataset.keys():
                            subdataset = dataset[subkey]
                            if isinstance(subdataset, h5py.Dataset):
                                data[key][subkey] = np.array(subdataset[start_idx:end_idx:self.downsample_obs])
                else:
                    # Key not found, create zeros or skip
                    print(colored(f"Warning: Key '{key}' not found in {hdf5_path}/{demo_key}", "yellow"))
                    continue
        task_langauges = misc_utils.get_task_language_from_hdf5(hdf5_path, load_generated_instructions=True)
        if not demo_key in task_langauges:
            demo_number = '_'.join(demo_key.split('_')[-2:])
            if demo_number in task_langauges:
                task_language = np.random.choice(task_langauges[demo_number])
            else:
                raise ValueError(f"Demo key {demo_number} or {demo_key} not found in task_langauges: {task_langauges.keys()}")
        else:
            task_language = np.random.choice(task_langauges[demo_key])
        
        # Create timestep indices
        timestep_indices = np.arange(start_idx, end_idx, self.downsample_obs, dtype=np.int64)
        timestep_times = timestep_indices.astype(np.float32) / self.control_freq
        relative_timestep_indices = (timestep_indices - start_idx) // self.downsample_obs
        relative_timestep_times = relative_timestep_indices.astype(np.float32) / self.control_freq

        assert len(relative_timestep_indices) == len(data[self.bbox_keys[0] + '_mask']), f"{len(relative_timestep_indices)=} != {len(data[self.bbox_keys[0] + '_mask'])=}"
        
        # Convert to tensors
        output = {}
        for key, value in data.items():
            if isinstance(value, dict):
                # Nested dictionary (e.g., observations/images)
                output[key] = {
                    k: torch.from_numpy(v).float() if v.dtype != np.uint8 else torch.from_numpy(v)
                    for k, v in value.items()
                }
            else:
                if hasattr(value, "dtype") and (value.dtype == np.object_ or getattr(value.dtype, "kind", "") == "O") or value.dtype.kind == "S":
                    output[key] = value
                else:
                    output[key] = torch.from_numpy(value).float() if value.dtype != np.uint8 else torch.from_numpy(value)
        
        # save: 1 for gripper close action & 0 for gripper open action
        gripper_action = output["actions"][:, self.action_index_mapping["gripper"]["index"]]
        gripper_action = (gripper_action == self.action_index_mapping["gripper"]["close"]).to(torch.int32)
        output["gripper_action"] = gripper_action 
        output["timestep_index"] = torch.from_numpy(timestep_indices)
        output["timestep_time"] = torch.from_numpy(timestep_times)
        output["relative_timestep_index"] = torch.from_numpy(relative_timestep_indices)
        output["relative_timestep_time"] = torch.from_numpy(relative_timestep_times)
        output["task_language"] = task_language
        # store relative path to hdf5 file
        output["hdf5_path"] = os.path.relpath(hdf5_path, self.dataset_config.base_dir)
        output["demo_key"] = demo_key
        output["start_idx"] = torch.tensor(start_idx)
        output["end_idx"] = torch.tensor(end_idx)
        output["summary_list"] = summary_list
        return output
