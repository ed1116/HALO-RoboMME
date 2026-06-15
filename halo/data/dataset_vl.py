import os
import copy
import h5py
import pickle
import torch
import numpy as np
import torchvision.transforms as transforms
from typing import Optional, Union, List, Any
from dataclasses import dataclass
from termcolor import colored
import torch.distributed as dist
import psutil
import gc
import re
from collections import OrderedDict
from tqdm import tqdm

import halo.util.tensor_utils as TU
import halo.data.utils as data_utils
from halo.util.args import DatasetConfig, SharedConfig
import halo.util.misc as misc_utils
import halo.util.casa_utils as casa_utils
import halo.models.backbones.token_sequence_gen as TokenSequenceGen

SHARED_TENSORS = {}
SHARED_HANDLES = {}
SHM_META = {}  # key -> (name, shape, dtype)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class TaskGroupDataset(torch.utils.data.Dataset):
    """
    Dataset class for HDF5 datasets grouped by task.
    Each HDF5 file represents a group of similar task demonstrations.
    """

    def __init__(
        self,
        dataset_config: DatasetConfig,
        shared_config: SharedConfig,
        vision_transform: transforms.Compose,
        no_aug_vision_transform: Optional[transforms.Compose] = None,
        split: str = "train",
        split_ratio: float = -1.0,
        lazy_image_convert: bool = True,
    ):
        # 1. Basic configuration setup
        self.dataset_config = dataset_config
        self.shared_config = shared_config
        self.split = split

        # Vision transforms
        self.vision_transform = vision_transform
        self.no_aug_vision_transform = (
            no_aug_vision_transform
            if no_aug_vision_transform is not None
            else vision_transform
        )
        # When True, defer float()/255 + permute + transform + stack until after
        # downsampling so we only process T/downsample_obs frames instead of T.
        self.lazy_image_convert = lazy_image_convert
        self.image_mean = IMAGENET_MEAN
        self.image_std = IMAGENET_STD

        # Sequence length configuration
        self.seq_length = shared_config.seq_length
        self.num_pred_steps = shared_config.num_pred_steps
        self.total_seq_length = self.seq_length + self.num_pred_steps
        self.downsample_obs = shared_config.downsample_obs
        self.downsample_act = self.downsample_obs
        assert self.seq_length >= self.downsample_obs, f"seq_length must be greater than downsample_obs: {self.seq_length} is not greater than {self.downsample_obs}"
        self.single_trajectory = False
        self.pad_to_max_length = False
        self.add_language_embedding = False
        self.add_language_tokens = True
        self.sample_each_state = dataset_config.sample_each_state
        # used for debugging or reducing the number of training samples for a given dataset
        self.n_examples_only = dataset_config.n_examples_only
        self.train_on_exploration = dataset_config.train_on_exploration
        self.add_state_supervision = shared_config.add_state_supervision
        self.state_supervision_mode = shared_config.state_supervision_mode
        self.max_state_supervision_len = shared_config.max_state_supervision_len # if str, this is the max length of the state supervision string

        # 2. Load dataset metadata from the specified JSON file
        json_path = dataset_config.dataset_json if split == "train" else dataset_config.dataset_val_json
        if isinstance(json_path, list):
            assert (
                len(json_path) == 1
            ), "TaskGroupDataset only supports one dataset JSON configuration file."
            json_path = json_path[0]
        dataset_metadata = data_utils.load_json(json_path)

        # 3. Store data keys and direct HDF5 paths from metadata
        self.dataset_paths = dataset_metadata["dataset_path"]
        self.image_keys = dataset_metadata.get("image_keys", [])
        self.low_dim_keys = dataset_metadata.get("low_dim_keys", [])
        self.extra_keys = ['policy_mode'] if not dataset_config.train_on_exploration else []
        # we are assuming that the gripper is a separate key
        self.proprio_keys = dataset_metadata.get("proprio_keys", [])
        assert len(self.proprio_keys) >= 2, "TaskGroupDataset supports proprio keys with separate gripper key."
        assert any("gripper" in k for k in self.proprio_keys), "Atleast one proprio key must contain 'gripper' in the key name."
        self.action_keys = dataset_metadata.get("action_keys", [])
        assert len(self.action_keys) == 1, "TaskGroupDataset only supports one action key."

        # 4. Scan HDF5 files and prepare trajectory mappings
        self._traj_lengths = {}
        self._n_samples = 0
        print(f"Processing {len(self.dataset_paths)} HDF5 files for split '{split}'...")
        for hdf5_path in tqdm(self.dataset_paths, desc="Scanning HDF5 files"):
            try:
                data_dir = misc_utils.get_data_base_dir(hdf5_path, shared_config=self.shared_config)
                print(f"data_dir: {data_dir}")
                assert os.path.exists(data_dir), colored(f"CASAPLAY_DATAROOT does not exist: {data_dir}", "red")
                hdf5_path = os.path.join(data_dir, hdf5_path)
                assert os.path.exists(hdf5_path), colored(f"HDF5 file does not exist: {hdf5_path}", "red")
                self._traj_lengths[hdf5_path] = []  # each hdf5 path is a group
                with h5py.File(hdf5_path, "r") as f:
                    data_group = f["data"]
                    demo_keys = list(data_group.keys())
                    # sort by splitting key into numbers by '_'[-1]
                    demo_keys = sorted(demo_keys, key=lambda x: int(x.split('_')[-1]))
                    if self.n_examples_only > -1:
                        self._n_samples += min(len(demo_keys), self.n_examples_only)
                    else:
                        self._n_samples += len(demo_keys)

                    # if split_ratio > 0; use it to split the spit_ratio if train and 1-split_ratio from behind if val
                    if split_ratio > 0:
                        if split == "train":
                            demo_keys = demo_keys[:int(len(demo_keys) * split_ratio)]
                        else:
                            demo_keys = demo_keys[int(len(demo_keys) * split_ratio):]

                    for demo_key in demo_keys:
                        traj_len = data_group[demo_key][self.action_keys[0]].shape[0]
                        self._traj_lengths[hdf5_path].append((demo_key, traj_len))
                        if self.n_examples_only > -1 and len(self._traj_lengths[hdf5_path]) >= self.n_examples_only:
                            break
                # find the maximum trajectory length
                self._max_traj_len = max(traj_len for traj_list in self._traj_lengths.values() for _, traj_len in traj_list)
                print(colored(f"Maximum trajectory length: {self._max_traj_len}", "green"))
            except Exception as e:
                print(f"\nWarning: Could not process file {hdf5_path}. Error: {e}")
                raise e
        # print a summary: for each hdf5_path, print the number of demo_keys and the maximum trajectory length
        print(colored("Summary of the dataset:", "green"))
        for hdf5_path, traj_list in self._traj_lengths.items():
            print(colored(f"HDF5 path: {hdf5_path}", "green"))
            print(colored(f"    Number of demo_keys: {len(traj_list)}", "yellow"))
            print(colored(f"    Maximum trajectory length: {max(traj_len for _, traj_len in traj_list)}", "yellow"))
        self._generate_sample_mappings()

        # 5. Setup extra configurations
        self.proprio_noise = dataset_config.proprio_noise
        self.action_noise = dataset_config.action_noise
        self.num_repeat_traj = dataset_config.num_repeat_traj
        if isinstance(self.num_repeat_traj, list) and len(self.num_repeat_traj) == 1:
            self.num_repeat_traj = self.num_repeat_traj[0]

        # 6. Shared memory setup
        self.load_in_mem = dataset_config.load_in_mem
        self.share_ram = getattr(dataset_config, "share_ram", False)
        if self.load_in_mem:
            print("Loading the entire dataset in memory using shared memory")
            self._load_whole_dataset_shared()

        self.random_patch_masking = dataset_config.random_patch_masking
        self.only_first_obs = shared_config.only_first_obs
        self.k_ptp = shared_config.k_ptp

    def _load_whole_dataset_shared(self):
        """
        Rank‑0 loads HDF5 → builds ONE tensor per key →
        puts each tensor in /dev/shm → broadcasts metadata.
        Other ranks just attach by name.
        """
        world_rank = dist.get_rank() if dist.is_initialized() else 0

        if dist.is_initialized():
            dist.barrier()

        # ------------------------------------------------------------
        # 1. Rank‑0: read once, build per‑key arrays
        # ------------------------------------------------------------
        meta = {}          # {key: (name, shape, dtype)}
        if world_rank == 0:
            before = psutil.virtual_memory().available
            print("Loading entire dataset into RAM on rank‑0 …")

            # Load all data from HDF5 files organized by [hdf5_path][demo_key]
            all_keys = self.image_keys + self.proprio_keys + self.low_dim_keys + self.action_keys + self.extra_keys

            # Create a nested structure to store data by hdf5_path and demo_key
            dataset_data = {}

            for hdf5_path, demos in self._traj_lengths.items():
                dataset_data[hdf5_path] = {}
                with h5py.File(hdf5_path, 'r') as f:
                    data_group = f['data']
                    for demo_key, traj_len in demos:
                        dataset_data[hdf5_path][demo_key] = {}
                        for key in all_keys:
                            if key not in data_group[demo_key]:
                                # extra keys may not be present in the dataset
                                assert key in self.extra_keys, f"Key {key} not found in {hdf5_path} and {demo_key}"
                                if key == 'policy_mode':
                                    # set all of them to 0, i.e., teleoperation
                                    trajectory_data = np.zeros((traj_len,), dtype=np.int32)
                                else:
                                    raise NotImplementedError(f"Key {key} not found in {hdf5_path} and {demo_key}")
                            else:
                                # Load entire trajectory for this key
                                trajectory_data = np.array(data_group[demo_key][key])
                            dataset_data[hdf5_path][demo_key][key] = trajectory_data

            # Flatten the nested structure for shared memory storage
            # Create one shared memory block per (hdf5_path, key) combination
            flattened_data = {}
            self._demo_key_to_index = {}  # Mapping from (hdf5_path, demo_key) to demo index
            # Track (hdf5_path, key) -> intended shm key (before possible conflict rename)
            _hdf5_key_to_intended_shm = {}

            for hdf5_path, demo_dict in dataset_data.items():

                for key in all_keys:
                    # Collect all trajectories for this (hdf5_path, key) combination
                    all_trajectories = []
                    demo_keys_list = []

                    for demo_key, key_dict in demo_dict.items():
                        trajectory = key_dict[key]
                        all_trajectories.append(trajectory)
                        demo_keys_list.append(demo_key)

                        # Store the mapping from (hdf5_path, demo_key) to the demo_key position in the list
                        self._demo_key_to_index[(hdf5_path, demo_key)] = len(demo_keys_list) - 1

                    # Create a safe key name
                    safe_key = key.replace('/', '_')
                    shm_key = f"libero_{hdf5_path.replace('/', '_').replace('.', '_')}_{safe_key}"

                    # Concatenate all trajectories into a single array
                    # Each trajectory has shape (traj_len, ...), so we concatenate along the first axis
                    concatenated_trajectories = np.concatenate(all_trajectories, axis=0)

                    # Store the concatenated data and metadata
                    flattened_data[shm_key] = concatenated_trajectories
                    _hdf5_key_to_intended_shm[(hdf5_path, key)] = shm_key

                    # Store trajectory boundaries for later indexing
                    trajectory_boundaries = []
                    start_idx = 0
                    for traj in all_trajectories:
                        end_idx = start_idx + traj.shape[0]
                        trajectory_boundaries.append(np.array([start_idx, end_idx]))
                        start_idx = end_idx

                    # Store boundaries in a separate key
                    boundaries_key = f"{shm_key}_boundaries"
                    flattened_data[boundaries_key] = np.stack(trajectory_boundaries)
            # Dump the _demo_key_to_index data to a file
            with open("/tmp/demo_key_to_index.pkl", "wb") as f:
                pickle.dump(self._demo_key_to_index, f)

            del dataset_data
            gc.collect()

            print("Moving arrays to shared memory …")
            _intended_to_actual = {}  # intended_shm_key -> actual_shm_key (may differ on conflict)
            for intended_k, arr in flattened_data.items():
                actual_k, tensor, shm = data_utils.create_shared(intended_k, arr)
                print(f"Created shared memory block: {actual_k}")
                print(actual_k, arr.shape, str(arr.dtype))
                meta[actual_k] = (actual_k, arr.shape, str(arr.dtype))
                _intended_to_actual[intended_k] = actual_k

            used = (psutil.virtual_memory().available - before) / 1e9
            print(f"rank‑0: dataset ready, ΔRAM≈{-used:.2f} GB")
            global SHM_META
            SHM_META = meta

            # Populate cross-dataset registry so QA datasets can reuse these blocks
            if self.share_ram:
                _registry_hdf5_paths = set(k[0] for k in _hdf5_key_to_intended_shm)
                for hdf5_path in _registry_hdf5_paths:
                    data_meta = {}
                    bounds_meta = {}
                    for key in all_keys:
                        intended_sk = _hdf5_key_to_intended_shm.get((hdf5_path, key))
                        intended_bk = f"{intended_sk}_boundaries" if intended_sk else None
                        if intended_sk and intended_sk in _intended_to_actual:
                            actual_sk = _intended_to_actual[intended_sk]
                            data_meta[key] = meta[actual_sk]
                        if intended_bk and intended_bk in _intended_to_actual:
                            actual_bk = _intended_to_actual[intended_bk]
                            bounds_meta[key] = meta[actual_bk]
                    path_demo_idx = {k: v for k, v in self._demo_key_to_index.items() if k[0] == hdf5_path}
                    data_utils.register_shared_hdf5(hdf5_path, path_demo_idx, data_meta, bounds_meta)
                    print(f"[share_ram] Registered VL shared memory for {hdf5_path}")

        # ------------------------------------------------------------
        # 2. Broadcast metadata so every rank can attach
        # ------------------------------------------------------------
        obj = [meta]
        if dist.is_initialized():
            dist.barrier()
            dist.broadcast_object_list(obj, src=0)
            # Load the _demo_key_to_index data from the file
            with open("/tmp/demo_key_to_index.pkl", "rb") as f:
                self._demo_key_to_index = pickle.load(f)
        meta = obj[0] # same on all ranks now

        if world_rank != 0:
            SHM_META = meta

    def _lazy_attach(self):
        """Lazy attachment to shared memory blocks"""
        global SHARED_TENSORS
        global SHARED_HANDLES
        if not getattr(self, "_loaded", False):
            world_rank = dist.get_rank() if dist.is_initialized() else 0
            print(f"[Rank {world_rank}] Attaching shared memory blocks: {list(SHM_META.keys())}")
            for k in SHM_META.keys():
                if k not in SHARED_TENSORS:
                    name, shape, dtype = SHM_META[k]
                    tensor, shm = data_utils.attach_shared(name, shape, dtype)
                    SHARED_TENSORS[k] = tensor  # only tensor, not shm
                    SHARED_HANDLES[k] = shm
            self._loaded = True

    def _generate_sample_mappings(self):
        '''
        This function creates the following mappings from the self._traj_lengths:
        WARNING: Must be called after reshuffling the dataset.
        - _index2path: maps the index to the (hdf5_path, demo_key, traj_length)
        - _index2group: maps the index to the group of the sample
        - _group2indices: maps the group to the indices of the samples in that group
        '''
        self._index2path = {}
        self._index2group = {}
        self._group2indices = {}
        sample_idx, group_idx = 0, 0
        for hdf5_path, demos in self._traj_lengths.items():
            self._group2indices[group_idx] = []
            for demo_key, traj_length in demos:
                if (self.sample_each_state): # useful for cross attention; where each sample only contains the sample
                    for traj_ind in range(traj_length):
                        self._index2path[sample_idx] = (hdf5_path, demo_key, traj_ind)
                        self._index2group[sample_idx] = group_idx
                        self._group2indices[group_idx].append(sample_idx)
                        sample_idx += 1
                else:
                    self._index2path[sample_idx] = (hdf5_path, demo_key, traj_length)
                    self._index2group[sample_idx] = group_idx
                    self._group2indices[group_idx].append(sample_idx)
                    sample_idx += 1
            group_idx += 1
        # assert sample_idx == self._n_samples, "Sample mapping generation failed."
        self._n_samples = sample_idx # overrite this number
        return

    def _create_state_supervision(self, hdf5_path, demo_key, data_group, timesteps):
        raise NotImplementedError("State supervision creation is not implemented for this dataset. Should be implemented in the subclass.")

    def _get_data_from_h5(self, hdf5_path, demo_key, timesteps):
        """
        Get data from HDF5 file or shared memory for specific timesteps.

        Args:
            hdf5_path: Path to the HDF5 file
            demo_key: Key of the demonstration
            timesteps: List of timestep indices to access
        """
        all_keys = self.image_keys + self.proprio_keys + self.low_dim_keys + self.action_keys + self.extra_keys
        if self.load_in_mem:
            # Use shared memory tensors
            self._lazy_attach()

            data = {}
            for key in all_keys:
                # Use the same safe key naming scheme as in _load_whole_dataset_shared
                safe_key = key.replace('/', '_')
                shm_key = f"libero_{hdf5_path.replace('/', '_').replace('.', '_')}_{safe_key}"
                boundaries_key = f"{shm_key}_boundaries"
                # get the corresponding key from the SHM_META with an ending integer: shm_key_0, shm_key_1, etc.
                if shm_key not in SHM_META.keys():
                    shm_key = [k for k in SHM_META.keys() if re.match(f"{shm_key}_\d+", k)][0]

                # Get the concatenated trajectory data
                concatenated_data = SHARED_TENSORS[shm_key]

                if boundaries_key not in SHM_META.keys():
                    boundaries_key = [k for k in SHM_META.keys() if re.match(f"{boundaries_key}_\d+", k)][0]
                trajectory_boundaries = SHARED_TENSORS[boundaries_key]

                # Get the demo index
                demo_idx = self._demo_key_to_index[(hdf5_path, demo_key)]

                # Get the start and end indices for this demo
                start_idx, end_idx = trajectory_boundaries[demo_idx].tolist()

                # Extract the trajectory for this demo
                trajectory = concatenated_data[start_idx:end_idx]

                # Extract the specific timesteps
                data[key] = trajectory[timesteps]
        else:
            # Load from HDF5 file directly
            with h5py.File(hdf5_path, 'r') as f:
                data_group = f['data'][demo_key]
                data_local = {}
                for key in all_keys:
                    # Convert HDF5 dataset to numpy array safely
                    if key in data_group:
                        dataset = data_group[key]
                        data_local[key] = np.array(dataset[timesteps])
                    else:
                        # extra keys may not be present in the dataset
                        assert key in self.extra_keys, f"Key {key} not found in {hdf5_path} and {demo_key}"
                        if key == 'policy_mode':
                            # set all of them to 0, i.e., teleoperation
                            data_local[key] = np.zeros((len(timesteps),), dtype=np.int32)
                        else:
                            raise NotImplementedError(f"Key {key} not found in {hdf5_path} and {demo_key}")
                if self.add_state_supervision:
                    extra_state_info = {}
                    if self.shared_config.ss_create_mode == "inst_generic":
                        task_name = misc_utils.get_task_name_from_hdf5_path(hdf5_path, actual_task_name=True)
                        state_supervision_key = casa_utils.TASK_NAME_TO_IMP_OBJ_MAP[task_name]
                        state_supervision_key = state_supervision_key.encode('utf-8') # convert it to b'string'
                        value = np.zeros((len(timesteps), len(self.image_keys)*4))
                        # data_group['obs'][img_key][ts] == b(state_supervision_key))
                        for img_ind, img_key in enumerate(self.image_keys):
                            bbox_val = np.zeros((4,), dtype=np.float32)
                            bbox_name_key = img_key.replace('image', 'bbox_names')
                            bbox_key = img_key.replace('image', 'bbox')
                            ts_indices, pos_indices = np.where(data_group[bbox_name_key][timesteps] == state_supervision_key)
                            prev_ts_ind = 0
                            for p_ind, ts_ind in zip(pos_indices, ts_indices):
                                # all values should be less than 128.0
                                assert np.all(data_group[bbox_key][timesteps][ts_ind, p_ind] < 128.0), f"Bbox value is {data_group[bbox_key][timesteps][ts_ind, p_ind]} which is greater than 128.0 for {hdf5_path} {demo_key} {img_key} {ts_ind} {p_ind}"
                                assert np.all(data_group[bbox_key][timesteps][ts_ind, p_ind] >= 0.0), f"Bbox value is {data_group[bbox_key][timesteps][ts_ind, p_ind]} which is less than 0.0 for {hdf5_path} {demo_key} {img_key} {ts_ind} {p_ind}"
                                bbox_val = data_group[bbox_key][timesteps][ts_ind, p_ind] / 128.0
                                # fill the value for the timesteps between the previous and current timestep
                                if ts_ind > prev_ts_ind:
                                    value[prev_ts_ind:ts_ind, img_ind*4:(img_ind+1)*4] = bbox_val.reshape(1, 4)
                                elif ts_ind == prev_ts_ind:
                                    value[prev_ts_ind, img_ind*4:(img_ind+1)*4] = bbox_val
                                else:
                                    raise ValueError(f"ts_ind > prev_ts_ind: {ts_ind} > {prev_ts_ind}")
                                prev_ts_ind = ts_ind
                    elif self.shared_config.ss_create_mode == "time":
                        # the output is the last timestep of the instance when the gripper was closed.
                        value = np.zeros((len(timesteps), 1))
                        action_data = data_group['actions'][timesteps]
                        assert action_data.shape[-1] == 12, f"Action data must have 12 dimensions for {hdf5_path} {demo_key}: {action_data.shape}"
                        gripper_one_position = np.where(action_data[:, 6] == 1)[0]
                        # we want to predict the position of the gripper where the last timestep where the gripper was closed and divide by the seq_length
                        for grip_pos in gripper_one_position:
                            value[grip_pos + 1:] = grip_pos
                        value = value / len(timesteps)
                    else:
                        value, extra_state_info = self._create_state_supervision(hdf5_path, demo_key, data_group, timesteps)
                        data_local['extra_state_info'] = [extra_state_info]*len(timesteps)
                    data_local['state_supervision'] = value
                # assert np.allclose(data[key], data_local[key]), f"Data mismatch for key: {key} in {hdf5_path} and {demo_key}"
                # print(colored(f"Data verified for key: {key} in {hdf5_path} and {demo_key}", "green"))
            data = data_local
        return data

    def _encode_text(self, text: Union[str, List[str]]) -> List[int]:
        """Encode text using tokenizer."""
        if hasattr(self.tokenizer, 'encode'):
            return self.tokenizer.encode(text, add_special_tokens=False)
        result = self.tokenizer(text, add_special_tokens=False)
        return result['input_ids'] if isinstance(result, dict) else result

    def _get_subsequence(self, index):
        '''
        This function returns a subsequence of length self.total_seq_length.
        '''
        main_hdf5_path, main_demo_key = "", ""
        if len(self._index2path[index]) == 3:
            main_hdf5_path, main_demo_key, main_traj_len = self._index2path[index]
        elif len(self._index2path[index]) == 4:
            main_hdf5_path, main_demo_key, main_traj_len, _ = self._index2path[index]
        else:
            raise ValueError(f"Unknown number of dimensions in _index2path: {len(self._index2path[index])}")

        task_language = ""
        if self.add_language_tokens:
            assert self.single_trajectory, "Language tokens are only supported for single trajectory."
            task_langauges = misc_utils.get_task_language_from_hdf5(main_hdf5_path, load_generated_instructions=not self.dataset_config.use_og_inst)
            if main_demo_key not in task_langauges:
                # the main_demo_key is of the form: TASKNME_DEMO_NUMBER
                # check if DEMO_NUMBER is in the task_langauges
                demo_number = '_'.join(main_demo_key.split('_')[-2:])
                if demo_number in task_langauges:
                    task_language = np.random.choice(task_langauges[demo_number])
                else:
                    raise ValueError(f"Demo key {demo_number} or {main_demo_key} not found in task_langauges: {task_langauges.keys()}")
            else:
                task_language = np.random.choice(task_langauges[main_demo_key])

        group_id = self._index2group[index]
        similar_indices = self._group2indices[group_id]

        # Build a sequence of (path, demo_key, timestep)
        trajectory_step_info = []
        subsequence_info = []
        eos = []

        if not self.single_trajectory:
            # Add other trajectories from the group
            current_len = 0
            while current_len < self.total_seq_length - main_traj_len:
                rand_idx = np.random.choice(similar_indices)
                hdf5_path, demo_key, traj_len = self._index2path[rand_idx]
                subsequence_info.extend([(hdf5_path, demo_key, ts) for ts in range(traj_len)])
                trajectory_step_info.extend([ts for ts in range(traj_len)])
                eos.extend([0.0] * (traj_len - 1) + [1.0])
                current_len += traj_len

        # Insert the main trajectory
        main_traj_info = [(main_hdf5_path, main_demo_key, ts) for ts in range(main_traj_len)]
        trajectory_step_info = [ts for ts in range(main_traj_len)] + trajectory_step_info
        # put the main trajectory at the start of the sequence acting as the prompt
        subsequence_info = main_traj_info + subsequence_info
        # Correctly mark the end of the prompt trajectory
        eos = ([0.0] * (main_traj_len - 1) + [1.0]) + eos

        if not self.single_trajectory:
            # Truncate to the final desired length, ensuring some of the prompt is visible
            if len(subsequence_info) > self.total_seq_length:
                # Randomly start from a point in the first half of the prompt
                max_start_offset = min(main_traj_len // 2, len(subsequence_info) - self.total_seq_length)
                start_idx = np.random.randint(0, max_start_offset + 1)
                subsequence_info = subsequence_info[start_idx : start_idx + self.total_seq_length]
                eos = eos[start_idx : start_idx + self.total_seq_length]
                trajectory_step_info = trajectory_step_info[start_idx : start_idx + self.total_seq_length]
        else:
            if len(subsequence_info) > self.total_seq_length:
                # randomly sample a subsequence of length self.total_seq_length
                start_idx = np.random.randint(0, len(subsequence_info) - self.total_seq_length)
                subsequence_info = subsequence_info[start_idx : start_idx + self.total_seq_length]
                eos = eos[start_idx : start_idx + self.total_seq_length]
                trajectory_step_info = trajectory_step_info[start_idx : start_idx + self.total_seq_length]

        # Group subsequence_info by trajectory to maintain trajectory structure
        # Use a list-based approach to handle multiple trajectories from the same demo
        grouped_data = []
        current_trajectory = []
        current_trajectory_info = None

        for i, (hdf5_path, demo_key, timestep) in enumerate(subsequence_info):
            # Check if this is the start of a new trajectory (timestep == 0 or previous was end of trajectory)
            if i == 0 or timestep == 0 or (eos[i-1] > 0.5):
                # Save the previous trajectory if it exists
                if current_trajectory:
                    grouped_data.append((current_trajectory_info, current_trajectory))

                # Start a new trajectory
                current_trajectory_info = (hdf5_path, demo_key)
                current_trajectory = [timestep]
            else:
                # Continue the current trajectory
                current_trajectory.append(timestep)
        # Don't forget to add the last trajectory
        if current_trajectory:
            grouped_data.append((current_trajectory_info, current_trajectory))

        # Load data in batches, maintaining trajectory structure
        subseq = []
        hdf5_paths = [hdf5_path for hdf5_path, _ in grouped_data]
        # # find unique hdf5 paths
        # unique_hdf5_paths = list(set(hdf5_paths))
        for (hdf5_path, demo_key), timesteps in grouped_data:
            batch_data = self._get_data_from_h5(hdf5_path, demo_key, timesteps)
            # Convert batch data to individual timesteps
            for i, timestep in enumerate(timesteps):
                timestep_data = {key: batch_data[key][i] for key in batch_data.keys()}
                subseq.append(timestep_data)
        obs_subseq = subseq
        language_embedding = np.zeros(13, dtype=np.float32) # random oddly specific number to avoid broadcasting issues
        if self.add_language_embedding:
            hdf5_path_example = grouped_data[0][0][0]
            demo_keys = [demo_key for (_, demo_key), _ in grouped_data]
            unique_demo_keys = list(set(demo_keys))
            assert len(unique_demo_keys) == 1, "Language embedding is only supported for single demo. this should be enforced by self.single_trajectory"
            language_embedding = self.helper_load_language_embedding(hdf5_path_example, unique_demo_keys[0])
        metadata = {
            'task_language': task_language,
            'language_embedding': language_embedding,
            'trajectory_step_info': trajectory_step_info,
            'main_hdf5_path': main_hdf5_path,
            'main_demo_key': main_demo_key,
        }
        if 'extra_state_info' in subseq[0]:
            metadata['extra_state_info'] = [s['extra_state_info'] for s in subseq]
        return subseq, obs_subseq, torch.tensor(eos)[:, None], metadata

    def convert_float_to_string(self, val: np.ndarray) -> str:
        token_str = ', '.join([f"{x:.2f}" for x in val])
        return token_str

    def convert_strs_to_token_ids(self, token_strs: List[str]) -> np.ndarray:
        '''
        Convert a list of strings to token ids.
        '''
        token_ids = self.tokenizer(
            token_strs,
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            max_length=self.max_state_supervision_len,
            padding_side="right",
            return_tensors="pt",
        )['input_ids']
        return token_ids

    def helper_load_state_supervision(self, subseq):
        # add a noise of 0.01 gaussian noise to the state supervision
        state_supervision = np.zeros((len(subseq), *subseq[0]['state_supervision'].shape))
        noise_val = 0.01
        # if the type is time, then the noise should be around 1/seq_length
        if self.shared_config.ss_create_mode == "time":
            noise_val = 5.0/len(subseq)
        for i, s in enumerate(subseq):
            state_supervision[i] = s['state_supervision'] + np.random.normal(0, noise_val, s['state_supervision'].shape)
        return torch.from_numpy(state_supervision).float()

    def __len__(self):
        return self._n_samples * self.num_repeat_traj

    def __getitem__(self, index):
        index = index % self._n_samples
        subseq, obs_subseq, eos, metadata = self._get_subsequence(index)
        trajectory_step_info = metadata['trajectory_step_info']
        metadata.pop('trajectory_step_info')

        proprio = self.helper_load_proprio(subseq)
        action = self.helper_load_action(subseq)
        state_supervision = None
        if self.add_state_supervision:
            state_supervision = self.helper_load_state_supervision(subseq)

        action = torch.cat([action, eos], dim=-1)
        # Multi-step prediction formatting
        if self.k_ptp > 0:
            # action: T, num_pred_steps, action_dim
            flipped_actions = torch.flip(action, dims=[0])
            eos_flipped = torch.flip(eos, dims=[0])
            eos_flipped = torch.cat([eos_flipped[1:], eos_flipped[0:1]], dim=0) # shift by one step
            flipped_actions = self.convert_multi_step(flipped_actions, eos_flipped, num_pred_steps=self.k_ptp+1)
            # remove the first step of the flipped actions
            flipped_actions = flipped_actions[:, 1:] # the first step is the same as the current action
            history_action = torch.flip(flipped_actions, dims=[0])

        proprio = self.convert_multi_step(proprio, eos)
        action = self.convert_multi_step(action, eos)
        if self.k_ptp > 0:
            action = torch.cat([history_action, action], dim=1)

        observation = None
        if len(self.image_keys) > 0:
            observation = self.helper_load_image(obs_subseq)
        else: # low-dim only
             observation = self.helper_load_low_dim(obs_subseq)

        # Downsample final tensors
        prompt_mask, weight_mask = self._get_prompt_weight_mask(action)
        if not self.train_on_exploration:
            # get the policy_mode == 0 mask
            policy_mode = np.stack([s['policy_mode'] for s in subseq], axis=0)
            policy_mode = torch.from_numpy(policy_mode)
            assert prompt_mask.shape == policy_mode.shape, f"Prompt mask and policy mode must have the same shape. {prompt_mask.shape} != {policy_mode.shape}"
            prompt_mask = prompt_mask * (policy_mode == 0)

        ### Downsample the data
        # we can robust to different start positions
        start_idx = np.random.randint(0, self.downsample_obs)
        if isinstance(observation, dict):
            for k in observation:
                observation[k] = observation[k][start_idx::self.downsample_obs]
        else:
            observation = observation[start_idx::self.downsample_obs]
        trajectory_step_info = trajectory_step_info[start_idx::self.downsample_obs]
        proprio = proprio[start_idx::self.downsample_act]
        action = action[start_idx::self.downsample_act]
        prompt_mask = prompt_mask[start_idx::self.downsample_act]
        weight_mask = weight_mask[start_idx::self.downsample_act]
        if state_supervision is not None:
            state_supervision = state_supervision[start_idx::self.downsample_act]
            if '_str' in self.state_supervision_mode:
                state_supervision = [self.convert_float_to_string(val) for val in state_supervision]
                state_supervision = self.convert_strs_to_token_ids(state_supervision).to(torch.long)
        # change the frame index number to be wrt to the downsample frames and start index if present in the metadata
        if 'extra_state_info' in metadata:
            # change only for 0; rest all should also change
            metadata['extra_state_info'][0]['ss_frame_index'] = (metadata['extra_state_info'][0]['ss_frame_index'] - start_idx)//self.downsample_act
        if self.only_first_obs:
            assert self.single_trajectory, \
                f"only_first_obs is only supported with language conditioning; since otherwise the first observation is the prompt"
            for ind, ts in enumerate(trajectory_step_info):
                # only keep the exploration part of the trajectory or the first observation
                if ts > 0 and prompt_mask[ind] > 0.5:
                    for k in observation.keys():
                        observation[k][ind:] = 0.0
                    break

        ### After downsampling, we will handle the short-term and sensory memory
        clip_length = self.seq_length//self.downsample_act
        empty_len = 7
        hist_action, hist_proprio, hist_mask = torch.empty(empty_len, 1), torch.empty(empty_len, 1), torch.empty(empty_len, dtype=torch.bool)
        if isinstance(observation, dict):
            hist_observation = {k: torch.empty(empty_len, 1) for k in observation.keys()}
        else:
            hist_observation = torch.empty(empty_len, 1)

        ### Clip the data to seq_length//downsample_act
        clip_length = self.seq_length//self.downsample_act
        if isinstance(observation, dict):
            for k in observation:
                observation[k] = observation[k][:clip_length]
        else:
            observation = observation[:clip_length]
        proprio = proprio[:clip_length]
        action = action[:clip_length]
        prompt_mask = prompt_mask[:clip_length]
        weight_mask = weight_mask[:clip_length]
        trajectory_step_info = trajectory_step_info[:clip_length]
        if state_supervision is not None:
            state_supervision = state_supervision[:clip_length]

        # Finalize images deferred by lazy_image_convert: float/255 + permute + stack
        # on T/downsample_obs frames instead of the original T frames.
        if self.lazy_image_convert and isinstance(observation, dict):
            observation = self._finalize_images(observation)

        ### Pad the data to the max length for batching
        if self.pad_to_max_length:
            # we always do right padding.
            assert self.downsample_act == self.downsample_obs, "Downsample act and downsample obs must be the same"
            max_length = self.seq_length//self.downsample_act
            proprio = data_utils.pad_data_to_max_length(proprio, max_length)
            action = data_utils.pad_data_to_max_length(action, max_length)
            prompt_mask = data_utils.pad_data_to_max_length(prompt_mask, max_length)
            weight_mask = data_utils.pad_data_to_max_length(weight_mask, max_length)
            observation = data_utils.pad_data_to_max_length(observation, max_length)
            if state_supervision is not None:
                state_supervision = data_utils.pad_data_to_max_length(state_supervision, max_length)

        return {
            "observation": observation,
            "proprio": proprio,
            "action": action,
            "prompt_mask": prompt_mask,
            "weight_mask": weight_mask,
            "state_supervision": state_supervision if state_supervision is not None else torch.empty(0),
            "hist_observation": hist_observation, # backward compatibility
            "hist_action": hist_action, # backward compatibility
            "hist_proprio": hist_proprio, # backward compatibility
            "hist_mask": hist_mask, # backward compatibility
            "trajectory_step_info": trajectory_step_info,
            **metadata, # task language, language embedding, main hdf5 path, main demo key
        }

    def helper_load_language_embedding(self, hdf5_path, demo_key):
        task_name = misc_utils.get_task_name_from_hdf5_path(hdf5_path)
        valid_language_embeddings = self.language_embeddings[task_name]
        if isinstance(valid_language_embeddings[0], tuple):
            valid_language_embeddings = [t[1] for t in valid_language_embeddings if t[0] == demo_key]
        else:
            assert len(valid_language_embeddings) == 1, "Language embedding is only supported for single demo. this should be enforced by self.single_trajectory"
        # pick a random language embedding from the valid language embeddings
        try:
            lang_idx = np.random.randint(0, len(valid_language_embeddings))
        except Exception as e:
            import ipdb; ipdb.set_trace()
        language_embedding = valid_language_embeddings[lang_idx]
        if self.language_embedding_noise > 0:
            language_embedding += np.random.normal(0, self.language_embedding_noise, language_embedding.shape)
        return torch.from_numpy(language_embedding).float()

    # --- Helper functions copied from GroupSequenceDataset ---
    def helper_load_proprio(self, subseq):
        proprio = {}
        gripper_state = None
        n_grip_vals = 1
        for k in self.proprio_keys:
            if 'gripper' in k:
                gripper_state = [s[k][None, -n_grip_vals:] for s in subseq]
                gripper_state = np.concatenate(gripper_state, axis=0)
            else:
                data = [s[k][None, :] for s in subseq]
                proprio[k] = np.concatenate(data, axis=0)

        if self.proprio_noise > 0:
            for k in proprio.keys():
                proprio[k] += np.random.normal(0, self.proprio_noise, proprio[k].shape)

        proprio_vec = np.concatenate([proprio[k] for k in proprio.keys()], axis=-1)
        proprio_vec = np.concatenate([proprio_vec, gripper_state], axis=-1)
        return torch.from_numpy(proprio_vec).float()

    def helper_load_action(self, subseq):
        n_grip_vals = 1
        # action is a combined key for all the actions. The last n dimensions are the gripper actions.
        k = self.action_keys[0]
        action_shape = subseq[0][k].shape

        no_noise_indices = []
        if action_shape[0] == 7:
            no_noise_indices = [6]
        elif action_shape[0] == 12:
            # 6 eef pose + gripper action + 4 base action + 1 switch mode
            no_noise_indices = [6, 11]
        else:
            raise ValueError(f"Unknown action shape: {action_shape}")

        action_data = [s[k][None, :] for s in subseq]
        action = np.concatenate(action_data, axis=0)

        if self.action_noise > 0:
            # Add noise to all dimensions except the no_noise_indices
            noise = np.random.normal(0, self.action_noise, action.shape)
            noise[..., no_noise_indices] = 0.0
            action += noise

        return torch.from_numpy(action).float()

    def helper_load_image(self, subseq):
        # When lazy_image_convert is enabled, skip float()/255 + permute + transform + stack
        # here and defer to _finalize_images, which runs after downsampling on
        # T/downsample_obs frames instead of T.
        image = {}
        for k in self.image_keys:
            data = [s[k][None] for s in subseq]
            imgs = np.concatenate(data, axis=0)   # (T, H, W, C) uint8

            if (self.split == 'train') and self.random_patch_masking and (torch.rand(1) < 0.8):
                # TODO: never tested this.
                imgs = np.array(data_utils.random_patch_mask(imgs.clone()))

            if self.lazy_image_convert:
                image[k] = imgs   # stay as uint8 numpy (T, H, W, C)
                continue

            imgs_tensor = torch.from_numpy(imgs).float() / 255.0
            imgs_tensor = imgs_tensor.permute(0, 3, 1, 2)
            if ("wrist" in k) or ("hand" in k) or ("eye" in k):
                imgs_tensor = self.no_aug_vision_transform(imgs_tensor) if self.no_aug_vision_transform is not None else imgs_tensor
            else:
                imgs_tensor = self.vision_transform(imgs_tensor) if self.vision_transform is not None else imgs_tensor
            image[k] = imgs_tensor

        if self.lazy_image_convert:
            return image   # dict of uint8 numpy (T, H, W, C) — caller must call _finalize_images

        return torch.stack([image[k] for k in self.image_keys], dim=1).float()

    def _finalize_images(self, image_dict):
        """float()/255 + permute + transform + stack on already-downsampled uint8 numpy arrays."""
        tensors = []
        for k in self.image_keys:
            t = torch.from_numpy(image_dict[k]).float() / 255.0   # (T', H, W, C)
            t = t.permute(0, 3, 1, 2)                             # (T', C, H, W)
            if ("wrist" in k) or ("hand" in k) or ("eye" in k):
                t = self.no_aug_vision_transform(t) if self.no_aug_vision_transform is not None else t
            else:
                t = self.vision_transform(t) if self.vision_transform is not None else t
            tensors.append(t)
        return torch.stack(tensors, dim=1).float()                 # (T', N_CAMS, C, H, W)

    def helper_load_low_dim(self, subseq):
        low_dim_vec = np.concatenate([s[k][None, :] for k in self.low_dim_keys for s in subseq], axis=-1)
        return torch.from_numpy(low_dim_vec).float()

    def convert_multi_step(self, data: torch.Tensor, eos: Union[torch.Tensor, np.ndarray], num_pred_steps: Optional[int] = None) -> torch.Tensor:
        if num_pred_steps is None:
            num_pred_steps = self.num_pred_steps
        if num_pred_steps == 1:
            return data.unsqueeze(1)
        if isinstance(eos, torch.Tensor):
            eos = eos.numpy().flatten()

        # Find indices where trajectories end
        pos = np.concatenate([np.array([0]), np.where(eos > 0.5)[0] + 1, np.array([len(data)])])

        data_chunked = []
        for i in range(len(pos) - 1):
            demo_start, demo_end = pos[i], pos[i+1]
            if demo_start >= demo_end:
                continue
            chunk = data_utils.convert_multi_step(data[demo_start:demo_end], num_pred_steps)
            data_chunked.append(chunk)

        if not data_chunked:
             return torch.empty(0, num_pred_steps, data.shape[-1])

        return torch.cat(data_chunked, dim=0)

    def _get_prompt_weight_mask(self, action):
        # TODO: remove num_steps as it is only used for the weight_mask; and weight_mask is not used at all
        prompt_mask, weight_mask = data_utils.create_prompt_mask(
            action[..., 0, -1], num_steps=1, skip_first=True, deterministic=True
        )
        return torch.from_numpy(prompt_mask).float(), torch.from_numpy(weight_mask).float()

    def shuffle_dataset(self, seed=0):
        rng = np.random.RandomState(seed=seed)
        # shuffle the _traj_lengths dictionary items
        shuffled_items = list(self._traj_lengths.items())
        rng.shuffle(shuffled_items)
        self._traj_lengths = dict(shuffled_items)
        # regenerate mappings based on the new group order
        self._generate_sample_mappings()
        return

    def save_split(self, path : str):
        """
            Not implemented for TaskGroupDataset
        """
        pass

    def _load_language_embeddings(self):
        # we will iterate over all the hdf5 files in the dataset and load the language embeddings
        self.language_embeddings = {}
        pbar = tqdm(self._traj_lengths, desc="Loading language embeddings")
        for hdf5_path, _ in self._traj_lengths.items():
            pbar.update(1)
            task_name = misc_utils.get_task_name_from_hdf5_path(hdf5_path)
            if task_name in self.language_embeddings:
                continue
            pkl_file = misc_utils.get_task_embeddings_path(hdf5_path, model='clip')
            with open(pkl_file, 'rb') as f:
                task_embeddings = pickle.load(f)
            if isinstance(task_embeddings[task_name], dict): # this implies that the task has only one language annotation for all the demos
                task_embed = [task_embeddings[task_name]['embedding']]
                self.language_embeddings[task_name] = task_embed
            elif isinstance(task_embeddings[task_name], list):
                task_embed = [(t['demo_key'], t['embedding'], t['language']) for t in task_embeddings[task_name]]
                self.language_embeddings[task_name] = task_embed
            else:
                raise ValueError(f"Unknown type of task embeddings: {type(task_embeddings[task_name])}")
        print(f"Loaded {len(self.language_embeddings)} language embeddings")
        # print the variance of language embeddings, and its mean of the vairance of the language embeddings
        # language_embeddings_var = np.var(list(self.language_embeddings.values()), axis=0)
        # print(f"Mean of the variance of the language embeddings: {np.mean(language_embeddings_var)}")
        return

class TaskDatasetWithTokenizer(TaskGroupDataset):
    '''
    TaskGroupDataset with a tokenizer
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # it also create the special tokens
        self.tokenizer = data_utils.build_tokenizer(self.shared_config.tokenizer_name, self.image_keys)
        self.tokenizer.backend_tokenizer.model.dropout = self.shared_config.tokenizer_dropout
        self.single_trajectory = True
        self.image_token_ids = self.tokenizer.convert_tokens_to_ids(data_utils.get_img_token_str_list(self.image_keys))
        self.action_token_ids = self.tokenizer.convert_tokens_to_ids(data_utils.get_action_token_str_list())
        self.add_language_embedding = False
        self.add_language_tokens = self.dataset_config.add_language_tokens
        self.language_token_drop_prob = 0.01
        self.sequence_generator = TokenSequenceGen.LangTrajSequence(
            image_keys=self.image_keys,
            image_token_ids=self.image_token_ids,
            tokens_per_frame=self.shared_config.attn_latent_len,
            action_token_ids=self.action_token_ids,
            tokens_per_action=1,
            action_in_inputs=not self.shared_config.remove_action,
            pad_inst_tokens=self.shared_config.pad_inst_tokens,
            max_inst_tokens=self.shared_config.max_inst_tokens,
            inst_token_pad_value=self.tokenizer.pad_token_id,
        )

    def _get_prompt_weight_mask(self, action):
        ## only the task tokens are prompt. rest all are not.
        prompt_mask = np.ones_like(action[..., 0, -1])
        weight_mask = np.zeros_like(action[..., 0, -1])
        return torch.from_numpy(prompt_mask).float(), torch.from_numpy(weight_mask).float()

    def _create_token_sequence(self, inst_ids: torch.Tensor, num_frames: int, prompt_mask: torch.Tensor):
        full_ids, img_token_positions, action_inp_token_pos, action_out_token_pos = self.sequence_generator(inst_ids, num_frames)
        n_prompt_mask = torch.zeros(len(full_ids), dtype=torch.bool)
        n_prompt_mask[action_out_token_pos] = prompt_mask.bool()
        return full_ids, img_token_positions, action_inp_token_pos, action_out_token_pos, n_prompt_mask

    def __getitem__(self, index):
        data = super().__getitem__(index)
        # pop all the history data
        data.pop('hist_observation')
        data.pop('hist_action')
        data.pop('hist_proprio')
        data.pop('hist_mask')
        data.pop('language_embedding')
        data.pop('weight_mask')
        data.pop('trajectory_step_info')
        task_language = data.pop('task_language')
        prompt_mask = data.pop('prompt_mask')
        # convert the observation tensor to a dictoonary of images: T, N, C, H, W -> {key: T, C, H, W}
        num_frames = data['observation'].shape[0]
        data['observation'] = {k: data['observation'][:, cam_ind] for cam_ind, k in enumerate(self.image_keys)}
        # tokenize the instruction and img action tokens
        inst_ids = self._encode_text(task_language) if self.add_language_tokens else []
        full_ids, img_token_positions, action_inp_token_pos, action_out_token_pos, action_prompt_mask = \
                self._create_token_sequence(inst_ids, num_frames, prompt_mask)
        state_supervision = data['state_supervision']
        state_supervision_out_token_pos = action_out_token_pos.clone() if state_supervision.shape[0] > 0 else torch.empty(0, dtype=torch.long)
        data['input_ids'] = full_ids
        data['instruction'] = task_language
        data['inst_ids'] = inst_ids
        data['proprio'] = data['proprio'][:, :1, :] # only the first timestep is used
        data['image_token_positions'] = img_token_positions
        data['text_prompt_mask'] = torch.zeros(len(full_ids), dtype=torch.bool)
        data['action_inp_token_pos'] = action_inp_token_pos
        data['action_out_token_pos'] = action_out_token_pos
        data['action_prompt_mask'] = action_prompt_mask
        # add the state supervision out token positions
        data['state_supervision_out_token_pos'] = state_supervision_out_token_pos
        return data

class DALITaskGroupDataset(TaskGroupDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keys_order = list(self.image_keys) + ['proprio', 'action', 'prompt_mask', 'language_embedding']
    def helper_load_image(self, start_end_epi):
        """
        Load image data from the dataset
        """
        image = {}
        dtype = None
        for k in self.image_keys:
            subsequence = [s[k][None] for s in start_end_epi]
            subsequence = np.concatenate(subsequence, axis=0)
            image[k] = subsequence
        return image
    def __getitem__(self, *args, **kwargs):
        data = super().__getitem__(*args, **kwargs)
        return_data = ()
        for key in self.keys_order:
            if ('observation' in key) or ('rgb' in key) or ('image' in key) or ('hist_observation' in key): # TODO: remove this
                access_key = 'observation' if not key.startswith('hist/') else 'hist_observation'
                key = key.replace('hist/', '')
                if isinstance(data[access_key], dict):
                    obs_data = data[access_key][key]
                else:
                    obs_data = data[access_key]
                if (self.split == 'train') and self.random_patch_masking and (torch.rand(1) < 0.8).item():
                    obs_data = np.array(data_utils.random_patch_mask(torch.from_numpy(obs_data).clone()))
                return_data += (obs_data,)
            else:
                return_data += (data[key],)
        # for ind, key in enumerate(self.keys_order):
        #     print(f"Key: {key}, Shape: {return_data[ind].shape}, Type: {type(return_data[ind])}, Dtype: {return_data[ind].dtype}")
        return return_data

class TaskDatasetLanguage(TaskGroupDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.single_trajectory = True
        self.pad_to_max_length = kwargs['shared_config'].pad_to_max_length # if false, it should pad it during the collate function
        self.language_embedding_noise = 5e-4
        self.add_language_embedding = True
        self._load_language_embeddings()

    def _get_prompt_weight_mask(self, action):
        ## only the language embeddings are prompt. rest all are not.
        prompt_mask = np.ones_like(action[..., 0, -1])
        weight_mask = np.zeros_like(action[..., 0, -1])
        return torch.from_numpy(prompt_mask).float(), torch.from_numpy(weight_mask).float()

class DALITaskDatasetLanguage(TaskDatasetLanguage, DALITaskGroupDataset):
    """
    DALI wrapper for TaskDatasetLanguage that combines the language functionality
    with DALI-accelerated data loading.
    """
    def __init__(self, *args, **kwargs):
        # Initialize TaskDatasetLanguage first (which calls TaskGroupDataset.__init__)
        TaskDatasetLanguage.__init__(self, *args, **kwargs)
        # Then initialize DALI-specific attributes from DALITaskGroupDataset
        self.keys_order = list(self.image_keys) + ['proprio', 'action', 'prompt_mask', 'language_embedding']
    # No need to override helper_load_image or __getitem__ -
    # they will automatically use DALITaskGroupDataset versions due to MRO

