import json
import os
import pickle
import gc
import re
import psutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable
from collections import OrderedDict
import h5py
import numpy as np
import torch
import torch.utils.data as data
import torch.distributed as dist
from halo.util.args import SharedConfig, DatasetConfig
from PIL import Image
from termcolor import colored
import halo.data.utils as data_utils
import halo.models.backbones.token_sequence_gen as TokenSequenceGen

SHARED_TENSORS_QA = {}   # key → numpy array view into shared memory
SHARED_HANDLES_QA = {}   # key → SharedMemory handle
SHM_META_QA = {}         # key → (name, shape, dtype_str)


class QADataset(data.Dataset):
    """
    Dataset for Q&A with structure: instruction, img1, img2, img1, img2, ..., query, answer
    """

    def __init__(
        self,
        dataset_config: DatasetConfig, # only for matching the other dataset configs
        shared_config: SharedConfig, # only for shared config
        vision_transform: Optional[Callable] = None,
        no_aug_vision_transform: Optional[Callable] = None,
        split: str = "train", # unused
        split_ratio: float = -1.0,
    ):
        assert vision_transform is None, "vision_transform is not supported for QADataset"
        assert no_aug_vision_transform is None, "no_aug_vision_transform is not supported for QADataset"
        self.shared_config = shared_config
        self.dataset_config = dataset_config
        self.hdf5_base_dir = Path(os.environ.get("CASAPLAY_DATAROOT"))
        self.tokens_per_image = shared_config.attn_latent_len
        self.add_answer_ids = True

        # 2. Load dataset metadata from the specified JSON file
        json_path = dataset_config.dataset_json
        if isinstance(json_path, list):
            assert (
                len(json_path) == 1
            ), "Dataset only supports one dataset JSON configuration file."
            json_path = json_path[0]
        dataset_metadata = data_utils.load_json(json_path)
        self.image_keys = dataset_metadata.get("image_keys", [])
        self.proprio_keys = dataset_metadata.get("proprio_keys", [])
        self.action_keys = dataset_metadata.get("action_keys", [])
        json_files: Union[str, List[str]] = dataset_metadata.get("json_files", [])
        assert len(json_files) > 0, f"json_files must be provided inside the config: {json_path}"
        # Load JSON entries
        if isinstance(json_files, str):
            json_files = [json_files]
        self.json_files = json_files
        tokenizer_name = shared_config.tokenizer_name
        self.tokenizer = data_utils.build_tokenizer(tokenizer_name, self.image_keys)
        self.tokenizer.backend_tokenizer.model.dropout = shared_config.tokenizer_dropout
        self.proprio_noise = dataset_config.proprio_noise
        self.action_noise = dataset_config.action_noise
        self.remove_proprio = shared_config.remove_proprio
        self.remove_action = shared_config.remove_action
        self.downsample_obs = shared_config.downsample_obs

        # Setup image tokens
        self.image_tokens = data_utils.get_img_token_str_list(self.image_keys)
        self.image_token_ids = [self.tokenizer.convert_tokens_to_ids(tok) for tok in self.image_tokens]
        self.action_token_ids = self.tokenizer.convert_tokens_to_ids(data_utils.get_action_token_str_list())

        self.split_ratio = split_ratio
        self.split = split
        # build the entries
        self.entries = self._build_entries(split_ratio, split)
        self.num_pred_steps = shared_config.num_pred_steps
        self.max_state_supervision_len = shared_config.max_state_supervision_len

        self.load_in_mem = dataset_config.load_in_mem
        self.share_ram = getattr(dataset_config, "share_ram", False)
        if self.load_in_mem:
            print("Loading the entire QA dataset in memory using shared memory")
            self._build_traj_info()
            self._load_whole_dataset_shared()

    def _build_traj_info(self):
        """Build {full_path: [(demo_key, traj_len), ...]} from entries."""
        seen = {}
        for entry in self.entries:
            full_path = str(self.hdf5_base_dir / entry['hdf5_path'])
            demo_key = entry['demo_key']
            seen.setdefault(full_path, set()).add(demo_key)

        self._traj_info = {}
        for full_path, demo_keys in seen.items():
            self._traj_info[full_path] = []
            with h5py.File(full_path, 'r') as f:
                dg = f['data'] if 'data' in f else f
                for demo_key in demo_keys:
                    img_key = self.image_keys[0]
                    resolved = img_key if img_key in dg[demo_key] else f"obs/{img_key.split('/')[-1]}"
                    traj_len = len(dg[demo_key][resolved])
                    self._traj_info[full_path].append((demo_key, traj_len))

    def _load_whole_dataset_shared(self):
        """
        Rank-0 loads HDF5 → builds ONE tensor per key →
        puts each tensor in /dev/shm → broadcasts metadata.
        Other ranks just attach by name.
        """
        global SHM_META_QA
        world_rank = dist.get_rank() if dist.is_initialized() else 0

        if dist.is_initialized():
            dist.barrier()

        meta = {}
        if world_rank == 0:
            before = psutil.virtual_memory().available
            print("Loading entire QA dataset into RAM on rank-0 ...")

            all_keys = self.image_keys + self.proprio_keys + self.action_keys
            self._demo_key_to_index = {}

            # Separate paths that can reuse existing VL shared memory from those
            # that must be loaded from HDF5 directly.
            paths_from_registry = {}   # full_path -> registry_entry
            paths_to_load = {}         # full_path -> demos list

            for full_path, demos in self._traj_info.items():
                if self.share_ram:
                    reg = data_utils.lookup_shared_hdf5(full_path)
                    if reg is not None:
                        missing = [k for k in all_keys if k not in reg["data"] or k not in reg["bounds"]]
                        if not missing:
                            paths_from_registry[full_path] = reg
                            continue
                        else:
                            print(f"[share_ram] {full_path}: missing keys {missing} in registry, loading from HDF5")
                paths_to_load[full_path] = demos

            # --- Reuse registry blocks ---
            for full_path, reg in paths_from_registry.items():
                for key in all_keys:
                    safe_key = key.replace('/', '_')
                    qa_shm_key = f"qa_{full_path.replace('/', '_').replace('.', '_')}_{safe_key}"
                    qa_boundaries_key = f"{qa_shm_key}_boundaries"
                    # Point qa_ keys at VL's actual shared memory blocks
                    meta[qa_shm_key] = reg["data"][key]
                    meta[qa_boundaries_key] = reg["bounds"][key]
                # Reuse VL's demo ordering for this path
                for k, v in reg["demo_key_to_index"].items():
                    self._demo_key_to_index[k] = v
                print(f"[share_ram] Reusing VL shared memory for {full_path}")

            # --- Load remaining paths from HDF5 ---
            dataset_data = {}
            for full_path, demos in paths_to_load.items():
                dataset_data[full_path] = {}
                with h5py.File(full_path, 'r') as f:
                    dg = f['data'] if 'data' in f else f
                    for demo_key, traj_len in demos:
                        dataset_data[full_path][demo_key] = {}
                        for key in all_keys:
                            if key in dg[demo_key]:
                                dataset_data[full_path][demo_key][key] = np.array(dg[demo_key][key])
                            else:
                                # try obs/ prefix fallback for image keys
                                fallback = f"obs/{key.split('/')[-1]}"
                                if fallback in dg[demo_key]:
                                    dataset_data[full_path][demo_key][key] = np.array(dg[demo_key][fallback])
                                else:
                                    raise KeyError(f"Key '{key}' not found in {full_path}/{demo_key}")

            flattened_data = {}

            for full_path, demo_dict in dataset_data.items():
                for key in all_keys:
                    all_trajectories = []
                    demo_keys_list = []
                    for demo_key, key_dict in demo_dict.items():
                        trajectory = key_dict[key]
                        all_trajectories.append(trajectory)
                        demo_keys_list.append(demo_key)
                        self._demo_key_to_index[(full_path, demo_key)] = len(demo_keys_list) - 1

                    safe_key = key.replace('/', '_')
                    shm_key = f"qa_{full_path.replace('/', '_').replace('.', '_')}_{safe_key}"

                    concatenated_trajectories = np.concatenate(all_trajectories, axis=0)
                    flattened_data[shm_key] = concatenated_trajectories

                    trajectory_boundaries = []
                    start_idx = 0
                    for traj in all_trajectories:
                        end_idx = start_idx + traj.shape[0]
                        trajectory_boundaries.append(np.array([start_idx, end_idx]))
                        start_idx = end_idx

                    boundaries_key = f"{shm_key}_boundaries"
                    flattened_data[boundaries_key] = np.stack(trajectory_boundaries)

            with open("/tmp/demo_key_to_index_qa.pkl", "wb") as f:
                pickle.dump(self._demo_key_to_index, f)

            del dataset_data
            gc.collect()

            print("Moving QA arrays to shared memory ...")
            for k, arr in flattened_data.items():
                k, tensor, shm = data_utils.create_shared(k, arr)
                print(f"Created shared memory block: {k}, shape={arr.shape}, dtype={arr.dtype}")
                meta[k] = (k, arr.shape, str(arr.dtype))

            used = (psutil.virtual_memory().available - before) / 1e9
            print(f"rank-0: QA dataset ready, ΔRAM≈{-used:.2f} GB")
            SHM_META_QA = meta

        obj = [meta]
        if dist.is_initialized():
            dist.barrier()
            dist.broadcast_object_list(obj, src=0)
            with open("/tmp/demo_key_to_index_qa.pkl", "rb") as f:
                self._demo_key_to_index = pickle.load(f)
        meta = obj[0]

        if world_rank != 0:
            SHM_META_QA = meta

    def _lazy_attach(self):
        """Lazy attachment to QA shared memory blocks."""
        global SHARED_TENSORS_QA, SHARED_HANDLES_QA
        if not getattr(self, "_loaded", False):
            world_rank = dist.get_rank() if dist.is_initialized() else 0
            print(f"[Rank {world_rank}] Attaching QA shared memory blocks: {list(SHM_META_QA.keys())}")
            for k in SHM_META_QA.keys():
                if k not in SHARED_TENSORS_QA:
                    name, shape, dtype = SHM_META_QA[k]
                    tensor, shm = data_utils.attach_shared(name, shape, dtype)
                    SHARED_TENSORS_QA[k] = tensor
                    SHARED_HANDLES_QA[k] = shm
            self._loaded = True

    def _get_raw_from_shared(self, full_path: str, demo_key: str, key: str, timestep_indices: List[int]) -> np.ndarray:
        """Retrieve raw numpy data for a single key/demo from shared memory."""
        self._lazy_attach()
        safe_key = key.replace('/', '_')
        shm_key = f"qa_{full_path.replace('/', '_').replace('.', '_')}_{safe_key}"
        boundaries_key = f"{shm_key}_boundaries"
        if shm_key not in SHM_META_QA:
            shm_key = [k for k in SHM_META_QA if re.match(f"{shm_key}_\\d+$", k)][0]
        if boundaries_key not in SHM_META_QA:
            boundaries_key = [k for k in SHM_META_QA if re.match(f"{boundaries_key}_\\d+$", k)][0]
        demo_idx = self._demo_key_to_index[(full_path, demo_key)]
        start, end = SHARED_TENSORS_QA[boundaries_key][demo_idx].tolist()
        return SHARED_TENSORS_QA[shm_key][start:end][timestep_indices].numpy()

    def _build_entries(self, split_ratio: float = -1.0, split: str = "train"):
        entries = []
        json_files = [self.hdf5_base_dir / json_file for json_file in self.json_files]
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
                entries.extend(data if isinstance(data, list) else [data])

        entries = [e for e in entries if e.get('decision', 'keep') == 'keep']
        print(f"Loaded {len(entries)} entries")
        # if split_ratio > 0; use it to split the spit_ratio if train and 1-split_ratio from behind if val
        if split_ratio > 0:
            if split == "train":
                entries = entries[:int(len(entries) * split_ratio)]
            else:
                entries = entries[int(len(entries) * split_ratio):]
        return entries

    def _encode_text(self, text: str) -> List[int]:
        """Encode text using tokenizer."""
        if hasattr(self.tokenizer, 'encode'):
            return self.tokenizer.encode(text, add_special_tokens=False)
        result = self.tokenizer(text, add_special_tokens=False)
        return result['input_ids'] if isinstance(result, dict) else result

    def convert_strs_to_token_ids(self, token_strs: Union[str, List[str]]) -> torch.Tensor:
        '''
        Convert a string or list of strings to token ids with max_length padding.
        Similar to dataset_vl.py convert_strs_to_token_ids.
        '''
        if isinstance(token_strs, str):
            token_strs = [token_strs]
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

    def _create_token_sequence(self, instruction: str, query: str, answer: str, num_frames: int, action_len: int):
        """Create token sequence: instruction, img1, img2, ..., query, answer"""
        # Tokenize instruction
        inst_ids = self._encode_text(instruction) if instruction else []
        if self.shared_config.pad_inst_tokens:
            if len(inst_ids) > self.shared_config.max_inst_tokens:
                print(f"Warning: Inst ids are longer than the max length, cutting to {self.shared_config.max_inst_tokens} tokens")
                inst_ids = inst_ids[:self.shared_config.max_inst_tokens]
            inst_ids = inst_ids + [self.tokenizer.pad_token_id] * (self.shared_config.max_inst_tokens - len(inst_ids))

        # Create image token pattern for each frame: img1 x N, img2 x N
        img_token_pattern = []
        for img_token_id in self.image_token_ids:
            img_token_pattern.extend([img_token_id] * self.tokens_per_image)

        num_action_tokens = 1
        action_token_pattern = self.action_token_ids * num_action_tokens if action_len > 0 else []

        # Repeat pattern for each frame
        all_img_tokens = (img_token_pattern + action_token_pattern) * num_frames

        # Tokenize query and answer
        query_ids = self._encode_text(query) if query else []
        answer_ids = self._encode_text(answer) if answer else []

        # Combine: instruction + images + query + answer
        full_ids = []
        input_ids = inst_ids + all_img_tokens
        if not self.dataset_config.qa_remove_query:
            input_ids += query_ids
        if self.add_answer_ids:
            full_ids = input_ids + answer_ids
        else:
            full_ids = input_ids

        # Create prompt mask: 0 for input (instruction + images + query), 1 for answer
        prompt_mask = torch.zeros(len(full_ids), dtype=torch.bool)
        prompt_mask[len(input_ids):] = 1

        # Find image token positions
        image_positions = OrderedDict({img_str: [] for img_str in self.image_keys})
        action_inp_token_pos = []
        text_out_token_pos = []
        for i, token_id in enumerate(full_ids):
            if token_id in self.image_token_ids:
                img_str = self.image_keys[self.image_token_ids.index(token_id)]
                image_positions[img_str].append(i) # order is preserved
            elif token_id in self.action_token_ids:
                action_inp_token_pos.append(i)
            if i >= len(input_ids):
                text_out_token_pos.append(i)

        return full_ids, prompt_mask, image_positions, action_inp_token_pos, text_out_token_pos

    # --- Helper functions for loading data from HDF5 ---
    def _helper_load_images_from_hdf5(self, hdf5_path: str, demo_key: str, timestep_indices: List[int]):
        """Helper: Load images from HDF5 file."""
        full_path = self.hdf5_base_dir / hdf5_path if self.hdf5_base_dir else Path(hdf5_path)
        if not full_path.exists():
            raise FileNotFoundError(f"HDF5 not found: {full_path}")

        images_dict = OrderedDict({key: [] for key in self.image_keys})

        if self.load_in_mem:
            for img_key in self.image_keys:
                raw = self._get_raw_from_shared(str(full_path), demo_key, img_key, timestep_indices)
                assert raw.dtype == np.uint8, f"Expected uint8, got {raw.dtype}"
                raw = raw.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
                images_dict[img_key] = torch.from_numpy(raw)
            return images_dict

        # dict order is same as self.image_keys
        with h5py.File(full_path, 'r') as f:
            data_group = f['data'] if 'data' in f else f
            demo_group = data_group[demo_key]

            for img_key in self.image_keys:
                # Try to find the key
                found_key = None
                if img_key in demo_group:
                    found_key = img_key
                elif f"obs/{img_key.split('/')[-1]}" in demo_group:
                    found_key = f"obs/{img_key.split('/')[-1]}"

                if found_key is None:
                    raise KeyError(f"Image key '{img_key}' not found in {full_path}/{demo_key}")

                dataset = demo_group[found_key]
                # # get the min and max of the indices, gather from the dataset, and last keep the ones in timestep_indices
                min_ti = min(timestep_indices)
                max_ti = max(timestep_indices)
                dataset_images = np.array(dataset[min_ti:max_ti + 1])
                dataset_images = np.array(dataset_images[np.array(timestep_indices) - min_ti])
                assert dataset_images.shape[0] == len(timestep_indices), f"dataset_images.shape[0]= {dataset_images.shape[0]} != {len(timestep_indices)=}"
                assert dataset_images.dtype == np.uint8, f"dataset_images.dtype= {dataset_images.dtype}, expected uint8"
                assert dataset_images.shape[3] == 3, f"dataset_images.shape[2]= {dataset_images.shape[2]}, expected 3"
                dataset_images = dataset_images.transpose(0, 3, 1, 2) # (H, W, 3) -> (3, H, W)
                dataset_images = dataset_images.astype(np.float32) / 255.0
                images_dict[img_key] = torch.from_numpy(dataset_images)

        return images_dict

    def _helper_load_proprio_from_hdf5(self, hdf5_path: str, demo_key: str, timestep_indices: List[int]):
        """Helper: Load and process proprioceptive data from HDF5 file."""
        full_path = self.hdf5_base_dir / hdf5_path if self.hdf5_base_dir else Path(hdf5_path)
        if not full_path.exists():
            raise FileNotFoundError(f"HDF5 not found: {full_path}")

        n_grip_vals = 1
        proprio = {}
        gripper_state = None

        if self.load_in_mem:
            for k in self.proprio_keys:
                raw = self._get_raw_from_shared(str(full_path), demo_key, k, timestep_indices)
                if 'gripper' in k:
                    gripper_state = raw[:, -n_grip_vals:].astype(np.float32)
                else:
                    proprio[k] = raw.astype(np.float32)
        else:
            with h5py.File(full_path, 'r') as f:
                data_group = f['data'] if 'data' in f else f
                demo_group = data_group[demo_key]

                for k in self.proprio_keys:
                    assert k in demo_group, f"Proprio key '{k}' not found in {full_path}/{demo_key}"
                    if 'gripper' in k:
                        gripper_data = []
                        for idx in timestep_indices:
                            gripper_data.append(demo_group[k][idx][None, -n_grip_vals:])
                        gripper_state = np.concatenate(gripper_data, axis=0)
                    else:
                        proprio_data = []
                        for idx in timestep_indices:
                            proprio_data.append(demo_group[k][idx][None, :])
                        proprio[k] = np.concatenate(proprio_data, axis=0)

        # Apply noise to non-gripper keys only
        if self.proprio_noise > 0:
            for k in proprio.keys():
                proprio[k] += np.random.normal(0, self.proprio_noise, proprio[k].shape)

        # Concatenate all proprio keys
        if len(proprio) > 0:
            proprio_vec = np.concatenate([proprio[k] for k in proprio.keys()], axis=-1)
            if gripper_state is not None:
                proprio_vec = np.concatenate([proprio_vec, gripper_state], axis=-1)
        else:
            proprio_vec = gripper_state

        return torch.from_numpy(proprio_vec).float()

    def _helper_load_action_from_hdf5(self, hdf5_path: str, demo_key: str, timestep_indices: List[int]):
        """Helper: Load and process action data from HDF5 file."""
        full_path = self.hdf5_base_dir / hdf5_path if self.hdf5_base_dir else Path(hdf5_path)
        if not full_path.exists():
            raise FileNotFoundError(f"HDF5 not found: {full_path}")

        k = self.action_keys[0]

        if self.load_in_mem:
            action = self._get_raw_from_shared(str(full_path), demo_key, k, timestep_indices).astype(np.float32)
        else:
            with h5py.File(full_path, 'r') as f:
                data_group = f['data'] if 'data' in f else f
                demo_group = data_group[demo_key]
                assert k in demo_group, f"Action key '{k}' not found in {full_path}/{demo_key}"
                action_data = []
                for idx in timestep_indices:
                    action_data.append(demo_group[k][idx][None, :])
                action = np.concatenate(action_data, axis=0)

        # Determine no-noise indices based on action shape (gripper/switch dims)
        action_shape = action.shape[1]  # shape is (T, action_dim)
        no_noise_indices = []
        if action_shape == 7:
            no_noise_indices = [6]
        elif action_shape == 12:
            # 6 eef pose + gripper action + 4 base action + 1 switch mode
            no_noise_indices = [6, 11]
        else:
            raise ValueError(f"Unknown action shape: {action_shape}")

        # Apply noise to all dimensions except gripper/switch
        if self.action_noise > 0:
            noise = np.random.normal(0, self.action_noise, action.shape)
            noise[..., no_noise_indices] = 0.0
            action += noise

        # add all zeros to the action of shape (T,1) for EOS token position
        # TODO: very risky hack
        action = np.concatenate([action, np.zeros((action.shape[0], 1))], axis=1)
        return torch.from_numpy(action).float()

    def _load_data_from_hdf5(self, hdf5_path: str, demo_key: str, timestep_indices: List[int]):
        """Load images, proprio, and action data from HDF5 file."""
        images_dict = self._helper_load_images_from_hdf5(hdf5_path, demo_key, timestep_indices)
        proprio = torch.empty(0)
        action = torch.empty(0)
        if not self.remove_proprio:
            proprio = self._helper_load_proprio_from_hdf5(hdf5_path, demo_key, timestep_indices)
            if proprio.ndim == 2: proprio = proprio[:, None, :]
        if not self.remove_action:
            action = self._helper_load_action_from_hdf5(hdf5_path, demo_key, timestep_indices)
            if action.ndim == 2: action = action[:, None, :]
        return images_dict, proprio, action

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        instruction = entry.get('task-instruction', '')
        query = entry.get('query', '')
        answer = entry.get('answer', '')
        if not isinstance(answer, str):
            answer = str(answer)
        if not answer.endswith('.'):
            answer = answer + '.'

        # Load all data from HDF5
        timestep_indices = entry['timestep_index']
        if isinstance(timestep_indices, str):
            timestep_indices = [int(x) for x in timestep_indices.split(",")]
        # check the difference between the timestep_indices and match it with downsample
        diff = timestep_indices[1] - timestep_indices[0]
        factor = self.downsample_obs // diff
        # pick a random start number between 0 and factor
        if factor > 0:
            start_number = np.random.randint(0, factor)
            timestep_indices = timestep_indices[start_number::factor]

        images_dict, proprio, action = self._load_data_from_hdf5(
            entry['hdf5_path'], entry['demo_key'], timestep_indices
        )

        # Create token sequence
        input_ids, text_prompt_mask, image_positions, action_inp_token_pos, text_out_token_pos = \
                self._create_token_sequence(
                    instruction, query, answer, len(timestep_indices), action_len=action.shape[0]
                )

        timestep_index = entry.get('timestep_index')
        if isinstance(timestep_index, str):
            timestep_index = [int(x) for x in timestep_index.split(",")]
        return {
            'instruction': instruction,
            'observation': images_dict,
            'proprio': proprio[:, :1, :], # only the first timestep is used
            'action': action.repeat(1, self.num_pred_steps, 1), # TODO: very risky hack
            'query': query,
            'answer': answer,
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'image_token_positions': OrderedDict({img_str: torch.tensor(positions, dtype=torch.long) for img_str, positions in image_positions.items()}),
            'action_inp_token_pos': torch.tensor(action_inp_token_pos, dtype=torch.long),
            'action_out_token_pos': torch.zeros(action.shape[0], dtype=torch.long)-1, # if this is set to be non-zero, ensure the action is not just repeated self.num_pred_steps times.
            'text_prompt_mask': text_prompt_mask,
            'action_prompt_mask': torch.zeros(len(input_ids), dtype=torch.bool),
            'state_supervision': torch.empty(0),
            'state_supervision_out_token_pos': torch.empty(0, dtype=torch.long),
            'metadata': {
                'id': entry.get('id'),
                'timestep_index': timestep_index,
                'hdf5_path': entry.get('hdf5_path'),
                'demo_key': entry.get('demo_key'),
            }
        }

    def save_split(self, split_dir: str):
        pass

    def shuffle_dataset(self, seed: int = 0):
        # rebuild the entries, because the original entries might have been split if max_qa_size is set
        self.entries = self._build_entries(split_ratio=self.split_ratio, split=self.split)
        # shuffle the entries
        rng = np.random.RandomState(seed=seed)
        shuffled_indices = list(range(len(self.entries)))
        rng.shuffle(shuffled_indices)
        self.entries = [self.entries[i] for i in shuffled_indices]
        if self.dataset_config.max_qa_size_post_shuffle > 0: # each epoch rank will still see the same data points
            print(colored(f"Truncating the entries to (max_qa_size_post_shuffle) {self.dataset_config.max_qa_size_post_shuffle} from {len(self.entries)}", "yellow"))
            self.entries = self.entries[:self.dataset_config.max_qa_size_post_shuffle] # truncate the entries to the max_qa_size
        return

class QADatasetGPTState(QADataset):
    '''
    Structure of an output data point is the following:
    Input: <query> <images> (use the langsequence tokenizer) (the action out token pos is actually the answer out token pos)
    Answer: <answer>
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_answer_ids = False
        # Initialize LangTrajSequence generator similar to TaskDatasetWithTokenizer
        self.sequence_generator = TokenSequenceGen.LangTrajSequence(
            image_keys=self.image_keys,
            image_token_ids=self.image_token_ids,
            tokens_per_frame=self.tokens_per_image,
            action_token_ids=self.action_token_ids,
            tokens_per_action=1,
            action_in_inputs=not self.remove_action,
            pad_inst_tokens=self.shared_config.pad_inst_tokens,
            max_inst_tokens=self.shared_config.max_inst_tokens,
            inst_token_pad_value=self.tokenizer.pad_token_id,
        )
        self.mode = self.shared_config.state_supervision_mode
        self.max_ss_size = self.dataset_config.max_ss_size
        # bbox_str means query will be used, but bbox_inst_str means instruction will be used
        assert self.mode in ['bbox_str'], f"Invalid mode: {self.mode}"

    def __getitem__(self, idx):
        entry = self.entries[idx]
        query = entry.get('query', '')
        answer = entry.get('answer', '')
        if not isinstance(answer, str):
            answer = str(answer)
        if not isinstance(query, str):
            query = str(query)
        # if anwere does not end with a period, add a period
        if not answer.endswith('.'):
            answer = answer + '.'
        rel_frame_index = entry.get('rel_frame_index', 0)

        # Load all data from HDF5
        timestep_indices = entry['timestep_index']
        if isinstance(timestep_indices, str):
            timestep_indices = [int(x) for x in timestep_indices.split(",")]
        # check the difference between the timestep_indices and match it with downsample
        diff = timestep_indices[1] - timestep_indices[0] if len(timestep_indices) > 1 else 1
        if self.shared_config.downsample_obs != diff:
            assert self.shared_config.downsample_obs % diff == 0, f"{self.shared_config.downsample_obs=} and {diff=}"
            start_number = np.random.randint(0, self.shared_config.downsample_obs // diff)
            timestep_indices = timestep_indices[start_number::self.shared_config.downsample_obs // diff]
        diff = timestep_indices[1] - timestep_indices[0] if len(timestep_indices) > 1 else 1
        assert diff == self.shared_config.downsample_obs, f"{self.shared_config.downsample_obs=} and {diff=}"

        images_dict, proprio, action = self._load_data_from_hdf5(
            entry['hdf5_path'], entry['demo_key'], timestep_indices
        )

        # Tokenize query and use it as instruction for LangTrajSequence
        query_ids = self._encode_text(query) if query else []
        num_frames = len(timestep_indices)

        # Use LangTrajSequence to create token sequence: query + images
        full_ids, img_token_positions, action_inp_token_pos, answer_out_token_pos = \
            self.sequence_generator(query_ids, num_frames)

        answer_out_token_pos = answer_out_token_pos[rel_frame_index:]
        if self.max_ss_size > 0:
            # sample the answer of length min(self.max_ss_size, len(answer_out_token_pos)) by sampling random points
            ss_size = min(self.max_ss_size, len(answer_out_token_pos))
            ss_indices = np.sort(np.random.choice(len(answer_out_token_pos), ss_size, replace=False))
            answer_out_token_pos = answer_out_token_pos[ss_indices]

        # Tokenize answer separately (not included in input_ids) with max_length padding
        answer_ids = self.convert_strs_to_token_ids(answer)
        # find the frame_index from the data point. make sure the action_out_token_pos is -1 before the frame_index
        # repeat answer_ids for the the number of positions in answer_out_token_pos
        # we are assuming a condition here: given the instruction, the answer will always remain the same after the specified frame_index
        answer_ids = answer_ids.repeat(len(answer_out_token_pos), 1)

        # Create prompt mask: all input tokens (query + images) are prompt, answer is not
        text_prompt_mask = torch.zeros(len(full_ids), dtype=torch.bool)
        # action_out_token_pos represents where the answer should be generated
        # For separate answer, we don't include answer in input_ids, so prompt_mask stays all False

        timestep_index = entry.get('timestep_index')
        if isinstance(timestep_index, str):
            timestep_index = [int(x) for x in timestep_index.split(",")]
        return {
            'instruction': query,  # query acts as instruction
            'observation': images_dict,
            'proprio': proprio[:, :1, :] if proprio.numel() > 0 else proprio,  # only the first timestep is used
            'action': action.repeat(1, self.num_pred_steps, 1) if action.numel() > 0 else action,  # TODO: very risky hack
            'query': query,
            'answer': answer,
            'state_supervision': answer_ids,
            'state_supervision_out_token_pos': answer_out_token_pos,
            'input_ids': full_ids,  # query + images (no answer)
            'image_token_positions': img_token_positions,
            'action_inp_token_pos': action_inp_token_pos,
            'action_out_token_pos': torch.zeros(action.shape[0], dtype=torch.long)-1,  # these are the answer output positions
            'text_prompt_mask': text_prompt_mask,
            'action_prompt_mask': torch.zeros(len(full_ids), dtype=torch.bool), # no action loss calculation
            'answer_ids': answer_ids,
            'metadata': {
                'id': entry.get('id'),
                'timestep_index': timestep_index,
                'hdf5_path': entry.get('hdf5_path'),
                'demo_key': entry.get('demo_key'),
            }
        }

class QADatasetStateSupervision(QADataset):

    def __init__(self, dataset_config: DatasetConfig, shared_config: SharedConfig, **kwargs):
        super().__init__(dataset_config=dataset_config, shared_config=shared_config, **kwargs)
        # store the mode of the state qa dataset: float_all, string_rnd
        self.mode = shared_config.state_supervision_mode
        # bbox_str means query will be used, but bbox_inst_str means instruction will be used
        assert self.mode in ['bbox_str', 'bbox_inst_str'], f"Invalid mode: {self.mode}"
        self.entries = [e for e in self.entries]
        self.add_answer_ids = False
        # Initialize LangTrajSequence generator similar to TaskDatasetWithTokenizer
        self.sequence_generator = TokenSequenceGen.LangTrajSequence(
            image_keys=self.image_keys,
            image_token_ids=self.image_token_ids,
            tokens_per_frame=self.tokens_per_image,
            action_token_ids=self.action_token_ids,
            tokens_per_action=1,
            action_in_inputs=not self.remove_action,
            pad_inst_tokens=self.shared_config.pad_inst_tokens,
            max_inst_tokens=self.shared_config.max_inst_tokens,
            inst_token_pad_value=self.tokenizer.pad_token_id,
        )

    def __getitem__(self, idx):
        entry = self.entries[idx]
        instruction = entry.get('task-instruction', '')
        # Load all data from HDF5
        timestep_indices = entry['timestep_index']
        if isinstance(timestep_indices, str):
            timestep_indices = [int(x) for x in timestep_indices.split(",")]

        # randomly pick a query from the query_list
        query_list = entry.get('query_list', []) # TODO: make two different modes here
        query = query_list[np.random.randint(0, len(query_list))]
        if self.mode == 'bbox_inst_str':
            query = instruction

        state_supervision = entry.get('state_supervision')
        # convert it to a string
        answer_list = []
        for state in state_supervision:
            state_np = np.array(state)
            # state_np = state_np + np.random.normal(0, 0.01, state_np.shape) # adding noise to the state supervision, already added in each json file
            answer = ' '.join([str(round(x, 2)) for x in state_np.tolist()])
            answer_list.append(answer)

        # check the difference between the timestep_indices and match it with downsample
        diff = timestep_indices[1] - timestep_indices[0]
        assert diff == self.shared_config.downsample_obs, f"{self.shared_config.downsample_obs=} and {diff=}"

        images_dict, proprio, action = self._load_data_from_hdf5(
            entry['hdf5_path'], entry['demo_key'], timestep_indices
        )

        # Tokenize query and use it as instruction for LangTrajSequence
        query_ids = self._encode_text(query) if query else []
        num_frames = len(timestep_indices)

        # Use LangTrajSequence to create token sequence: query + images
        full_ids, img_token_positions, action_inp_token_pos, answer_out_token_pos = \
            self.sequence_generator(query_ids, num_frames)

        answer_ids = self.convert_strs_to_token_ids(answer_list)
        text_prompt_mask = torch.zeros(len(full_ids), dtype=torch.bool)

        timestep_index = entry.get('timestep_index')
        if isinstance(timestep_index, str):
            timestep_index = [int(x) for x in timestep_index.split(",")]
        return {
            'instruction': query,  # query acts as instruction
            'observation': images_dict,
            'proprio': proprio[:, :1, :] if proprio.numel() > 0 else proprio,  # only the first timestep is used
            'action': action.repeat(1, self.num_pred_steps, 1) if action.numel() > 0 else action,  # TODO: very risky hack
            'query': query,
            'answer': answer,
            'state_supervision': answer_ids,
            'state_supervision_out_token_pos': answer_out_token_pos,
            'input_ids': full_ids,  # query + images (no answer)
            'image_token_positions': img_token_positions,
            'action_inp_token_pos': action_inp_token_pos,
            'action_out_token_pos': torch.zeros(action.shape[0], dtype=torch.long)-1,  # these are the answer output positions
            'text_prompt_mask': text_prompt_mask,
            'action_prompt_mask': torch.zeros(len(full_ids), dtype=torch.bool), # no action loss calculation
            'answer_ids': answer_ids,
            'metadata': {
                'id': entry.get('id'),
                'timestep_index': timestep_index,
                'hdf5_path': entry.get('hdf5_path'),
                'demo_key': entry.get('demo_key'),
            }
        }


# Example usage:
if __name__ == "__main__":
    from transformers import AutoTokenizer

    # Use Llama or Qwen tokenizer
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")  # or "Qwen/Qwen2-7B"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")  # or "Qwen/Qwen2-7B-Instruct"

    # dataset = QADataset(
    #     json_files=["scripts/dataset_test.json"],
    #     hdf5_base_dir=os.environ.get("CASAPLAY_DATAROOT"),
    #     tokenizer=tokenizer,
    #     tokens_per_image=1,
    # )

    dataset = QADatasetStateSupervision(
        dataset_config=DatasetConfig(dataset_json="config/qa/qa_robocasa_debug.json"),
        shared_config=SharedConfig(tokenizer_name="Qwen/Qwen2-7B-Instruct"),
    )

    sample = dataset[0]
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    # print(f"Prompt mask shape: {sample['prompt_mask'].shape}")
    import ipdb; ipdb.set_trace()
