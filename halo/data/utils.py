import os
import h5py
import pickle
import torch
import copy
from typing import Union, List, Tuple, Literal, Any, Dict, OrderedDict
import json
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm, trange
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from multiprocessing import shared_memory
import multiprocessing.resource_tracker as rt
from transformers import AutoTokenizer
from timm.data.transforms import RandomResizedCropAndInterpolation, ToTensor
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ColorJitter

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Cross-dataset shared-RAM registry
# Allows a QA dataset to reuse shared-memory blocks already created by a VL
# dataset that loaded the same HDF5 files (and vice-versa).
#
# Structure:
#   _SHARED_RAM_REGISTRY[abs_hdf5_path] = {
#       "demo_key_to_index": {(abs_hdf5_path, demo_key): int, ...},
#       "data":  {original_key_str: (shm_name, shape_tuple, dtype_str)},
#       "bounds": {original_key_str: (shm_name, shape_tuple, dtype_str)},
#   }
# ---------------------------------------------------------------------------
_SHARED_RAM_REGISTRY: dict = {}


def register_shared_hdf5(
    abs_hdf5_path: str,
    demo_key_to_index: dict,
    data_meta: dict,
    bounds_meta: dict,
):
    """Register shared-memory blocks for one HDF5 file.

    Args:
        abs_hdf5_path: Absolute path to the HDF5 file.
        demo_key_to_index: Mapping (abs_hdf5_path, demo_key) -> index.
        data_meta:   {original_key_str: (shm_name, shape, dtype_str)} for data.
        bounds_meta: {original_key_str: (shm_name, shape, dtype_str)} for boundaries.
    """
    _SHARED_RAM_REGISTRY[abs_hdf5_path] = {
        "demo_key_to_index": demo_key_to_index,
        "data": data_meta,
        "bounds": bounds_meta,
    }


def lookup_shared_hdf5(abs_hdf5_path: str):
    """Return registry entry for *abs_hdf5_path*, or None if not registered."""
    return _SHARED_RAM_REGISTRY.get(abs_hdf5_path, None)

def get_relevant_summary_data(traj_start_idx: int, traj_end_idx: int, summary_list: List[Dict[str, Any]], downsample_factor: int):
    retrieved_summaries = []
    for data in summary_list:
        if traj_start_idx >= data["end_idx"] or traj_end_idx <= data["start_idx"]: # if the summary is not relevant, skip it
            continue
        assign_start_idx = data["start_idx"]
        assign_end_idx = data["end_idx"]
        assign_start_idx = max(assign_start_idx, traj_start_idx)
        assign_end_idx = min(assign_end_idx, traj_end_idx)

        data_pass = copy.deepcopy(data)
        data_pass["start_idx"] = (assign_start_idx - traj_start_idx) // downsample_factor
        data_pass["end_idx"] = (assign_end_idx - traj_start_idx) // downsample_factor
        retrieved_summaries.append(data_pass)
    return retrieved_summaries


def get_action_token_str_list() -> List[str]:
    return ['<action>']

def get_img_token_str_list(image_keys: List[str]) -> List[str]:
    return [f"<{key.split('/')[-1]}>" for key in image_keys]

def add_image_tokens_to_tokenizer(tokenizer: Any, image_tokens: List[str]) -> Any:
    """Add image tokens to tokenizer if not present."""
    special_tokens = [tok for tok in image_tokens 
                        if tok not in tokenizer.get_vocab()]
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    return tokenizer

def build_tokenizer(tokenizer_name: str, image_keys: List[str]) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding_side="right")
    image_tokens = get_img_token_str_list(image_keys)
    action_tokens = get_action_token_str_list()
    tokenizer = add_image_tokens_to_tokenizer(tokenizer, image_tokens + action_tokens)
    return tokenizer

class MaybeToTensor:
    def __call__(self, x):
        # If NumPy array
        if isinstance(x, np.ndarray):
            tensor = torch.from_numpy(x)
            if tensor.dtype != torch.float32:
                tensor = tensor.float()
            if tensor.max() > 5.0:  # most likely in [0, 255]
                tensor = tensor / 255.0
            return tensor

        # If PIL Image
        if not isinstance(x, torch.Tensor):
            return transforms.functional.to_tensor(x)

        # If Tensor
        if torch.is_floating_point(x):
            if x.max() > 5.0: # most likely in [0, 255]
                return x / 255.0
        else:  # integer tensor
            return x.float() / 255.0
        
        return x


def get_vision_transform(
        size=(128, 128),
        scale=(0.65, 1.0),
        ratio=(1.0, 1.0),
        interpolation='bicubic',
        brightness=(0.6, 1.4),
        contrast=(0.6, 1.4),
        saturation=(0.6, 1.4),
        hue=0.0,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    ):
    """
    Get the vision transform for the dataset.
    """
    
    # Remove ToTensor() since input is already tensor
    vision_transform = Compose([
        RandomResizedCropAndInterpolation(
            size=size,
            scale=scale,
            ratio=ratio,
            interpolation=interpolation  # Keep as string for timm
        ),
        ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        ),
        # ToTensor() removed - input is already tensor
        Normalize(
            mean=mean,
            std=std
        )
    ])
    
    # Vision transform without augmentation
    no_aug_vision_transform = Compose([
        Resize(
            size=142 if size[0] == 128 else 256,
            interpolation=InterpolationMode.BICUBIC,  # Use enum for torchvision
            antialias=True
        ),
        CenterCrop(size=size),
        # ToTensor() removed - input is already tensor
        Normalize(
            mean=mean,
            std=std
        )
    ])
    
    return vision_transform, no_aug_vision_transform

def collate_fn_tokenizer(batch) -> Dict[str, torch.Tensor]:
    # gather all the keys needed for padding
    max_len = max(item['input_ids'].shape[0] for item in batch)
    max_obs_key_len = {key: max(len(item['observation'][key]) for item in batch) for key in batch[0]['observation'].keys()}
    max_img_token_positions_len = {key: max(len(item['image_token_positions'][key]) for item in batch) for key in batch[0]['image_token_positions'].keys()}
    max_action_len = max(len(item['action']) if 'action' in item else 0 for item in batch)
    max_proprio_len = max(len(item['proprio']) if 'proprio' in item else 0 for item in batch)
    max_action_inp_token_pos_len = max(len(item['action_inp_token_pos']) if 'action_inp_token_pos' in item else 0 for item in batch)
    max_action_out_token_pos_len = max(len(item['action_out_token_pos']) if 'action_out_token_pos' in item else 0 for item in batch)
    max_ss_len = max(len(item['state_supervision']) if 'state_supervision' in item else 0 for item in batch)
    max_ss_out_token_pos_len = max(len(item['state_supervision_out_token_pos']) if 'state_supervision_out_token_pos' in item else 0 for item in batch)
    assert max_action_out_token_pos_len == max_action_len, f"max_action_out_token_pos_len should be equal to max_action_len: {max_action_out_token_pos_len=} {max_action_len=}"
    assert max_ss_out_token_pos_len == max_ss_len, f"max_ss_out_token_pos_len should be equal to max_action_len: {max_ss_out_token_pos_len=} {max_ss_len=}"
    batch_size = len(batch)
    
    input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    text_prompt_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    action_prompt_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    action_tar_pad_mask = torch.zeros(batch_size, max_action_len, dtype=torch.bool)
    
    # action and proprio may be not present in all the items
    images_dict = OrderedDict({key: [] for key in batch[0]['observation'].keys()})
    action_list = []
    proprio_list = []

    # new position tokens with the paddings
    image_token_positions_dict = OrderedDict({key: [] for key in batch[0]['image_token_positions'].keys()})
    action_inp_token_pos = []
    action_out_token_pos = []
    state_supervision_list = []
    state_supervision_out_token_pos = []
    if max_ss_len > 0:
        ss_shape = [item for item in batch if 'state_supervision' in item and item['state_supervision'].shape[0] > 0][0]['state_supervision'].shape
    
    for i, item in enumerate(batch):
        # Read-only access: we only read from item, never modify it
        seq_len = item['input_ids'].shape[0]
        # Copy data into new tensors (input_ids, masks are new tensors created above)
        input_ids[i, :seq_len] = item['input_ids']
        text_prompt_mask[i, :seq_len] = item['text_prompt_mask']
        action_prompt_mask[i, :seq_len] = item['action_prompt_mask']
        for key, value in item['observation'].items():
            max_obs_len = max_obs_key_len[key]
            assert value.ndim == 4, f"Observation should be a 4D tensor: (T, C, H, W), got {value.shape}"
            # value shape is (T, C, H, W)
            # torch.nn.functional.pad creates a NEW tensor, does not modify the original
            value = torch.nn.functional.pad(value, (0, 0, 0, 0, 0, 0, 0, max_obs_len - value.shape[0]), value=0)
            # positions shape is (T',)
            max_img_token_pos_len = max_img_token_positions_len[key]
            positions = item['image_token_positions'][key]  # Read-only: just reading the tensor
            assert positions.ndim == 1, f"Image token positions should be a 1D tensor, got {positions.shape}"
            # print(f"padding image token positions from {positions.shape[0]} to {max_img_token_pos_len}")
            # Creates NEW tensor, original item['image_token_positions'][key] is unchanged
            positions = torch.nn.functional.pad(positions, (0, max_img_token_pos_len - positions.shape[0]), value=-1)

            images_dict[key].append(value)
            image_token_positions_dict[key].append(positions)
        
        # actions is output tokens, but can also be input tokens if action_token_positions are non-empty
        if max_action_len > 0:
            # action is expected to be of form T, N, D, where N is the number of action tokens
            # print(f"padding action from {item['action'].shape} to {max_action_len}")
            assert item['action'].ndim == 3, f"Action should be a 3D tensor: (T, N, D), got {item['action'].shape}"
            # Creates NEW tensor, original item['action'] is unchanged
            action_value = torch.nn.functional.pad(item['action'], (0, 0, 0, 0, 0, max_action_len - item['action'].shape[0]), value=0)
            action_tar_pad_mask[i, :item['action'].shape[0]] = True
            action_list.append(action_value)

        if max_action_out_token_pos_len > 0:
            # Creates NEW tensor, original item['action_out_token_pos'] is unchanged
            action_out_pos_val = torch.nn.functional.pad(item['action_out_token_pos'], (0, max_action_out_token_pos_len - item['action_out_token_pos'].shape[0]), value=-1)
            action_out_token_pos.append(action_out_pos_val)

        if max_ss_len > 0:
            if item['state_supervision'].shape[0] == 0:
                item['state_supervision'] = torch.zeros(*ss_shape, dtype=torch.long) - 1
            assert item['state_supervision'].ndim == 2, f"State supervision should be a 2D tensor: (T, D): {item['state_supervision'].shape=}"
            pad_amount = max_ss_len - item['state_supervision'].shape[0]
            state_supervision_val = torch.nn.functional.pad(
                item['state_supervision'],
                (0, 0, 0, pad_amount),  # (left_D, right_D, left_T, right_T)
                value=-1
            )
            state_supervision_list.append(state_supervision_val)

        if max_ss_out_token_pos_len > 0:
            # Creates NEW tensor, original item['state_supervision_out_token_pos'] is unchanged
            ss_out_pos_val = torch.nn.functional.pad(item['state_supervision_out_token_pos'], (0, max_ss_out_token_pos_len - item['state_supervision_out_token_pos'].shape[0]), value=-1)
            state_supervision_out_token_pos.append(ss_out_pos_val)

        if max_action_inp_token_pos_len > 0: # max_action_token_positions_len may be different from max_action_len
            # print(f"padding action_token_positions from {item['action_inp_token_pos'].shape[0]} to {max_action_inp_token_pos_len}")
            # Creates NEW tensor, original item['action_inp_token_pos'] is unchanged
            action_positions = torch.nn.functional.pad(
                item['action_inp_token_pos'], (0, max_action_inp_token_pos_len - item['action_inp_token_pos'].shape[0]), value=-1)
            action_inp_token_pos.append(action_positions)

        if max_proprio_len > 0:
            # proprio is expected to be of form T, N, D, where N is the number of proprio tokens
            # print(f"padding proprio from {item['proprio'].shape[0]} to {max_proprio_len}")
            assert item['proprio'].ndim == 3, f"Proprio should be a 3D tensor: (T, N, D), got {item['proprio'].shape}"
            # Creates NEW tensor, original item['proprio'] is unchanged
            proprio_value = torch.nn.functional.pad(item['proprio'], (0, 0, 0, 0, 0, max_proprio_len - item['proprio'].shape[0]), value=0)
            proprio_list.append(proprio_value)

    for key in images_dict:
        images_dict[key] = torch.stack(images_dict[key], dim=0)
    
    for key in image_token_positions_dict:
        image_token_positions_dict[key] = torch.stack(image_token_positions_dict[key], dim=0)

    action_tensor = torch.stack(action_list, dim=0) if len(action_list) > 0 else torch.empty(0)
    proprio_tensor = torch.stack(proprio_list, dim=0) if len(proprio_list) > 0 else torch.empty(0)
    action_inp_token_pos = torch.stack(action_inp_token_pos, dim=0) if len(action_inp_token_pos) > 0 else torch.empty(0)
    action_out_token_pos = torch.stack(action_out_token_pos, dim=0) if len(action_out_token_pos) > 0 else torch.empty(0)
    state_supervision_tensor = torch.stack(state_supervision_list, dim=0) if len(state_supervision_list) > 0 else torch.empty(0)
    state_supervision_out_token_pos = torch.stack(state_supervision_out_token_pos, dim=0) if len(state_supervision_out_token_pos) > 0 else torch.empty(0)
    return {
        'input_ids': input_ids,
        'text_prompt_mask': text_prompt_mask,
        'action_prompt_mask': action_prompt_mask, # wrt full sequence and contains 0 also for exploratory actions
        'observation': images_dict,
        'action': action_tensor,
        'proprio': proprio_tensor,
        'image_token_positions': image_token_positions_dict,
        'action_inp_token_pos': action_inp_token_pos, # token positions to get input from
        'action_out_token_pos': action_out_token_pos, # token positions to get output from
        'action_tar_pad_mask': action_tar_pad_mask, # mask for the action target (because we pad it to the max length)
        'state_supervision': state_supervision_tensor,
        'state_supervision_out_token_pos': state_supervision_out_token_pos,
    }



def collate_fn_lerobot(batch):
    keys = batch[0].keys()
    padded_data = {k: [] for k in keys}
    for k in keys:
        if k == 'task':
            padded_data[k] = [sample[k] for sample in batch]
        elif k == 'task_embedding' or k == 'task_clip_embedding' or k == 'language_embedding':
            padded_data[k] = torch.stack([sample[k] for sample in batch], dim=0)
        else:
            max_length = max(len(sample[k]) for sample in batch)
            if max_length == 0: max_length = 1
            fill_value = True if k.endswith('_is_pad') else 0
            padded_samples = [pad_data_to_max_length(sample[k], max_length, fill_value=fill_value) for sample in batch]
            padded_data[k] = torch.stack(padded_samples, dim=0)
    return padded_data

def collate_history_batch(batch):
    """
    Custom collate function to handle history-related values with variable lengths.
    Uses the existing pad_data_to_max_length function from utils.py.
    Assumes history keys are never None.
    """
    # Separate history and non-history data
    history_keys = ['hist_observation', 'hist_action', 'hist_proprio', 'hist_mask']
    non_history_keys = [k for k in batch[0].keys() if k not in history_keys]
    # Handle non-history data (use default collate)
    non_history_batch = [{k: sample[k] for k in non_history_keys} for sample in batch]
    # collated_non_history = torch.utils.data.dataloader.default_collate(non_history_batch)
    collated_non_history = {}
    max_length = max(max(len(sample[non_history_keys[0]]) for sample in batch), 1)
    for k in non_history_keys:
        if k == 'language_embedding':
            collated_non_history[k] = torch.stack([sample[k] for sample in batch], dim=0)
        elif k == 'eagle_input_ids' or k == 'eagle_attention_mask':
            # left padding is to be done in the model with new type of max length
            special_max_length = max(max(sample[k].shape[-1] for sample in batch), 1)
            collated_non_history[k] = torch.stack([left_pad_data_to_max_length(sample[k], special_max_length, dim=1) for sample in batch], dim=0)
        else:
            collated_non_history[k] = torch.stack([pad_data_to_max_length(sample[k], max_length) for sample in batch], dim=0)

    # Handle history data with proper padding
    collated_history = {}
    for key in history_keys:
        if key not in batch[0]:
            continue
        # Find the maximum length in this batch
        max_length = max(len(sample[key]) for sample in batch)
        if max_length == 0:
            max_length = 1
        # Use the existing pad_data_to_max_length function
        padded_samples = [pad_data_to_max_length(sample[key], max_length) for sample in batch]
        # Stack the padded samples
        collated_history[key] = torch.stack(padded_samples, dim=0)
    # Combine history and non-history data
    result = {**collated_non_history, **collated_history}
    return result

def collate_history_batch_dali(batch, num_img_keys):
    """
    Custom collate function to handle history-related values with variable lengths.
    Uses the existing pad_data_to_max_length function from utils.py.
    Assumes history keys are never None.
    - each sample is a tuple
    - in each tuple, the last num_img_keys + 3 are the history keys to be collated using padding
    """
    total_non_history_keys = num_img_keys + 3 + 1 # + 1 for language embedding
    total_history_keys = num_img_keys + 3

    assert len(batch[0]) == total_non_history_keys or len(batch[0]) == total_non_history_keys + total_history_keys, "Each sample must have exactly total_non_history_keys or total_non_history_keys + total_history_keys keys"
    # non_history is a list
    # collated_non_history = torch.utils.data.dataloader.default_collate([sample[:total_non_history_keys] for sample in batch])
    max_length = max(max(len(sample[0]) for sample in batch), 1)
    collated_non_history = []
    for k in range(total_non_history_keys):
        if k == total_non_history_keys - 1:
            # this is the language embedding
            collated_non_history.append(torch.stack([sample[k] for sample in batch], dim=0))
        else:
            if isinstance(batch[0][k], np.ndarray):
                collated_non_history.append(np.stack([pad_data_to_max_length(sample[k], max_length) for sample in batch], axis=0))
            else:
                collated_non_history.append(torch.stack([pad_data_to_max_length(sample[k], max_length) for sample in batch], dim=0))

    max_length = max(max(len(sample[-total_history_keys]) for sample in batch), 1)
    collated_history = []
    for i in range(len(batch[0])):
        if i < total_non_history_keys:
            continue
        padded_samples = [pad_data_to_max_length(sample[i], max_length) for sample in batch]
        if isinstance(padded_samples[0], np.ndarray):
            collated_history.append(np.stack(padded_samples, axis=0))
        else:
            collated_history.append(torch.stack(padded_samples, dim=0))
    return collated_non_history + collated_history

def left_pad_data_to_max_length(data, max_length, dim):
    '''
    Left pad the data to the max length on the left side with zeros
    '''
    if isinstance(data, torch.Tensor):
        if data.shape[dim] < max_length:
            data = torch.cat([torch.zeros((*data.shape[:dim], max_length - data.shape[dim], *data.shape[dim+1:]), dtype=data.dtype), data], dim=dim)
    elif isinstance(data, np.ndarray):
        if data.shape[dim] < max_length:
            data = np.concatenate([np.zeros((*data.shape[:dim], max_length - data.shape[dim], *data.shape[dim+1:]), dtype=data.dtype), data], axis=dim)
    elif isinstance(data, dict):
        for k in data:
            data[k] = left_pad_data_to_max_length(data[k], max_length, dim)
    else:
        raise ValueError(f"Unknown type of data: {type(data)}")
    return data

def pad_data_to_max_length(data, max_length, dim=0, fill_value=0):
    '''
    Pad the data to the max length on the right side with zeros
    '''
    assert dim == 0, "Padding to max length is only supported for the first dimension. Easy to extend."
    if isinstance(data, dict):
        for k in data:
            data[k] = pad_data_to_max_length(data[k], max_length, dim=0)
    elif isinstance(data, torch.Tensor):
        if len(data) < max_length:
            data = torch.cat([data, torch.full((max_length - len(data), *data.shape[1:]), fill_value=fill_value, dtype=data.dtype)], dim=0)
    elif isinstance(data, np.ndarray):
        if len(data) < max_length:
            data = np.concatenate([data, np.full((max_length - len(data), *data.shape[1:]), fill_value=fill_value, dtype=data.dtype)], axis=0)
    else:
        raise ValueError(f"Unknown type of data: {type(data)}")
    return data

def create_shared(name: str, np_arr: np.ndarray):
    # try:
    #     existing_shm = shared_memory.SharedMemory(name=name)
    #     existing_shm.close()
    #     existing_shm.unlink()
    # except FileNotFoundError:
    #     pass  # No preexisting shm block — good
    # except Exception as e:
    #     print(f"Warning: Could not unlink existing shared memory '{name}': {e}")
    try:
        shm = shared_memory.SharedMemory(create=True, size=np_arr.nbytes, name=name)
    except Exception as e:
        # create a loop to find a new name
        for i in range(16):
            name = f"{name}_{np.random.randint(0, 1000000)}"
            print(f"ERROR! Could not create shared memory block. Creating new name: {name}")
            try:
                shm = shared_memory.SharedMemory(create=True, size=np_arr.nbytes, name=name)
            except Exception as e:
                print(f"ERROR! Could not create shared memory block. Creating new name: {name}")
                continue
            if i == 15:
                raise Exception(f"Could not create shared memory block. Creating new name: {name}")
            break
    # HACK: Unregister from resource_tracker to avoid deletion on exit
    try:
        rt.unregister(shm._name, "shared_memory")
    except Exception as e:
        print(f"Warning: failed to unregister shared memory '{shm._name}' from resource tracker: {e}")
    buf = np.ndarray(np_arr.shape, dtype=np_arr.dtype, buffer=shm.buf)
    buf[:] = np_arr
    tensor = torch.as_tensor(buf).share_memory_()
    return name, tensor, shm


def attach_shared(name: str, shape, dtype) -> torch.Tensor:
    """Return a Tensor view of an existing SharedMemory block."""
    shm = shared_memory.SharedMemory(name=name)
    try:
        rt.unregister(shm._name, "shared_memory")
    except Exception as e:
        print(f"[WARN] Could not unregister {shm._name}: {e}")

    buf = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    return torch.as_tensor(buf), shm


def count_indices(final_indices, seq_len):
    '''
        indices is a dictionary with keys as indices and values as list of (start, end, score)
    '''
    indices = []
    pbar = tqdm(total=len(final_indices), desc="Counting indices")
    for key, matches in final_indices.items():
        indices.extend(list(range(key, key+seq_len)))
        for match in matches:
            start, end, sim_val = match[0], match[1], match[2]
            indices.extend(list(range(start, end)))
        pbar.update(1)
    pbar.close()
    print("finding unique indices")
    indices = np.array(indices)
    unique_indices, counts = np.unique(indices, return_counts=True)
    # create a dictionary with keys as unique indices and values as counts
    index_counts = {}
    print("updating index counts")
    for i in trange(len(unique_indices)):
        index_counts[unique_indices[i]] = counts[i]
    return index_counts

def find_index_counts(final_indices, seq_len, index_count_path=None):
    '''
    final_indices is a dictionary with keys as indices and values as list of (start, end, score)
    '''

    index_counts = count_indices(final_indices, seq_len)
    if index_count_path is not None:
        pickle.dump(index_counts, open(index_count_path, 'wb'))
    return index_counts

def random_patch_mask(
    images: torch.Tensor,
    min_patches: int = 0,
    max_patches: int = 16,
    min_frac: float = 0.01,
    max_frac: float = 0.04,
    patch_counts: torch.Tensor = None,
) -> torch.Tensor:
    """
    Randomly mask patches of each image in a batch.

    Args:
        images (torch.Tensor): Input images of shape (B, H, W, C).
        min_patches (int): Minimum number of patches to mask in each image.
        max_patches (int): Maximum number of patches to mask in each image.
        min_frac (float): Minimum fraction of the image area per patch.
        max_frac (float): Maximum fraction of the image area per patch.
        patch_counts (torch.Tensor, optional): Pre-defined patch counts per image.
    Returns:
        torch.Tensor: The input images with random patches masked.
    """
    B, H, W, C = images.shape
    # we assume that the images have not been normalized using imagenet mean and std yet.
    assert C == 3, "Random masking only works with RGB images with channels last. Current shape: {}".format(images.shape)
    device = images.device
    norm_val = torch.tensor([0.4850+0.2290, 0.4560+0.2240, 0.4060+0.2250], device=device).view(1, 1, 1, 3)
    # norm_val = torch.tensor([1.0, 1.0, 1.0], device=device).view(1, 1, 1, 3)
    norm_val = norm_val * 255.0 if  images.max() > 1.0 else 1.0

    # 1. Randomly determine how many patches each image gets
    if patch_counts is None:
        patch_counts = torch.randint(min_patches, max_patches + 1, (B,), device=device)

    # 2. Prepare a mask tensor filled with ones -> shape (B, 1, H, W)
    # mask = torch.ones((B, 1, H, W), device=device, dtype=images.dtype)

    # 3. Generate all random parameters for patches at once
    max_count = patch_counts.max().item()
    if max_count == 0:
        return images

    # Generate fractions for all potential patches
    frac_values = min_frac + torch.rand((B, max_count), device=device) * (max_frac - min_frac)

    # Calculate patch sizes for all patches
    patch_area = frac_values * (H * W)
    patch_sizes = torch.floor(torch.sqrt(patch_area)).long().clamp(min=1)

    # Generate random positions for all patches
    # We'll create a mask of valid indices first
    valid_indices = torch.arange(max_count, device=device).expand(B, max_count) < patch_counts.unsqueeze(1)

    # Total number of patches to create
    total_patches = valid_indices.sum().item()
    if total_patches == 0:
        return images

    # Create batch indices for scatter operations
    batch_indices, patch_indices = torch.nonzero(valid_indices, as_tuple=True)[:2]

    # Get patch sizes for valid patches
    patch_sizes_flat = patch_sizes[batch_indices, patch_indices]

    # Generate random top-left corners for each patch
    x_max = H - patch_sizes_flat + 1
    y_max = W - patch_sizes_flat + 1

    # Handle edge case where patch size equals or exceeds image dimensions
    x_max = torch.clamp(x_max, min=1)
    y_max = torch.clamp(y_max, min=1)

    x_coords = torch.floor(torch.rand(total_patches, device=device) * x_max).long()
    y_coords = torch.floor(torch.rand(total_patches, device=device) * y_max).long()

    # Apply masks using advanced indexing
    for i in range(total_patches):
        b = batch_indices[i]
        ps = patch_sizes_flat[i]
        x = x_coords[i]
        y = y_coords[i]
        # in-place operation; make it IMAGENET MEAN + STD;
        images[b, x:x+ps, y:y+ps, :] = norm_val

    # # Apply the final mask
    return images


def add_noise_and_clip_bboxes(boxes, noise, base_mask, img_size):
    """
    Add per‐box noise, zeroing it out on padded slots, then clip
    and repair any invalid (xmin>xmax or ymin>ymax) after noise.

    Args:
        boxes      np.ndarray (B, L, 4), the output of pad_bboxes_to_max_length
        noise      np.ndarray (B, L, 4), same shape
        base_mask  torch.IntTensor/np.array (B, L), 1=real box, 0=padded
        img_size   Tuple[int,int] or int, max (width,height) or single dim

    Returns:
        noisy_boxes  np.ndarray (B, L, 4)
    """
    # ensure img_size is a pair
    # H = W = img_size if isinstance(img_size, int) else img_size[0]
    H, W = img_size[0], img_size[1] if isinstance(img_size, tuple) else (img_size, img_size)

    # zero out noise on padded slots
    nm = base_mask[..., None]           # (B,L,1)
    if isinstance(noise, torch.Tensor):
        nm = base_mask.numpy()
    noise = noise * nm

    # add noise
    noisy = boxes + noise

    # clip coords to [0, W-1] / [0, H-1]
    noisy[..., [0,2]] = np.clip(noisy[..., [0,2]], 0, W-1)
    noisy[..., [1,3]] = np.clip(noisy[..., [1,3]], 0, H-1)

    # fix any xmin>=xmax or ymin>=ymax by nudging
    # do x-axis
    invalid = noisy[...,2] <= noisy[...,0]
    while invalid.any():
        noisy[invalid, 0] = np.maximum(noisy[invalid, 0]-1, 0)
        noisy[invalid, 2] = np.minimum(noisy[invalid, 2]+1, W-1)
        invalid = noisy[...,2] <= noisy[...,0]

    # do y-axis
    invalid = noisy[...,3] <= noisy[...,1]
    while invalid.any():
        noisy[invalid, 1] = np.maximum(noisy[invalid, 1]-1, 0)
        noisy[invalid, 3] = np.minimum(noisy[invalid, 3]+1, H-1)
        invalid = noisy[...,3] <= noisy[...,1]

    return noisy

def generate_xyz_range(ranges, step=0.001):
    return [(start, start + step) for start in ranges]
# XYZ_ACTION_TRAJ_NOISE_RANGE = [(-0.09, -0.07), (-0.05, -0.03), (-0.01, 0.01), (0.03, 0.05), (0.07, 0.09)]
XYZ_ACTION_TRAJ_NOISE_RANGE = generate_xyz_range([-0.09, -0.07, -0.05, -0.03, -0.01, 0.01, 0.03, 0.05, 0.07, 0.09])
RPY_ACTION_TRAJ_NOISE_RANGE = [(-0.1, -0.1)]

def store_quantiles(q1, q99, save_path):
    if isinstance(q1, torch.Tensor):
        q1 = q1.cpu().numpy()
    if isinstance(q99, torch.Tensor):
        q99 = q99.cpu().numpy()
    # path ends with .npz
    np.savez(save_path, q1=q1, q99=q99)
    return

def load_quantiles(load_path):
    data = np.load(load_path)
    return data["q1"], data["q99"]

def get_quantiles(data):
    q1 = torch.quantile(data.reshape(-1, data.shape[-1]), 0.01, dim=0, keepdim=False)
    q99 = torch.quantile(data.reshape(-1, data.shape[-1]), 0.99, dim=0, keepdim=False)
    return q1, q99

def normalize_actions_q(actions, q1, q99):
    """
    Normalize the input actions such that the 1st and 99th quantiles of values
    in the training dataset for each action dimension map to the range [-1, 1].
    """
    # convert q1, q99 to either torch Tensor or numpy array depending on the input actions
    if isinstance(actions, torch.Tensor):
        q1 = torch.tensor(q1, device=actions.device, dtype=actions.dtype)
        q99 = torch.tensor(q99, device=actions.device, dtype=actions.dtype)
    else:
        q1 = np.array(q1)
        q99 = np.array(q99)
    if q1.ndim == 1:
        for _ in range(actions.ndim - 1):
            if isinstance(q1, torch.Tensor):
                q1 = q1.unsqueeze(0)
                q99 = q99.unsqueeze(0)
            else:
                q1 = np.expand_dims(q1, axis=0)
                q99 = np.expand_dims(q99, axis=0)
    assert actions.ndim == q1.ndim == q99.ndim, f"actions: {actions.ndim}, q1: {q1.ndim}, q99: {q99.ndim}"
    # Avoid division by zero by ensuring no identical quantiles
    scale = q99 - q1
    scale[scale == 0] = 1e-6

    # Normalize actions to the range [-1, 1]
    normalized_actions = 2 * (actions - q1) / scale - 1
    if isinstance(normalized_actions, torch.Tensor):
        normalized_actions = torch.clamp(normalized_actions, -1, 1)
    elif isinstance(normalized_actions, np.ndarray):
        normalized_actions = np.clip(normalized_actions, -1, 1)

    return normalized_actions

def unnormalize_actions_q(actions, q1, q99):
    # convert q1, q99 to either torch Tensor or numpy array depending on the input actions
    if isinstance(actions, torch.Tensor):
        q1 = torch.tensor(q1, device=actions.device, dtype=actions.dtype)
        q99 = torch.tensor(q99, device=actions.device, dtype=actions.dtype)
    else:
        q1 = np.array(q1)
        q99 = np.array(q99)
    if q1.ndim == 1:
        for _ in range(actions.ndim - 1):
            if isinstance(q1, torch.Tensor):
                q1 = q1.unsqueeze(0)
                q99 = q99.unsqueeze(0)
            else:
                q1 = np.expand_dims(q1, axis=0)
                q99 = np.expand_dims(q99, axis=0)
    assert actions.ndim == q1.ndim == q99.ndim, f"actions: {actions.ndim}, q1: {q1.ndim}, q99: {q99.ndim}"
    scale = q99 - q1
    scale[scale == 0] = 1e-6
    unnormalized_actions = (actions + 1) / 2 * scale + q1
    return unnormalized_actions

def load_all_npz_files(dataset_path, keys, indices=None):
    if indices is None:
        # listdir all the files in the dataset_path ending with .npz
        state_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if (f.endswith(".npz") and ('quantile' not in f))]
        state_paths.sort()
        indices = [int(os.path.splitext(os.path.basename(f))[0].split("_")[-1]) for f in state_paths]
    else:
        state_paths = [os.path.join(dataset_path, f"episode_{ind:07d}.npz") for ind in indices]
    # create a dictionary to store the data with fixed memory
    traj_idx2list_idx = {ind: idx for idx, ind in enumerate(indices)}
    # preallocate the memory for the data
    with np.load(state_paths[0]) as f:
        first_data = {k: f[k].copy() for k in f.files if k in keys}
    print(f"First data: {first_data.keys()}")
    # data = [{k: None for k in first_data.keys()} for _ in state_paths]
    data = [{k: first_data[k].copy() for k in first_data.keys()} for _ in state_paths]
    pbar = tqdm(total=len(state_paths))
    pbar.set_description("Loading data")
    for idx, state_path in enumerate(state_paths):
        # perform the lazy loading
        with np.load(state_path) as f:
            for k in keys:
                data[idx][k] = f[k]
        pbar.update(1)
    pbar.close()
    return data, traj_idx2list_idx

def load_entire_hdf5(dct):
    if isinstance(dct, h5py.Dataset):
        return dct[()]
    ret = {}
    for k, v in dct.items():
        ret[k] = load_entire_hdf5(v)
    return ret

def rot_mat_to_rot_6d(rot_mat : np.ndarray) -> np.ndarray:
    """
    Convert a rotation matrix to 6d representation
    rot_mat: N, 3, 3

    return: N, 6
    """
    rot_6d = rot_mat[:, :2, :] # N, 2, 3
    return rot_6d.reshape(-1, 6) # N, 6

def rot_6d_to_rot_mat(rot_6d : np.ndarray) -> np.ndarray:
    """
    Convert a 6d representation to rotation matrix
    rot_6d: N, 6

    return: N, 3, 3
    """
    rot_6d = rot_6d.reshape(-1, 2, 3)
    # assert the first two vectors are orthogonal
    if not np.allclose(np.sum(rot_6d[:, 0] * rot_6d[:, 1], axis=-1), 0):
        rot_6d = gram_schmidt(rot_6d)

    rot_mat = np.zeros((rot_6d.shape[0], 3, 3))
    rot_mat[:, :2, :] = rot_6d
    rot_mat[:, 2, :] = np.cross(rot_6d[:, 0], rot_6d[:, 1])
    return rot_mat

def euler_to_rot_6d(euler : np.ndarray, format="XYZ") -> np.ndarray:
    """
    Convert euler angles to 6d representation
    euler: N, 3
    """
    rot_mat = Rotation.from_euler(format, euler, degrees=False).as_matrix()
    return rot_mat_to_rot_6d(rot_mat)

def rot_6d_to_euler(rot_6d : np.ndarray, format="XYZ"):
    """
    Convert 6d representation to euler angles
    rot_6d: N, 6
    """
    rot_mat = rot_6d_to_rot_mat(rot_6d)
    return Rotation.from_matrix(rot_mat).as_euler(format, degrees=False)

def quat_to_rot_6d(quat : np.ndarray, format : str = "wxyz") -> np.ndarray:
    """
    Convert quaternion to 6d representation
    quat: N, 4
    robomimic:
    https://mujoco.readthedocs.io/en/2.2.1/programming.html#:~:text=To%20represent%203D%20orientations%20and,cos(a%2F2).
    To represent 3D orientations and rotations, MuJoCo uses unit quaternions - namely 4D unit vectors arranged as q = (w, x, y, z).
    Here (x, y, z) is the rotation axis unit vector scaled by sin(a/2), where a is the rotation angle in radians, and w = cos(a/2).
    Thus the quaternion corresponding to a null rotation is (1, 0, 0, 0). This is the default setting of all quaternions in MJCF.
    """
    assert format in ["wxyz", "xyzw"], "Invalid quaternion format, only support wxyz or xyzw"
    if format == "wxyz":
        quat = quat[:, [1, 2, 3, 0]]
    rot_mat = Rotation.from_quat(quat).as_matrix()
    return rot_mat_to_rot_6d(rot_mat)

def rot_6d_to_quat(rot_6d : np.ndarray, format : str = "wxyz") -> np.ndarray:
    """
    Convert 6d representation to quaternion
    rot_6d: N, 6
    """
    rot_mat = rot_6d_to_rot_mat(rot_6d)
    quat = Rotation.from_matrix(rot_mat).as_quat()
    if format == "wxyz":
        quat = quat[:, [3, 0, 1, 2]]
    return quat

def euler_to_quat(euler : np.ndarray, format_euler="XYZ", format_quat="wxyz") -> np.ndarray:
    """
    Convert euler angles to quaternion
    euler: N, 3
    """
    assert format_quat in ["wxyz", "xyzw"], "Invalid quaternion format, only support wxyz or xyzw"
    quat = Rotation.from_euler(format_euler, euler, degrees=False).as_quat()
    if format_quat == "wxyz":
        quat = quat[:, [3, 0, 1, 2]]
    return quat

def gram_schmidt(vectors : np.ndarray) -> np.ndarray:
    """
    Apply Gram-Schmidt process to a set of vectors
    vectors are indexed by rows

    vectors: batchsize, N, D

    return: batchsize, N, D
    """
    if len(vectors.shape) == 2:
        vectors = vectors[None]

    basis = np.zeros_like(vectors)
    basis[:, 0] = vectors[:, 0] / np.linalg.norm(vectors[:, 0], axis=-1, keepdims=True)
    for i in range(1, vectors.shape[1]):
        v = vectors[:, i]
        for j in range(i):
            v -= np.sum(v * basis[:, j], axis=-1, keepdims=True) * basis[:, j]
        basis[:, i] = v / np.linalg.norm(v, axis=-1, keepdims=True)
    return basis

def combine_dicts(dlist,key):
    """
    Combine a list of dictionaries into a single dictionary.
    """
    d = {}
    for k in key:
        dk = [d[k] for d in dlist]
        d[k] = np.concatenate(dk, axis=0)
    return d

def calculate_delta_rot(euler_rot_start : np.ndarray, euler_rot_end : np.ndarray, format="XYZ") -> np.ndarray:
    """
    Calculate the delta rotation between two euler angles
    euler_rot_start: N, 3
    euler_rot_end: N, 3

    return: N, 3
    """
    r = Rotation.from_euler(format, euler_rot_start, degrees=False)
    r2 = Rotation.from_euler(format, euler_rot_end, degrees=False)
    delta_rot = r2 * r.inv()
    euler_rot = delta_rot.as_euler(format, degrees=False)
    return euler_rot

def load_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def convert_multi_step(data : torch.Tensor, num_pred_steps: int):
    """Chunk data for predicting data `num_pred_steps` steps into the future.
    The resulting data have shape (batch, data.shape[-2] - (num_pred_steps - 1), num_pred_steps, action_dim)
    For example: chunk_data([a_1, a_2, a_3, a_4, a_5], 3) ->
        [
            [a_1, a_2, a_3],
            [a_2, a_3, a_4],
            [a_3, a_4, a_5],
            [a_4, a_5, a_5],
            [a_5, a_5, a_5],
        ]
    adapted from https://github.com/octo-models/octo/blob/7480a2a90160122b7a02459fc6f56ceefa501ebf/octo/model/components/action_heads.py#L59
    """
    assert (
        data.ndim == 2
    ), f"Expected data to have shape (seq length, action_dim), but got shape {data.shape}"
    window_size = data.shape[0]
    chunk_window_size = window_size

    curr_step = torch.arange(chunk_window_size, device=data.device)
    action_offset = torch.arange(num_pred_steps, device=data.device)
    chunk_indices = torch.minimum(curr_step[:, None] + action_offset[None, :], torch.tensor(chunk_window_size - 1))
    return data[chunk_indices]

def convert_delta_action(action, proprio):
    """
    Calculate the delta action given the action and proprioception
    Gripper action remains as absolute action
    action: S, T, action_dim
    proprio: S, T, proprio_dim
    """
    trans = action[:, :, :3].reshape(-1, 3)
    rot = action[:, :, 3:9].reshape(-1, 6)

    rot =  Rotation.from_matrix(rot_6d_to_rot_mat(rot))

    current_state = np.repeat(proprio[:, 0:1],action.shape[1],1)
    current_trans = current_state[:, :, :3].reshape(-1, 3)
    current_rot = current_state[:,:, 3:9]# S, T, 6
    current_rot =  Rotation.from_matrix(rot_6d_to_rot_mat(current_rot.reshape(-1, 6)))

    delta_rot = (current_rot.inv()*rot).as_matrix()
    delta_trans = np.einsum('ijk,ik->ij', current_rot.inv().as_matrix(),(trans-current_trans))

    delta_rot = rot_mat_to_rot_6d(delta_rot).reshape(-1,action.shape[1],6)
    delta_trans = delta_trans.reshape(-1,action.shape[1],3)

    if action.shape[-1] == proprio.shape[-1]:
        #no eos
        delta_action = np.concatenate([delta_trans, delta_rot, action[:,:,-1:]], axis=-1)
    else:
        #with eos
        delta_action = np.concatenate([delta_trans, delta_rot, action[:,:,-2:]], axis=-1)

    return delta_action

def convert_abs_action(action,proprio):
    '''
    Calculate the next state from the delta action and the current proprioception
    action: S, T, action_dim
    proprio: S, T, proprio_dim
    '''
    delta_trans = action[:, :, :3].reshape(-1, 3)
    delta_rot = action[:, :, 3:9].reshape(-1,6)
    delta_rot =  Rotation.from_matrix(rot_6d_to_rot_mat(delta_rot))

    current_state = np.repeat(proprio[:, 0:1],action.shape[1],1)
    current_trans = current_state[:, :, :3].reshape(-1, 3)
    current_rot = Rotation.from_matrix(rot_6d_to_rot_mat(current_state[:,:, 3:9].reshape(-1,6)))

    trans = np.einsum('ijk,ik->ij',current_rot.as_matrix(),delta_trans) + current_trans
    rot = (current_rot*delta_rot).as_matrix()

    rot = rot_mat_to_rot_6d(rot).reshape(-1,action.shape[1],6)
    trans = trans.reshape(-1,action.shape[1],3)

    if action.shape[-1] == proprio.shape[-1]:
        #no eos
        desired_mat = np.concatenate([trans, rot, action[:,:,-1:]], axis=-1)
    else:
        #with eos
        desired_mat = np.concatenate([trans, rot, action[:,:,-2:]], axis=-1)
    return desired_mat

def find_increasing_subsequences(arr : List[int]) -> List[Tuple[int, int]]:
    """
    4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3
    Find the all increasing subsequence in the order present in the dataset and return the values
    which should be [(4, 9), (1,9), (1,3)],

    args:
        arr: List[int] - list of integers
    """
    subsequences = []
    start = arr[0]
    for i in range(1, len(arr)):
        if arr[i] - arr[i-1] <= 0:
            subsequences.append((start, arr[i-1]))
            start = arr[i]
    subsequences.append((start, arr[-1]))
    return subsequences

def _get_weight_mask(eos_vector, eos_position, num_steps, mask_position):
    weight_mask = np.zeros_like(eos_vector)
    eos_position = eos_position[eos_position.index(mask_position):]
    eos_position.append(len(eos_vector))
    # the first one is the mask position
    if num_steps >= 1:
        for idx, eos in enumerate(eos_position[:-1]):
            end_pos = min(eos+num_steps+1, eos_position[idx+1])
            weight_mask[eos+1:end_pos] = 1
    else:
        # num_step is a ratio
        for idx, eos in enumerate(eos_position[:-1]):
            seq_len = eos_position[idx+1] - eos
            weight_steps = int(num_steps * seq_len)
            end_pos = eos+weight_steps+1
            weight_mask[eos+1:end_pos] = 1
    return weight_mask

def create_prompt_mask(eos_vector, num_steps, skip_first=True, skip_two=False, deterministic=False):
    assert not (skip_first and skip_two), "Cannot skip both first and second"
    eos_position = sorted(np.where(eos_vector == 1)[0])
    if len(eos_position) == 0:
        return np.zeros_like(eos_vector), np.ones_like(eos_vector)
    ## randomly select a position to mask
    if len(eos_position) > 1 and skip_first:
        eos_position = eos_position[1:]
    if len(eos_position) > 2 and skip_two:
        eos_position = eos_position[2:]

    mask_position = np.random.choice(eos_position) if not deterministic else eos_position[0]
    prompt_mask = np.zeros_like(eos_vector)
    prompt_mask[mask_position+1:] = 1 #if 1 then not prompt
    weight_mask = _get_weight_mask(eos_vector, eos_position, num_steps, mask_position)
    return prompt_mask, weight_mask

def scale_action(
        action : torch.Tensor,
        stat : dict,
        type : Literal["minmax", "standard"] = "standard"
) -> torch.Tensor:
    """
    action: S, T, action_dim
    stat: dictionary
    """
    # move stats to action device
    for k, v in stat.items():
        stat[k] = v.to(action.device)
    action_dim = stat["min"].shape[0]
    if type == "minmax":
        action[..., :action_dim] = (action[..., :action_dim] - stat["min"]) / (stat["max"] - stat["min"])
    elif type == "standard":
        action[..., :action_dim] = (action[..., :action_dim] - stat["mean"]) / stat["std"]
    return action

def unscale_action(
        action : torch.Tensor,
        stat : dict,
        type : Literal["minmax", "standard"] = "standard"
) -> torch.Tensor:
    """
    action: S, T, action_dim
    stat: dictionary
    """
    # move stats to action device
    for k, v in stat.items():
        stat[k] = v.to(action.device)
    action_dim = stat["min"].shape[0]
    if type == "minmax":
        action[..., :action_dim] = action[..., :action_dim] * (stat["max"] - stat["min"]) + stat["min"]
    elif type == "standard":
        action[..., :action_dim] = action[..., :action_dim] * stat["std"] + stat["mean"]
    return action

