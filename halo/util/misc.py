# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit

## additional installations: pip install lerobot robosuite==1.4.0
# --------------------------------------------------------

import sys
import builtins
import datetime
import h5py
import json
import os
import time
from collections import defaultdict, deque
from pathlib import Path
import inspect, datetime
from PIL import Image, ImageFont, ImageDraw

import torch
import torch.utils.data
import torch.distributed as dist
from torch.distributed import FileStore
from torch import inf
import numpy as np
from halo.util.args import ExperimentConfig, SharedConfig
from typing import Optional

robocasa_benchmark_names = ["v0.1/single_stage", "memory"]

# Global flag to enable/disable timing
ENABLE_TIMING = True

def convert_image_keys_to_bbox_keys(image_keys):
    bbox_keys = []
    bbox_names_keys = []
    bbox_mask_keys = []
    bbox_cc_keys = []
    for key in image_keys:
        bbox_key = key.replace("image", "bbox").replace("rgb", "bbox")
        bbox_keys.append(bbox_key)
        bbox_names_keys.append(bbox_key + "_names")
        bbox_mask_keys.append(bbox_key + "_mask")
        bbox_cc_keys.append(bbox_key + "_cc")
    return bbox_keys, bbox_names_keys, bbox_mask_keys, bbox_cc_keys

def _generate_curriculum_topk_stages(max_topk: int, min_topk: int) -> list[int]:
    """
    Generate the list of topk values for the curriculum schedule.

    At each stage, the topk value is divided by 2, capped at min_topk.
    Both max_topk and min_topk are guaranteed to be included.

    Example: max_topk=16, min_topk=4 -> [16, 8, 4]
    Example: max_topk=20, min_topk=4 -> [20, 10, 4] (5 would be capped to 4)

    Args:
        max_topk: Maximum topk value (starting value)
        min_topk: Minimum topk value (final value)

    Returns:
        List of topk values for each stage
    """
    topk_values = []
    current = max_topk
    while current >= min_topk:
        # Cap current value to at least min_topk
        topk_values.append(max(current, min_topk))
        if current <= min_topk:
            break
        # Divide by 2 for next stage
        next_value = current // 2
        if next_value < min_topk:
            # If next value would be below min, jump to min_topk (final stage)
            if min_topk not in topk_values:
                topk_values.append(min_topk)
            break
        current = next_value

    # Remove duplicates while preserving order
    seen = set()
    topk_values_unique = []
    for val in topk_values:
        if val not in seen:
            seen.add(val)
            topk_values_unique.append(val)

    return topk_values_unique

def get_curriculum_topk(epoch: int, min_topk: int, max_topk: Optional[int], total_epochs: int) -> Optional[int]:
    """
    Compute the current topk value based on stagewise curriculum learning schedule.

    At each stage, the topk value is divided by 2, capped at min_topk.
    The number of stages is determined by the number of topk values in [max_topk, min_topk]
    (dividing by 2 at each step), and epochs are divided almost equally among stages.
    Both max_topk and min_topk are guaranteed to be included.

    Example: max_topk=16, min_topk=4 -> stages: [16, 8, 4] (3 stages)
    Example: max_topk=20, min_topk=4 -> stages: [20, 10, 4] (3 stages, 5 capped to 4)

    Args:
        epoch: Current epoch number
        min_topk: Minimum topk value (final value)
        max_topk: Maximum topk value (starting value, None = no curriculum, always use min_topk)
        total_epochs: Total number of training epochs

    Returns:
        Current topk value to use, or None if no curriculum (use default)
    """
    if max_topk is None:
        return None  # No curriculum, use default ret_topk

    if max_topk <= min_topk:
        return None  # Invalid config, use default

    topk_values = _generate_curriculum_topk_stages(max_topk, min_topk)

    num_stages = len(topk_values)
    if num_stages == 0:
        return None

    # Calculate epochs per stage (almost equal distribution)
    epochs_per_stage = total_epochs / num_stages

    # Determine which stage the current epoch belongs to
    # Stage 0: epochs [0, epochs_per_stage)
    # Stage 1: epochs [epochs_per_stage, 2*epochs_per_stage)
    # ...
    # Stage n-1: epochs [(n-1)*epochs_per_stage, total_epochs)
    stage = min(int(epoch / epochs_per_stage), num_stages - 1)

    return topk_values[stage]


def convert_str_to_torch_dtype(compute_dtype_str: str):
    if compute_dtype_str == "bfloat16":
        return torch.bfloat16
    elif compute_dtype_str == "float16":
        return torch.float16
    elif compute_dtype_str == "float32":
        return torch.float32
    else:
        raise ValueError(f"Invalid compute dtype: {compute_dtype_str}")


def create_language_image(language, resolution, font_size):
    # make it 40 characters per line
    lines = [language[i:i+40] for i in range(0, len(language), 40)]
    language = "\n".join(lines)
    black_img = np.zeros(resolution, dtype=np.uint8)
    black_img[:, :] = [0, 0, 0]
    # use default font and make the text more readable by upscaling
    font = ImageFont.load_default(size=24)
    base_h, base_w, _ = black_img.shape
    # draw on a smaller canvas so that the default font appears larger when upscaled
    scale = 2.0
    canvas_w = max(1, int(base_w * scale))
    canvas_h = max(1, int(base_h * scale))
    text_img = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
    text_draw = ImageDraw.Draw(text_img)
    text = language
    # get text size in a Pillow-version-compatible way
    try:
        # Pillow >= 8.0: use font.getbbox if available
        if hasattr(font, "getbbox"):
            bbox = font.getbbox(text)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        else:
            text_w, text_h = font.getsize(text)
    except Exception:
        # conservative fallback
        text_w, text_h = canvas_w, canvas_h // 4
    text_x = max(0, (canvas_w - text_w) // 2)
    text_y = max(0, (canvas_h - text_h) // 2)
    text_draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)
    # upscale back to the rollout image size
    img = text_img.resize((base_w, base_h), resample=Image.NEAREST)
    return np.array(img)

def timed_operation(description):
    """Context manager for timing operations."""
    class TimedOperation:
        def __init__(self, desc):
            self.description = desc
            self.start_time = None

        def __enter__(self):
            if ENABLE_TIMING:
                self.start_time = time.time()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if ENABLE_TIMING and self.start_time:
                elapsed_minutes = (time.time() - self.start_time) / 60
                print(f"Time taken to {self.description}: {elapsed_minutes:.2f} minutes")

    return TimedOperation(description)

def get_task_language_from_hdf5(hdf5_file: str, load_generated_instructions: bool = False) -> dict[str, list[str]]:
    all_task_languages = {}
    if load_generated_instructions:
        base_dir = os.path.dirname(hdf5_file)
        generated_instructions_file = os.path.join(base_dir, 'task_queries', 'all_queries_v1.json')
        assert os.path.exists(generated_instructions_file), f"Generated instructions file {generated_instructions_file} does not exist"
        with open(generated_instructions_file, 'r') as f:
            generated_instructions = json.load(f)
        for episode_key, episode_data in generated_instructions.items():
            all_task_languages[episode_key] = episode_data["generated_instructions"] + episode_data["original_instructions"]
    else:
        domain_name = get_dataset_domain_name(hdf5_file)
        if domain_name == 'robocasa':
            with h5py.File(hdf5_file, 'r') as f:
                for key in f['data'].keys():
                    all_task_languages[key] = [json.loads(f['data'][key].attrs['ep_meta'])['lang']]
        elif domain_name == 'mutex':
            language = hdf5_file.split('/')[-1].replace('.hdf5', '')
            # replace the MEX_
            language = language.split('_')[1:]
            language = ' '.join(language)
            with h5py.File(hdf5_file, 'r') as f:
                keys = list(f['data'].keys())
            for key in keys:
                all_task_languages[key] = [language]
    return all_task_languages


def get_dataset_domain_name(path):
    if 'mutex' in path:
        return 'mutex' # still inside the robocasa dataset
    if 'robocasa' in path:
        return 'robocasa'
    if any(name in path for name in robocasa_benchmark_names):
        return 'robocasa'
    return None

def get_benchmark_name(path, domain_name):
    if domain_name == 'robocasa':
        # find the one that mathes from robocasa_benchmark_names
        for bm in robocasa_benchmark_names:
            if bm in path:
                return bm
        raise ValueError(f"Unknown benchmark: {path}")
    raise ValueError(f"Unknown benchmark: {domain_name}")

def get_task_name_from_hdf5_path(hdf5_file, actual_task_name=False):
    domain_name = get_dataset_domain_name(hdf5_file)
    if domain_name == 'mutex':
        task_name = hdf5_file.split('/')[-1].replace('.hdf5', '')
    elif domain_name == 'robocasa':
        if actual_task_name:
            task_name = hdf5_file.split('/')[-3]
        else:
            task_name = hdf5_file.split('/')[-3] + '_' + hdf5_file.split('/')[-2]
    else:
        raise ValueError(f"Unknown domain: {domain_name}")
    return task_name

def get_task_embeddings_path(hdf5_file, model='clip'):
    assert model in ['clip']
    domain_name = get_dataset_domain_name(hdf5_file)
    if domain_name == 'mutex':
        pkl_file = os.path.join(os.path.dirname(hdf5_file), f"task_embeds_{model}_v3.pickle")
    elif domain_name == 'robocasa':
        bm = get_benchmark_name(hdf5_file, domain_name)
        pkl_file = os.path.join(os.environ['CASAPLAY_DATAROOT'], bm, f"task_embeds_{model}_v3.pickle")
    else:
        raise ValueError(f"Unknown domain: {domain_name}")
    return pkl_file

def get_data_base_dir(hdf5_path, shared_config: Optional[SharedConfig] = None):
    domain_name = get_dataset_domain_name(hdf5_path)
    if domain_name == 'robocasa':
        data_dir = os.environ["CASAPLAY_DATAROOT"]
        if shared_config is not None:
            assert shared_config.has_base_action, "Robocasa dataset has base action"
    elif domain_name == 'mutex':
        data_dir = os.environ["CASAPLAY_DATAROOT"]
    else:
        raise ValueError(f"Unknown domain name: {domain_name}")
    return data_dir


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None, warmup_calls=2):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt
        self.warmup_calls = warmup_calls
        self.call_count = 0

    def update(self, value, n=1):
        self.call_count += 1
        # Only add to sliding window after warmup
        if self.call_count > self.warmup_calls:
            # Always update global statistics
            self.count += n
            self.total += value * n
            self.deque.append(value)

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        if not self.deque:
            return 0.0
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        if not self.deque:
            return 0.0
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        if self.count == 0:
            return 0.0
        return self.total / self.count

    @property
    def max(self):
        if not self.deque:
            return 0.0
        return max(self.deque)

    @property
    def value(self):
        if not self.deque:
            return 0.0
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print
    return


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def sync_all_processes(timeout_hours=1):
    if is_dist_avail_and_initialized():
        if "timeout" in inspect.signature(dist.barrier).parameters:
            dist.barrier(timeout=datetime.timedelta(hours=timeout_hours))
        else:
            # Older PyTorch: per-call timeout unsupported.
            # Timeout must be set via dist.init_process_group(timeout=...) instead.
            dist.barrier()
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args, print_only_in_master=True, file_based_sync=False, timeout=None):
    if file_based_sync: # assumes the RANK AND WORLD_SIZE are set in the environment
        if not args.distributed:
            return
        assert 'RANK' in os.environ, f"RANK must be set in the environment."
        assert 'WORLD_SIZE' in os.environ, f"WORLD_SIZE must be set in the environment."
        assert 'LOCAL_RANK' in os.environ, f"LOCAL_RANK must be set in the environment."
        # Use FileStore for file-based synchronization
        sync_dir = os.path.join(args.logging_cfg.output_dir, 'eval_sync')
        os.makedirs(sync_dir, exist_ok=True)
        store_path = os.path.join(sync_dir, "torch_dist_store")

        # Get rank and world_size from environment
        rank = int(os.environ.get('RANK'))
        world_size = int(os.environ.get('WORLD_SIZE'))
        args.rank = rank
        args.world_size = world_size
        args.gpu = int(os.environ.get('LOCAL_RANK'))

        args.distributed = True
        num_gpus = torch.cuda.device_count()
        print("GPU::", args.gpu, "| Num GPUs::", num_gpus)
        torch.cuda.set_device(args.gpu % num_gpus)
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible_devices:
            os.environ["MUJOCO_EGL_DEVICE_ID"] = str(cuda_visible_devices.split(",")[args.gpu % num_gpus])
        else:
            os.environ["MUJOCO_EGL_DEVICE_ID"] = str(args.gpu % num_gpus)
        args.dist_backend = 'nccl'

        print(f'| distributed init (rank {rank}): file_store={store_path}, gpu {args.gpu}', flush=True)
        store = FileStore(store_path, world_size)
        ipg_kwargs = dict(backend=args.dist_backend, store=store, world_size=world_size, rank=rank)
        if timeout is not None:
            ipg_kwargs["timeout"] = timeout
        torch.distributed.init_process_group(**ipg_kwargs)
        torch.distributed.barrier()
        if print_only_in_master:
            setup_for_distributed(args.rank == 0)
        return

    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        print("Using RANK and WORLD_SIZE")
        print(f"RANK: {os.environ['RANK']}")
        print(f"WORLD_SIZE: {os.environ['WORLD_SIZE']}")
        print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK', 0)}")
        print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR', '127.0.0.1')}")
        print(f"MASTER_PORT: {os.environ.get('MASTER_PORT', '29500')}")
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ.get('LOCAL_RANK', 0))
        master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
        master_port = os.environ.get('MASTER_PORT', '29500')
        args.dist_url = f"tcp://{master_addr}:{master_port}"
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
    elif 'SLURM_PROCID' in os.environ:
        print("Using SLURM_PROCID")
        print(f"SLURM_PROCID: {os.environ['SLURM_PROCID']}")
        print(f"SLURM_NTASKS: {os.environ['SLURM_NTASKS']}")
        print(f"SLURM_LOCALID: {os.environ['SLURM_LOCALID']}")
        print(f"SLURM_NODEID: {os.environ['SLURM_NODEID']}")
        print(f"SLURM_JOB_NODELIST: {os.environ['SLURM_JOB_NODELIST']}")
        print(f"SLURM_NTASKS_PER_NODE: {os.environ['SLURM_NTASKS_PER_NODE']}")
        print(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}")
        print(f"MASTER_PORT: {os.environ['MASTER_PORT']}")
        args.rank = int(os.environ['SLURM_PROCID'])
        args.world_size = int(os.environ['SLURM_NTASKS'])
        # Use SLURM_LOCALID for per-node local rank if available
        local_id = int(os.environ.get('SLURM_LOCALID', 0))
        args.gpu = local_id if torch.cuda.is_available() else 0
        print(f"Rank: {args.rank} | World size: {args.world_size} | GPU: {args.gpu}")
        # Populate standard env vars for downstream libs
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
        master_port = os.environ.get('MASTER_PORT', '29500')
        args.dist_url = f"tcp://{master_addr}:{master_port}"
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        if print_only_in_master:
            setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    num_gpus = torch.cuda.device_count()
    print("GPU::", args.gpu, "| Num GPUs::", num_gpus)
    torch.cuda.set_device(args.gpu % num_gpus)
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible_devices:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(cuda_visible_devices.split(",")[args.gpu % num_gpus])
    else:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(args.gpu % num_gpus)
    args.dist_backend = 'nccl'
    print(f'| distributed init {args.rank=} {args.dist_url=} {args.gpu=}', flush=True)
    ipg_kwargs = dict(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    if timeout is not None:
        ipg_kwargs["timeout"] = timeout
    torch.distributed.init_process_group(**ipg_kwargs)
    print(f'| distributed init done {args.rank=} {args.dist_url=} {args.gpu=}', flush=True)
    torch.distributed.barrier()
    if print_only_in_master:
        setup_for_distributed(args.rank == 0)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self, set_scaler=True):
        # we do not need gradscaler for bfloat16 parameters
        # check: https://github.com/pytorch/pytorch/issues/127176
        if set_scaler:
            self._scaler = torch.amp.GradScaler('cuda')
        else:
            self._scaler = None

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        if self._scaler is None:
            loss.backward(create_graph=create_graph)
            if update_grad:
                if clip_grad is not None:
                    assert parameters is not None
                    norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
                else:
                    norm = get_grad_norm_(parameters)
                optimizer.step()
            else:
                norm = None
            return norm
        else:
            # Original AMP scaling logic for Float16
            self._scaler.scale(loss).backward(create_graph=create_graph)
            if update_grad:
                if clip_grad is not None:
                    assert parameters is not None
                    self._scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
                else:
                    self._scaler.unscale_(optimizer)
                    norm = get_grad_norm_(parameters)
                self._scaler.step(optimizer)
                self._scaler.update()
            else:
                norm = None
            return norm

    def state_dict(self):
        if self._scaler is not None:
            return self._scaler.state_dict()
        else:
            return None

    def load_state_dict(self, state_dict):
        if self._scaler is not None:
            self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, best_val_loss=float('inf'), file_name=None):
    output_dir = Path(args.logging_cfg.output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        if file_name is None:
            file_name = 'checkpoint-%s.pth' % epoch_name
        checkpoint_paths = [output_dir / file_name]
        for checkpoint_path in checkpoint_paths:
            ## check the keys in the model_without_ddp.state_dict()
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
                'best_val_loss': best_val_loss,
            }

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        model.save_checkpoint(save_dir=args.logging_cfg.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)


def load_model(model_without_ddp, path, base_ckpt_path=None, phase=None):
    if path.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            path, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    new_checkpoint = {}
    for key, value in checkpoint['model'].items():
        key = key.replace("llma", "llama")
        new_checkpoint[key] = value

    if base_ckpt_path is not None:
        base_checkpoint = torch.load(base_ckpt_path, map_location='cpu', weights_only=False)
        if (phase == 'llama_block_finetune') or ('llama_block_fullft' in phase): # it will not work for step-wise finetuning. need to find better fix.
            assert phase != None, "phase should be provided for llama_block_finetune"
            base_checkpoint['model'] = model_without_ddp.rename_state_dict_keys(base_checkpoint['model'], phase)
        for key, value in base_checkpoint['model'].items():
            key = key.replace("llma", "llama")
            if key not in new_checkpoint:
                new_checkpoint[key] = value

    missing_keys, unexpected_keys = model_without_ddp.load_state_dict(new_checkpoint, strict=False)
    assert len(unexpected_keys) == 0, f"Unexpected keys: {unexpected_keys}"
    actual_missing_keys = []
    for key in missing_keys:
        if key.startswith("icrt_action_decoder.tokenizer"):
            continue
        assert 'halo' not in key, f"Missing keys: {missing_keys}"
    for key in missing_keys:
        # if 'vision_encoder' in key or 'llama' in key:
        #     continue
        # if 'llama' in key:
        #     continue
        # else:
            actual_missing_keys.append(key)
    # check llama should not be missing from any of the keys
    llama_missing_keys = [key for key in actual_missing_keys if 'llama' in key]
    # if any of these keys include: mean_lr0, mean_lr1, mean_lr2, max_corr0, max_corr1, max_corr2, bm then we can ignore them as it is not a trained weights but rather just buffers
    ignore_keys = ['mean_lr0', 'mean_lr1', 'mean_lr2', 'max_corr0', 'max_corr1', 'max_corr2', 'bm', 'avg_writes_w_time']
    ignore_keys.extend(['llama.layers.0.attention.long_term_bank.cache_k', 'llama.layers.0.attention.long_term_bank.cache_v', 'llama.layers.0.attention.long_term_bank.mask', 'llama.layers.0.attention.long_term_bank.n_valid_entries', 'llama.layers.2.attention.long_term_bank.cache_k', 'llama.layers.2.attention.long_term_bank.cache_v', 'llama.layers.2.attention.long_term_bank.mask', 'llama.layers.2.attention.long_term_bank.n_valid_entries', 'div_term','long_term_bank.cache_time', 'long_term_bank.n_valid', 'long_term_bank.cache_time', 'long_term_bank.n_valid'])
    ignore_keys.extend(['attention.long_term_bank.causal_mask'])
    llama_missing_keys = [key for key in llama_missing_keys if not any(ignore_key in key for ignore_key in ignore_keys)]
    assert len(llama_missing_keys) == 0, f"Llama should not be missing from any of the keys: {llama_missing_keys}"
    # assert len(actual_missing_keys) == 0, f"Missing keys: {actual_missing_keys}"
    print("Load checkpoint %s" % path)
    print("Missing keys: ", actual_missing_keys)
    print("Unexpected keys: ", unexpected_keys)
    return

def resume_from_ckpt(args : ExperimentConfig, model_without_ddp, optimizer, loss_scaler):
    best_val_loss = float('inf')
    if args.shared_cfg.resume:
        if args.shared_cfg.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.shared_cfg.resume, map_location='cpu', check_hash=True)
        elif args.shared_cfg.resume == 'last':
            ckpt_location = os.path.join(args.logging_cfg.output_dir, "last_epoch.pth")
            if not os.path.exists(ckpt_location):
                print(f"\033[91m*** No checkpoint found at {ckpt_location}, starting from scratch. ***\033[0m")
                return best_val_loss
            checkpoint = torch.load(ckpt_location, map_location='cpu', weights_only=False)
        else:
            checkpoint = torch.load(args.shared_cfg.resume, map_location='cpu', weights_only=False)
        if args.model_cfg.policy_cfg.phase != "pretrain":
            new_checkpoint = model_without_ddp.rename_state_dict_keys(checkpoint['model'], args.model_cfg.policy_cfg.phase)
        else:
            new_checkpoint = checkpoint['model']
        # compare the keys in the checkpoint['model'] and the model_without_ddp.state_dict()
        checkpoint_keys = list(new_checkpoint.keys())
        ##################### note the state_dict mode here will return all the state_dict to save
        model_keys = list(model_without_ddp.state_dict(mode=args.model_cfg.policy_cfg.phase).keys())
        for key in checkpoint_keys:
            if key not in model_keys:
                print(f"Key {key} not found in model_without_ddp.state_dict()")
        ##################################################################
        if args.shared_cfg.resume_new_exp:
            print("Resuming with new experiment. Not loading optimizer, epoch, etc.")
        else:
            model_without_ddp.load_state_dict(new_checkpoint, strict=False) # some non-trainable parameters might be missing
            print("Resume checkpoint %s" % args.shared_cfg.resume)
            if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
                if ('finetune' in args.model_cfg.policy_cfg.phase) or ('_fullft' in args.model_cfg.policy_cfg.phase):
                    print("Finetuning from a base model. Not loading optimizer.")
                else:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                args.shared_cfg.start_epoch = checkpoint['epoch'] + 1
                if 'scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['scaler'])
                print("With optim & sched!")
            if 'best_val_loss' in checkpoint:
                best_val_loss = checkpoint['best_val_loss']
    return best_val_loss

def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x

def all_reduce_sum(x_reduce):
    world_size = get_world_size()
    if world_size > 1:
        dist.all_reduce(x_reduce, op=dist.ReduceOp.SUM)
        return x_reduce
    else:
        return x_reduce

def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

class DistributedWeightedSubEpochSampler(torch.utils.data.Sampler):
    """
        This is an extension of DistributedSubEpochSampler that can sample based on assigned weights.
    """
    def __init__(self, dataset, num_replicas, rank, shuffle, split_epoch=1, seed=0, weights=None):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.split_epoch = split_epoch
        self.seed = seed

        self.epoch = 0

        self.num_samples = len(dataset) // (num_replicas * split_epoch)

        if weights is not None:
            assert len(weights) == len(dataset), \
                    f"weights are of length {len(weights)} where dataset is of length {len(dataset)}"
            self.weights = torch.tensor(weights, dtype=torch.float32)
        else:
            self.weights = None

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch // self.split_epoch)
        if self.weights is not None:
            all_indices = None
            if self.shuffle:
                all_indices = torch.randperm(len(self.dataset), generator=g).tolist()
            else:
                all_indices = list(range(len(self.dataset)))
            worker_indices = all_indices[self.rank * self.split_epoch + self.epoch % self.split_epoch::self.num_replicas * self.split_epoch]
            worker_weights = self.weights[worker_indices]
            indices = torch.multinomial(worker_weights, self.num_samples, replacement=True, generator=g).tolist()
            indices = [worker_indices[i] for i in indices]
        elif self.shuffle:
            # deterministically shuffle based on epoch and seed
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
            indices = indices[self.rank * self.split_epoch + self.epoch % self.split_epoch::self.num_replicas * self.split_epoch]
            assert len(indices) >= self.num_samples
            indices = indices[:self.num_samples]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]
            indices = indices[self.rank * self.split_epoch + self.epoch % self.split_epoch::self.num_replicas * self.split_epoch]
            assert len(indices) >= self.num_samples
            indices = indices[:self.num_samples]

        return iter(indices)

    def set_epoch(self, epoch):
        self.epoch = epoch

class DistributedSubEpochSampler(torch.utils.data.Sampler):

    def __init__(self, dataset, num_replicas, rank, shuffle, split_epoch=1, seed=0):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.split_epoch = split_epoch
        self.seed = seed

        self.epoch = 0

        self.num_samples = len(dataset) // (num_replicas * split_epoch)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch // self.split_epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        indices = indices[self.rank * self.split_epoch + self.epoch % self.split_epoch::self.num_replicas * self.split_epoch]
        assert len(indices) >= self.num_samples
        indices = indices[:self.num_samples]

        return iter(indices)

    def set_epoch(self, epoch):
        self.epoch = epoch

