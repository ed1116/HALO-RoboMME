import os
import copy
import json
import yaml
import torch
from pathlib import Path
from termcolor import colored
import tyro
import datetime

from easydict import EasyDict

from halo.util.args import ExperimentConfig
from halo.util.model_constructor import model_constructor
from halo.util import misc
from halo.util import casa_rollouts
from halo.models.augmentation.torchvision_aug import MultiViewTorchDictTransform
import halo.data.utils as data_utils

import dataclasses
from dataclasses import is_dataclass
from typing import Any, get_args, get_origin, Union
import yaml

class ObjectAsDictLoader(yaml.SafeLoader):
    pass

def _construct_python_object_as_dict(loader, tag_suffix, node):
    # For !!python/object:... the node is typically a mapping of fields.
    return loader.construct_mapping(node, deep=True)

ObjectAsDictLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/object:",
    _construct_python_object_as_dict,
)

def _is_dc_type(t) -> bool:
    try:
        return is_dataclass(t)
    except TypeError:
        return False

def _strip_optional(t):
    # Optional[T] is Union[T, NoneType]
    if get_origin(t) is Union:
        args = [a for a in get_args(t) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return t

def merge_into_dataclass(obj: Any, data: dict, *, strict: bool = False) -> Any:
    """
    Merge dict `data` into dataclass instance `obj`.
    - Keeps defaults for missing keys (so new fields survive).
    - If strict=False, ignores unknown keys in YAML.
    """
    if not is_dataclass(obj):
        raise TypeError(f"obj must be a dataclass instance, got {type(obj)}")

    field_map = {f.name: f for f in dataclasses.fields(obj)}

    for k, v in (data or {}).items():
        f = field_map.get(k)
        if f is None:
            if strict:
                raise KeyError(f"Unknown config key: {k}")
            continue

        cur = getattr(obj, k)
        ft = _strip_optional(f.type)
        # nested dataclass
        if is_dataclass(cur) and isinstance(v, dict):
            merge_into_dataclass(cur, v, strict=strict)
            continue
        if _is_dc_type(ft) and isinstance(v, dict):
            # field is dataclass type but current value might be None
            new_child = ft()  # requires defaults in that dataclass
            merge_into_dataclass(new_child, v, strict=strict)
            setattr(obj, k, new_child)
            continue

        # list of dataclasses (optional, handy)
        if get_origin(ft) in (list, tuple) and isinstance(v, list):
            elem_t = _strip_optional(get_args(ft)[0]) if get_args(ft) else None
            if elem_t and _is_dc_type(elem_t) and v and isinstance(v[0], dict):
                lst = []
                for item in v:
                    child = elem_t()
                    merge_into_dataclass(child, item, strict=strict)
                    lst.append(child)
                setattr(obj, k, lst if get_origin(ft) is list else tuple(lst))
                continue

        # plain assignment
        setattr(obj, k, v)

    return obj

def set_new_args(old_args: ExperimentConfig) -> ExperimentConfig:
    num_procs =  old_args.world_size
    eval_only_directory = old_args.eval_cfg.eval_only_directory
    eval_config_old = old_args.eval_cfg
    assert old_args.eval_cfg.eval_only_directory is not None, colored("eval_only_directory must be specified", "red")
    assert os.path.exists(old_args.eval_cfg.eval_only_directory), colored(f"Eval only directory does not exist: {old_args.eval_cfg.eval_only_directory}", "red")
    # Load config from directory if eval_only_directory is specified
    config_path = os.path.join(old_args.eval_cfg.eval_only_directory, "run.yaml")
    assert os.path.exists(config_path), colored(f"Config file does not exist: {config_path}", "red")
    print(colored(f"Loading config from: {config_path}", "green"))

    # args : ExperimentConfig = yaml.load(Path(config_path).read_text(), Loader=yaml.Loader)
    # 1) defaults (new schema)
    # 2) yaml -> dict
    # 3) overlay yaml onto defaults
    args = copy.deepcopy(old_args)
    text = Path(config_path).read_text()
    data = yaml.load(text, Loader=ObjectAsDictLoader) or {}
    dataset_json_contents = data['dataset_json_contents']
    data.pop('dataset_json_contents', None)
    merge_into_dataclass(args, data, strict=True)
    args.dataset_json_contents = dataset_json_contents

    # change the output_dir, log_dir
    dataset_jsons = args.dataset_cfg.dataset_json
    dataset_val_jsons = args.dataset_cfg.dataset_val_json
    # make the dataset_jsons relative to the args.logging_cfg.output_dir
    dataset_jsons = [os.path.relpath(dataset_json, args.logging_cfg.output_dir) for dataset_json in dataset_jsons]
    dataset_val_jsons = [os.path.relpath(dataset_val_json, args.logging_cfg.output_dir) if dataset_val_json != '' else '' for dataset_val_json in dataset_val_jsons]

    dataset_jsons = [os.path.join(eval_only_directory, dataset_json) for dataset_json in dataset_jsons]
    dataset_val_jsons = [os.path.join(eval_only_directory, dataset_val_json) if dataset_val_json != '' else '' for dataset_val_json in dataset_val_jsons]
    args.world_size = num_procs
    args.logging_cfg.output_dir = eval_only_directory
    args.logging_cfg.log_dir = eval_only_directory
    args.dataset_cfg.dataset_json = dataset_jsons
    args.dataset_cfg.dataset_val_json = dataset_val_jsons
    casaplay_data_root = os.environ['CASAPLAY_DATAROOT']
    if not ("resnet" in args.model_cfg.vision_encoder_cfg.vision_encoder):
        vision_encoder_path = args.model_cfg.vision_encoder_cfg.vision_encoder.split("/")[-2:]
        args.model_cfg.vision_encoder_cfg.vision_encoder = os.path.join(casaplay_data_root, *vision_encoder_path)
    args.eval_cfg = eval_config_old
    return args

def main():
    # Parse args with eval_only flags
    args = tyro.cli(ExperimentConfig)
    args = set_new_args(args)

    # Set checkpoint path based on eval_only_ckpt_num
    if args.eval_cfg.eval_only_ckpt_num is not None and args.eval_cfg.eval_only_ckpt_num != -1:
        ckpt_path = os.path.join(
            args.logging_cfg.output_dir,
            f"checkpoint-{args.eval_cfg.eval_only_ckpt_num}.pth"
        )
        ckpt_num = args.eval_cfg.eval_only_ckpt_num
    else:
        # Use last_epoch.pth if no specific checkpoint is specified
        ckpt_path = os.path.join(args.eval_cfg.eval_only_directory, "last_epoch.pth")
        ckpt_num = -1

    args.eval_cfg.ckpt_path = ckpt_path
    args.eval_cfg.latest_epoch = ckpt_num

    assert os.path.exists(args.eval_cfg.ckpt_path), f"Checkpoint does not exist: {args.eval_cfg.ckpt_path}"
    print(f"Using checkpoint: {args.eval_cfg.ckpt_path}")

    # Initialize distributed mode
    misc.init_distributed_mode(args, print_only_in_master=False, timeout=datetime.timedelta(hours=2))
    device = torch.device(args.device)

    # Load dataset config to get image keys (similar to train.py)
    image_keys = []
    for dataset_json in args.dataset_cfg.dataset_json:
        if dataset_json:
            data_cfg = json.load(open(dataset_json, 'r'))
            rgb_observations = data_cfg.get("image_keys", {})
            image_keys = list(rgb_observations)
            break

    # Construct model
    model = model_constructor(
        model_config=args.model_cfg,
        shared_config=args.shared_cfg,
        train=False,
        extra_kwargs={
            'image_keys': image_keys,
        },
    )
    model.use_cache(args.eval_cfg.eval_caching)
    model.to(device)
    model_without_ddp = model
    model.eval()
    if args.trainer_cfg.compile_model:
        # keep memory usage low
        model.compile()

    # Set up video transform
    kornia_video_transform = None
    if args.shared_cfg.use_lerobot or args.shared_cfg.use_kornia_augmentation:
        img_gripper_flags = []
        for key in image_keys:
            if ("gripper" in key) or ("hand" in key) or ("wrist" in key):
                img_gripper_flags.append(True)
            else:
                img_gripper_flags.append(False)
        name = "torch_aug.MultiViewTorchVideoTransform" if not args.shared_cfg.use_tokenizer_dataset else "torch_aug.MultiViewTorchDictTransform"
        # if resnet is in the model name, use 128 x 128 image size
        final_image_size = 224
        resize = 248
        if "resnet" in model.vision_encoder.name:
            final_image_size = 128
            resize = 148

        kornia_video_transform = MultiViewTorchDictTransform(
            img_gripper_flags,
            final_image_size=final_image_size,
            resize=resize,
            share_bc_across_frames_and_views=True,
            mean=data_utils.IMAGENET_MEAN,
            std=data_utils.IMAGENET_STD,
        ).to(device).eval()

    # Load checkpoint
    print(f"Loading checkpoint from: {args.eval_cfg.ckpt_path}")
    misc.load_model(model_without_ddp, args.eval_cfg.ckpt_path)

    # Run evaluation
    print("Starting evaluation...")
    with torch.no_grad():
        casa_rollouts.perform_env_evals(
            model=model_without_ddp,
            args=args,
            video_transform=kornia_video_transform,
            epoch_num=args.eval_cfg.latest_epoch if args.eval_cfg.latest_epoch is not None else 0,
            task_name=args.eval_cfg.task_name
        )
    print("Evaluation completed!")

if __name__ == '__main__':
    main()
