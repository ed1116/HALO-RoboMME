import torch
from termcolor import colored
from .dataset_vl import *
from .dataset_qa import QADataset, QADatasetStateSupervision, QADatasetGPTState
from .concat_dataset import CustomConcatDataset
import copy

def get_dataset(args, dataset_kwargs, val_only=False, train_only=False):
    dataset_train, dataset_val = None, None
    # if ('calvin' in args.dataset_cfg.dataset_json) and ('group' not in args.dataset_cfg.dataset_json):
    if 'lerobot' in args.dataset_cfg.dataset_json.split('/')[-1]:
        from .dataset_lerobot import LeRobotDatasetWrapper
        print(colored("Using LeRobotDataset", "red"))
        dataset_train = []
        name = "LeRobotDatasetWrapper"
    elif 'qa_' in args.dataset_cfg.dataset_json.split('/')[-1]:
        name = "QADataset"
        if 'gptstate' in args.dataset_cfg.dataset_json.split('/')[-1]:
            name = "QADatasetGPTState"
        if 'statesuper' in args.dataset_cfg.dataset_json.split('/')[-1]:
            name = "QADatasetStateSupervision"
    elif 'task_' in args.dataset_cfg.dataset_json.split('/')[-1]:
        print(colored("Using TaskDataset", "red"))
        dataset_train = []
        name = "TaskGroupDataset"
        if args.shared_cfg.use_language_conditioning:
            name = "TaskDatasetLanguage"
        if args.shared_cfg.use_tokenizer_dataset:
            name = "TaskDatasetWithTokenizer"
    else:
        raise NotImplementedError(f"Dataset {args.dataset_cfg.dataset_json} not supported")
    print("Dataset name: ", name)
    if not val_only:
        dataset_train = eval(name)(
            split="train",
            **dataset_kwargs
        )
    if not train_only:
        dataset_val = eval(name)(
            split="val",
            **dataset_kwargs
        )
    return dataset_train, dataset_val

def load_datasets(args, vision_transform, no_aug_vision_transform, val_only=False, train_only=False):
    assert len(args.dataset_cfg.dataset_json) == len(args.dataset_cfg.dataset_val_json), "Number of train and val datasets should be the same: {} != {}".format(len(args.dataset_cfg.dataset_json), len(args.dataset_cfg.dataset_val_json))
    assert len(args.dataset_cfg.dataset_json) == len(args.dataset_cfg.num_repeat_traj), "Number of train datasets and num_repeat_traj should be the same: {} != {}".format(len(args.dataset_cfg.dataset_json), len(args.dataset_cfg.num_repeat_traj))
    assert len(args.dataset_cfg.dataset_json) == len(args.dataset_cfg.non_overlapping), "Number of train datasets and non_overlapping should be the same: {} != {}".format(len(args.dataset_cfg.dataset_json), len(args.dataset_cfg.non_overlapping))
    dataset_train, dataset_val = None, None
    if len(args.dataset_cfg.dataset_json) == 1:
        args.dataset_cfg.dataset_json = args.dataset_cfg.dataset_json[0]
        args.dataset_cfg.dataset_val_json = args.dataset_cfg.dataset_val_json[0]
        args.dataset_cfg.num_repeat_traj = args.dataset_cfg.num_repeat_traj[0]
        args.dataset_cfg.non_overlapping = args.dataset_cfg.non_overlapping[0]
        print("*"*20)
        print(args.dataset_cfg.dataset_json)
        dataset_kwargs = {
            "shared_config": args.shared_cfg,
            "dataset_config": args.dataset_cfg,
            "vision_transform": vision_transform,
            "no_aug_vision_transform": no_aug_vision_transform,
        }
        if args.dataset_cfg.dataset_val_json == "":
            train_only = True
        else:
            dataset_kwargs["split_ratio"] = 0.9

        dataset_train, dataset_val = get_dataset(args, dataset_kwargs, val_only, train_only)
        if not val_only:
            print("Length of dataset_train: ", len(dataset_train))
        if not train_only:
            print("Length of dataset_val: ", len(dataset_val))
        print("*"*20)
    else:
        dataset_train_list, dataset_val_list = [], []
        dataset_jsons = args.dataset_cfg.dataset_json
        dataset_val_json = args.dataset_cfg.dataset_val_json
        num_repeat_trajs = args.dataset_cfg.num_repeat_traj
        non_overlapping = args.dataset_cfg.non_overlapping
        args_to_pass = copy.deepcopy(args)
        for dataset_index, (dataset_json, dataset_val_json) in enumerate(zip(dataset_jsons, dataset_val_json)):
            args_to_pass.dataset_cfg.dataset_json = dataset_json
            args_to_pass.dataset_cfg.dataset_val_json = dataset_val_json
            args_to_pass.dataset_cfg.num_repeat_traj = num_repeat_trajs[dataset_index]
            args_to_pass.dataset_cfg.non_overlapping = non_overlapping[dataset_index]
            print("*"*20)
            print(dataset_json)
            dataset_kwargs = {
                "shared_config": args_to_pass.shared_cfg,
                "dataset_config": args_to_pass.dataset_cfg,
                "vision_transform": vision_transform,
                "no_aug_vision_transform": no_aug_vision_transform,
            }
            train_only = True if args_to_pass.dataset_cfg.dataset_val_json == "" else False
            _dataset_train, _dataset_val = get_dataset(args_to_pass, dataset_kwargs, val_only, train_only)
            print("*"*20)
            if _dataset_train is None:
                print("Length of dataset_train: ", len(_dataset_train))
            if _dataset_val is not None:
                print("Length of dataset_val: ", len(_dataset_val))
            print("*"*20)
            dataset_train_list.append(_dataset_train)
            dataset_val_list.append(_dataset_val)
        if not val_only:
            dataset_train = CustomConcatDataset(dataset_train_list, weight_by_dataset=args.dataset_cfg.weight_by_dataset)
        if not train_only:
            dataset_val = CustomConcatDataset(dataset_val_list, weight_by_dataset=args.dataset_cfg.weight_by_dataset)

        print("*"*20)
        if dataset_train is not None:
            print("Length of dataset_train: ", len(dataset_train))
        if dataset_val is not None:
            print("Length of dataset_val: ", len(dataset_val))
    return dataset_train, dataset_val
