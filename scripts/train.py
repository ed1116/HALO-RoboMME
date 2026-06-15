import gc
import math
from pathlib import Path
import datetime
import json
import certifi
import numpy as np
import os
import time
import tyro
import wandb
import yaml
from functools import partial
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import subprocess
import sys
import pickle

import timm
from timm.data.loader import MultiEpochsDataLoader
from timm.data.transforms import RandomResizedCropAndInterpolation, ToTensor
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ColorJitter

from halo.data import load_datasets

import halo.util.misc as misc
from halo.util.misc import NativeScalerWithGradNormCount as NativeScaler
from halo.util.args import ExperimentConfig
from halo.util.engine import train_one_epoch
from halo.util.model_constructor import model_constructor
from halo.models.backbones.encoders import VisionPatchEncoder
import halo.data.utils as data_utils
from halo.data.concat_dataset import CustomConcatDataset
import halo.models.augmentation.kornia_aug as kornia_aug
import halo.models.augmentation.torchvision_aug as torch_aug
import halo.util.casa_rollouts as casa_rollouts

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def env_evals(args, optimizer, loss_scaler, kornia_video_transform, epoch, image_keys, device):
    # we want to reinstantiate the model without ddp for the evaluations with setting train to False
    model_without_ddp = model_constructor(
        model_config=args.model_cfg,
        shared_config=args.shared_cfg,
        train=False,
        extra_kwargs={
            'image_keys': image_keys,
        },
    )
    model_without_ddp.to(device)
    model_without_ddp.eval()
    misc.resume_from_ckpt(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    casa_rollouts.perform_env_evals(model_without_ddp, args, kornia_video_transform, epoch_num=epoch)
    return

def _run_env_evals_in_subprocess(args_path, epoch, rank):
    """Helper function to run env_evals in a subprocess. This will be called by the subprocess."""
    print(f"[Subprocess Rank {rank}] Starting eval subprocess...", flush=True)
    with open(args_path, 'rb') as f:
        args, extra_kwargs = pickle.load(f)
    image_keys, img_gripper_flags = extra_kwargs['image_keys'], extra_kwargs['image_gripper_flags']

    # Synchronize all subprocesses before initializing distributed mode
    # Each subprocess writes a ready marker, then waits for all others
    world_size = misc.get_world_size()

    # Use FileStore for file-based synchronization (no port needed)
    print(f"[Subprocess Rank {rank}] Initializing distributed mode with FileStore...", flush=True)
    misc.init_distributed_mode(args, print_only_in_master=False, file_based_sync=True)
    print(f"[Subprocess Rank {rank}] Distributed mode initialized", flush=True)

    device = torch.device(args.device)
    _, _, kornia_video_transform = get_video_transforms(args, img_gripper_flags, device, model=None)
    print(f"[Subprocess Rank {rank}] Starting env_evals...", flush=True)
    env_evals(args, optimizer=None, loss_scaler=None, kornia_video_transform=kornia_video_transform, epoch=epoch, image_keys=image_keys, device=device)
    print(f"[Subprocess Rank {rank}] Eval subprocess completed", flush=True)

    return

def launch_env_evals_in_subprocess(args, epoch, extra_kwargs):
    """Launch env_evals in a subprocess from each rank, wait for completion, and synchronize."""
    rank = misc.get_rank()
    world_size = misc.get_world_size()

    # # Destroy parent distributed group so subprocesses can reuse the same setup
    if world_size > 1 and torch.distributed.is_initialized():
        torch.distributed.barrier()
        print(f"[Rank {rank}] All ranks synchronized, destroying parent process group...", flush=True)
        torch.distributed.destroy_process_group()
        print(f"[Rank {rank}] Parent process group destroyed, launching subprocesses...", flush=True)

    # Save args and image_keys to a temporary file that will be shared across ranks
    # Use a file in the output directory so all ranks can access it
    args_path = os.path.join(args.logging_cfg.output_dir, f"eval_args_rank{rank}.pkl")
    with open(args_path, 'wb') as f:
        pickle.dump((args, extra_kwargs), f, protocol=pickle.HIGHEST_PROTOCOL)

    # Create the subprocess command
    script_path = os.path.abspath(__file__)
    cmd = [
        sys.executable,
        script_path,
        "--eval-mode",
        "--args-path", args_path,
        "--epoch", str(epoch),
        "--rank", str(rank),
    ]

    # Launch subprocess with current environment (which includes RANK, WORLD_SIZE, etc.)
    # Environment inherits all NCCL settings from parent, so it will use the same network interface
    env = os.environ.copy()
    print(f"[Rank {rank}] Launching eval subprocess...", flush=True)
    process = subprocess.Popen(cmd, env=env)

    # Wait for this rank's subprocess to complete
    return_code = process.wait()

    if return_code != 0:
        print(f"[Rank {rank}] Eval subprocess failed with return code {return_code}", flush=True)
    else:
        print(f"[Rank {rank}] Eval subprocess completed successfully", flush=True)

    # Note: We don't re-initialize the parent process group since training is complete after evals

    # Clean up temp files
    if os.path.exists(args_path):
        os.remove(args_path)

    return return_code

def get_collate_fn(args):
    collate_fn = torch.utils.data.default_collate
    if args.shared_cfg.use_tokenizer_dataset:
        collate_fn = data_utils.collate_fn_tokenizer
    elif not args.shared_cfg.use_lerobot:
        collate_fn = data_utils.collate_history_batch if not args.dataset_cfg.use_dali else partial(data_utils.collate_history_batch_dali, num_img_keys=args.shared_cfg.num_cameras)
    elif args.shared_cfg.use_lerobot:
        collate_fn = data_utils.collate_fn_lerobot
    return collate_fn

def get_video_transforms(args, img_gripper_flags, device, model):
    no_aug_vision_transform, vision_transform = None, None
    kornia_video_transform = None
    if args.shared_cfg.use_lerobot or args.shared_cfg.use_kornia_augmentation:
        # name = "kornia_aug.MultiViewVideoTransform" if not args.shared_cfg.use_tokenizer_dataset else "kornia_aug.MultiViewDictTransform"
        name = "torch_aug.MultiViewTorchVideoTransform" if not args.shared_cfg.use_tokenizer_dataset else "torch_aug.MultiViewTorchDictTransform"
        # if resnet is in the model name, use 128 x 128 image size
        final_image_size = 224
        resize = 248
        if "resnet" in model.vision_encoder.name:
            final_image_size = 128
            resize = 148
        kornia_video_transform = eval(name)(
            img_gripper_flags,
            final_image_size=final_image_size,
            resize=resize,
            share_bc_across_frames_and_views=True,
            mean=data_utils.IMAGENET_MEAN,
            std=data_utils.IMAGENET_STD,
        ).to(device).eval()
    elif (not args.shared_cfg.no_img) and (not args.shared_cfg.use_bboxes) and (not isinstance(model.vision_encoder, VisionPatchEncoder)):
        timm_data_cfg = timm.data.resolve_data_config(model.vision_encoder.model.pretrained_cfg)
        no_aug_vision_transform = timm.data.create_transform(**timm_data_cfg)
        if args.dataset_cfg.vision_aug:
            timm_data_cfg["is_training"] = True
            timm_data_cfg["hflip"] = 0.0
            timm_data_cfg["scale"] = (0.65, 1.0)
            timm_data_cfg["ratio"] = (1.0, 1.0)
            vision_transform = timm.data.create_transform(**timm_data_cfg)
    else:
        vision_transform, no_aug_vision_transform = data_utils.get_vision_transform(size=(224, 224))

    return vision_transform, no_aug_vision_transform, kornia_video_transform

def get_sampler(epoch, train_weights, val_weights, dataset_train, dataset_val, args):
    sampler_train, sampler_val = None, None
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    if train_weights is not None:
        print("Using distributed weighted sub epoch sampler for training")
        sampler_train = misc.DistributedWeightedSubEpochSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, split_epoch=args.shared_cfg.split_epoch, shuffle=True, weights=train_weights
        )
    else:
        sampler_train = misc.DistributedSubEpochSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, split_epoch=args.shared_cfg.split_epoch, shuffle=True
        )
    if dataset_val:
        if val_weights is not None:
            print("Using distributed weighted sub epoch sampler for val")
            sampler_val = misc.DistributedWeightedSubEpochSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, split_epoch=args.shared_cfg.split_epoch, shuffle=False, weights=val_weights
            )
        else:
            sampler_val = misc.DistributedSubEpochSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, split_epoch=args.shared_cfg.split_epoch, shuffle=False
            )
    if args.distributed:
        sampler_train.set_epoch(epoch)
        if sampler_val is not None:
            sampler_val.set_epoch(epoch)
    return sampler_train, sampler_val

def main(args : ExperimentConfig):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.shared_cfg.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Loading data config
    image_keys = []
    img_gripper_flags = []
    for dataset_json in args.dataset_cfg.dataset_json:
        print("Loading dataset config from: ", dataset_json)
        data_cfg = json.load(open(dataset_json, 'r'))

        # make sure the number of cameras is correct
        rgb_observations = data_cfg["image_keys"]
        image_keys = list(rgb_observations)
        img_gripper_flags = []
        for key in rgb_observations:
            if ("gripper" in key) or ("hand" in key) or ("wrist" in key):
                img_gripper_flags.append(True)
            else:
                img_gripper_flags.append(False)
        if args.shared_cfg.no_img:
            args.shared_cfg.num_cameras = 0
            assert len(rgb_observations) == 0, "Number of cameras must be 0 for no_img. Check your json file"
        assert len(rgb_observations) == args.shared_cfg.num_cameras, "Number of cameras must match the number of rgb observations"
        if len(rgb_observations) == 0:
            args.shared_cfg.num_cameras = 0

    model = model_constructor(
        model_config=args.model_cfg,
        shared_config=args.shared_cfg,
        train=args.train,
        extra_kwargs={
            'image_keys': image_keys,
        },
    )
    model.to(device)

    vision_transform, no_aug_vision_transform, kornia_video_transform = get_video_transforms(args, img_gripper_flags, device, model)
    model_without_ddp = model

    # controlled by --model-cfg.policy-cfg.pretrained_path flag
    if args.model_cfg.policy_cfg.pretrained_path is not None:
        print("Finetuning from %s" % args.model_cfg.policy_cfg.pretrained_path)
        misc.load_model(model_without_ddp, args.model_cfg.policy_cfg.pretrained_path)

    print("Model trainable params: ")
    print(model_without_ddp.state_dict().keys())

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # training detail
    eff_batch_size = args.shared_cfg.batch_size * args.trainer_cfg.accum_iter * misc.get_world_size()

    if args.optimizer_cfg.lr is None:  # only base_lr is specified
        args.optimizer_cfg.lr = args.optimizer_cfg.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.optimizer_cfg.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.optimizer_cfg.lr)

    print("accumulate grad iterations: %d" % args.trainer_cfg.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # following timm: set wd as 0 for bias and norm layers
    param_groups = misc.add_weight_decay(model_without_ddp, args.optimizer_cfg.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.optimizer_cfg.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler(set_scaler=True)

    total, trainable = model_without_ddp.get_total_parameters(), model_without_ddp.get_trainable_parameters()
    print("trainable: ", trainable/1e6, "M")
    print("Total params: ", total/1e6, "M")
    print("percentage trainable: ", trainable / total)
    # print all the trainable parameters
    # for name, param in model_without_ddp.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.shape)
    # --resume
    best_val_loss = float('inf')
    best_val_loss = misc.resume_from_ckpt(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    effective_bs =  args.shared_cfg.batch_size * args.trainer_cfg.accum_iter * misc.get_world_size()
    seq_length = args.shared_cfg.seq_length
    token_batch_size = effective_bs * seq_length

    dataset_train, dataset_val = load_datasets(args, vision_transform, no_aug_vision_transform)
    save_epoch_list = []
    # if n_gradients_steps is mentioned, calculate the epochs to satisfy the total number of gradient steps
    if args.trainer_cfg.n_gradients_steps > 0:
        # args.trainer_cfg.epochs = args.trainer_cfg.n_gradients_steps // (len(dataset_train) + len(dataset_val))
        gradients_steps_per_epoch = len(dataset_train) // (args.trainer_cfg.accum_iter * misc.get_world_size())
        # take the ceiling of the epochs
        args.trainer_cfg.epochs = math.ceil(args.trainer_cfg.n_gradients_steps / gradients_steps_per_epoch)
        # save-every should be set to the 5 checkpoints per run
        print(f"Calculated epochs to satisfy the total number of gradient steps: {args.trainer_cfg.epochs}")
        # add the save-every to the save_epoch_list with total of 5 checkpoints per run using the new epochs calculated; do not include 0
        spaced_epochs = np.linspace(0, args.trainer_cfg.epochs, 5).astype(int) - 1
        save_epoch_list = spaced_epochs.tolist()
        print(f"Save epochs list: {save_epoch_list}")

    # save the train the val splits
    dataset_train.save_split(os.path.join(args.logging_cfg.output_dir, "train_split.json"))
    if dataset_val is not None:
        dataset_val.save_split(os.path.join(args.logging_cfg.output_dir, "val_split.json"))

    ca = certifi.where(); os.environ.setdefault('REQUESTS_CA_BUNDLE', ca); os.environ.setdefault('SSL_CERT_FILE', ca)
    if not misc.is_main_process():
        os.environ["WANDB_MODE"] = "offline"
    # Start a wandb run with `sync_tensorboard=True`
    if global_rank == 0 and args.logging_cfg.log_name is not None:
        wandb_project_name = args.trainer_cfg.wandb_project
        wandb_entity = args.trainer_cfg.wandb_entity

        # Prepare wandb init kwargs
        wandb_init_kwargs = {
            "project": wandb_project_name,
            "config": args,
            "name": args.logging_cfg.log_name,
            "sync_tensorboard": False,
        }
        if wandb_entity:
            wandb_init_kwargs["entity"] = wandb_entity
        # Add group if specified
        if args.logging_cfg.wandb_group is not None:
            wandb_init_kwargs["group"] = args.logging_cfg.wandb_group

        if (not args.shared_cfg.resume) or args.shared_cfg.resume_new_exp:
            resume = None
            wandb_init_kwargs["resume"] = resume
            wandb_run = wandb.init(**wandb_init_kwargs)
            run_id = wandb_run.id
            # save the run id to a txt file in the output directory
            with open(os.path.join(args.logging_cfg.output_dir, "run_id.txt"), "w") as f:
                f.write(run_id)
        elif os.path.exists(os.path.join(args.logging_cfg.output_dir, "run_id.txt")):
            resume = "must"
            with open(os.path.join(args.logging_cfg.output_dir, "run_id.txt"), "r") as f:
                run_id = f.read()
            print("*"*100)
            print("Resuming wandb run with id: ", run_id)
            wandb_init_kwargs["resume"] = resume
            wandb_init_kwargs["id"] = run_id
            wandb.init(**wandb_init_kwargs)
        else:
            # resume requested but no prior run exists — start fresh
            resume = None
            wandb_init_kwargs["resume"] = resume
            wandb_run = wandb.init(**wandb_init_kwargs)
            run_id = wandb_run.id
            os.makedirs(args.logging_cfg.output_dir, exist_ok=True)
            with open(os.path.join(args.logging_cfg.output_dir, "run_id.txt"), "w") as f:
                f.write(run_id)
        wandb.log({
            "trainable_params": trainable,
            "total_params": total,
            "effective_bs": effective_bs,
            "token_batch_size": token_batch_size,
            "seq_length": seq_length,
        }) # log the number of trainable and total parameters
        if args.trainer_cfg.wandb_watch:
            wandb.watch(model, log="all", log_freq=1000) # default value of log_freq is 1000

    # SummaryWrite
    if global_rank == 0 and args.logging_cfg.log_dir is not None:
        os.makedirs(args.logging_cfg.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.logging_cfg.log_dir)
        log_writer.add_scalar('effective_bs', effective_bs)
        log_writer.add_scalar('trainable_params', trainable)
        log_writer.add_scalar('total_params', total)
        log_writer.add_scalar('token_batch_size', token_batch_size)
        log_writer.add_scalar('seq_length', seq_length)
    else:
        log_writer = None

    print(f"Start training for {args.trainer_cfg.epochs} epochs")
    start_time = time.time()

    # for resume, we need to instantiate new samplers
    resume_reload = args.shared_cfg.resume is not None
    if args.trainer_cfg.compile_model:
        # keep memory usage low
        # model = torch.compile(model)
        model.compile()

    data_loader_train, data_loader_val = None, None
    assert args.shared_cfg.split_epoch == 1, "Currently only supporting split_epoch = 1 because set_epoch is not implemented for DALI as it requires recreating the pipeline."
    train_stats, val_stats = {}, {}

    start_epoch = args.shared_cfg.start_epoch
    dataset_train.shuffle_dataset(start_epoch)
    if dataset_val is not None:
        dataset_val.shuffle_dataset(start_epoch)
    assert not args.dataset_cfg.use_dali, "DALI is not supported for the current dataset"
    # get the sampler for the start epoch
    train_weights, val_weights = None, None
    if isinstance(dataset_train, CustomConcatDataset):
        train_weights = dataset_train.balanced_sampling_weights
    if dataset_val and isinstance(dataset_val, CustomConcatDataset):
        val_weights = dataset_val.balanced_sampling_weights
    sampler_train, sampler_val = get_sampler(
        epoch=start_epoch,
        train_weights=train_weights,
        val_weights=val_weights,
        dataset_train=dataset_train,
        dataset_val=dataset_val,
        args=args,
    )
    collate_fn = get_collate_fn(args)
    data_loader_train = MultiEpochsDataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.shared_cfg.batch_size,
        num_workers=args.trainer_cfg.num_workers // misc.get_world_size(),
        pin_memory=args.trainer_cfg.pin_memory,
        drop_last=True,
        collate_fn=collate_fn
        # persistent_workers=True if (args.trainer_cfg.num_workers // misc.get_world_size()) > 1 else False,
    )
    print("Done loading train dataloader")
    if sampler_val and (len(sampler_val) > args.shared_cfg.batch_size):
        print("Sampler_val = %s" % str(sampler_val))
        print("length of val sampler: ", len(sampler_val))
        data_loader_val = MultiEpochsDataLoader(
            dataset_val,
            sampler=sampler_val,
            batch_size=min(args.shared_cfg.batch_size, len(dataset_val)),
            num_workers=args.trainer_cfg.num_workers // misc.get_world_size(),
            pin_memory=args.trainer_cfg.pin_memory,
            drop_last=True,
            collate_fn=collate_fn
            # persistent_workers=True if (args.trainer_cfg.num_workers // misc.get_world_size()) > 1 else False,
        )
    else:
        data_loader_val = None

    # save an initial checkpoint (checkpoint-0.pth) before any training happens
    # (skip when resuming, so we don't overwrite it with resumed weights)
    if args.logging_cfg.output_dir and args.shared_cfg.resume is None:
        misc.save_model(
            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
            loss_scaler=loss_scaler, epoch=0, best_val_loss=best_val_loss, file_name='checkpoint-0.pth',
        )

    for epoch in range(start_epoch, args.trainer_cfg.epochs):
        # perform environment evaluations
        if args.eval_cfg.eval_every > 0 and ((epoch % args.eval_cfg.eval_every == 0) or (epoch == args.trainer_cfg.epochs - 1)):
            if epoch != 0 or (args.eval_cfg.eval_at_zero == True):
                gc.collect()
                torch.cuda.empty_cache()
                casa_rollouts.perform_env_evals(model_without_ddp, args, kornia_video_transform, epoch_num=epoch)

        if resume_reload or epoch % args.shared_cfg.split_epoch == 0:
            if (data_loader_train is not None) and (args.dataset_cfg.use_dali):
                data_loader_train.close()
            if (data_loader_val is not None) and (args.dataset_cfg.use_dali):
                data_loader_val.close()
            print(f"Shuffling sequences every {args.shared_cfg.split_epoch} epochs, epoch: {epoch}")
            data_loader_train.dataset.shuffle_dataset(epoch)
            if data_loader_val is not None:
                data_loader_val.dataset.shuffle_dataset(epoch)
            # dataset_train.shuffle_dataset(epoch) # this is very important especially if we are only using a small subset of the dataset for training in each epoch
            # if dataset_val is not None:
            #     dataset_val.shuffle_dataset(epoch)
            # set train weights if rebalance_indices is set to True
            print("Updating sampler epoch ...")
            sampler_train.set_epoch(epoch)
            if sampler_val is not None:
                sampler_val.set_epoch(epoch)
            print("Sampler_train = %s" % str(sampler_train))
            print("length of train sampler: ", len(sampler_train))
            print("length of dataset: ", len(dataset_train))
            print("length of dataloader: ", len(data_loader_train))
            gc.collect() # garbage collection for cleaning up memory
            resume_reload = False

        # if args.distributed: # this should be avoided for dali as the pipeline is already created
        #     data_loader_train.sampler.set_epoch(epoch)
        #     if data_loader_val is not None:
        #         data_loader_val.sampler.set_epoch(epoch)


        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args,
            kornia_video_transform=kornia_video_transform
        )

        if (data_loader_val is not None) and (epoch % args.trainer_cfg.val_every == 0):
            with torch.no_grad():
                val_stats = train_one_epoch(
                    model, data_loader_val,
                    optimizer, device, epoch, loss_scaler,
                    log_writer=log_writer, validate=True,
                    args=args,
                    kornia_video_transform=kornia_video_transform
                )
                is_best_val_loss = val_stats["loss"] < best_val_loss
                best_val_loss = min(best_val_loss, val_stats["loss"])
                if is_best_val_loss:
                    print("Best val loss: %.4f" % best_val_loss)
                    misc.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch, best_val_loss=best_val_loss, file_name='best_val_loss.pth',
                    )

            print("Validation Epoch {}".format(epoch))

        if args.logging_cfg.output_dir and (epoch % args.shared_cfg.save_every == 0 or epoch + 1 == args.trainer_cfg.epochs or epoch == args.trainer_cfg.break_after_n_epochs or epoch in save_epoch_list):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, best_val_loss=best_val_loss,
            )
        # always save the last epoch
        misc.save_model(
            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
            loss_scaler=loss_scaler, epoch=epoch, best_val_loss=best_val_loss, file_name='last_epoch.pth'
        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if (data_loader_val is not None) and (epoch % args.trainer_cfg.val_every == 0):
            log_stats.update({f'val_{k}': v for k, v in val_stats.items()})

        if args.logging_cfg.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.logging_cfg.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        # if break_after_n_epochs is set to 50, it will first save the 50th epoch and then break the training.
        if args.trainer_cfg.break_after_n_epochs > -1 and epoch >= args.trainer_cfg.break_after_n_epochs:
            print(f"Breaking training after {epoch} epochs")
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    epoch = args.trainer_cfg.epochs - 1
    # perform environment evaluations (-2 means no evaluation at all)
    if args.eval_cfg.eval_every != -2:
        del model
        del model_without_ddp
        gc.collect()
        torch.cuda.empty_cache()
        ## setting up the environment for the subprocess
        args.shared_cfg.resume = 'last' # use the last checkpoint
        args.shared_cfg.resume_new_exp = True
        args.trainer_cfg.compile_model = False
        # close wandb if it is initialized
        if wandb.run is not None:
            wandb.finish()

        extra_kwargs = {
            'image_keys': image_keys,
            'image_gripper_flags': img_gripper_flags,
        }
        launch_env_evals_in_subprocess(args, epoch=epoch, extra_kwargs=extra_kwargs)

def edit_and_dump_dataset_jsons(dataset_jsons, args):
    new_dataset_jsons = []
    for dataset_json in dataset_jsons:
        if dataset_json == "":
            new_dataset_jsons.append("")
            continue
        print("Loading dataset config from: ", dataset_json)
        data_cfg = json.load(open(dataset_json, 'r'))
        # no img is set to True, then set the image keys to empty
        if args.shared_cfg.no_img:
            data_cfg["image_keys"] = []

        basename = dataset_json.split('config/')[-1]
        new_dataset_json = os.path.join(args.logging_cfg.output_dir, basename)
        # make the directory if it doesn't exist
        os.makedirs(os.path.dirname(new_dataset_json), exist_ok=True)
        new_dataset_jsons.append(new_dataset_json)
        with open(new_dataset_json, 'w') as f:
            json.dump(data_cfg, f, indent=4)
    return new_dataset_jsons

def wait_for_valid_json(path, retries=10, delay=1.0):
    for _ in range(retries):
        try:
            with open(path, 'r') as f:
                content = f.read()
                if content.strip() == "":
                    raise ValueError("File is empty")
                return json.loads(content)
        except (json.JSONDecodeError, ValueError):
            time.sleep(delay)
    raise RuntimeError(f"Could not load valid JSON from {path}")

if __name__ == '__main__':
    # Check if we're in eval mode (launched as subprocess)
    if '--eval-mode' in sys.argv:
        # Parse eval-specific arguments
        eval_idx = sys.argv.index('--eval-mode')
        args_path = sys.argv[sys.argv.index('--args-path', eval_idx) + 1]
        epoch = int(sys.argv[sys.argv.index('--epoch', eval_idx) + 1])
        rank = int(sys.argv[sys.argv.index('--rank', eval_idx) + 1])
        _run_env_evals_in_subprocess(args_path, epoch, rank)
        sys.exit(0)

    # parsing args
    args = tyro.cli(ExperimentConfig)

    # Preserve hyper_tune flag before loading config (if load_config is used)
    hyper_tune_flag = args.hyper_tune

    if args.load_config is not None:
        print("loading configs from file: ", args.load_config)
        assert os.path.exists(args.load_config), f"Config file does not exist: {args.load_config}"
        args : ExperimentConfig = yaml.load(Path(args.load_config).read_text(), Loader=yaml.Loader)

    if args.hyper_tune:
        args.optimizer_cfg.warmup_epochs = 0
        args.trainer_cfg.epochs = 1
        args.optimizer_cfg.min_lr = args.optimizer_cfg.lr

    if args.model_cfg.policy_cfg.phase == 'finetune':
        raise ValueError("Finetune is not tested yet in this branch. For original tested code -- @ 7192b301d0f5a4a425252bdae463bb76b1c961af")

    # if using offline augmentation, do not use DALI loader.
    assert (not args.shared_cfg.rot_6d) or (not args.shared_cfg.rot_euler), "Currently only supporting either rot_6d_quat or rot_euler or None"
    if args.shared_cfg.no_img:
        args.shared_cfg.num_cameras = 0

    # creating the output directory and logging directory
    if args.logging_cfg.log_name is not None:
        args.logging_cfg.output_dir = os.path.join(args.logging_cfg.output_dir, args.logging_cfg.log_name)
    if args.logging_cfg.log_dir is None:
        args.logging_cfg.log_dir = args.logging_cfg.output_dir
    if args.logging_cfg.output_dir:
        Path(args.logging_cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # check if the dataset json files exist
    args.dataset_cfg.dataset_json = edit_and_dump_dataset_jsons(args.dataset_cfg.dataset_json, args)
    args.dataset_cfg.dataset_val_json = edit_and_dump_dataset_jsons(args.dataset_cfg.dataset_val_json, args)

    dataset_json_contents = []
    for dataset_json in args.dataset_cfg.dataset_json:
        print("Loading dataset config from: ", dataset_json)
        data_cfg = wait_for_valid_json(dataset_json)
        dataset_json_contents.append(data_cfg)
    args.dataset_json_contents = dataset_json_contents # this will be used during evaluations

    # dump the args into a yaml file
    with open(os.path.join(args.logging_cfg.output_dir, "run.yaml"), 'w') as f:
        yaml.dump(args, f)

    main(args)
