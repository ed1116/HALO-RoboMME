import time
import math
import sys
from typing import Iterable, Union, Optional

import torch
import torch.nn as nn
from . import misc, lr_sched
import contextlib

from halo.util.args import ExperimentConfig
import torch
import numpy as np
import wandb
from transformers import Trainer, TrainingArguments
from torch.utils.tensorboard import SummaryWriter
import halo.util.misc as misc_utils
from halo.util.misc import NativeScalerWithGradNormCount as NativeScaler
from halo.util.test_runs import TestRunRecorder
from halo.models.augmentation.kornia_aug import MultiViewVideoTransform

def log_wandb_values(wandb_log_dict):
    def flatten_dict(d, prefix="", sep="/"):
        out = {}
        for k, v in d.items():
            key = f"{prefix}{sep}{k}" if prefix else str(k)
            if isinstance(v, dict):
                out.update(flatten_dict(v, key, sep))
            elif isinstance(v, np.ndarray) and v.ndim == 1:
                out[key] = wandb.Histogram(v)
            else:
                out[key] = v
        return out
    wandb_log_dict = flatten_dict(wandb_log_dict)
    wandb.log(wandb_log_dict)
    return

def update_mean_logging_dict(mean_logging_dict, wandb_log_dict, log_step_dict):
    '''
    For every value in wandb_log_dict, we maintain a running mean (not full list) with number of values seen so far = log_step
    Also maintain the timesteps for each value logged and use that instead of log_step in the running mean calculation
    '''
    for k, v in wandb_log_dict.items():
        new_k = f"mean_{k}"
        if new_k not in mean_logging_dict:
            mean_logging_dict[new_k] = v
            log_step_dict[new_k] = {} if isinstance(v, dict) else 1
        elif isinstance(v, dict):
            if not isinstance(log_step_dict[new_k], dict):
                log_step_dict[new_k] = {}
            mean_logging_dict[new_k], log_step_dict[new_k] = update_mean_logging_dict(mean_logging_dict[new_k], v, log_step_dict[new_k])
        elif isinstance(v, np.ndarray):
            if v.ndim == 1:
                # 1D arrays: compute running mean element-wise
                mean_logging_dict[new_k] = (mean_logging_dict[new_k] * (log_step_dict[new_k] - 1) + v) / log_step_dict[new_k]
                log_step_dict[new_k] += 1
            else:
                # Multi-dimensional arrays or lists: just update to latest value (skip running mean)
                pass
        elif isinstance(v, (list, tuple)):
            # Lists/tuples: just update to latest value (skip running mean)
            pass
        else:
            # Handle scalars (int, float, etc.) - same as log_wandb_values treats everything else
            mean_logging_dict[new_k] = (mean_logging_dict[new_k] * (log_step_dict[new_k] - 1) + float(v)) / log_step_dict[new_k]
            log_step_dict[new_k] += 1
    return mean_logging_dict, log_step_dict

def train_one_epoch(
    model: nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler : NativeScaler,
    log_writer: Optional[SummaryWriter] = None,
    validate: Optional[bool] = False,
    args : Optional[ExperimentConfig] = None,
    kornia_video_transform: Optional[MultiViewVideoTransform] = None
):
    if validate:
        model.eval()
    else:
        model.train()
        optimizer.zero_grad() # Clear gradients only during training

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    accum_iter = args.trainer_cfg.accum_iter
    global_steps_per_epoch = len(data_loader)
    grad_steps_per_epoch = len(data_loader) // accum_iter
    print(f"Number of gradient steps per epoch: {grad_steps_per_epoch}")
    print(f"Accumulation factor: {accum_iter}")
    global_grad_step = grad_steps_per_epoch * epoch
    global_steps = global_steps_per_epoch * epoch
    print(f"Gradient steps globally {global_grad_step}")

    test_run_recorder = TestRunRecorder(getattr(args.trainer_cfg, "test_runs", "none")) if not validate else TestRunRecorder("none")

    # Compute curriculum topk if enabled
    curriculum_topk = None
    if (args is not None and args.model_cfg.policy_cfg.use_topk_attention) and (args.model_cfg.policy_cfg.ret_topk_max is not None):
        max_topk = args.model_cfg.policy_cfg.ret_topk_max
        min_topk = args.model_cfg.policy_cfg.ret_topk

        curriculum_topk = misc_utils.get_curriculum_topk(
            epoch=epoch,
            min_topk=min_topk,
            max_topk=max_topk,
            total_epochs=args.trainer_cfg.epochs
        )
        if curriculum_topk is not None:
            print(f"Epoch {epoch}: Using curriculum topk = {curriculum_topk}")

    log_step = 0
    mean_logging_dict, mean_log_step_dict = {}, {}
    per_update_logging_dict, per_update_log_step_dict = {}, {}
    # breakpoint()
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    for data_iter_step, dataset_item in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # if data_iter_step == 100:
        #     break
        dataset_item['topk'] = curriculum_topk
        epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        for k, v in dataset_item.items():
            if isinstance(v, torch.Tensor):
                dataset_item[k] = v.to(device, non_blocking=True)
            elif isinstance(v, dict):
                dataset_item[k] = {k2: v2.to(device, non_blocking=True) if isinstance(v2, torch.Tensor) else v2
                                   for k2, v2 in v.items()}
        if kornia_video_transform is not None:
            dataset_item['observation'] = kornia_video_transform(dataset_item['observation'])

        logging_dict = {'epoch_1000x': epoch_1000x, 'log_writer': log_writer}
        # TODO: Add context, remove unnecessary cuda.synchronize()
        # context = model.no_sync() if (validate) or ((data_iter_step+1) % accum_iter != 0) else contextlib.nullcontext()
        # with context:
        with torch.amp.autocast('cuda', dtype=misc_utils.convert_str_to_torch_dtype(args.shared_cfg.compute_dtype)):
            model_outputs = model(dataset_item, log_dict=logging_dict, log_writer=log_writer)
            loss, loss_dict = model_outputs[0], model_outputs[1]
            model_out_tensor = model_outputs[2] if len(model_outputs) > 2 else None

        loss_value = loss.item()
        loss_value_dict = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
        if not math.isfinite(loss_value):
            print(f"dataset_item: {dataset_item}")
            print(f"loss_value: {loss_value}, rank: {misc.get_rank()}")
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        # loss_value_dict = {k: v / accum_iter for k, v in loss_value_dict.items()}
        # print(f"loss_value_dict: {loss_value_dict}")
        # print(f"loss : {loss}")
        if not validate:
            loss_scaler(loss, optimizer, parameters=model.parameters(),
                        update_grad=(data_iter_step + 1) % accum_iter == 0)


        if (data_iter_step + 1) % accum_iter == 0 and not validate:
            global_grad_step += 1
            # print(f"Gradient steps globally {global_grad_step} at data_iter_step {data_iter_step}")
            optimizer.zero_grad()
        global_steps += 1

        if test_run_recorder.active and (data_iter_step + 1) % accum_iter == 0 and not validate:
            test_run_recorder.record_step(global_grad_step, loss_value, loss_value_dict, model_out_tensor)
            if test_run_recorder.done():
                test_run_recorder.finalize()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        # print(f"loss_value: {loss_value:.4f}; rank={misc.get_rank()}; step={data_iter_step}")
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        # print(f"loss_value_reduce: {loss_value_reduce:.4f}; rank={misc.get_rank()}; step={data_iter_step}")
        loss_value_dict_reduce = {k: misc.all_reduce_mean(v) for k, v in loss_value_dict.items()}
        if (args.logging_cfg.log_name is not None) and (misc.get_rank() == 0):
            mean_logging_dict, mean_log_step_dict = update_mean_logging_dict(mean_logging_dict, loss_value_dict_reduce, mean_log_step_dict)
            per_update_logging_dict, per_update_log_step_dict = update_mean_logging_dict(per_update_logging_dict, loss_value_dict_reduce, per_update_log_step_dict)
            log_step += 1
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            wandb_log_dict = {
                'epoch': epoch,
                'epoch_1000x': epoch_1000x,
                'global_grad_step': global_grad_step,
                'global_steps': global_steps,
            }
            # Log curriculum topk if enabled
            if curriculum_topk is not None:
                wandb_log_dict['train/curriculum_topk'] = curriculum_topk
                log_writer.add_scalar('train/curriculum_topk', curriculum_topk, global_grad_step)
            log_writer.add_scalar('epoch', epoch, global_grad_step)
            log_writer.add_scalar('global_grad_step', global_grad_step, global_grad_step)
            if misc.get_rank() == 0 and (args.logging_cfg.log_name is not None):
                # wandb.log({'epoch': epoch, 'epoch_1000x': epoch_1000x, 'global_grad_step': global_grad_step, 'global_steps': global_steps}) # this is to trigger the wandb watch logging
                log_writer.add_scalar('epoch_1000x', epoch_1000x, global_grad_step)
                log_writer.add_scalar('global_steps', global_steps, global_grad_step)
                log_writer.add_scalar('global_grad_step', global_grad_step, global_grad_step)
            if not validate:
                log_writer.add_scalar('train_loss', loss_value_reduce, global_grad_step)
                for k, v in loss_value_dict_reduce.items():
                    log_writer.add_scalar('train_{}'.format(k), v, global_grad_step)
                log_writer.add_scalar('train_orig/loss', loss_value, global_grad_step) # TODO: Remove this
                log_writer.add_scalar(f'train/loss_orig_worker{misc.get_rank()}', loss_value, global_grad_step)
                log_writer.add_scalar(f'train/loss_red_worker{misc.get_rank()}', loss_value_reduce, global_grad_step)
                log_writer.add_scalar('lr', lr, global_grad_step)
                wandb_log_dict['train/loss'] = loss_value_reduce
                wandb_log_dict['train/loss_orig_worker{}'.format(misc.get_rank())] = loss_value
                wandb_log_dict['train/loss_red_worker{}'.format(misc.get_rank())] = loss_value_reduce
                wandb_log_dict['train/lr'] = lr
                if hasattr(model, 'get_logging_values'):
                    wandb_log_dict['train/icrt'] = model.get_logging_values()
                else:
                    wandb_log_dict['train/icrt'] = model.module.get_logging_values()
                for k, v in loss_value_dict_reduce.items():
                    wandb_log_dict['train/train_{}'.format(k)] = v
                for k, v in per_update_logging_dict.items():
                    wandb_log_dict[f'train_accum/{k}'] = v
                per_update_logging_dict, per_update_log_step_dict = {}, {} # empty it
            else:
                log_writer.add_scalar('val_loss', loss_value_reduce, global_steps)
                for k, v in loss_value_dict_reduce.items():
                    log_writer.add_scalar('val_{}'.format(k), v, global_steps)
                log_writer.add_scalar('val_orig/loss', loss_value, global_steps) # TODO: Remove this
                log_writer.add_scalar(f'val/loss_orig_worker{misc.get_rank()}', loss_value, global_steps)
                log_writer.add_scalar(f'val/loss_red_worker{misc.get_rank()}', loss_value_reduce, global_steps)
                wandb_log_dict['val/loss'] = loss_value_reduce
                wandb_log_dict['val/loss_orig_worker{}'.format(misc.get_rank())] = loss_value
                wandb_log_dict['val/loss_red_worker{}'.format(misc.get_rank())] = loss_value_reduce
                for k, v in loss_value_dict_reduce.items():
                    wandb_log_dict['val/val_{}'.format(k)] = v
                for k, v in per_update_logging_dict.items():
                    wandb_log_dict[f'val_accum/{k}'] = v
                per_update_logging_dict, per_update_log_step_dict = {}, {} # empty it
            if args.logging_cfg.log_name is not None:
                log_wandb_values(wandb_log_dict)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if (args.logging_cfg.log_name is not None) and misc.get_rank() == 0:
        log_wandb_values(mean_logging_dict)
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
