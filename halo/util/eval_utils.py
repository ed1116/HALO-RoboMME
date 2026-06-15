import os
import cv2
from filelock import FileLock
from PIL import Image
from termcolor import colored
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import torch
import wandb
from typing import Tuple
from halo.util.args import ExperimentConfig

# @eval_utils
def get_filename_from_args(args: ExperimentConfig)->Tuple[str, str]:

    filename = f"results_s{args.shared_cfg.seed}_evals{args.eval_cfg.n_eval}"
    if not args.eval_cfg.replay_prompt_traj:
        if args.eval_cfg.latest_epoch is not None:
            filename += f"_ep{args.eval_cfg.latest_epoch}"
        elif args.eval_cfg.ckpt_path is not None:
            filename += '_' + os.path.basename(args.eval_cfg.ckpt_path).replace(".pth", "")
        else:
            raise ValueError("Either latest_epoch or ckpt_path must be provided")

        if args.eval_cfg.replay_train_traj:
            filename += "_train_resets"
        filename += ".csv"
    elif args.eval_cfg.replay_prompt_traj:
        filename += "_replayprompt"
    result_path = os.path.join(args.logging_cfg.output_dir, filename)

    return result_path, filename

def wandb_init(run_id: int, wandb_entity: str, wandb_project_name: str, log_name: str):
    # if wandb is already initialized, return
    close = True
    if wandb.run is not None:
        close = False
        return wandb.run.id, close
    init_kwargs = dict(project=wandb_project_name, name=log_name, sync_tensorboard=True, resume="must", id=run_id)
    if wandb_entity:
        init_kwargs["entity"] = wandb_entity
    wandb.init(**init_kwargs)
    return wandb.run.id, close

def save_wandb_result(task_result: dict, result_path: str, rollout_imgs: list, args: ExperimentConfig):
    wandb_lock_path = result_path + ".wandb.lock"
    lock = FileLock(wandb_lock_path)
    with lock:
        # ckpt_num = int(args.ckpt_path.split("/")[-1].split("-")[-1].split(".")[0])
        log_name = args.logging_cfg.log_name
        output_dir = args.logging_cfg.output_dir
        wandb_project_name = args.trainer_cfg.wandb_project
        wandb_entity = args.trainer_cfg.wandb_entity
        resume = "must"
        if not os.path.exists(output_dir):
            print(colored(f"Output directory {output_dir} does not exist, skipping wandb", "red"))
            args.use_wandb = False
        else:
            with open(os.path.join(output_dir, "run_id.txt"), "r") as f:
                run_id = f.read()
            run_id, close = wandb_init(run_id, wandb_entity, wandb_project_name, log_name)

            ckpt_num = 0
            if args.eval_cfg.ckpt_path is not None:
                ckpt_num = args.eval_cfg.ckpt_path.split("/")[-1].split("-")[-1].split(".")[0]
                if ckpt_num == "last" or ckpt_num == "last_epoch":
                    ckpt_num = "-1"
                ckpt_num = int(ckpt_num)
            elif args.eval_cfg.latest_epoch is not None:
                ckpt_num = args.eval_cfg.latest_epoch
            else:
                raise ValueError(f"No ckpt path or latest epoch provided")

            # set the plotting of the success rate w.r.t. the epoch number
            success_rate_metric_name = f"success_n/{task_result['task_name']}_{args.eval_cfg.n_eval}evals"
            wandb.define_metric(success_rate_metric_name, step_metric="ckpt_num")
            wandb_log_dict = {
                success_rate_metric_name: float(task_result['success_rate']),
                "ckpt_num": ckpt_num
            }
            wandb.log(wandb_log_dict)

            if (len(rollout_imgs) > 0) and args.eval_cfg.save_videos_in_wandb:
                task_name = task_result['task_name']
                # max 10 videos
                wandb_video = np.stack(rollout_imgs, axis=0)
                wandb_video = wandb_video.transpose(0, 3, 1, 2)
                wandb.log({
                    f"success_n/videos/{task_name}": wandb.Video(wandb_video, fps=30, format="mp4"),
                    f"ckpt_num": ckpt_num
                })

            # if newly initialized wandb, close it
            if close:
                wandb.finish()
    try:
        os.remove(wandb_lock_path)
    except FileNotFoundError:
        # Lock file may already be removed on shared/NFS filesystems.
        pass
    return

def save_task_result(task_result: dict, result_path: str):
    lock_path = result_path + ".lock"
    lock = FileLock(lock_path)
    with lock:
        if os.path.exists(result_path):
            results = pd.read_csv(result_path)
        else:
            results = pd.DataFrame(columns=["n_eval", "success_rate", "task_name"])
        print(colored(f"Appending results to: {result_path}", "green"))
        task_result = pd.DataFrame([task_result])
        results = pd.concat([results, task_result], ignore_index=True)
        results.to_csv(result_path, index=False)
    try:
        os.remove(lock_path)
    except FileNotFoundError:
        # Lock file may already be removed on shared/NFS filesystems.
        pass
    return

def verify_result_path(result_path: str, task_name: str) -> bool:
    if os.path.exists(result_path):
        csv_content = pd.read_csv(result_path)
        # check if the entries have the task name
        if task_name in csv_content['task_name'].tolist():
            return True
    return False

def save_rollout_video(imgs, video_path, fps=30, dowsample_factor=-1):
    if len(imgs) == 0:
        return
    if len(imgs)<fps:
        fps = 1
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # calculate the resolution and then downsample if factor > -1
    resolution = (imgs[0].shape[1], imgs[0].shape[0])
    if dowsample_factor > 1.0:
        resolution = (resolution[0] // dowsample_factor, resolution[1] // dowsample_factor)
    out = cv2.VideoWriter(video_path, fourcc, fps, resolution)
    for i in range(len(imgs)):
        frame = imgs[i]

        # If the input is a torch tensor, convert it to a numpy array first
        if hasattr(frame, 'cpu'): # A simple check for torch.Tensor
            frame = frame.cpu().numpy()

        # If the image is in a float format (e.g., normalized between 0 and 1),
        # we scale it to the 0-255 range.
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0 and frame.min() >= 0.0:
                frame = frame * 255.0

            # Clip values to ensure they are within the valid range and convert to uint8
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if dowsample_factor > -1:
            img = cv2.resize(img, resolution, interpolation=cv2.INTER_LINEAR)
        out.write(img)
    out.release()
    return

def display_image(image, wait_time):
    """
    Displays a single image using Matplotlib, waits for 1 second, and clears the plot.

    Args:
    - image: The image to display (a NumPy array or similar image object).
    """
    plt.imshow(image)
    plt.axis('off')  # Hides axes for better visualization
    plt.show(block=False)  # Non-blocking display
    plt.pause(wait_time)  # Wait for 1 second
    plt.clf()  # Clear the figure for the next image

def control_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

