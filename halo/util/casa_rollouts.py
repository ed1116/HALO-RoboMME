import os
import time
import copy
import json
import torch
import tyro
from collections import OrderedDict
from typing import Optional, Dict, Any, List
import wandb
import numpy as np
import h5py
import fcntl
from PIL import Image
from tqdm import tqdm, trange
from termcolor import colored
from torch.utils.data import DataLoader
from PIL import ImageFont, ImageDraw
from dataclasses import dataclass, field

import robocasa
import robocasa.macros as macros
from robocasa.models.exploration_policy import RotateExplorationPolicy, LRExplorationPolicy

from halo.models.policy.mem_model import MemModel
from halo.models.augmentation.torchvision_aug import MultiViewTorchDictTransform
from halo.util.args import ExperimentConfig
import halo.util.misc as misc_utils
import halo.util.bbox_utils as bbox_utils
import halo.util.casa_utils as casa_utils
import halo.util.eval_utils as eval_utils
from halo.data import load_datasets
macros.SHOW_SITES = False
macros.VERBOSE = False
assert macros.SHOW_SITES == False, "SHOW_SITES should be False"

@dataclass
class Trajectory:
    obs: Dict[str, torch.Tensor] = field(default_factory=dict)
    actions: List[np.ndarray] = field(default_factory=list)
    # reward is a list of rewards for each step
    rewards: List[float] = field(default_factory=list)
    # done is a list of done flags for each step
    dones: List[bool] = field(default_factory=list)
    policy_mode: List[int] = field(default_factory=list)
    # HDF5 file path for saving trajectories
    hdf5_path: Optional[str] = None
    # Episode counter for generating episode names (local to the rank)
    episode_counter_local: int = 0
    # Whether to compress observations
    compress: bool = True

    # success is a list of success flags for each step
    success: List[bool] = field(default_factory=list)

    # id2name dictionary
    id2name: Dict[int, str] = field(default_factory=dict)

    def add_step(self, obs: Dict, policy_mode: int, action: np.array, reward: float, done: bool, success: bool, flip_images: bool = True):
        if flip_images:
            # flip the image keys
            for key in obs.keys():
                if "image" in key:
                    assert obs[key].ndim == 3
                    assert obs[key].shape[-1] == 3, f"Image should have channel dimension 3, got {obs[key].shape}"
                    # To avoid mutating obs globally, copy the array before modifying.
                    obs[key] = obs[key][::-1].copy()

        # add all the bbox related observations
        bbox_obs = bbox_utils.get_all_bbox_observations(obs, self.id2name)
        for k, v in bbox_obs.items():
            obs[k] = v
        assert action.ndim == 1, f"Action should be 1D tensor: (action_dim,), got {action.shape}"
        # policy_mode: 0 = to imitate, 1 = exploration
        # self.obs.append(obs)
        for k, v in obs.items():
            if k not in self.obs:
                self.obs[k] = []
            self.obs[k].append(v)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.policy_mode.append(policy_mode)
        self.success.append(success)

    def reset(self, episode_name: Optional[str] = None, episode_counter: Optional[int] = None):
        """
        Reset the trajectory by saving current data to HDF5 file if any data exists,
        then clearing all fields.

        Args:
            episode_name: Optional episode name. If None, uses "episode_{episode_counter}"
        """
        # Check if there's any data to save
        has_data = (
            len(self.obs) > 0 or
            len(self.actions) > 0 or
            len(self.rewards) > 0 or
            len(self.dones) > 0 or
            len(self.policy_mode) > 0
        )

        if has_data:
            if self.hdf5_path is None:
                raise ValueError("hdf5_path must be set before calling reset(). Use set_hdf5_path() or pass it to reset().")

            # Generate episode name if not provided
            if episode_name is None:
                if episode_counter is None:
                    episode_name = f"episode_{self.episode_counter_local}"
                else:
                    episode_name = f"episode_{episode_counter}"
                self.episode_counter_local += 1

            # Save to HDF5 with file locking
            self._save_to_hdf5_with_lock(self.hdf5_path, episode_name)

        # Clear all fields
        self.obs.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.policy_mode.clear()
        self.success.clear()

    def set_hdf5_path(self, path: str, compress: bool = True):
        """
        Set the HDF5 file path for saving trajectories.

        Args:
            path: Path to the HDF5 file
            compress: Whether to compress observations (default: True)
        """
        self.hdf5_path = path
        self.compress = compress
        # Ensure directory exists
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

    def _save_to_hdf5_with_lock(self, path: str, episode_name: str):
        """
        Internal method to save trajectory to HDF5 file with file locking.

        Args:
            path: Path to the HDF5 file
            episode_name: Name for the episode group
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        # Create lock file path
        lock_path = path + ".lock"

        # Open lock file and acquire exclusive lock
        lock_file = open(lock_path, 'w')
        try:
            # Acquire exclusive lock (blocks until available)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

            try:
                # Open or create HDF5 file
                file_mode = 'a' if os.path.exists(path) else 'w'
                with h5py.File(path, file_mode) as f:
                    # Create data group if it doesn't exist
                    if 'data' not in f:
                        data_grp = f.create_group('data')
                    else:
                        data_grp = f['data']

                    # Create episode group
                    if episode_name in data_grp:
                        # If episode already exists, remove it first
                        del data_grp[episode_name]
                    ep_data_grp = data_grp.create_group(episode_name)

                    # Convert observations from list of dicts to dict of lists
                    if len(self.obs) > 0:
                        # Get all keys from all observations
                        obs_keys = list(self.obs.keys())

                        # Convert lists to numpy arrays and store
                        for k, v in self.obs.items():
                            if len(v) > 0:
                                try:
                                    obs_array = np.array(v)
                                except ValueError:
                                    # If shapes are incompatible, store as object array
                                    obs_array = np.array(v, dtype=object)

                                if self.compress:
                                    ep_data_grp.create_dataset(
                                        f"obs/{k}",
                                        data=obs_array,
                                        compression="gzip",
                                    )
                                else:
                                    ep_data_grp.create_dataset(
                                        f"obs/{k}",
                                        data=obs_array,
                                    )

                    # Convert actions to numpy array
                    actions_array = np.stack(self.actions, axis=0)
                    ep_data_grp.create_dataset("actions", data=actions_array)

                    # Convert rewards, dones, and policy_mode to numpy arrays
                    ep_data_grp.create_dataset("rewards", data=np.array(self.rewards))
                    ep_data_grp.create_dataset("dones", data=np.array(self.dones, dtype=np.int32))
                    ep_data_grp.create_dataset("policy_mode", data=np.array(self.policy_mode, dtype=np.int32))

                    ep_data_grp.create_dataset("success", data=np.array(self.success, dtype=np.int32))
                    # Store metadata
                    num_samples = len(self.actions)
                    ep_data_grp.attrs["num_samples"] = num_samples

                    print(f"Saved trajectory to {path} in group {episode_name} with {num_samples} samples")

            finally:
                # Release lock before closing lock file
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

        finally:
            # Close lock file
            lock_file.close()
            # Remove lock file
            try:
                os.remove(lock_path)
            except OSError:
                pass  # Lock file might have been removed by another process

def online_obs_processing(obs: dict, keys_info: dict, args: ExperimentConfig, device: torch.device, video_transform: MultiViewTorchDictTransform = None) -> tuple[OrderedDict, torch.Tensor]:

    # combine any preprocessing of the observation here.
    obs = casa_utils.preprocess_casa_obs(obs, keys_info, args)

    # image calculation
    image = None
    if not args.shared_cfg.no_img: # robomimic env wrapper in used casa_s2o.py has flips the images
        image = {}
        for cam_key in keys_info["image_keys"]:
            cam_name = cam_key.split("/")[-1]
            cam_obs = obs[cam_name]
            # flip the image to match the robomimic env wrapper in used casa_s2o.py
            # convert H, W, C to C, H, W
            image[cam_key] = torch.from_numpy(cam_obs[::-1].copy()).permute(2, 0, 1)

        # adding the time dimension
        image = OrderedDict({k: torch.stack([o[k] for o in [image]], axis=0).to(device, non_blocking=True) for k in keys_info["image_keys"]})

        if video_transform is not None:
            image = video_transform(image)

    # proprio calculation
    proprio = np.concatenate(
        [obs[key.replace("obs/", "")] for key in keys_info["proprio_keys"]],
        axis=-1
    )
    proprio = torch.from_numpy(proprio.copy()).float().to(device, non_blocking=True)

    # add the time dimension and N (number of proprio tokens) dimension
    assert proprio.ndim == 1, f"Proprio should be a 1D tensor: (D,), got {proprio.shape} with shape {proprio.shape=}"
    proprio = proprio.view(1, 1, -1)

    return image, proprio

def explore_eval_loop(
        keys_info,
        args,
        explore_policy,
        model,
        env,
        rollout_imgs,
        video_transform,
        pbar=None,
        data_trajectory: Optional[Trajectory] = None,
        gt_explore_actions: Optional[np.ndarray] = None
    ):
    rank = misc_utils.get_rank()
    explore_policy.begin()
    obs = env._get_observations(force_update=True)
    step_count = 0
    print(colored(f"[Rank {rank}] Exploration started", "green"), flush=True)
    action_list = []
    image_list = []
    proprio_list = []
    while True:

        # process the observation
        image, proprio = online_obs_processing(obs, keys_info, args=args, device=torch.device('cuda'), video_transform=video_transform)

        # if saving videos, add the images to the rollout
        if args.eval_cfg.save_videos:
            rollout_imgs.append(casa_utils.get_images_to_save(env, keys_info))

        # get the ground truth action
        gt_action = explore_policy.get_action(obs)
        if gt_explore_actions is not None:
            if step_count >= len(gt_explore_actions): # break if we have reached the end of the ground truth actions
                break
            gt_action = gt_explore_actions[step_count]

        # break if done
        if gt_action is None:
            break

        # take a step in the environment
        obs, _, _, _ = env.step(gt_action)

        # save the data to the trajectory
        if data_trajectory is not None:
            # this will be raw observations, flip images inside the add_step function
            data_trajectory.add_step(obs, policy_mode=1, action=gt_action, reward=0.0, done=False, success=False, flip_images=True)

        # store the action, image, and proprio
        to_store_action = torch.from_numpy(gt_action).unsqueeze(dim=0).unsqueeze(dim=0).to(device='cuda', non_blocking=True).float()
        assert to_store_action.ndim == 3, f"Action should be 3D tensor: (1, 1, action_dim), got {to_store_action.shape}"
        assert proprio.ndim == 3, f"Proprio should be 3D tensor: (1, num_proprio_tokens=1, proprio_dim), got {proprio.shape}"
        image_keys = list(image.keys())
        for image_key in image_keys:
            assert image[image_key].ndim == 4, f"Image {image_key} should be 4D tensor: (1, 3, H, W), got {image[image_key].shape}"
        image_list.append(image)
        proprio_list.append(proprio)
        action_list.append(to_store_action)
        step_count += 1

        # debugging
        if args.eval_cfg.debug and step_count > 10:
            break

        # Warn if exploration goes beyond 1000 steps (potential infinite loop)
        if step_count > 1000:
            if step_count == 1001:
                print(colored(f"WARNING: Exploration has exceeded 1000 steps (current: {step_count}). gt_action is None: {gt_action is None}. This may indicate an infinite loop!", "red"), flush=True)
            # Print every 100 steps after 1000 to avoid spam
            elif step_count % 100 == 0:
                print(colored(f"Exploration still running at step {step_count}, gt_action is None: {gt_action is None}...", "yellow"), flush=True)

        if pbar is not None:
            pbar.update(1)

    print(colored(f"[Rank {rank}] Exploration completed after {step_count} steps", "green"), flush=True)

    # convert to sequence format
    image = OrderedDict({k: torch.cat([o[k] for o in image_list], axis=0)[::args.shared_cfg.downsample_obs] for k in keys_info["image_keys"]})
    action = torch.cat(action_list, axis=0)[::args.shared_cfg.downsample_obs]
    proprio = torch.cat(proprio_list, axis=0)[::args.shared_cfg.downsample_obs]

    # add to model: maybe cache or no cache
    with torch.autocast('cuda', dtype=misc_utils.convert_str_to_torch_dtype(args.shared_cfg.compute_dtype)):
        model.add_sequence(
            sequences=[{
                'observation': image,
                'proprio': proprio,
                'action': action
            }]
        )

    return step_count, rollout_imgs

def do_rollout(i_eval, args, env, model, video_transform, keys_info, rollout_imgs, ep_metas, env_states, task_name):
    # set to eval
    rank = misc_utils.get_rank()
    model.eval()
    video_transform.eval()
    model.reset(batch_size=1)

    # reset
    print(f"[Rank {rank}] Resetting environment for {task_name} episode {i_eval + 1} of {len(ep_metas)}", flush=True)
    ep_meta = ep_metas[(i_eval + 1) % len(ep_metas)]
    demo_key, hdf5_path, gt_explore_actions, gt_teleop_actions = None, None, None, None
    if args.eval_cfg.replay_train_traj:
        state_dict = env_states[(i_eval + 1) % len(env_states)]
        casa_utils.reset_to_original(env, state_dict)
        demo_key = state_dict['demo_key']
        hdf5_path = state_dict['dataset_path']
        return_action_dict = casa_utils.get_exploratory_and_teleop_actions_from_dataset(hdf5_path, demo_key)
        gt_explore_actions = return_action_dict['exploratory_actions']
        gt_teleop_actions = return_action_dict['teleop_actions']
    else:
        env.set_ep_meta(ep_meta)
        env.reset()
        # make up demo_key and hdf5_path
        demo_key = f"demo_{i_eval + 1}"

    # check if save_as_hdf5 is true
    data_trajectory = None
    if args.eval_cfg.save_as_hdf5:
        if args.eval_cfg.hdf5_path is None:
            # path name would be loggin_cfg.output_dir/rollouts_dataset/<task_name>-ep<epoch_num>.hdf5
            filename = f"{task_name}-ep{args.eval_cfg.latest_epoch}.hdf5"
            args.eval_cfg.hdf5_path = os.path.join(args.logging_cfg.output_dir, f"rollouts_dataset/{filename}")
        # Trajectory class to save the rollouts
        name2id = {inst: i for i, inst in enumerate(list(env.model.instances_to_ids.keys()))}
        id2name = {v: k for k, v in name2id.items()}
        data_trajectory = Trajectory(id2name=id2name)
        data_trajectory.set_hdf5_path(args.eval_cfg.hdf5_path)

    # prompt
    rank = misc_utils.get_rank()
    eval_ep_meta = env.get_ep_meta()
    language = [eval_ep_meta['lang']]
    model.prompt(language)
    print(f"[Rank {rank}] {language=}", flush=True)

    pred_steps, steps = 0, 0
    max_steps = ep_meta['action_len'] * 1.5
    pbar = tqdm(total=max_steps, dynamic_ncols=True)

    # save as hdf5 file is true but path is None, create a path
    if args.eval_cfg.save_as_hdf5:
        if args.eval_cfg.latest_epoch is not None:
            # add timestamp-ckptnum
            args.eval_cfg.hdf5_path = os.path.join(args.logging_cfg.output_dir, f"rollouts_dataset/{task_name}/{time.strftime('%Y%m%d_%H%M%S')}-ep{args.eval_cfg.latest_epoch}.hdf5")

    # set up exploration policy
    retrieve_oils_list = ["MemRetrieveOilsFromCounterLL", "MemRetrieveOilsFromCounterLR", "MemRetrieveOilsFromCounterRL", "MemRetrieveOilsFromCounterRR"]
    retrieve_fruit_list = ["MemFruitInSinkRightFar", "MemFruitInSinkLeftFar", "MemFruitPickRightFar", "MemFruitPickLeftFar"]
    if (task_name in retrieve_fruit_list) or (task_name in retrieve_oils_list):
        explore_policy: Optional[LRExplorationPolicy | RotateExplorationPolicy] = None
        if task_name in retrieve_oils_list:
            print("*"*40)
            print(f"Using LRExplorationPolicy for {task_name}")
            explore_policy = LRExplorationPolicy(env)
        elif task_name in retrieve_fruit_list:
            print("*"*40)
            print(f"Using RotateExplorationPolicy for {task_name}")
            explore_policy = RotateExplorationPolicy(env)
        steps, rollout_imgs = explore_eval_loop(
            keys_info=keys_info,
            args=args,
            explore_policy=explore_policy,
            model=model,
            env=env,
            rollout_imgs=rollout_imgs,
            video_transform=video_transform,
            pbar=pbar,
            data_trajectory=data_trajectory,
            gt_explore_actions=gt_explore_actions,
        )

    # set everything for use_gt mode
    teacher_forcing = False #args.eval_cfg.use_gt
    if args.eval_cfg.use_gt:
        assert gt_teleop_actions is not None, "No ground truth teleop actions provided"

    # initializations for the loop
    success = False
    device = torch.device('cuda')

    # get the new observation
    obs = env._get_observations(force_update=True)
    if args.eval_cfg.debug:
        max_steps = steps + 10

    while steps < max_steps:

        # add any images to save
        if args.eval_cfg.save_videos:
            rollout_imgs.append(casa_utils.get_images_to_save(env, keys_info))

        # preprocess the observation
        image, proprio = online_obs_processing(obs, keys_info, args=args, device=device, video_transform=video_transform)

        assert not teacher_forcing, "Teacher forcing is not supported for action prediction training"

        # predict the action: add batch dimension.
        # during eval, it is a list of dictionary that we pass
        with torch.amp.autocast('cuda', dtype=misc_utils.convert_str_to_torch_dtype(args.shared_cfg.compute_dtype)):
            pred_action, _ = model.forward_inference(
                sequences=[{
                    'observation': image,
                    'proprio': proprio,
                    'action': None,
                }]
            )
            # B, T=1, self.action_dim
            pred_action = pred_action.detach().cpu().float()
            pred_action = pred_action.squeeze(0).squeeze(0).numpy()

        # check if ground truth actions should be used
        if args.eval_cfg.use_gt:
            if pred_steps >= len(gt_teleop_actions): # break if we have reached the end of the ground truth actions
                break
            pred_action = gt_teleop_actions[pred_steps]

        # take a step in the environment
        obs, reward, done, info = env.step(pred_action)

        # check success
        success = env._check_success()

        # save the data to the trajectory
        if data_trajectory is not None:
            # this will be raw observations, flip images inside the add_step function
            data_trajectory.add_step(obs, policy_mode=0, action=pred_action, reward=reward, done=done, success=success, flip_images=True)

        # update and render
        pbar.update(1)
        if args.eval_cfg.render:
            env.render()

        steps += 1
        pred_steps += 1
        if success:
            break

    if data_trajectory is not None:
        print(f"[Rank {rank}] Saving trajectory data (episode_{i_eval})...", flush=True)
        data_trajectory.reset(episode_name=f"episode_{i_eval}")
        print(colored(f"[Rank {rank}] Trajectory data saved successfully", "green"), flush=True)

    if args.eval_cfg.save_videos:
        language_img = misc_utils.create_language_image(language[0], rollout_imgs[0].shape, font_size=28)
        for _ in range(10):
            rollout_imgs.append(language_img)

        if success:
            # add a green image to the rollout_imgs
            green_img = np.zeros(rollout_imgs[0].shape, dtype=np.uint8)
            green_img[:, :] = [0, 255, 0]
            rollout_imgs.append(green_img)
        else:
            red_img = np.zeros(rollout_imgs[0].shape, dtype=np.uint8)
            red_img[:, :] = [255, 0, 0]
            rollout_imgs.append(red_img)

    video_transform.train()
    model.train()
    model.reset(batch_size=1)
    return {
        'success': success,
        'steps': steps,
        'rollout_imgs': rollout_imgs,
        'ep_meta': ep_meta,
    }


def perform_env_evals(model: MemModel, args: ExperimentConfig, video_transform: MultiViewTorchDictTransform, epoch_num: int, task_name: str | None = None):
    task_result = {}

    # load the dataset_train
    args.eval_cfg.latest_epoch = epoch_num

    dataset_train_json = args.dataset_cfg.dataset_json[0] if isinstance(args.dataset_cfg.dataset_json, list) else args.dataset_cfg.dataset_json

    # gather all the keys
    dataset_train_json = json.load(open(dataset_train_json, "r"))
    env_files = dataset_train_json['dataset_path']
    image_keys = dataset_train_json['image_keys']
    proprio_keys = dataset_train_json['proprio_keys']
    action_keys = dataset_train_json['action_keys']
    keys_info = {
        'image_keys': image_keys,
        'proprio_keys': proprio_keys,
        'action_keys': action_keys,
    }

    # result path is the path to save the results
    result_path, filename = eval_utils.get_filename_from_args(args)

    # group the env files by task
    env_files = casa_utils.casa_group_env_files_from_path(env_files)
    if task_name is not None:
        assert task_name in env_files, f"Task {task_name} not found in env_files"
        env_files = {task_name: env_files[task_name]}

    # perform the evals for each task
    for task_name, env_files_list in env_files.items():

        # find the n_eval for the task
        n_eval = args.eval_cfg.n_eval
        if n_eval == -1:
            n_eval = casa_utils.DEFAULT_TASK_NAME_TO_N_EVALS[task_name]

        # fix the path of the env files
        env_files_list = [os.path.join(misc_utils.get_data_base_dir(hdf5_path, shared_config=args.shared_cfg), hdf5_path) for hdf5_path in env_files_list]
        if args.eval_cfg.skip_n_hdf5 > 0:
            env_files_list = env_files_list[args.eval_cfg.skip_n_hdf5:]
            if len(env_files_list) == 0:
                print(
                    colored(
                        f"[Rank {misc_utils.get_rank()}] Skipped all env files for task {task_name} "
                        f"(skip_n_hdf5={args.eval_cfg.skip_n_hdf5}), skipping task",
                        "yellow"
                    )
                )
                continue

        # skip if result path already exists
        if eval_utils.verify_result_path(result_path, task_name) and not args.eval_cfg.redo_evals:
            print(colored(f"[Rank {misc_utils.get_rank()}] Result path {result_path} has task {task_name}, skipping", "yellow"))
            continue

        # ep_meta is used to set the environment to the correct episode
        # env_state maybe used to reset the environment if train_traj is True
        ep_metas = []
        env_states = []
        for env_file in env_files_list:
            ep_metas.extend(casa_utils.get_ep_metadata_from_dataset(env_file, key=None, mode='all'))
            env_states.extend(casa_utils.get_env_states_from_dataset(env_file, key=None, mode='all'))

        # make the environment with the current task
        env, env_kwargs = casa_utils.make_env(
            env_files_list[0], # task is the same everywhere.
            keys_info,
            args.eval_cfg
        )
        env.set_ep_meta(ep_metas[0])

        # video filename
        rollout_imgs = []
        video_filename = None
        if args.eval_cfg.save_videos:
            video_filename = result_path.replace(".csv", f"_{task_name}.mp4")
            if args.eval_cfg.store_video_from_every_rank:
                # add rank000.mp4 to the file name
                video_filename = video_filename.replace(".mp4", f"_r{misc_utils.get_rank():03d}.mp4")
            # add a videos dir
            video_base_dir = os.path.dirname(video_filename)
            video_filename = os.path.join(video_base_dir, "videos", video_filename.split("/")[-1])
            os.makedirs(os.path.dirname(video_filename), exist_ok=True)

        # rollout, performance evaluation, and save videos if needed
        num_evals = 0
        num_success = 0
        rank = misc_utils.get_rank()
        world_size = misc_utils.get_world_size()
        rollout_indices = list(range(rank, n_eval, world_size))
        for i_eval in rollout_indices:
            print(f"[Rank {rank}] Starting rollout {i_eval + 1}/{n_eval} (local {rollout_indices.index(i_eval) + 1}/{len(rollout_indices)})", flush=True)
            output_dict = do_rollout(
                i_eval=i_eval,
                args=args,
                env=env,
                model=model,
                video_transform=video_transform,
                keys_info=keys_info,
                rollout_imgs=rollout_imgs, # in-place
                ep_metas=ep_metas,
                env_states=env_states,
                task_name=task_name,
            )
            num_success += int(output_dict['success'])
            num_evals += 1
            print(f"[Rank {rank}] Completed rollout {i_eval + 1}, success={output_dict['success']}", flush=True)

        # close the environment
        env.close()

        # communicate the results to DDP = 1 if needed. everything rest should be done by rank 0.
        if world_size > 1:
            # Sync all processes before collective operations to prevent NCCL timeouts
            # This ensures all ranks have finished their rollouts before attempting all_reduce
            print(f"[Rank {rank}] Finished all rollouts. {num_success=}, {num_evals=}, syncing before all_reduce...", flush=True)
            misc_utils.sync_all_processes()
            print(colored(f"[Rank {rank}] Successfully synced, proceeding with all_reduce...", "green"), flush=True)

            # Create CUDA tensors for all_reduce
            num_success_tensor = torch.tensor(num_success, dtype=torch.int64, device='cuda')
            num_evals_tensor = torch.tensor(num_evals, dtype=torch.int64, device='cuda')

            num_success = misc_utils.all_reduce_sum(num_success_tensor).cpu().item()
            num_evals = misc_utils.all_reduce_sum(num_evals_tensor).cpu().item()
            print(colored(f"After all reduce: {num_success=}, {num_evals=} for {misc_utils.get_rank()=}", "green"), flush=True)

        # save videos if needed
        if args.eval_cfg.save_videos and (misc_utils.is_main_process() or args.eval_cfg.store_video_from_every_rank):
            print(colored("saving video to: {}".format(video_filename), "yellow"))
            eval_utils.save_rollout_video(rollout_imgs, video_filename, fps=30, dowsample_factor=args.eval_cfg.video_downsample_factor)

        # save the task results to the pandas dataframe
        task_result = {
            'task_name': task_name,
            'n_eval': n_eval,
            'success_rate': num_success/n_eval,
        }

        # save to csv
        if (not args.eval_cfg.no_save_csv) and misc_utils.is_main_process():
            eval_utils.save_task_result(task_result, result_path)

        # save to wandb
        if args.logging_cfg.log_name is not None and misc_utils.is_main_process() and args.eval_cfg.save_wandb:
            eval_utils.save_wandb_result(task_result, result_path, rollout_imgs=rollout_imgs, args=args)

        # sync the results to all processes
        if world_size > 1:
            misc_utils.sync_all_processes()

    return task_result
