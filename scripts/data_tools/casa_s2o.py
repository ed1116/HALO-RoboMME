
"""
Script to extract observations from low-dimensional simulation states in a robocasa dataset.
Adapted from robomimic's dataset_states_to_obs.py script.
"""
import os
import json
import h5py
import argparse
from tqdm import tqdm
import numpy as np
from copy import deepcopy
import queue
import time
import traceback
from PIL import Image
import torch
from termcolor import colored
import xml.etree.ElementTree as ET

from robocasa import macros
import robocasa.utils.robomimic.robomimic_tensor_utils as TensorUtils
import robocasa.utils.robomimic.robomimic_env_utils as EnvUtils
import robocasa.utils.robomimic.robomimic_dataset_utils as DatasetUtils
from robocasa.environments.kitchen.memory.memory_env import MemWashAndReturnLeft, MemWashAndReturnRight 

from halo.util.bbox_utils import get_all_bbox_observations

# from robomimic.utils.log_utils import log_warning
macros.SHOW_SITES = False
assert macros.SHOW_SITES == False, "SHOW_SITES should be False"

def is_none_object_array(x) -> bool:
    return (
        isinstance(x, np.ndarray)
        and x.dtype == object
        and x.size == 1
        and x.item() is None
    )

def extract_trajectory(
    env,
    initial_state,
    states,
    actions,
    done_mode,
    add_datagen_info=False,
    policy_mode=None,
):
    """
    Helper function to extract observations, rewards, and dones along a trajectory using
    the simulator environment.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load to extract information
        actions (np.array): array of actions
        done_mode (int): how to write done signal. If 0, done is 1 whenever s' is a
            success state. If 1, done is 1 at the end of each trajectory.
            If 2, do both.
    """
    assert states.shape[0] == actions.shape[0]

    # load the initial state
    # try:
    #     env.reset()
    # except Exception as e:
    #     print("Error loading initial state")
    #     print(e)
    #     pass
    if isinstance(env.env, MemWashAndReturnLeft) or isinstance(env.env, MemWashAndReturnRight):
        print(colored("Overwriting the mjcf_path of the plates for the env instance: ", "red"))
        print(colored("Removed this if not needed", "red"))
        print("Overwriting the mjcf_path of the plates for the env instance: ", env.env)
        ep_meta_in_init_state = json.loads(initial_state["ep_meta"])
        obj_cfgs = ep_meta_in_init_state["object_cfgs"]
        # always pick the fruit_container index
        indices = [i for i, obj_cfg in enumerate(obj_cfgs) if obj_cfg["name"] in ["fruit_container", "fruit_container2"]]
        fruit_container_index = [i for i, obj_cfg in enumerate(obj_cfgs) if obj_cfg["name"] == "fruit_container"][0]
        fruit_container2_index = [i for i, obj_cfg in enumerate(obj_cfgs) if obj_cfg["name"] == "fruit_container2"][0]
        selected_mjcf_path = obj_cfgs[fruit_container_index]['info']['mjcf_path']
        obj_cfgs[fruit_container2_index]['info']['mjcf_path'] = selected_mjcf_path
        ep_meta_in_init_state['object_cfgs'] = obj_cfgs
        initial_state['ep_meta'] = json.dumps(ep_meta_in_init_state, indent=4)

        model_xml = initial_state['model']
        # find the block of text block in model xml that contains the fruit_container2 and replace it with the fruit_container
        # we cannot just replace the strings. we need to load the block and replace it with the fruit_container
        model_xml_loaded = ET.fromstring(model_xml)
        # # write the model xml to a file
        # ET.ElementTree(model_xml_loaded).write("model_xml_loaded.xml", encoding='utf8', method='xml')
        # Copy geoms from fruit_container_main to fruit_container2_main with name replacement
        fruit_container2_body = model_xml_loaded.find(".//body[@name='fruit_container2_main']")
        fruit_container_body = model_xml_loaded.find(".//body[@name='fruit_container_main']")
        if fruit_container2_body is not None and fruit_container_body is not None:
            # Remove existing geoms from fruit_container2_main
            for geom in fruit_container2_body.findall("geom"):
                fruit_container2_body.remove(geom)
            # Copy geoms from fruit_container_main and replace names
            for geom in fruit_container_body.findall("geom"):
                geom_copy = ET.fromstring(ET.tostring(geom))
                geom_copy.set("name", geom_copy.get("name").replace("fruit_container_", "fruit_container2_"))
                fruit_container2_body.append(geom_copy)
        # # write the model xml to a file
        # ET.ElementTree(model_xml_loaded).write("model_xml_loaded_2.xml", encoding='utf8', method='xml')
        model_xml = ET.tostring(model_xml_loaded, encoding='utf8', method='xml')
        initial_state['model'] = model_xml
    obs = env.reset_to(initial_state)

    ep_meta = json.loads(initial_state["ep_meta"])
    # hack: add the cam configs in, since it's been modified
    ep_meta["cam_configs"] = deepcopy(env.env._cam_configs)
    ep_meta["lang"] = env.env.get_ep_meta()['lang'] # this is only to make sure that the language is overwritten to the latest version
    print("New lang: ", ep_meta["lang"])
    initial_state["ep_meta"] = json.dumps(ep_meta, indent=4)

    traj = dict(
        obs=[],
        next_obs=[],
        rewards=[],
        dones=[],
        actions=np.array(actions),
        # actions_abs=[],
        states=np.array(states),
        initial_state_dict=initial_state,
        datagen_info=[],
        policy_mode=policy_mode,
    )
    traj_len = states.shape[0]
    # iteration variable @t is over "next obs" indices

    name2id = {inst: i for i, inst in enumerate(list(env.env.model.instances_to_ids.keys()))}
    id2name = {v: k for k, v in name2id.items()}
    for t in tqdm(range(traj_len)):
        obs = deepcopy(env.reset_to({"states": states[t]}))
        bbox_obs = get_all_bbox_observations(obs, id2name)
        for k, v in bbox_obs.items():
            obs[k] = v

        # use matplotlib to plot all the images in the obs in the order of the camera names concatenated.
        # print("obs keys: ", obs.keys())
        # import matplotlib.pyplot as plt
        # img = np.concatenate([obs[cam_name+"_image"] for cam_name in args.camera_names], axis=1)
        # plt.imshow(img)
        # plt.savefig("obs_images.png")
        # plt.close()

        # extract datagen info
        if add_datagen_info:
            datagen_info = env.base_env.get_datagen_info(action=actions[t])
        else:
            datagen_info = {}

        # infer reward signal
        # note: our tasks use reward r(s'), reward AFTER transition, so this is
        #       the reward for the current timestep
        r = env.get_reward()
        # joint_name = "cab_2_left_group_doorhinge"
        # joint_idx = env.env.sim.model.joint_names.index(joint_name)
        # joint_value = env.env.sim.get_state().qpos[env.env.sim.model.jnt_qposadr[joint_idx]]

        # infer done signal
        done = False
        if (done_mode == 1) or (done_mode == 2):
            # done = 1 at end of trajectory
            done = done or (t == traj_len)
        if (done_mode == 0) or (done_mode == 2):
            # done = 1 when s' is task success state
            done = done or env.is_success()["task"]
        done = int(done)

        # get the absolute action
        # action_abs = env.base_env.convert_rel_to_abs_action(actions[t])

        # collect transition
        traj["obs"].append(obs)
        traj["rewards"].append(r)
        traj["dones"].append(done)
        traj["datagen_info"].append(datagen_info)
        # traj["actions_abs"].append(action_abs)

    is_success = env.is_success()["task"]
    print(colored("Episode is successful: {}".format(is_success), "green"))
    # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
    traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
    traj["datagen_info"] = TensorUtils.list_of_flat_dict_to_dict_of_list(
        traj["datagen_info"]
    )

    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return traj, is_success


""" The process that writes over the generated files to memory """


def write_traj_to_file(
    args, output_path, mul_queue, env_meta
):
    f = h5py.File(args.dataset, "r")
    f_out = h5py.File(output_path, "w")
    data_grp = f_out.create_group("data")
    start_time = time.time()
    num_processed = 0
    total_samples = 0

    try:
        print("Starting to write to file")
        for item in tqdm(mul_queue):
            num_processed = num_processed + 1
            ep = item[0]
            env_name = env_meta["env_kwargs"]["env_name"]
            robot_name = env_meta["env_kwargs"]["robots"][0]
            ep = "{}_{}_{}".format(env_name, robot_name, ep)
            traj = item[1]
            process_num = item[2]
            # try:
            print("[debug]: writing episode {}".format(ep))
            ep_data_grp = data_grp.create_group(ep)
            ep_data_grp.create_dataset(
                "actions", data=np.array(traj["actions"])
            )
            ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            ep_data_grp.create_dataset(
                "rewards", data=np.array(traj["rewards"])
            )
            ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
            # ep_data_grp.create_dataset(
            #     "actions_abs", data=np.array(traj["actions_abs"])
            # )
            for k in traj["obs"]:
                if args.no_compress:
                    ep_data_grp.create_dataset(
                        "obs/{}".format(k), data=np.array(traj["obs"][k])
                    )
                else:
                    ep_data_grp.create_dataset(
                        "obs/{}".format(k),
                        data=np.array(traj["obs"][k]),
                        compression="gzip",
                    )
                if args.include_next_obs:
                    if args.no_compress:
                        ep_data_grp.create_dataset(
                            "next_obs/{}".format(k),
                            data=np.array(traj["next_obs"][k]),
                        )
                    else:
                        ep_data_grp.create_dataset(
                            "next_obs/{}".format(k),
                            data=np.array(traj["next_obs"][k]),
                            compression="gzip",
                        )

            if "datagen_info" in traj:
                for k in traj["datagen_info"]:
                    ep_data_grp.create_dataset(
                        "datagen_info/{}".format(k),
                        data=np.array(traj["datagen_info"][k]),
                    )

            # copy action dict (if applicable)
            if "data/{}/action_dict".format(ep) in f:
                action_dict = f["data/{}/action_dict".format(ep)]
                for k in action_dict:
                    ep_data_grp.create_dataset(
                        "action_dict/{}".format(k),
                        data=np.array(action_dict[k][()]),
                    )

            if ('policy_mode' in traj) and (not is_none_object_array(traj["policy_mode"])):
                ep_data_grp.create_dataset(
                    "policy_mode",
                    data=np.array(traj["policy_mode"]),
                )

            print(ep_data_grp["obs"].keys())
            # episode metadata
            ep_data_grp.attrs["model_file"] = traj["initial_state_dict"][
                "model"
            ]  # model xml for this episode
            ep_data_grp.attrs["ep_meta"] = traj["initial_state_dict"][
                "ep_meta"
            ]  # ep meta data for this episode
            # if "ep_meta" in f["data/{}".format(ep)].attrs:
            #     ep_data_grp.attrs["ep_meta"] = f["data/{}".format(ep)].attrs["ep_meta"]
            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[
                0
            ]  # number of transitions in this episode
            if traj["initial_qpos"] is not None:
                ep_data_grp.attrs["initial_qpos"] = traj["initial_qpos"]
            if traj["non_robot_qpos_idx"] is not None:
                ep_data_grp.attrs["non_robot_qpos_idx"] = traj["non_robot_qpos_idx"]
            if ("ep_meta_info" in traj) and (traj["ep_meta_info"] is not None):
                ep_data_grp.attrs["ep_meta_info"] = json.dumps(
                    traj["ep_meta_info"], indent=4
                )

            total_samples += traj["actions"].shape[0]
            # except Exception as e:
            #     print("++" * 50)
            #     print(
            #         f"Error at Process {process_num} on episode {ep} with \n\n {e}"
            #     )
            #     print("++" * 50)
            #     raise Exception("Write out to file has failed")
            print(
                "ep {}: wrote {} transitions to group {} at process {} with {} finished. Datagen rate: {:.2f} sec/demo".format(
                    num_processed,
                    ep_data_grp.attrs["num_samples"],
                    ep,
                    process_num,
                    num_processed,
                    (time.time() - start_time) / num_processed,
                )
            )
    except KeyboardInterrupt:
        print("Control C pressed. Closing File and ending \n\n\n\n\n\n\n")

    if "mask" in f:
        f.copy("mask", f_out)

    # global metadata
    data_grp.attrs["total"] = total_samples
    env_meta = DatasetUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    if args.generative_textures:
        env_meta["env_kwargs"]["generative_textures"] = "100p"
    if args.randomize_cameras:
        env_meta["env_kwargs"]["randomize_cameras"] = True
    env_meta['env_kwargs']['translucent_robot'] = 0.00 if args.transparent_robot else False
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=args.camera_names,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        reward_shaping=args.shaped,
        segmentation_level="instance",
    )
    data_grp.attrs["env_args"] = json.dumps(
        env.serialize(), indent=4
    )  # environment info

    f_out.close()
    f.close()
    print("Wrote all samples samples to {}".format(output_path))

    print("Extracting action dict")
    DatasetUtils.extract_action_dict(dataset=output_path)
    return

# runs multiple trajectory. If there has been an unrecoverable error, the system puts the current work back into the queue and exits
def extract_multiple_trajectories_with_error(
    process_num, args
):
    # create environment to use for data processing

    mul_queue = []
    if args.add_datagen_info:
        import mimicgen.utils.file_utils as MG_FileUtils

        env_meta = MG_FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    else:
        env_meta = DatasetUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    if args.generative_textures:
        env_meta["env_kwargs"]["generative_textures"] = "100p"
    if args.randomize_cameras:
        env_meta["env_kwargs"]["randomize_cameras"] = True
    env_meta['env_kwargs']['translucent_robot'] = 0.00 if args.transparent_robot else False
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=args.camera_names,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        reward_shaping=args.shaped,
        segmentation_level="instance",
    )

    start_time = time.time()

    print("==== Using environment with the following metadata ====")
    print(json.dumps(env.serialize(), indent=4))
    print("")

    # list of all demonstration episodes (sorted in increasing number order)
    f = h5py.File(args.dataset, "r")
    if args.filter_key is not None:
        print("using filter key: {}".format(args.filter_key))
        demos = [
            elem.decode("utf-8")
            for elem in np.array(f["mask/{}".format(args.filter_key)])
        ]
    else:
        demos = list(f["data"].keys())
    print("Number of demos: {}, {}".format(len(demos), demos))
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        demos = demos[: args.n]

    for ind in range(len(demos)):
        # try:
        print("Running {}/{}".format(ind, len(demos)))
        ep = demos[ind]

        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
        initial_state["ep_meta"] = f["data/{}".format(ep)].attrs.get(
            "ep_meta", None
        )

        # extract obs, rewards, dones
        actions = f["data/{}/actions".format(ep)][()]
        policy_modes = f["data/{}/policy_mode".format(ep)][()] if "policy_mode" in f["data/{}".format(ep)] else None

        traj, is_success = extract_trajectory(
            env=env,
            initial_state=initial_state,
            states=states,
            actions=actions,
            done_mode=args.done_mode,
            add_datagen_info=args.add_datagen_info,
            policy_mode=policy_modes,
        )
        if (not is_success) and (args.skip_unsuccessful):
            print(colored(f"Episode {ep} is not successful. Skipping...", "red"))
            continue
        traj["non_robot_qpos_idx"] = f["data/{}".format(ep)].attrs.get(
            "non_robot_qpos_idx", None
        )
        traj["initial_qpos"] = f["data/{}".format(ep)].attrs.get("initial_qpos")
        traj["ep_meta_info"] = f["data/{}".format(ep)].attrs.get("ep_meta_info", None)
        print(f"[debug] {actions.shape}")

        # maybe copy reward or done signal from source file
        if args.copy_rewards:
            traj["rewards"] = f["data/{}/rewards".format(ep)][()]
        if args.copy_dones:
            traj["dones"] = f["data/{}/dones".format(ep)][()]

        ep_grp = f["data/{}".format(ep)]

        states = ep_grp["states"][()]
        initial_state = dict(states=states[0])
        initial_state["model"] = ep_grp.attrs["model_file"]
        initial_state["ep_meta"] = ep_grp.attrs.get("ep_meta", None)

        # store transitions

        # IMPORTANT: keep name of group the same as source file, to make sure that filter keys are
        #            consistent as well
        mul_queue.append([ep, traj, process_num])

    f.close()
    print("Process {} finished".format(process_num))
    return mul_queue



def dataset_states_to_obs_single_process(args):
    # create environment to use for data processing

    # output file in same directory as input file
    output_name = args.output_name
    if output_name is None:
        if len(args.camera_names) == 0:
            output_name = os.path.basename(args.dataset)[:-5] + "_ld.hdf5"
        else:
            image_suffix = str(args.camera_width)
            image_suffix = (
                image_suffix + "_randcams" if args.randomize_cameras else image_suffix
            )
            if args.generative_textures:
                output_name = os.path.basename(args.dataset)[
                    :-5
                ] + "_gentex_im{}.hdf5".format(image_suffix)
            elif not args.transparent_robot:
                output_name = os.path.basename(args.dataset)[:-5] + "_im{}_notp.hdf5".format(
                    image_suffix
                )
            else:
                output_name = os.path.basename(args.dataset)[:-5] + "_im{}.hdf5".format(
                    image_suffix
                )
    if args.name_suffix != "":
        output_name = output_name.replace(".hdf5", "_" + args.name_suffix + ".hdf5")
    print(f"[debug] output_name: {output_name}")

    output_path = os.path.join(os.path.dirname(args.dataset), output_name)

    print("input file: {}".format(args.dataset))
    print("output file: {}".format(output_path))

    # f = h5py.File(args.dataset, "r")
    # if args.filter_key is not None:
    #     print("using filter key: {}".format(args.filter_key))
    #     demos = [
    #         elem.decode("utf-8")
    #         for elem in np.array(f["mask/{}".format(args.filter_key)])
    #     ]
    # else:
    #     demos = list(f["data"].keys())
    # inds = np.argsort([int(elem[5:]) for elem in demos])
    # demos = [demos[i] for i in inds]
    # if args.n is not None:
    #     demos = demos[:args.n]
    # num_demos = len(demos)
    # f.close()

    env_meta = DatasetUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    mul_queue = extract_multiple_trajectories_with_error(0, args)
    write_traj_to_file(
        args=args,
        output_path=output_path,
        mul_queue=mul_queue,
        env_meta=env_meta,
    )
    print("Finished Processing")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to input hdf5 dataset",
    )
    # name of hdf5 to write - it will be in the same directory as @dataset
    parser.add_argument(
        "--output_name",
        type=str,
        help="name of output hdf5 dataset",
    )

    parser.add_argument(
        "--filter_key",
        type=str,
        help="filter key for input dataset",
    )

    # specify number of demos to process - useful for debugging conversion with a handful
    # of trajectories
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are processed",
    )

    # flag for reward shaping
    parser.add_argument(
        "--shaped",
        action="store_true",
        help="(optional) use shaped rewards",
    )

    # camera names to use for observations
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs="+",
        default=[
            # "robot0_agentview_left",
            # "robot0_agentview_right",
            "robot0_agentview_center",
            "robot0_eye_in_hand",
        ],
        help="(optional) camera name(s) to use for image observations. Leave out to not use image observations.",
    )

    parser.add_argument(
        "--camera_height",
        type=int,
        default=128,
        help="(optional) height of image observations",
    )

    parser.add_argument(
        "--camera_width",
        type=int,
        default=128,
        help="(optional) width of image observations",
    )
    parser.add_argument(
        "--name_suffix",
        type=str,
        default="",
        help="(optional) suffix to add to output name",
    )

    # specifies how the "done" signal is written. If "0", then the "done" signal is 1 wherever
    # the transition (s, a, s') has s' in a task completion state. If "1", the "done" signal
    # is one at the end of every trajectory. If "2", the "done" signal is 1 at task completion
    # states for successful trajectories and 1 at the end of all trajectories.
    parser.add_argument(
        "--done_mode",
        type=int,
        default=0,
        help="how to write done signal. If 0, done is 1 whenever s' is a success state.\
            If 1, done is 1 at the end of each trajectory. If 2, both.",
    )

    # flag for copying rewards from source file instead of re-writing them
    parser.add_argument(
        "--copy_rewards",
        action="store_true",
        help="(optional) copy rewards from source file instead of inferring them",
    )

    # flag for copying dones from source file instead of re-writing them
    parser.add_argument(
        "--copy_dones",
        action="store_true",
        help="(optional) copy dones from source file instead of inferring them",
    )

    # flag to include next obs in dataset
    parser.add_argument(
        "--include-next-obs",
        action="store_true",
        help="(optional) include next obs in dataset",
    )

    # flag to disable compressing observations with gzip option in hdf5
    parser.add_argument(
        "--no_compress",
        action="store_true",
        help="(optional) disable compressing observations with gzip option in hdf5",
    )


    parser.add_argument(
        "--add_datagen_info",
        action="store_true",
        help="(optional) add datagen info (used for mimicgen)",
    )

    parser.add_argument("--transparent_robot", action="store_true")

    parser.add_argument("--generative_textures", action="store_true")

    parser.add_argument("--randomize_cameras", action="store_true")

    parser.add_argument("--skip_unsuccessful", action="store_true", help="skip unsuccessful episodes")

    args = parser.parse_args()
    dataset_states_to_obs_single_process(args)
