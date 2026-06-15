# Description: Utility functions for loading and processing data from the Casa dataset
from termcolor import colored
import robosuite
import numpy as np
import os
import json
import h5py
from collections import OrderedDict
import robocasa
from robocasa.utils.robomimic.robomimic_env_wrapper import fix_asset_paths_relative_to_robocasa

if robosuite.__version__ > "1.4.0":
    from robosuite.controllers import load_composite_controller_config
    from robosuite.controllers import load_part_controller_config

TASK_NAME_TO_IMP_OBJ_MAP = {
    'MemRetrieveOilsFromCounterLL': 'olive_oil',
    'MemRetrieveOilsFromCounterLR': 'olive_oil',
    'MemRetrieveOilsFromCounterRL': 'olive_oil',
    'MemRetrieveOilsFromCounterRR': 'olive_oil',
    'MemWashAndReturnLeft': 'fruit_container',
    'MemWashAndReturnRight': 'fruit_container',
}

DEFAULT_TASK_NAME_TO_N_EVALS = {
    'MemRetrieveOilsFromCounterLL': 15,
    'MemRetrieveOilsFromCounterLR': 15,
    'MemRetrieveOilsFromCounterRL': 15,
    'MemRetrieveOilsFromCounterRR': 15,
    'MemWashAndReturnLeft': 25,
    'MemWashAndReturnRight': 25,
    'MemHeatPot': 50,
    'MemHeatPotMultiple': 50,
    'MemHeatPotLong': 50,
    'MemPutKBowlInCabinet': 50,
    'MemPutKBreadInMicrowave': 50,
    'MemFruitInSinkRightFar': 25,
    'MemFruitInSinkLeftFar': 25,
    'MemFruitPickRightFar': 25,
    'MemFruitPickLeftFar': 25,
}

TASK_DESCRIPTION_DICT = {
    "MemWashAndReturn": "There are two containers in the scene with one of them containing a fruit. The robot picks up a fruit from a specific container (the original container). It places the fruit into the sink, positioning it directly under the faucet. The robot waits for a short duration to simulate washing. After waiting, it picks up the same fruit from the sink. It then places the fruit back into the same original container it was taken from. The task is successful only if the fruit is returned to its same original container and not a different container.",
    "MemRetrieveOilsFromCounter": "The robot first explores the environment by moving left and right to observe the locations of objects on the counter. After completing the exploration, the robot returns to its initial position in front of the stove. Using only the information observed during exploration, it must decide which direction to move to reach the olive oil bottle. The robot then moves in the chosen direction and picks up the olive oil bottle. The task is successful only if the robot moves in the correct direction and retrieves the olive oil bottle.",
    "MemPutKBreadInMicrowave": "The robot may see one or more breads on the kitchen counter. It then picks up one bread at a time, and places it inside the microwave. The task is successful only if all the breads are placed inside the microwave before shutting the microwave door. If any of the breads are not placed, the task is unsuccessful.",
    "MemHeatPot": "The robot sees a container on the stove with meat on it. It turns on the stove. It waits for an appriate amount of time for the pot to heat up. After the specified amount of time, it turns off the stove. Only if the pot is turned off after the specified amount of time, the task is successful. The has to interact only with the stove knob.",
    "MemWashBowlAndReturn": "There are two plates in the scene with one of them containing a bowl. The robot first picks up the bowl from the plate (original plate). It places the bowl into the sink. After placing, it retracts its arm. Later, it picks up the same bowl from the sink. It then places the bowl back into the same original plate it was taken from. The task is successful only if the bowl is returned to its same original plate and not a different one.",
}

TASK_NAME_TO_GPT_OBJECT_SELECTION = {
    "MemPutKBreadInMicrowave": ["bread", "microwave"],
    "MemHeatPot": ["meat", "stove"],
    "MemHeatPotLong": ["meat", "stove"],
    "MemRetrieveOilsFromCounterLL": ["olive oil"],
    "MemRetrieveOilsFromCounterLR": ["olive oil"],
    "MemRetrieveOilsFromCounterRL": ["olive oil"],
    "MemRetrieveOilsFromCounterRR": ["olive oil"],
    "MemWashAndReturnLeft": ["fruit container", "fruit"],
    "MemWashAndReturnRight": ["fruit container", "fruit"],
    "MemWashBowlAndReturn": ["pink plate", "bowl", "green plate"],
}
TASK_NAME_TO_GPT_OBJECT_SELECTION["MEM1_Wash_the_bowl_and_place_it_back_in_the_same_container"] = TASK_NAME_TO_GPT_OBJECT_SELECTION["MemWashBowlAndReturn"]
TASK_NAME_TO_GPT_OBJECT_SELECTION["MEM2_Wash_the_bowl_and_place_it_back_in_the_same_container"] = TASK_NAME_TO_GPT_OBJECT_SELECTION["MemWashBowlAndReturn"]
TASK_NAME_TO_GPT_OBJECT_SELECTION["MEM3_Wash_the_bowl_and_place_it_back_in_the_same_container"] = TASK_NAME_TO_GPT_OBJECT_SELECTION["MemWashBowlAndReturn"]
TASK_NAME_TO_GPT_OBJECT_SELECTION["MEM4_Wash_the_bowl_and_place_it_back_in_the_same_container"] = TASK_NAME_TO_GPT_OBJECT_SELECTION["MemWashBowlAndReturn"]

TASK_NAME_TO_TASK_DESCRIPTION = {
    "MemWashAndReturnLeft": TASK_DESCRIPTION_DICT["MemWashAndReturn"],
    "MemWashAndReturnRight": TASK_DESCRIPTION_DICT["MemWashAndReturn"],
    "MemRetrieveOilsFromCounterLL": TASK_DESCRIPTION_DICT["MemRetrieveOilsFromCounter"],
    "MemRetrieveOilsFromCounterLR": TASK_DESCRIPTION_DICT["MemRetrieveOilsFromCounter"],
    "MemRetrieveOilsFromCounterRL": TASK_DESCRIPTION_DICT["MemRetrieveOilsFromCounter"],
    "MemRetrieveOilsFromCounterRR": TASK_DESCRIPTION_DICT["MemRetrieveOilsFromCounter"],
    "MemPutKBreadInMicrowave": TASK_DESCRIPTION_DICT["MemPutKBreadInMicrowave"],
    "MemHeatPot": TASK_DESCRIPTION_DICT["MemHeatPot"],
}

def convert_bbox_name_to_str(bbox_name: np.dtype[bytes]) -> str:
    bbox_name_str = bbox_name.decode("utf-8")
    bbox_name_str = bbox_name_str.replace("_", " ")
    # remove " group" from the end of the string
    if bbox_name_str.endswith(" island group"):
        bbox_name_str = bbox_name_str.replace(" island group", "")
    if bbox_name_str.endswith(" main group"):
        bbox_name_str = bbox_name_str.replace(" main group", "")
    if bbox_name_str.endswith(" main group 1"):
        bbox_name_str = bbox_name_str.replace(" main group 1", "")
    if bbox_name_str.endswith(" main group 2"):
        bbox_name_str = bbox_name_str.replace(" main group 2", "")
    if bbox_name_str.endswith(" main group 3"):
        bbox_name_str = bbox_name_str.replace(" main group 3", "")
    if bbox_name_str.endswith(" main group 4"):
        bbox_name_str = bbox_name_str.replace(" main group 4", "")
    if bbox_name_str.endswith(" main group 5"):
        bbox_name_str = bbox_name_str.replace(" main group 5", "")
    if bbox_name_str == "top" or bbox_name_str == "bottom":
        bbox_name_str = None
    if bbox_name_str is not None and bbox_name_str.endswith(" housing"):
        bbox_name_str = None

    # if the name name ends with either 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, then remove the number
    if bbox_name_str is not None and bbox_name_str.endswith((" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9")):
        bbox_name_str = bbox_name_str[:-2]
    return bbox_name_str

def filter_bbox_names(bbox_names, task_name):
    if task_name not in TASK_NAME_TO_GPT_OBJECT_SELECTION:
        print(colored(f"Task name {task_name} not found in TASK_NAME_TO_GPT_OBJECT_SELECTION", "red"))
        return bbox_names
    assert task_name in TASK_NAME_TO_GPT_OBJECT_SELECTION, f"Task name {task_name} not found in TASK_NAME_TO_GPT_OBJECT_SELECTION"
    object_selection = TASK_NAME_TO_GPT_OBJECT_SELECTION[task_name]
    # if the bbox name have any matching of the object strings; keep it, otherwise remove it
    filtered_bbox_names = [bbox_name for bbox_name in bbox_names if any(object_string in bbox_name for object_string in object_selection)]
    return filtered_bbox_names

def get_images_to_save(env, keys_info) -> np.ndarray:
    img_keys_in_obs = keys_info["image_keys"]
    render_list = [env.sim.render(width=128, height=128, camera_name=key.split('/')[-1].replace('_image', '')) for key in img_keys_in_obs]
    img_to_append = np.concatenate([render_img[:,:,:] for render_img in render_list], axis=1)
    return img_to_append.copy()[::-1]

def preprocess_casa_obs(obs, keys_info, args):
    proprio_keys = keys_info["proprio_keys"]
    gripper_qpos_key = "robot0_gripper_qpos"
    obs[f'{gripper_qpos_key}'] = obs[gripper_qpos_key][..., -1:]
    return obs

def casa_group_env_files_from_path(dataset_paths):
    def path_to_task_name(path):
        return path.split('/')[-3] # specific hack for dataset in a specific data directory
    # sorting is important to maintain consistency across ranks
    task_names = sorted(set([path_to_task_name(dataset_path) for dataset_path in dataset_paths]))
    env_files = OrderedDict({})
    for task_name in task_names:
        env_files[task_name] = [dataset_path for dataset_path in dataset_paths if task_name == path_to_task_name(dataset_path)]
    return env_files

def get_ep_metadata_from_dataset(dataset_path, key=None, mode='first'):
    def get_ep_meta_from_hdf5(f, key, dataset_path):
        if "data" in f:
            data = f["data"]
        else:
            data = f
        ep_meta = data[key].attrs["ep_meta"]
        if isinstance(ep_meta, str):
            ep_meta = json.loads(ep_meta)
        action_len = data[key]['actions'].shape[0]
        ep_meta['action_len'] = action_len
        ep_meta['dataset_path'] = dataset_path
        ep_meta['demo_key'] = key
        return ep_meta

    with h5py.File(dataset_path, "r") as f:
        if key is None:
            keys = list(f['data'].keys())
            keys.sort(key=lambda x: int(x.split('_')[-1]))
            if mode == 'first':
                key = keys[0]
                ep_metas = [get_ep_meta_from_hdf5(f, key, dataset_path)]
            elif mode == 'all':
                ep_metas = [get_ep_meta_from_hdf5(f, k, dataset_path) for k in keys]
            else:
                raise NotImplementedError(f"Unknown mode: {mode}")
    return ep_metas

def get_exploratory_and_teleop_actions_from_dataset(dataset_path, key=None, mode='first'):
    def get_exploratory_and_teleop_actions_from_hdf5(f, key, dataset_path):
        action = f['data'][key]['actions'][()]
        policy_mode = np.zeros(action.shape[0], dtype=np.int32)
        if 'policy_mode' in f['data'][key]:
            policy_mode = f['data'][key]['policy_mode'][()]
        # any where policy_mode is 1, it is exploratory action, otherwise it is teleop action
        exploratory_actions = action[policy_mode == 1]
        teleop_actions = action[policy_mode == 0]
        return {
            'exploratory_actions': exploratory_actions,
            'teleop_actions': teleop_actions,
        }

    with h5py.File(dataset_path, "r") as f:
        if key is None:
            keys = list(f['data'].keys())
            keys.sort(key=lambda x: int(x.split('_')[-1]))
            if mode == 'first':
                key = keys[0]
                return get_exploratory_and_teleop_actions_from_hdf5(f, key, dataset_path)
            elif mode == 'all':
                return [get_exploratory_and_teleop_actions_from_hdf5(f, k, dataset_path) for k in keys]
            else:
                raise NotImplementedError(f"Unknown mode: {mode}")
        else:
            return get_exploratory_and_teleop_actions_from_hdf5(f, key, dataset_path)
    return None

def get_env_states_from_dataset(dataset_path, key=None, mode='first'):
    # get the first state from the dataset
    state_list: list[dict] = []
    def generate_state_dict(f, key, dataset_path):
        return {
            'states': f['data'][key]['states'][0],
            'model': f['data'][key].attrs["model_file"],
            'ep_meta': f['data'][key].attrs.get("ep_meta"),
            'dataset_path': dataset_path,
            'demo_key': key,
        }
    with h5py.File(dataset_path, "r") as f:
        if key is None:
            keys = list(f['data'].keys())
            keys.sort(key=lambda x: int(x.split('_')[-1]))
            if mode == 'first':
                key = keys[0]
                # we want to create state dict consisting of model, states, and ep_meta
                state_list.append(generate_state_dict(f, key, dataset_path))
            elif mode == 'all':
                for k in keys:
                    state_list.append(generate_state_dict(f, k, dataset_path))
            else:
                raise NotImplementedError(f"Unknown mode: {mode}")  
        else:
            state_list.append(generate_state_dict(f, key, dataset_path))
    return state_list

def get_env_args_from_dataset(dataset_path):
    dataset_path = os.path.expanduser(dataset_path)
    f = h5py.File(dataset_path, "r")
    env_args = json.loads(f["data"].attrs["env_args"]) if "data" in f else json.loads(f.attrs["env_args"])
    if isinstance(env_args, str):
        env_args = json.loads(env_args) # double leads to dict type
    f.close()
    # convert env_args to EasyDict
    return env_args

def get_env_meta_from_dataset(dataset_path, index=0):
    dataset_path = os.path.expanduser(dataset_path)
    f = h5py.File(dataset_path, "r")
    data = f["data"] if "data" in f else f
    keys = list(data.keys())
    env_meta = data[keys[index]].attrs["ep_meta"]
    env_meta = json.loads(env_meta)
    assert isinstance(env_meta, dict), f"Expected dict type but got {type(env_meta)}"
    return env_meta

def load_controller_config(controller, robot, control_type, ref_frame):
    if controller == "OSC_POSE":
        return load_part_controller_config(
            default_controller=controller,
        )
    if robot == "PandaOmron":
        raise ValueError(
            "Composite controller is not the one used for Robocasa default dataset. Use OSC_POSE for PandaOmron robots --controller OSC_POSE"
        )
    controller_config = load_composite_controller_config(
        controller=controller,
        robot=robot,
    )
    if "right" in controller_config["body_parts"]:
        controller_config["body_parts"]["right"]["input_type"] = control_type
        controller_config["body_parts"]["right"]["input_ref_frame"] = ref_frame
    if "left" in controller_config["body_parts"]:
        controller_config["body_parts"]["left"]["input_type"] = control_type
        controller_config["body_parts"]["left"]["input_ref_frame"] = ref_frame
    if ("WHOLE_BODY" in controller_config["type"]):
        controller_config['composite_controller_specific_configs']['ik_input_ref_frame'] = ref_frame
        controller_config['composite_controller_specific_configs']['ik_input_type'] = control_type
    return controller_config

def make_env(file_name, keys_info, args, env_meta=None):
    env_args = get_env_args_from_dataset(dataset_path=file_name)
    env_meta = get_env_meta_from_dataset(dataset_path=file_name, index=0)
    # f = h5py.File(file_name, "r")
    dataset_controller_config = env_args["env_kwargs"]["controller_configs"]
    if hasattr(args, "controller"):
        controller_config = load_controller_config(
            controller=args.controller,
            robot=args.robots if isinstance(args.robots, str) else args.robots[0],
            control_type=dataset_controller_config["body_parts"]["right"]["input_type"] if "body_parts" in dataset_controller_config else "delta",
            ref_frame=dataset_controller_config["body_parts"]["right"]["input_ref_frame"] if "body_parts" in dataset_controller_config else "base"
        )
    else:
        print("No controller specified. Using default controller from the dataset")
        controller_config = dataset_controller_config

    env_name = env_args["env_name"]
    # print("Env name: ", env_name)
    if hasattr(args, 'task_name') and args.task_name is not None:
        env_name = args.task_name
    env_kwargs = env_args["env_kwargs"]

    env_kwargs["env_name"] = env_name
    env_kwargs["ep_meta"] = env_meta # this should ideally reduce exploration for finding correct set of objects
    if hasattr(args, "robots"):
        env_kwargs["robots"] = args.robots if isinstance(args.robots, list) else [args.robots]
    env_kwargs["controller_configs"] = controller_config
    env_kwargs.pop("has_renderer", None)
    env_kwargs.pop("use_camera_obs", None)
    env_kwargs.pop("renderer", None)
    env_kwargs.pop("camera_segmentations", None)
    # print(f"Env args: {env_args}")

    # print(f"{args.render=}")
    control_freq = 20 if not hasattr(args, "control_freq") else args.control_freq
    # print(f"{control_freq=}")
    # print("Env kwargs:")
    # for key in env_kwargs:
    #     if isinstance(env_kwargs[key], dict):
    #         print(f"Key: {key}")
    #         for k in env_kwargs[key]:
    #             print(f"  {k}: {env_kwargs[key][k]}")
    #     else:
    #         print(f"Key: {key}, Value: {env_kwargs[key]}")
    # env_kwargs.pop("translucent_robot", None) ### TEMP: REMOVE TRANSLUCENT ROBOT
    env = robosuite.make(
        has_renderer=args.render,
        use_camera_obs=True if len(keys_info["image_keys"]) > 0 else False,
        renderer=args.renderer,
        camera_segmentations=args.camera_segmentations,
        control_freq=control_freq,
        # translucent_robot = 0.0, # TEMPORARY: REMOVE TRANSLUCENT ROBOT
        **env_kwargs,
    )
    return env, env_kwargs


def reset_to_original(env, state):
    """
    Reset to a specific simulator state.

    Args:
        state (dict): current simulator state that contains one or more of:
            - states (np.ndarray): initial state of the mujoco environment
            - model (str): mujoco scene xml

    Returns:
        observation (dict): observation dictionary after setting the simulator state (only
            if "states" is in @state)
    """
    should_ret = False
    if "model" in state:
        if state.get("ep_meta", None) is not None:
            # set relevant episode information
            ep_meta = json.loads(state["ep_meta"])
        else:
            ep_meta = {}
        if hasattr(env, "set_attrs_from_ep_meta"):  # older versions had this function
            env.set_attrs_from_ep_meta(ep_meta)
        elif hasattr(env, "set_ep_meta"):  # newer versions
            env.set_ep_meta(ep_meta)
        # this reset is necessary.
        # while the call to env.reset_from_xml_string does call reset,
        # that is only a "soft" reset that doesn't actually reload the model.
        env.reset()
        robosuite_version_id = int(robosuite.__version__.split(".")[1])
        if robosuite_version_id <= 3:
            from robosuite.utils.mjcf_utils import postprocess_model_xml

            xml = postprocess_model_xml(state["model"])
        else:
            # v1.4 and above use the class-based edit_model_xml function
            xml = env.edit_model_xml(state["model"])

        xml = fix_asset_paths_relative_to_robocasa(xml_string=xml, robocasa_module=robocasa)
        env.reset_from_xml_string(xml)
        env.sim.reset()
        # hide teleop visualization after restoring from model
        # env.sim.model.site_rgba[env.eef_site_id] = np.array([0., 0., 0., 0.])
        # env.sim.model.site_rgba[env.eef_cylinder_id] = np.array([0., 0., 0., 0.])
    if "states" in state:
        env.sim.set_state_from_flattened(state["states"])
        env.sim.forward()
        should_ret = True

    # update state as needed
    if hasattr(env, "update_sites"):
        # older versions of environment had update_sites function
        env.update_sites()
    if hasattr(env, "update_state"):
        # later versions renamed this to update_state
        env.update_state()

    # if should_ret:
    #     # only return obs if we've done a forward call - otherwise the observations will be garbage
    #     return get_observation()
    return None

TASK_SET_ALL_1 = {
    0:  'turn_on_faucet',
    1:  'turn_off_faucet', # this is generalization to new task?
    2:  'open_cabinet',
    3:  'close_cabinet', # this is generalization to new task?
    4:  'pnp_plate_to_cabinet',
    5:  'pnp_cabinet_to_plate',
    6:  'pnp_sink_to_cabinet',
    7:  'pnp_cabinet_to_sink',
    8:  'pnp_plate_to_sink',
    9:  'pnp_sink_to_plate',
    10: 'pnp_counter_to_sink', # this is generalization to new task?
    11: 'pnp_sink_to_counter',
    12: 'pnp_plate_to_counter',
    13: 'pnp_counter_to_plate', # this is generalization to new task?
    14: 'pnp_counter_to_cabinet',
    15: 'pnp_cabinet_to_counter'
}
TASK_SET_ALL_2 = {
    0: "l1_pnp_sink_to_plate",
    1: "l2_pnp_sink_to_plate",
    2: "l3_pnp_sink_to_plate",
    3: "l1_pnp_plate_to_sink",
    4: "l2_pnp_plate_to_sink",
    5: "l3_pnp_plate_to_sink",
    6: "l1_turn_off_faucet",
    7: "l2_turn_off_faucet",
    8: "l3_turn_off_faucet"
}
# task names, postfix
TASK_TRIM_DICT = {
    'SinkPlayEnvDebug': {
        1:  ([TASK_SET_ALL_1[i] for i in [8]], 'SinkPlayEnvDebug_1'),
        2:  ([TASK_SET_ALL_1[i] for i in [8, 9]], 'SinkPlayEnvDebug_2'),
        4:  ([TASK_SET_ALL_1[i] for i in [8, 9, 11, 15]], 'SinkPlayEnvDebug_4'),
        8:  ([TASK_SET_ALL_1[i] for i in [0, 2, 8, 9, 11, 12, 14, 15]], 'SinkPlayEnvDebug_8'),
        12: ([TASK_SET_ALL_1[i] for i in list(set(list(range(16))) - set([1, 3, 10, 13]))], 'SinkPlayEnvDebug_12'),
        16: ([TASK_SET_ALL_1[i] for i in range(16)], 'SinkPlayEnvDebug_16'),
    }
}
