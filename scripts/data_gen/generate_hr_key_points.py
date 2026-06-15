'''
Example usage:
python scripts/data_gen/generate_hr_key_points.py --hdf5_path $CASAPLAY_DATAROOT/memory/MemWashAndReturnLeft/2025-07-25-00-12-14/demo_im128_notp.hdf5

Additional: --resume 
'''
import os
import argparse
import copy
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from halo.util.casa_utils import TASK_NAME_TO_TASK_DESCRIPTION
from halo.util.misc import get_task_name_from_hdf5_path
from halo.models.policy.vlm import GPTQueryGenerator
task_to_rules = {
    "MemWashAndReturn": [
        "moving towards the fruit, the first frame is important.",
        "placing the fruit in the sink, do not select any frame.",
        "washing the fruit, do not select any frame.",
        "moving towards the fruit from the sink, do not select any frame.",
        "placing the fruit back in the container, do not select any frame.",
    ],
    "MemRetrieveOilsFromCounter": [
        "exploring the environment, the frame with the olive oil bottle is important.",
        "moving towards the olive oil bottle, do not select any frame.",
        "picking up the olive oil bottle, do not select any frame.",
    ],
    "MemPutKBreadInMicrowave": [
        "picking up the bread from the counter, the first frame is important.",
        "moving towards the microwave, do not select any frame.",
        "placing the bread in the microwave, do not select any frame.",
        "turning off the microwave, do not select any frame.",
    ],
}
task_to_rules["MemWashAndReturnLeft"] = task_to_rules["MemWashAndReturn"]
task_to_rules["MemWashAndReturnRight"] = task_to_rules["MemWashAndReturn"]
task_to_rules["MemRetrieveOilsFromCounterLL"] = task_to_rules["MemRetrieveOilsFromCounter"]
task_to_rules["MemRetrieveOilsFromCounterLR"] = task_to_rules["MemRetrieveOilsFromCounter"]
task_to_rules["MemRetrieveOilsFromCounterRL"] = task_to_rules["MemRetrieveOilsFromCounter"]
task_to_rules["MemRetrieveOilsFromCounterRR"] = task_to_rules["MemRetrieveOilsFromCounter"]

prompt_template = '''
You are a helpful assistant for a robot task. You are provided with the following information:
- Task instruction: the goal of the task.
- Description of all the activities performed by the robot from time step 0 to time step t (if t > 0).
- List of rules for each subtask to select the key frames. Keep it empty if nothing should be selected. Select only one per set of frames provided.
- List of images: a list of images from the robot's episodes from time step t to time step t+n.
  Each image has two views: (1) agent-view and (2) eye-in-hand.

Your job:
1. Analyze the images provided to you, and the rules for the task.
2. Using this information, try to understand the events happening from the complete task in these set of images. Do not repeat the same summary.
3. Use the summary and provided rules, select the key frames that are important for the subtask (time step t to time step t+n). Leave it empty if nothing should be selected.

OUTPUT FORMAT:
json{{
  "summary": "<description of the each image in the list and as a whole>",
  "key_frames": [<frame number important for the subtask>]
}}

Example 1:
json{{
  "summary": "exploring the environment to look around.",
  "key_frames": [8]
}}

Example 2:
json{{
  "summary": "after exploring the environment, moving towards the green sponge on the table.",
  "key_frames": []
}}

Example 3:
json{{
  "summary": "picking up the sponge",
  "key_frames": [0]
}}

Please think step by step and provide a list of frame numbers that are important for the task.

TASK INSTRUCTION: {task_instruction}
'''
# TASK DESCRIPTION: {task_description}

base_summary_template = '''
Frame {start_idx}-{end_idx}: {summary}
'''

def get_transition_values(arr, skip_indices=None):
    '''
    Given an array of shape (n,); it returns a list of positions where the value changes
    '''
    transition_values = []
    for i in range(1, len(arr)):
        if arr[i] != arr[i-1] and (skip_indices is None or i not in skip_indices):
            transition_values.append(i)
    return transition_values

def get_keyframes_from_actions(actions, include_last_frame=True):
    if actions.shape[-1] == 7:
        transition_indices = get_transition_values(actions[:, 6], skip_indices=list(np.arange(5)))
    if actions.shape[-1] == 12:
        gripper_transition_indices = get_transition_values(actions[:, 6], skip_indices=list(np.arange(5)))
        base_transition_indices = get_transition_values(actions[:, 11], skip_indices=list(np.arange(5)))
        transition_indices = gripper_transition_indices + base_transition_indices
    # sort theb transition indices in ascending order
    transition_indices.sort()
    if include_last_frame and (len(actions) - 1  > transition_indices[-1]):
        transition_indices.append(len(actions) - 1)
    return transition_indices

class HDF5ImageDataset(Dataset):
    def __init__(self, hdf5_path, image_keys):
        self.hdf5_path = hdf5_path
        self.image_keys = image_keys
        
        demo_keys = []
        with h5py.File(hdf5_path, "r") as f:
            demo_keys = list(f["data"].keys())
        # sort the demo_keys in ascending order
        demo_keys.sort(key=lambda x: int(x.split('_')[-1]))
        self.demo_keys = demo_keys

    
    def __len__(self):
        return len(self.demo_keys)
    
    def __getitem__(self, idx):
        demo_key = self.demo_keys[idx]
        with h5py.File(self.hdf5_path, "r") as f:
            demo_group = f["data"][demo_key]
            ep_metadata = demo_group.attrs["ep_meta"]
            if isinstance(ep_metadata, str):
                ep_metadata = json.loads(ep_metadata)
            lang = ep_metadata["lang"]
            images = {}
            for key in self.image_keys:
                images[key] = np.array(demo_group[key][:])
            
            actions = np.array(demo_group["actions"][:])
            
            return {"images": images, "actions": actions, "lang": lang, "demo_key": demo_key, "hdf5_path": self.hdf5_path}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5_path", type=str, required=True)
    parser.add_argument("--image_keys", type=str, nargs="+", default=["obs/robot0_agentview_center_image", "obs/robot0_eye_in_hand_image"])
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--resume", action="store_true")
    
    args = parser.parse_args()
    
    rel_path = os.environ["CASAPLAY_DATAROOT"]
    n_sampled_frame = 16
    dataset = HDF5ImageDataset(args.hdf5_path, args.image_keys)
    data_hdf5_path = os.path.relpath(args.hdf5_path, rel_path)
    basename = os.path.basename(args.hdf5_path).replace('.hdf5', '')
    save_path = os.path.join(os.path.dirname(args.hdf5_path), f"{basename}_memer_keypoints.json")

    store_summary = {}
    print(f"Dataset length: {len(dataset)}")
    start_sample_idx = 0
    if args.resume:
        with open(save_path, "r") as f:
            store_summary = json.load(f)
        start_sample_idx = len(list(store_summary[data_hdf5_path].keys()))

    for sample_idx in range(start_sample_idx, len(dataset), 1):
        sample = dataset[sample_idx]
        # print(f"Actions shape: {sample['actions'].shape}")
        # print(f"Image keys: {list(sample['images'].keys())}")
        # for key in args.image_keys:
        #     print(f"Image shape: {sample['images'][key].shape}")
        task_name = get_task_name_from_hdf5_path(args.hdf5_path, actual_task_name=True)
        task_instruction = sample['lang']
        task_description = TASK_NAME_TO_TASK_DESCRIPTION[task_name]
        print(f"Task name: {task_name}")
        print(f"Task instruction: {task_instruction}")
        print(f"Task description: {task_description}")

        images_agentview = sample['images'][args.image_keys[0]].astype(np.uint8)
        images_eyeinhand = sample['images'][args.image_keys[1]].astype(np.uint8)

        print(f"Images agentview shape: {images_agentview.shape}")
        print(f"Images eyeinhand shape: {images_eyeinhand.shape}")

        images_concatenated = np.concatenate([images_agentview, images_eyeinhand], axis=2)

        assert task_name in task_to_rules, f"Task name {task_name} not found in task_to_rules"
        base_prompt = prompt_template.format(task_instruction=task_instruction, task_description=task_description)
        vlm_generator = GPTQueryGenerator(model="gpt-4o")

        # create a prompt saying: "Frame X"
        prompts = [f"Frame {i}" for i in range(len(images_agentview))]
        transition_indices = get_keyframes_from_actions(sample['actions'])
        print(f"Transition indices: {transition_indices}")
        start_idx, n_retries = 0, 5
        ep_store_summary = []
        for end_idx in transition_indices:
            base_actual_prompt = copy.deepcopy(base_prompt)
            if len(ep_store_summary) > 0:
                base_actual_prompt += "\n\nHere is the summary of the previous events in this episode:\n"
                for hist_sum in ep_store_summary:
                    base_actual_prompt += base_summary_template.format(start_idx=hist_sum["start_idx"], end_idx=hist_sum["end_idx"], summary=hist_sum["summary"])
            base_actual_prompt += "\n\nHere are the rules for the task to select the key frames:\n"
            for rule in task_to_rules[task_name]:
                base_actual_prompt += f"- {rule}\n"
            base_actual_prompt += "\n\nPlease provide a summary and selected key frames important for the current subtask from frame {start_idx} to frame {end_idx}."
            

            if args.debug:
                print(f"{start_idx=}, {end_idx=}")
            # sample n_sampled_frames from start_idx to end_idx equidistant and both inclusive
            sampled_frames = np.arange(start_idx, end_idx + 1, dtype=int)
            if end_idx - start_idx + 1 > n_sampled_frame:
                sampled_frames = np.linspace(start_idx, end_idx, n_sampled_frame, dtype=int)
            sampled_images = images_concatenated[sampled_frames]

            sampled_prompts = [f"Frame {i}" for i in sampled_frames]

            # create a list of empty lists for the prompts
            prompts_list = [[] for _ in range(len(sampled_images))]
            for _ in range(n_retries):
                output = vlm_generator.generate_queries(
                    base_prompt=base_actual_prompt,
                    prompts=[prompts_list], # [sampled_prompts]
                    videos=[sampled_images],
                    n=1,
                    temperature=1.0,
                    debug=args.debug,
                )
                response_list = [vlm_generator.extract_json_from_response(out) for out in output]
                response_list = [res for res in response_list if len(res.keys()) > 0]
                if len(response_list) == 0:
                    if args.debug:
                        import ipdb; ipdb.set_trace()
                    continue 

                key_frames = response_list[0]["key_frames"]
                summary = response_list[0]["summary"]
                print(f"Summary for frame {start_idx}-{end_idx}: {summary}")
                print(f"Key frames for frame {start_idx}-{end_idx}: {key_frames}")
                # all of the key frames should be within the range of sampled_frames
                if not all(i < len(sampled_frames) for i in key_frames):
                    print(f"Key frames {key_frames} are not within the range of total sampled_frames {len(sampled_frames)}")
                    continue
                select_frame_idx = [int(sampled_frames[i]) for i in key_frames]
                ep_store_summary.append({
                    "start_idx": int(start_idx),
                    "end_idx": int(end_idx),
                    "key_frames": [int(i) for i in key_frames],
                    "selected_frame_idx": select_frame_idx,
                    "hdf5_path": os.path.relpath(sample["hdf5_path"], rel_path),
                    "demo_key": sample["demo_key"],
                    "summary": summary,
                })

                if args.debug:
                    # for debugging, concatenate all the sampled_frames along the height axis and save it to a file called /tmp/sampled_frames_<start_idx>_<end_idx>.png
                    plt_image = np.concatenate(sampled_images, axis=0)
                    plt.imshow(plt_image); plt.axis("off"); plt.tight_layout()
                    filename = f"/tmp/sampled_frames_{start_idx}_{end_idx}.png"
                    print(f"Saving image to {filename}")
                    plt.savefig(f"{filename}")
                    plt.close()
                    plt.clf()
        
                if args.debug:
                    import ipdb; ipdb.set_trace()
                break
            start_idx = end_idx + 1
        
        # we will consolidate the ep_store_summary into the store_summary
        ep_store_summary_dict = {}
        demo_key = ep_store_summary[0]["demo_key"]
        hdf5_path = ep_store_summary[0]["hdf5_path"]
        if hdf5_path not in store_summary:
            store_summary[hdf5_path] = {}
        store_summary[hdf5_path][demo_key] = ep_store_summary
        # store_summary.extend(ep_store_summary)
        with open(save_path, "w") as f:
            json.dump(store_summary, f, indent=4)
        print(f"Key frames saved to {save_path}")
