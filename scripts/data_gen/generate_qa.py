'''
Example usage:
python scripts/data_gen/generate_qa.py --dataset_config.hdf5_paths $CASAPLAY_DATAROOT/memory/MemPutKBreadInMicrowave/2025-07-25-10-40-55/demo_im128_notp.hdf5 --dataset-config.dataset-json config/task/task_robocasa_turn_on_stove.json  --shared_config.downsample_obs 8 --dataset-config.max_dataset_size -1 --seed 10 --prefix same_chunk_length512 --mode ts_text_only_bbox_v2 --max_timesteps -1 --dataset-config.min_chunk_length 3000 --dataset_config.use_same_chunk_length --debug
'''
import os
import time
import copy
import uuid
import pickle
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
import numpy as np
import json
import torch
import tyro
from tqdm import tqdm
from typing import Literal
from termcolor import colored
import matplotlib.pyplot as plt
# repository imports
from halo.data.gpt_qa_dataset import GPTDataset
from halo.models.policy.vlm import GPTQueryGenerator
from halo.util.args import GenerateQAGenericArgs
# prompts
import halo.prompts.qa_task_specific_text_only_promptsv2 as qa_task_specific_text_only_promptsv2
from halo.util.eval_utils import control_seed
from halo.util.casa_utils import TASK_NAME_TO_TASK_DESCRIPTION, filter_bbox_names, convert_bbox_name_to_str
from halo.util.misc import get_task_name_from_hdf5_path, convert_image_keys_to_bbox_keys
from halo.util.qa_website_render import visualize_data_point
base_summary_template = '''Frame {start_idx}-{end_idx}: {summary}'''

def get_query_base_prompt(mode: str, data: Dict[str, Any]):
    base_prompt = qa_task_specific_text_only_promptsv2.task_specific_bbox_only_prompt
    task_language = data["task_language"]
    demo_key = data["demo_key"]
    hdf5_path = data["hdf5_path"]
    actual_task_name = get_task_name_from_hdf5_path(hdf5_path, actual_task_name=True)
    task_names_avail = list(TASK_NAME_TO_TASK_DESCRIPTION.keys())
    task_selected = [task_name for task_name in task_names_avail if task_name in actual_task_name]
    assert len(task_selected) == 1, f"Multiple or no task names found for demo key: {demo_key}, actual task name: {actual_task_name}, task names available: {task_names_avail}"
    task_description = TASK_NAME_TO_TASK_DESCRIPTION[task_selected[0]]
    return base_prompt.format(task_language=task_language, task_description=task_description)

def get_eval_multi_prompt(mode: str, data: Dict[str, Any]):
    assert "demo_key" in data.keys(), f"demo_key not found in data: {data.keys()}"
    assert "task_language" in data.keys(), f"task_language not found in data: {data.keys()}"

    base_prompt = qa_task_specific_text_only_promptsv2.eval_multi_bbox_only_prompt
    demo_key = data["demo_key"]
    task_names_avail = list(TASK_NAME_TO_TASK_DESCRIPTION.keys())
    hdf5_path = data["hdf5_path"]
    actual_task_name = get_task_name_from_hdf5_path(hdf5_path, actual_task_name=True)
    task_selected = [task_name for task_name in task_names_avail if task_name in actual_task_name]
    assert len(task_selected) == 1, f"Multiple or no task names found for demo key: {demo_key}, actual task name: {actual_task_name}, task names available: {task_names_avail}"
    task_description = TASK_NAME_TO_TASK_DESCRIPTION[task_selected[0]]
    task_language = data["task_language"]
    return base_prompt.format(task_language=task_language, task_description=task_description)

def collate_fn(batch):
    # some of them is a list of strings, some of them is a list of tensors
    # we need to convert them to tensors
    # key to type mapping
    result = {}
    keys = batch[0].keys()
    for key in keys:
        result[key] = []
    for item in batch:
        for key in keys:
            result[key].append(item[key])
    return result

def create_max_timestep_prompts(prompts, videos, timestep_index, rng, max_timesteps=16, args: GenerateQAGenericArgs = None):
    if max_timesteps == -1:
        return prompts, videos, timestep_index
    num_cams = len(prompts)
    total_timesteps = len(videos[0])
    if total_timesteps < max_timesteps:
        return prompts, videos, timestep_index
    # this would make the images go from max_timesteps to 2*max_timesteps - 1
    skip_factor = total_timesteps//max_timesteps + 1
    start_index = rng.randint(0, skip_factor)
    prompts = [
        prompts[cam_ind][start_index::skip_factor]
        for cam_ind in range(num_cams)
    ]
    videos = [
        videos[cam_ind][start_index::skip_factor]
        for cam_ind in range(num_cams)
    ]
    # downsample the timesteps using start_index and skip_factor from data
    key_prompt_timesteps = timestep_index[start_index::skip_factor]
    key_timesteps = list(key_prompt_timesteps)
    assert len(key_timesteps) == len(prompts[0]) == len(videos[0])
    return prompts, videos, torch.tensor(np.array(key_timesteps))

def gather_selected_response_from_filtered_results(filter_json, real_response_list, frame_indices, data, debug: bool = False):
    selected_response = []
    if len(filter_json.keys()) > 0:
        # filter the response list based on the filter_json
        filter_results = filter_json["results"]
        # filter all results that do not contain the id, score, and decision in the filter_result
        filter_results = [filter_result for filter_result in filter_results if exists_all_filter_fields(filter_result)]
        # check the max id, if it greater than the length of the real_response_list, then something is wrong
        max_id = max(int(filter_result["id"]) for filter_result in filter_results)
        min_id = min(int(filter_result["id"]) for filter_result in filter_results)
        if max_id > len(real_response_list) or min_id < 1:
            if max_id > len(real_response_list):
                print(f"Max id {max_id} is greater than the length of the real_response_list {len(real_response_list)} for {data['hdf5_path']} {data['demo_key']}")
            if min_id < 1:
                print(f"Min id {min_id} is less than 1 for {data['hdf5_path']} {data['demo_key']}")
            print(f"{filter_results=}\n{real_response_list=}\n{frame_indices=}")
            print(f"Skipping this data point")
            if debug:
                import ipdb; ipdb.set_trace()
            return [] # return an empty list
        for filter_result in filter_results:
            id = int(filter_result["id"])
            selected_response.append({**real_response_list[id-1], **filter_result, "rel_frame_index": frame_indices[id-1]})
    return selected_response

def query_prompt_generator(
    camera_order: List[str],
    data: Dict[str, Any],
    base_prompt: str,
    image_keys: List[str],
    max_timesteps: int = 8,
    n_queries: int = 5,
    rng: Optional[np.random.RandomState] = None,
    vlm_generator: Optional[GPTQueryGenerator] = None,
    mode: str = "generic",
    args: GenerateQAGenericArgs = None,
):
    if vlm_generator is None:
        vlm_generator = GPTQueryGenerator()

    # Get bbox keys from image keys using the utility function
    bbox_keys, bbox_names_keys, bbox_mask_keys, bbox_cc_keys = convert_image_keys_to_bbox_keys(image_keys)
    
    num_cams = len(image_keys)
    # Dynamically load images and bbox data based on image_keys
    images = []
    images_bbox = []
    bbox_names_list = []
    bbox_mask_list = []
    
    for idx, img_key in enumerate(image_keys):
        images.append(data[img_key].cpu().numpy().astype(np.uint8))
        bbox_key = bbox_keys[idx]
        images_bbox.append(data[bbox_key].cpu().numpy().astype(np.uint8))
        bbox_names_list.append(data[bbox_names_keys[idx]])
        bbox_mask_list.append(data[bbox_mask_keys[idx]])
    
    hdf5_path = data["hdf5_path"]

    # the numbers pretend as if all the images were passed through the vlm
    frames_indices = data["relative_timestep_index"]
    summary_list = data["summary_list"]
    # print(images_agentview.shape, images_eyeinhand.dtype)
    base_prompt_to_pass = copy.deepcopy(base_prompt)
    summary_string = generate_summary_string(summary_list)
    base_prompt_to_pass = base_prompt_to_pass + summary_string
    # mention Action Gripper Close (1) or Open (0)
    base_prompt_to_pass = base_prompt_to_pass + f"\nNote: For each frame bounding box (bbox) of relevant objects are provided in the camera frames. For the Action Gripper, the value is 1 for close and 0 for open\n"
    base_prompt_to_pass += f'''
Data Annotation Guide
  Bounding Boxes (Bbox): Provided for all relevant objects in each camera frame.
  Action Gripper: Use 1 for closed and 0 for open.
    '''
    prompts = [
        [f"\nFrame {i}\n" if cam_ind == 0 else "" for i in frames_indices]
        for cam_ind in range(num_cams)
    ]
    frame_bbox_map: Dict[int, Dict[str, List[str]]] = {}  # rfi -> {cam_name -> [bbox strings]}
    # check the gripper closing stuff
    for cam_ind in range(num_cams):
        noise = np.random.normal(0, 0.01, (1, 4))
        for i in range(len(frames_indices)):
              # f"\n\nBbox names: {agentview_bbox_names[i] if cam_ind == 0 else eyeinhand_bbox_names[i]}"
            bbox_coords = images_bbox[cam_ind][i]
            bbox_mask = bbox_mask_list[cam_ind][i]
            bbox_name_np = bbox_names_list[cam_ind][i]
            H, W = images[cam_ind].shape[1], images[cam_ind].shape[2]
            valid_bbox_mask = bbox_mask > 0

            # normalize the bbox coordinates to be between 0 and 1: each of them are xmin, ymin, xmax, ymax
            bbox_coords = np.array(bbox_coords, dtype=np.float32)
            bbox_coords[:,0] = bbox_coords[:,0] / W
            bbox_coords[:,1] = bbox_coords[:,1] / H
            bbox_coords[:,2] = bbox_coords[:,2] / W
            bbox_coords[:,3] = bbox_coords[:,3] / H
            # adding noise to the bbox co-ordinates
            bbox_coords = bbox_coords + noise
            # clip the bbox co-ordinates to be between 0 and 1
            bbox_coords = np.clip(bbox_coords, 0, 1)
            # join the list into a string
            bbox_names = [
                convert_bbox_name_to_str(bbox_name) + f" ({coords[0]:.2f} {coords[1]:.2f} {coords[2]:.2f} {coords[3]:.2f})"
                for bbox_name, coords in zip(bbox_name_np[valid_bbox_mask], bbox_coords[valid_bbox_mask])
                if convert_bbox_name_to_str(bbox_name) is not None
            ]
            task_name = get_task_name_from_hdf5_path(hdf5_path, actual_task_name=True)
            bbox_names = filter_bbox_names(bbox_names, task_name)
            bbox_name_str = ", ".join(bbox_names)
            # add the camera name to the prompt - use camera_order if provided, otherwise use generic names
            if cam_ind < len(camera_order):
                camera_name = camera_order[cam_ind]
            else:
                camera_name = f"camera {cam_ind}"
            camera_name = camera_name # + " camera"
            prompts[cam_ind][i] = prompts[cam_ind][i] + f"Bbox in {camera_name}: {bbox_name_str}\n"
            rfi = int(frames_indices[i])
            if rfi not in frame_bbox_map:
                frame_bbox_map[rfi] = {}
            frame_bbox_map[rfi][camera_name] = bbox_names
            # add gripper_action to the prompt. 1 means close and 0 means open
            gripper_action = data["gripper_action"][i].item()
            if cam_ind == num_cams - 1:
                prompts[cam_ind][i] = prompts[cam_ind][i] + f"Gripper Action: {gripper_action}\n"

    if (mode == "ts_text_only_bbox") or (mode == "ts_text_only_bbox_v2") or (mode == "ts_text_only_time"):
        videos = [[] for _ in range(len(image_keys))]
    else:
        videos = images

    prompts, videos, timestep_index_prompt = create_max_timestep_prompts(prompts, videos, data["timestep_index"], rng=rng, max_timesteps=max_timesteps, args=args)
    # generate a random frame index from the frames_indices
    # create a string stating: Generate for the query-frame index {frame_index}.
    n_queries = min(n_queries, len(frames_indices))
    frame_indices = rng.choice(frames_indices, size=n_queries, replace=False).tolist()
    if args.debug:
        print(f"*****************\n{frame_indices=}\n*****************")
    frame_index_string = f"NOW GENERATE THE QUERIES FOR THE FOLLOWING QUERY-FRAME INDEXES: {frame_indices}. Think step by step."
    return {
        "base_prompt_to_pass": base_prompt_to_pass,
        "prompts": prompts,
        "videos": videos,
        "frame_indices": frame_indices,
        "frame_index_string": frame_index_string,
        "timestep_index_prompt": timestep_index_prompt,
        "frame_bbox_map": frame_bbox_map,
    }

def vqa_generator(
    base_prompt_to_pass: str,
    prompts: List[List[str]],
    videos: List[np.ndarray],
    frame_indices: List[int],
    frame_index_string: str,
    timestep_index_prompt: List[int],
    vlm_generator: GPTQueryGenerator,
    args: GenerateQAGenericArgs,
):
    outputs = []
    output = None
    for _ in range(args.n_retries):
        try:
            output = vlm_generator.generate_queries(
                base_prompt=base_prompt_to_pass,
                prompts=prompts,
                videos=videos,
                end_text=frame_index_string,
                n=1,
                temperature=1.0,
                debug=args.debug,
            )
        except Exception as e:
            print(colored(f"Error {e=}", "red"))
            output = None
            if args.debug:
                import ipdb; ipdb.set_trace()
            pass
            continue
        break
    if output is not None:
        outputs.extend(output)
    else:
        outputs = None
    return outputs, prompts, videos, timestep_index_prompt, frame_indices

def plot_images_grid(images, images_per_row=10):
    """
    Plots a list of images in a grid with a specified maximum number of images per row.

    Args:
        images (list): A list of images (e.g., numpy arrays or PIL Image objects).
        images_per_row (int): The maximum number of images to display per row.
    """
    num_images = len(images)
    num_rows = (num_images + images_per_row - 1) // images_per_row

    plt.figure(figsize=(images_per_row * 2, num_rows * 2)) # Adjust figure size as needed

    for i, img in enumerate(images):
        plt.subplot(num_rows, images_per_row, i + 1)
        plt.imshow(img)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def save_qa_results(qa_results, save_path):
    if len(save_path.split("/")) > 1:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(qa_results, f, indent=4)
    print(f"Dataset saved to {save_path}")

def exists_all_filter_fields(filter_result: Dict[str, Any], keys: List[str] = ['id', 'score', 'decision']) -> bool:
    if not all(key in filter_result.keys() for key in keys):
        return False
    return True

def save_img(img, path):
    plt.imshow(img)
    plt.savefig(path)
    plt.close()
    return

def generate_summary_string(summary_list: List[Dict[str, Any]]):
    summary_strings = [base_summary_template.format(start_idx=summary["start_idx"], end_idx=summary["end_idx"], summary=summary["summary"]) for summary in summary_list]
    return_str = "SUMMARY OF TASK EVENTS:\n" + "\n".join(summary_strings)
    return return_str

MODEL_NAME_TO_MAINTAIN_FREQUENCY = {}

def main():
    args = tyro.cli(GenerateQAGenericArgs)
    if args.batch_mode:
        generate_vqa_batch_file()
        return
    # dataset_config = GPTDatasetConfig(
    #     # hdf5_files="v0.1/single_stage/kitchen_stove/TurnOnStove/2024-05-02/demo_im128_notp.hdf5",
    #     dataset_json='configs/task_robocasa_turn_on_stove.json',
    #     control_freq=20,
    #     output_full_episode=False,
    #     min_chunk_length=10
    # )
    # shared_config = SharedConfig(
    #     downsample_obs=4,
    # )
    control_seed(args.seed)
    print(f"Saving to {args.save_path}")
    dataset = GPTDataset(
        dataset_config=args.dataset_config,
        shared_config=args.shared_config,
        action_index_mapping={
            'gripper': {
                'index': 6,
                'close': 1,
                'open': -1,
            }, # starting from zero; -1 open; 1 close
            'arm_base_control': {
                'index': -1,
                'move_arm': -1,
                'move_base': 1,
            },
        },
        seed=args.seed,
    )
    rng = np.random.RandomState(1)
    # Get image_keys from dataset config
    dataset_config_dict = dataset.get_dataset_config()
    image_keys = dataset_config_dict['image_keys']
    # Generate camera_order from image_keys (extract camera name from key, or use generic names)
    camera_order = []
    for img_key in image_keys:
        # Try to extract camera name from key (e.g., "agentview" from "obs/robot0_agentview_center_image")
        if "agentview" in img_key.lower() or "agent" in img_key.lower():
            camera_order.append("agentview")
        elif "eye" in img_key.lower() or "hand" in img_key.lower():
            camera_order.append("eyeinhand")
        else:
            raise NotImplementedError(f"Camera name {img_key} not found in image_keys")
    maintain_frequency = MODEL_NAME_TO_MAINTAIN_FREQUENCY.get(args.model_name, -1.0)
    vlm_generator = GPTQueryGenerator(model=args.model_name, maintain_frequency=maintain_frequency)
    vlm_generator_eval = GPTQueryGenerator(model="gpt-4o")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=2 if not args.debug else 0)
    # save the filtered response list to a json file
    dataset_all_values = []
    pbar = tqdm(dataloader, desc="Generating QA Generic", total=len(dataloader))
    n_steps = 0
    resumed_run = False
    # import ipdb; ipdb.set_trace()
    if args.resume:
        assert os.path.exists(args.save_path), f"Save path {args.save_path} does not exist"
        dataset_all_values = json.load(open(args.save_path, "r"))
        n_steps = int(dataset_all_values[-1]["data_idx"])
        resumed_run = True
    if args.debug:
        import ipdb; ipdb.set_trace()
    for data_idx, data in enumerate(pbar):
        if data_idx < n_steps:
            # try:
            #     assert data['hdf5_path'][0] == dataset_all_values[20*data_idx]["hdf5_path"]
            #     assert data['demo_key'][0] == dataset_all_values[20*data_idx]["demo_key"]
            #     assert data['start_idx'][0] == dataset_all_values[20*data_idx]["start_idx"]
            #     assert data['end_idx'][0] == dataset_all_values[20*data_idx]["end_idx"]
            # except Exception as e:
            #     print(colored(f"Error {e=}", "red"))
            #     import ipdb; ipdb.set_trace()
            #     continue
            continue
        n_steps += 1
        if n_steps % 50 == 0 and not resumed_run:
            save_qa_results(dataset_all_values, args.save_path)

        resumed_run = False
        # squeeze the data to remove the batch dimension
        data = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v[0] for k, v in data.items()}
        prompt_base = get_query_base_prompt(args.mode, data)
        query_generator_output = query_prompt_generator(
            data=data,
            camera_order=camera_order,
            base_prompt=prompt_base,
            image_keys=image_keys,
            max_timesteps=args.max_timesteps,
            n_queries=args.n_queries,
            vlm_generator=vlm_generator,
            rng=rng,
            mode=args.mode,
            args=args,
        )
        frame_bbox_map = query_generator_output.pop("frame_bbox_map")
        output, prompts, videos, timestep_index_prompt, frame_indices = vqa_generator(
            **query_generator_output,
            vlm_generator=vlm_generator,
            args=args,
        )
        query_generator_output["frame_bbox_map"] = frame_bbox_map
        if output is None:
            continue
        # add the timestep_prompt_index to the data
        data["timestep_prompt_index"] = timestep_index_prompt

        response_list = [vlm_generator.extract_json_from_response(out) for out in output]
        response_list = [res for res in response_list if len(res.keys()) > 0]
        if len(response_list) == 0:
            if args.debug:
                import ipdb; ipdb.set_trace()
            continue
        assert len(response_list) == 1, f"Expected 1 response list, got {len(response_list)}. If this changes, need to match frame_indices with the response list."
        real_response_list = [] 
        for res in response_list:
            real_response_list.extend(res["results"])
        if len(real_response_list) != len(frame_indices):
            print(f"Length of real_response_list {len(real_response_list)} and frame_indices {len(frame_indices)} do not match for {data['hdf5_path']} {data['demo_key']}")
            print(f"{real_response_list=}\n{frame_indices=}")
            print(f"Skipping this data point")
            continue
        # add id to each response consisting of the index of the response in the output list
        real_response_list = [{"id": i+1, **res} for i, res in enumerate(real_response_list)]
        response_string = json.dumps(real_response_list, indent=4)
        response_string = "Here are the candidate items to evaluate:\n" + response_string

        if args.skip_verification:
            # skip the second VLM call; treat all generated QA pairs as kept with a default score
            filter_json = {
                "results": [
                    {"id": res["id"], "score": 5, "decision": "keep"}
                    for res in real_response_list
                ]
            }
        else:
            for _ in range(args.n_retries):
                base_prompt = get_eval_multi_prompt(args.mode, data)
                summary_string = generate_summary_string(data["summary_list"])
                base_prompt = base_prompt + summary_string
                filter_output = vlm_generator_eval.generate_queries(
                    base_prompt=base_prompt,
                    prompts=prompts,
                    videos=videos,
                    end_text=response_string,
                    n=1,
                )
                filter_json = vlm_generator.extract_json_from_response(filter_output[0])
                if len(filter_json.keys()) > 0:
                    break

        # print(f"*****************\n{filter_json=}\n*****************")
        # print(filter_json)
        # print(len(real_response_list))

        dataset_meta = dataset.get_dataset_config()
        common_info = {
            'timestep_index': ",".join(map(str, data['timestep_index'].cpu().numpy().tolist())),
            'hdf5_path': data['hdf5_path'],
            'demo_key': data['demo_key'],
            'start_idx': data['start_idx'].cpu().numpy().tolist(),
            'end_idx': data['end_idx'].cpu().numpy().tolist(),
            'data_idx': data_idx,
        }
        selected_response = gather_selected_response_from_filtered_results(
            filter_json=filter_json,
            real_response_list=real_response_list,
            frame_indices=frame_indices,
            data=data,
            debug=args.debug,
        )
        if len(selected_response) == 0:
            print(f"No selected response found for {data['hdf5_path']} {data['demo_key']}")
            continue
        # print(f"*****************\n{selected_response=}\n*****************")
        if args.debug:
            print(f"*****************\n{len(selected_response)=}\n*****************")
            for res in selected_response:
                if res["decision"] == "skip":
                    print(colored(f"{res}", "red"))
            for res in selected_response:
                if res["decision"] == "keep":
                    print(f"{res}")

                
            # count number of keep and skip responses
            num_keep = sum(1 for res in selected_response if res["decision"] == "keep")
            num_skip = sum(1 for res in selected_response if res["decision"] == "skip")
            print(f"Number of keep responses: {num_keep}")
            print(f"Number of skip responses: {num_skip}")
            print(f"**********************************")
            import ipdb; ipdb.set_trace() # analyze the selected response

        if args.visualize:
            viz_dir = args.visualize_dir if args.visualize_dir else os.path.join(os.path.dirname(args.save_path), "viz")
            visualize_data_point(
                data=data,
                image_keys=image_keys,
                base_prompt=query_generator_output["base_prompt_to_pass"],
                vlm_output=output,
                real_response_list=real_response_list,
                filter_json=filter_json,
                selected_response=selected_response,
                frame_bbox_map=query_generator_output["frame_bbox_map"],
                output_dir=viz_dir,
                data_idx=data_idx,
            )
            if args.debug:
                import ipdb; ipdb.set_trace()

        # print(f"*****************\n{selected_response=}\n*****************")
        for res in selected_response:
            dataset_all_values.append({**res, **common_info})

    save_qa_results(dataset_all_values, args.save_path)

def generate_vqa_batch_file():
    args = tyro.cli(GenerateQAGenericArgs)
    control_seed(args.seed)
    print(f"Saving to {args.save_path}")
    dataset = GPTDataset(
        dataset_config=args.dataset_config,
        shared_config=args.shared_config,
        action_index_mapping={
            'gripper': {
                'index': 6,
                'close': 1,
                'open': -1,
            }, # starting from zero; -1 open; 1 close
            'arm_base_control': {
                'index': -1,
                'move_arm': -1,
                'move_base': 1,
            },
        },
        seed=args.seed,
    )
    save_path = args.save_path
    save_path = os.path.join(os.path.dirname(save_path), "batched_vqa", os.path.basename(save_path))
    base_dir = os.path.dirname(save_path)
    os.makedirs(base_dir, exist_ok=True)
    metadata_file = os.path.join(base_dir, os.path.basename(save_path).replace(".json", "_metadata.json"))

    rng = np.random.RandomState(1)
    # Get image_keys from dataset config
    dataset_config_dict = dataset.get_dataset_config()
    image_keys = dataset_config_dict['image_keys']
    # Generate camera_order from image_keys (extract camera name from key, or use generic names)
    camera_order = []
    for img_key in image_keys:
        # Try to extract camera name from key (e.g., "agentview" from "obs/robot0_agentview_center_image")
        if "agentview" in img_key.lower() or "agent" in img_key.lower():
            camera_order.append("agentview")
        elif "eye" in img_key.lower() or "hand" in img_key.lower():
            camera_order.append("eyeinhand")
        else:
            raise NotImplementedError(f"Camera name {img_key} not found in image_keys")
    vlm_generator = GPTQueryGenerator(model=args.model_name, is_batched=True)
    vlm_generator_eval = GPTQueryGenerator(model="gpt-4o", is_batched=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=20)
    pbar = tqdm(dataloader, desc="Generating VQA Batch", total=len(dataloader))
    n_steps = 0
    store_data_info = {}
    content_list = []
    # store the data info to be a pkl file
    data_info_file = save_path.replace(".json", "_data_info.pkl")
    vqa_input_jsonl_file = save_path.replace(".json", "_vqa_input.jsonl")
    vqa_output_jsonl_file = save_path.replace(".json", "_vqa_output.jsonl")
    print("*"*20)
    print(f"{vqa_input_jsonl_file=}")
    print(f"{vqa_output_jsonl_file=}")
    print(f"{data_info_file=}")
    print("*"*20)
    if not os.path.exists(vqa_input_jsonl_file):
        for data_idx, data in enumerate(pbar):
            data = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v[0] for k, v in data.items()}
            prompt_base = get_query_base_prompt(args.mode, data)
            custom_id = str(uuid.uuid4()) + "_" + str(data_idx)
            query_generator_output = query_prompt_generator(
                data=data,
                camera_order=camera_order,
                base_prompt=prompt_base,
                image_keys=image_keys,
                max_timesteps=args.max_timesteps,
                n_queries=args.n_queries,
                vlm_generator=vlm_generator,
                rng=rng,
                mode=args.mode,
                args=args,
            )
            content = vlm_generator.generate_queries(
                base_prompt=query_generator_output["base_prompt_to_pass"],
                prompts=query_generator_output["prompts"],
                videos=query_generator_output["videos"],
                end_text=query_generator_output["frame_index_string"],
                n=1,
                temperature=1.0,
                debug=args.debug,
                return_content=True,
                custom_id=custom_id,
            )
            content_list.append(content)
            # data-to-store should be everything but the keys containing image or rgb
            store_data_info[custom_id] = {
                "common_info": { # info to be stored for each data point
                    "timestep_index": query_generator_output["timestep_index_prompt"].cpu().numpy().tolist(),
                    "data_idx": data_idx,
                    "hdf5_path": data["hdf5_path"],
                    "demo_key": data["demo_key"],
                    "start_idx": data["start_idx"].cpu().numpy().tolist(),
                    "end_idx": data["end_idx"].cpu().numpy().tolist(),
                },
                "prompt_info": { # info to be stored later querying the vlm
                    "prompts": query_generator_output["prompts"],
                    "videos": query_generator_output["videos"],
                    "frame_indices": query_generator_output["frame_indices"],
                    "frame_index_string": query_generator_output["frame_index_string"],
                    "timestep_index_prompt": query_generator_output["timestep_index_prompt"].cpu().numpy().tolist(),
                },
                "data_info": { # to be reused for generating the data
                    "summary_list": data["summary_list"],
                    "task_language": data["task_language"],
                    "demo_key": data["demo_key"],
                    "hdf5_path": data["hdf5_path"],
                },
            }
        with open(data_info_file, "wb") as f:
            pickle.dump(store_data_info, f)
        print(f"Data info saved to {data_info_file}")
        with open(vqa_input_jsonl_file, "w") as f:
            for content in content_list:
                f.write(json.dumps(content) + "\n")
        print(f"Content list saved to {vqa_input_jsonl_file}")

    data = None # clear the data to avoid resusing it later
    # read the stored data info
    store_data_info = pickle.load(open(data_info_file, "rb"))
    
    metadata = {}
    if not os.path.exists(metadata_file):
        batch_id = vlm_generator.generate_queries_batch(
            jsonl_file=vqa_input_jsonl_file,
            display_name="vqa-batch-job",
        )
        metadata["vqa_batch_id"] = batch_id
        metadata["vqa_input_jsonl_file"] = vqa_input_jsonl_file
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=4)

    
    metadata = json.load(open(metadata_file, "r"))
    if not os.path.exists(vqa_output_jsonl_file):
        batch_id = metadata["vqa_batch_id"]
        print(f"Batch job ID: {batch_id}")
        start_time = time.time()
        vlm_generator.wait_and_save_for_batch_completion(
            batch_id=batch_id,
            output_file=vqa_output_jsonl_file,
            poll_interval=60,
        )
        end_time = time.time()
        metadata["vqa_output_jsonl_file"] = vqa_output_jsonl_file
        metadata["vqa_batch_time"] = (end_time - start_time) / 60
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=4)

    batch_id = metadata["vqa_batch_id"]
    vqa_output_jsonl_file = metadata["vqa_output_jsonl_file"]
    vqa_batch_time = metadata["vqa_batch_time"]
    print(f"Batch job ID: {batch_id}")
    print(f"VQA JSONL file: {vqa_output_jsonl_file}")
    print(f"VQA batch time: {vqa_batch_time} minutes")
    
    # loading the generated responses from the vqa_output_jsonl_file
    with open(vqa_output_jsonl_file, "r") as f:
        vqa_output_jsonl = [json.loads(line) for line in f]

    filter_output_list = []
    store_eval_data_info = {}
    # jsonl file for the filter output
    eval_input_jsonl_file = save_path.replace(".json", "_eval_input.jsonl")
    eval_store_data_info_file = save_path.replace(".json", "_eval_data_info.pkl")
    eval_output_jsonl_file = save_path.replace(".json", "_eval_output.jsonl")
    print("*"*20)
    print(f"{eval_input_jsonl_file=}")
    print(f"{eval_output_jsonl_file=}")
    print(f"{eval_store_data_info_file=}")
    print("*"*20)
    # assert the length of vqa_json is the same as the length of store_data_info
    if not os.path.exists(eval_input_jsonl_file):
        assert len(vqa_output_jsonl) == len(store_data_info), f"Length of vqa_output_jsonl {len(vqa_output_jsonl)} and store_data_info {len(store_data_info)} do not match"
        for vqa_response in vqa_output_jsonl:
            custom_id = vqa_response["custom_id"] if "custom_id" in vqa_response.keys() else vqa_response["key"]
            data_info = store_data_info[custom_id]
            frame_indices = data_info["prompt_info"]["frame_indices"]
            prompts = data_info["prompt_info"]["prompts"]
            videos = data_info["prompt_info"]["videos"]
            data_info_for_eval = data_info["data_info"]
            output = vlm_generator.gather_text_from_response(vqa_response['response'], 1) # done previously inside the vlm_generate.generate_queries
            response_list = [vlm_generator.extract_json_from_response(out) for out in output]
            response_list = [res for res in response_list if len(res.keys()) > 0]
            if len(response_list) == 0:
                continue
            assert len(response_list) == 1, f"Expected 1 response list, got {len(response_list)}. If this changes, need to match frame_indices with the response list."
            real_response_list = [] 
            for res in response_list:
                if "results" not in res.keys():
                    continue
                real_response_list.extend(res["results"])
            if len(real_response_list) != len(frame_indices):
                print(f"Length of real_response_list {len(real_response_list)} and frame_indices {len(frame_indices)} do not match for {data_info_for_eval['hdf5_path']} {data_info_for_eval['demo_key']}")
                print(f"{real_response_list=}\n{frame_indices=}")
                print(f"Skipping this data point")
                continue
            real_response_list = [{"id": i+1, **res} for i, res in enumerate(real_response_list)]
            response_string = json.dumps(real_response_list, indent=4)
            response_string = "Here are the candidate items to evaluate:\n" + response_string

            base_prompt = get_eval_multi_prompt(args.mode, data_info_for_eval)
            summary_string = generate_summary_string(data_info_for_eval["summary_list"])
            base_prompt = base_prompt + summary_string
            filter_output = vlm_generator_eval.generate_queries(
                base_prompt=base_prompt,
                prompts=prompts,
                videos=videos,
                end_text=response_string,
                n=1,
                return_content=True,
                custom_id=custom_id,
            )
            data_info["prompt_info"]["real_response_list"] = real_response_list
            filter_output_list.append(filter_output)
            store_eval_data_info[custom_id] = data_info
        with open(eval_input_jsonl_file, "w") as f:
            for filter_output in filter_output_list:
                f.write(json.dumps(filter_output) + "\n")
        with open(eval_store_data_info_file, "wb") as f:
            pickle.dump(store_eval_data_info, f)

        # submit the job to the batch queue
        eval_batch_id = vlm_generator_eval.generate_queries_batch(
            jsonl_file=eval_input_jsonl_file,
            display_name="vqa-eval-batch-job",
        )
        metadata["vqa_eval_batch_id"] = eval_batch_id
        metadata["vqa_eval_input_jsonl_file"] = eval_input_jsonl_file
        metadata["vqa_eval_store_data_info_file"] = eval_store_data_info_file
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=4)
    
    if not os.path.exists(eval_output_jsonl_file):
        # wait for the job to finish
        eval_batch_id = metadata["vqa_eval_batch_id"]
        start_time = time.time()
        vlm_generator_eval.wait_and_save_for_batch_completion(
            batch_id=eval_batch_id,
            output_file=eval_output_jsonl_file,
            poll_interval=60,
        )
        end_time = time.time()
        metadata["vqa_eval_batch_time"] = (end_time - start_time) / 60

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=4)

    if not os.path.exists(save_path):
        # loading the generated responses
        with open(eval_output_jsonl_file, "r") as f:
            eval_output_jsonl = [json.loads(line) for line in f]
        with open(eval_store_data_info_file, "rb") as f:
            store_eval_data_info = pickle.load(f)
        assert len(eval_output_jsonl) == len(store_eval_data_info), f"Length of eval_output_jsonl {len(eval_output_jsonl)} and store_eval_data_info {len(store_eval_data_info)} do not match"
        dataset_all_values = []
        for eval_output in eval_output_jsonl:
            custom_id = eval_output["custom_id"] if "custom_id" in eval_output.keys() else eval_output["key"]
            data_info = store_eval_data_info[custom_id]
            output = vlm_generator_eval.gather_text_from_response(eval_output['response'], 1) # done previously inside the vlm_generate.generate_queries
            assert len(output) == 1, f"Expected 1 output, got {len(output)}"
            filter_json = vlm_generator.extract_json_from_response(output[0])
            selected_response = gather_selected_response_from_filtered_results(
                filter_json=filter_json,
                real_response_list=data_info["prompt_info"]["real_response_list"],
                frame_indices=data_info["prompt_info"]["frame_indices"],
                data=data_info["data_info"],
                debug=args.debug,
            )
            if len(selected_response) == 0:
                print(f"No selected response found for {data_info['data_info']['hdf5_path']} {data_info['data_info']['demo_key']}")
                continue
            common_info = data_info["common_info"]
            common_info["timestep_index"] = ",".join(map(str, common_info["timestep_index"]))
            if len(common_info.keys()) == 0:
                raise Exception(f"Common info is empty for {data_info['data_info']['hdf5_path']} {data_info['data_info']['demo_key']}")
            if args.debug:
                print(f"{common_info.keys()=}, {selected_response[0].keys()=}")
            for res in selected_response:
                dataset_all_values.append({**res, **common_info})
        # saving the results
        save_qa_results(dataset_all_values, save_path)

    return None
if __name__ == "__main__":
    main()
