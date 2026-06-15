import os
import json
import pickle
import h5py
import numpy as np
import torch
from PIL import Image
from transformers import Sam3VideoModel, Sam3VideoProcessor
from accelerate import Accelerator
from tqdm import tqdm
import argparse
import cv2
from termcolor import colored

# CUDA_VISIBLE_DEVICES=7 python scripts/data_gen/generate_sam3_ids.py --hdf5_path $CASAPLAY_DATAROOT/memory/mutex/MEM1/MEM1_Wash_the_bowl_and_place_it_back_in_the_same_container.hdf5 --prompts "green plate" "pink plate" "white small bowl" "sink" "robot"
# CUDA_VISIBLE_DEVICES=7 python scripts/data_gen/generate_sam3_ids.py --hdf5_path $CASAPLAY_DATAROOT/memory/mutex/MEM1/MEM1_Wash_the_bowl_and_place_it_back_in_the_same_container.hdf5 --prompts "green plate" "pink plate" "white small bowl" "sink" "robot" --start_frame 400 --max_frames 10 --fps 1 --debug
# Import utilities from the codebase
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
robocasa_benchmark_names = ["v0.1/single_stage", "memory"]

# keep n different known colors for the object ids
known_colors = {
    0: (0, 0, 255),
    1: (0, 255, 0),
    2: (255, 0, 0),
    3: (255, 255, 0),
    4: (128, 0, 128),
    5: (255, 165, 0),
    6: (165, 42, 42),
    7: (255, 192, 203),
    8: (128, 128, 128),
}
max_object_id = 32
# extend it to 30 colors randomly
for obj_id in np.arange(max_object_id):
    if obj_id not in known_colors:
        known_colors[obj_id] = tuple(np.random.randint(0, 255, size=3, dtype=np.uint8))
colors = {obj_id: known_colors[obj_id] for obj_id in np.arange(max_object_id)}

def get_dataset_domain_name(path):
    if 'mutex' in path:
        return 'mutex' # still inside the robocasa dataset
    if 'robocasa' in path:
        return 'robocasa'
    if any(name in path for name in robocasa_benchmark_names):
        return 'robocasa'
    return None


def get_data_base_dir(hdf5_path, shared_config = None):
    domain_name = get_dataset_domain_name(hdf5_path)
    if domain_name == 'robocasa':
        data_dir = os.environ["CASAPLAY_DATAROOT"]
        if shared_config is not None:
            assert shared_config.has_base_action, "Robocasa dataset has base action"
    elif domain_name == 'mutex':
        data_dir = os.environ["CASAPLAY_DATAROOT"]
    else:
        raise ValueError(f"Unknown domain name: {domain_name}")
    return data_dir


def move_to_cpu(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.to("cpu", non_blocking=True)
    elif isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, list):
        return [move_to_cpu(v) for v in tensor]
    elif isinstance(tensor, dict):
        return {k: move_to_cpu(v) for k, v in tensor.items()}
    elif isinstance(tensor, tuple):
        return tuple(move_to_cpu(v) for v in tensor)
    elif isinstance(tensor, set):
        return set(move_to_cpu(v) for v in tensor)
    elif isinstance(tensor, (int, float, bool)):
        return tensor
    else:
        raise ValueError(f"Unsupported type: {type(tensor)}")

def separate_mask_by_connected_components(mask, min_distance_ratio=0.05):
    """
    Separate a mask into individual connected components.
    Components that are too close to the main component (largest area) are merged with it.
    
    Args:
        mask: Binary mask (numpy array, bool or uint8)
        min_distance_ratio: Minimum distance ratio (default: 0.05 = 5% of image dimension)
    
    Returns:
        List of masks, one for each separated component
    """
    # Convert mask to uint8 if needed
    if mask.dtype == bool:
        mask_uint8 = mask.astype(np.uint8) * 255
    elif mask.dtype == np.uint8:
        mask_uint8 = mask
    else:
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
    
    # Get image dimensions
    H, W = mask.shape[:2]
    min_dimension = min(H, W)
    min_distance = min_dimension * min_distance_ratio
    
    # Find connected components
    num_labels, labels = cv2.connectedComponents(mask_uint8, connectivity=8)
    
    if num_labels <= 1:
        # No components found
        return []
    
    # Extract each connected component and calculate its area
    component_masks = []
    component_areas = []
    component_coords = []  # Store coordinates for distance calculation
    
    for label_id in range(1, num_labels):  # Skip label 0 (background)
        component_mask = (labels == label_id).astype(np.bool_)
        component_masks.append(component_mask)
        
        # Calculate area
        area = np.sum(component_mask)
        component_areas.append(area)
        
        # Get coordinates of all pixels in this component
        coords = np.where(component_mask)
        component_coords.append(np.array(list(zip(coords[0], coords[1]))))
    
    # Find the main component (largest area)
    main_idx = np.argmax(component_areas)
    main_component = component_masks[main_idx].copy()
    main_coords = component_coords[main_idx]
    
    # Separate components: keep main component and others that are far enough
    separated_components = [main_component]
    
    for idx, component_mask in enumerate(component_masks):
        if idx == main_idx:
            continue  # Skip the main component
        
        # Calculate minimum distance between this component and the main component
        # More efficient: compute distance between closest points
        other_coords = component_coords[idx]
        
        # Compute minimum distance more efficiently
        # For each point in other component, find distance to nearest point in main component
        min_dist = float('inf')
        for other_point in other_coords:
            # Compute distances to all main points
            dy = main_coords[:, 0] - other_point[0]
            dx = main_coords[:, 1] - other_point[1]
            dists = np.sqrt(dy**2 + dx**2)
            min_dist_to_main = np.min(dists)
            min_dist = min(min_dist, min_dist_to_main)
        
        if min_dist > min_distance:
            # Keep as separate component
            separated_components.append(component_mask)
        else:
            # Merge with main component
            main_component = np.logical_or(main_component, component_mask)
            separated_components[0] = main_component
    
    return separated_components


def clean_bbox_and_masks(outputs_dict, min_mask_area_ratio=0.01, debug=False):
    """
    Clean bounding boxes and masks by removing:
    1. Masks that cover less than min_mask_area_ratio (default 1.2%) of the total image area
    2. Bounding boxes with zero height or width
    
    Args:
        outputs_dict: Dictionary with keys:
            - {bbox_key}_mask: (T, N) boolean array
            - {bbox_key}_names: (T, N) array of object names
            - {bbox_key}_cc: (T, N, 2) center coordinates
            - {bbox_key}: (T, N, 4) bounding boxes [x_min, y_min, x_max, y_max]
            - {bbox_key.replace('bbox', 'mask_instances')}: (T, N, H, W) segmentation masks
        min_mask_area_ratio: Minimum ratio of mask area to total image area (default: 0.012 = 1.2%)
    
    Returns:
        Cleaned outputs_dict with invalid entries zeroed out and masks set to False
    """
    # Find the mask_instances key
    mask_instances_key = None
    bbox_key = None
    for key in outputs_dict.keys():
        if 'mask_instances' in key:
            mask_instances_key = key
            # Extract bbox_key from mask_instances_key
            bbox_key = key.replace('mask_instances', 'bbox')
            break
    
    if mask_instances_key is None:
        print("[WARNING] No mask_instances key found, skipping cleaning")
        return outputs_dict
    
    # Get all related keys
    bbox_mask_key = f"{bbox_key}_mask"
    bbox_names_key = f"{bbox_key}_names"
    bbox_cc_key = f"{bbox_key}_cc"
    bbox_key_full = bbox_key
    
    # Verify all required keys exist
    required_keys = [mask_instances_key, bbox_mask_key, bbox_names_key, bbox_cc_key, bbox_key_full]
    missing_keys = [key for key in required_keys if key not in outputs_dict]
    if missing_keys:
        print(f"[WARNING] Missing required keys for cleaning: {missing_keys}, skipping cleaning")
        return outputs_dict
    
    # Extract arrays
    mask_instances = outputs_dict[mask_instances_key]  # (T, N, H, W)
    bbox_mask = outputs_dict[bbox_mask_key]  # (T, N)
    bbox_names = outputs_dict[bbox_names_key]  # (T, N)
    bbox_cc = outputs_dict[bbox_cc_key]  # (T, N, 2)
    bboxes = outputs_dict[bbox_key_full]  # (T, N, 4)
    
    T, N = bbox_mask.shape
    H, W = mask_instances.shape[2], mask_instances.shape[3]
    total_image_area = H * W
    min_mask_area = total_image_area * min_mask_area_ratio
    
    # Preserve original dtypes
    mask_instances_dtype = mask_instances.dtype
    bboxes_dtype = bboxes.dtype
    bbox_cc_dtype = bbox_cc.dtype
    
    num_cleaned = 0
    
    # Process each frame separately to reorganize valid/invalid entries
    for t in range(T):
        # First pass: identify valid and invalid entries
        valid_indices = []
        invalid_indices = []
        
        for n in range(N):
            # Skip if already marked as invalid
            if not bbox_mask[t, n]:
                invalid_indices.append(n)
                continue
            
            # Check mask area
            mask = mask_instances[t, n]  # (H, W)
            mask_area = np.sum(mask.astype(np.float32))
            
            # Check bounding box dimensions
            bbox = bboxes[t, n]  # [x_min, y_min, x_max, y_max]
            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]
            
            # Determine if this entry should be removed
            should_remove = False
            
            if mask_area < min_mask_area:
                # print the corresponding name
                name = bbox_names[t, n]
                if debug:
                    print(colored(f"Removing mask {n} with area {mask_area} because it is less than {min_mask_area} for {name}", "red"))
                should_remove = True
                num_cleaned += 1
            
            if (bbox_width <= 0 or bbox_height <= 0) and (mask_area >= min_mask_area):
                name = bbox_names[t, n]
                if debug:
                    print(colored(f"Removing mask {n} with area {mask_area} because it has zero width or height for {name}", "red"))
                should_remove = True
                num_cleaned += 1
            
            if should_remove:
                invalid_indices.append(n)
            else:
                valid_indices.append(n)
        
        # Reorganize: valid entries first, then invalid entries
        # Create new order: [valid_indices..., invalid_indices...]
        new_order = valid_indices + invalid_indices
        
        # Reorder all arrays for this frame
        bbox_mask[t, :] = bbox_mask[t, new_order]
        bboxes[t, :] = bboxes[t, new_order]
        bbox_cc[t, :] = bbox_cc[t, new_order]
        bbox_names[t, :] = bbox_names[t, new_order]
        mask_instances[t, :] = mask_instances[t, new_order]
        
        # Zero out invalid entries (now at the end)
        num_valid = len(valid_indices)
        for idx in range(num_valid, N):
            bbox_mask[t, idx] = False
            bboxes[t, idx] = np.zeros(4, dtype=bboxes_dtype)
            bbox_cc[t, idx] = np.zeros(2, dtype=bbox_cc_dtype)
            # Handle bbox_names - it's typically object dtype for byte strings
            bbox_names[t, idx] = b""
            mask_instances[t, idx] = np.zeros((H, W), dtype=mask_instances_dtype)
    
    if num_cleaned > 0:
        print(f"    Cleaned {num_cleaned} invalid bounding boxes/masks (out of {T * N} total)")
    
    return outputs_dict


def organize_outputs_by_id(all_frames_outputs, max_num_objects=32, image_key="", min_mask_area_ratio=0.012, debug=False):
    """
    Reverse prompt_to_obj_ids mapping, order strings by increasing IDs, and extract bounding boxes.

    Args:
        all_frames_outputs: Dictionary mapping frame_idx to SAM3 outputs
        max_num_objects: Maximum number of objects per frame (default: 32)
        image_key: Image observation key (e.g., "obs/agentview_rgb")
        min_mask_area_ratio: Minimum ratio of mask area to total image area (default: 0.012 = 1.2%)

    Returns:
        Dictionary with organized outputs including masks, bounding boxes, and names
    """
    all_frames_object_strings = []
    all_frames_object_boxes = []
    all_frames_segmentation_masks = []
    all_frames_masks = []
    max_id_value = -1
    for key, processed_outputs in all_frames_outputs.items():
        prompt_to_obj_ids = processed_outputs['prompt_to_obj_ids']
        print(f"{prompt_to_obj_ids=}")
        # Filter out empty lists and compute max only if there are non-empty lists
        non_empty_ids = [ids for ids in prompt_to_obj_ids.values() if len(ids) > 0]
        if len(non_empty_ids) > 0:
            frame_max_id = max(max(ids) for ids in non_empty_ids)
            max_id_value = max(frame_max_id, max_id_value)

    for key, processed_outputs in all_frames_outputs.items():
        masks = processed_outputs['masks']
        assert max_id_value >= len(masks) - 1, f"Max ID value {max_id_value} does not match the number of masks {len(masks)}"

        n_masks = len(masks)
        # for _ in range(max_id_value+1):
        # Reverse mapping: id -> list of strings
        object_ids = processed_outputs['object_ids']
        assert n_masks == len(object_ids), f"Number of masks {n_masks} does not match the number of object ids {len(object_ids)}"

        ordered_strings = []
        ordered_boxes = []
        ordered_masks = []
        ordered_segmentation_masks = []

        prompt_object_ids =processed_outputs['prompt_to_obj_ids']
        for pos, obj_id in enumerate(object_ids):
            string = [key for key, ids in prompt_object_ids.items() if obj_id in ids]
            assert len(string) == 1, f"Multiple strings found for object id {obj_id}, prompt_to_obj_ids: {prompt_to_obj_ids}"
            string = string[0]
            mask = masks[pos]

            # Extract mask as numpy array
            if isinstance(mask, torch.Tensor):
                mask_np = mask.numpy()
            else:
                mask_np = mask
            
            # Separate mask into connected components
            component_masks = separate_mask_by_connected_components(mask_np)
            
            # If no connected components found, skip this mask
            if len(component_masks) == 0:
                continue
            
            if len(component_masks) > 1:
                print(colored(f"Found {len(component_masks)} connected components for {string}", "red"))
            # Add each connected component as a separate object
            for component_mask in component_masks:
                # Find non-zero coordinates for this component
                coords = np.where(component_mask > 0)
                if len(coords[0]) > 0:
                    y_min, y_max = coords[0].min(), coords[0].max()
                    x_min, x_max = coords[1].min(), coords[1].max()
                    bbox = np.array([x_min, y_min, x_max, y_max])
                else:
                    bbox = np.zeros((4))

                ordered_strings.append(string.encode('utf-8'))
                ordered_boxes.append(bbox)
                ordered_masks.append(True)
                ordered_segmentation_masks.append(component_mask.astype(np.bool_))

        # we need to pad the lists to the max_num_objects
        while len(ordered_strings) < max_num_objects:
            ordered_strings.append(b"")
            ordered_boxes.append(np.zeros((4)))
            ordered_masks.append(False)
            height, width = 128, 128
            if len(ordered_segmentation_masks) > 0: # safer to use the last mask's shape
                height, width = ordered_segmentation_masks[-1].shape
            ordered_segmentation_masks.append(np.zeros((height, width), dtype=np.bool_))

        if len(ordered_strings) > max_num_objects:
            print(f"[WARNING]! More than {max_num_objects} objects detected in frame {key}")
            ordered_strings = ordered_strings[:max_num_objects]
            ordered_boxes = ordered_boxes[:max_num_objects]
            ordered_masks = ordered_masks[:max_num_objects]
            ordered_segmentation_masks = ordered_segmentation_masks[:max_num_objects]

        all_frames_object_strings.append(ordered_strings)
        all_frames_object_boxes.append(ordered_boxes)
        all_frames_segmentation_masks.append(ordered_segmentation_masks)
        all_frames_masks.append(ordered_masks)

    all_frames_object_strings = np.array(all_frames_object_strings)
    all_frames_object_boxes = np.array(all_frames_object_boxes)
    all_frames_segmentation_masks = np.array(all_frames_segmentation_masks)
    all_frames_masks = np.array(all_frames_masks)
    all_frames_cc = np.stack([(all_frames_object_boxes[..., 0] + all_frames_object_boxes[..., 2])/2, (all_frames_object_boxes[..., 1] + all_frames_object_boxes[..., 3])/2], axis=-1)

    bbox_key = image_key.replace("image", "bbox").replace("rgb", "bbox")
    outputs_dict = {
        f"{bbox_key}_mask": all_frames_masks,
        f"{bbox_key}_names": all_frames_object_strings,
        f"{bbox_key}_cc": all_frames_cc,
        f"{bbox_key}": all_frames_object_boxes,
        f"{bbox_key.replace('bbox', 'mask_instances')}": all_frames_segmentation_masks,
    }
    
    # Clean the outputs before returning
    outputs_dict = clean_bbox_and_masks(outputs_dict, min_mask_area_ratio=min_mask_area_ratio, debug=debug)
    
    return outputs_dict


def save_outputs_to_hdf5(outputs, hdf5_path, demo_key):
    """
    Safely save outputs to HDF5, writing to a temp file first and only moving to the original if successful.
    """
    import tempfile, shutil, os

    # Work in a temporary directory in the same location as the hdf5 to avoid cross-filesystem moves
    hdf5_dir = os.path.dirname(hdf5_path)
    temp_dir = tempfile.mkdtemp(dir=hdf5_dir)
    temp_hdf5_path = os.path.join(temp_dir, "temp_hdf5.h5")

    try:
        # Copy original to temp file
        shutil.copy2(hdf5_path, temp_hdf5_path)
        with h5py.File(temp_hdf5_path, "r+") as f:
            data_group = f["data"][demo_key]
            for key, value in outputs.items():
                # if key exists, delete it first
                if key in data_group:
                    del data_group[key]
                data_group.create_dataset(key, data=value, compression="gzip")
            f.flush()  # make sure everything is written

        # Move temp file over original atomically if possible
        shutil.move(temp_hdf5_path, hdf5_path)
        print(f"Safely saved outputs to {hdf5_path} in group {demo_key}")
        shutil.rmtree(temp_dir)
        return True

    except Exception as e:
        print(f"Error occurred while writing to HDF5 (no changes made to {hdf5_path}): {e}")
        if os.path.exists(temp_hdf5_path):
            os.remove(temp_hdf5_path)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return False


def process_video_with_sam3(video_frames, processor, model, device, text_prompts):
    """
    Process video frames with SAM3 using text prompts.
    Resets the predictor every 60 frames to avoid drifting errors.

    Args:
        video_frames: List of PIL Images or numpy arrays
        processor: SAM3VideoProcessor
        model: Sam3VideoModel
        device: torch device
        text_prompts: List of text prompts (e.g., ["sink", "plate", "bowl"])

    Returns:
        Dictionary mapping frame_idx to outputs with masks for each object ID
    """
    # Convert frames to PIL Images if needed
    if isinstance(video_frames[0], np.ndarray):
        # Ensure frames are in RGB format (not BGR) and uint8
        pil_frames = []
        for frame in video_frames:
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            # Convert to PIL Image (assumes RGB format)
            pil_frames.append(Image.fromarray(frame))
        video_frames = pil_frames

    total_frames = len(video_frames)
    batch_size = 30
    outputs_per_frame = {}
    
    # Process frames in batches of 60
    # Note: The min() ensures the last batch handles any remainder frames correctly
    # (e.g., if total_frames=626, the last batch will process frames[600:626] = 26 frames)
    for batch_start in range(0, total_frames, batch_size):
        batch_end = min(batch_start + batch_size, total_frames)
        batch_frames = video_frames[batch_start:batch_end]

        # # Close the session to avoid drifting errors
        # processor.clear_all_points_in_video(inference_session)
        
        # Initialize video inference session for this batch
        inference_session = processor.init_video_session(
            video=batch_frames,
            inference_device=device,
            processing_device="cuda",
            video_storage_device="cuda",
            dtype=torch.bfloat16,
        )

        # Add all text prompts
        for text in text_prompts:
            inference_session = processor.add_text_prompt(
                inference_session=inference_session,
                text=text,
            )

        # Try to get predictor from inference_session, otherwise use model
        if hasattr(inference_session, 'predictor'):
            predictor = inference_session.predictor
        elif hasattr(model, 'handle_request'):
            predictor = model
        elif hasattr(processor, 'predictor'):
            predictor = processor.predictor
        else:
            # Fallback: assume model has handle_request method
            predictor = model

        # Process frames in this batch
        for model_outputs in model.propagate_in_video_iterator(
            inference_session=inference_session, max_frame_num_to_track=len(batch_frames)
        ):
            processed_outputs = processor.postprocess_outputs(inference_session, model_outputs)
            processed_outputs = move_to_cpu(processed_outputs)
            # Adjust frame_idx to be relative to the original video
            original_frame_idx = batch_start + model_outputs.frame_idx
            outputs_per_frame[original_frame_idx] = processed_outputs

    return outputs_per_frame


def extract_frames_from_hdf5(hdf5_file, demo_key, image_key, max_frames=-1, start_frame=0, flip_image_channel=False):
    """
    Extract video frames from HDF5 file for a specific episode and camera.

    Args:
        hdf5_file: Open HDF5 file object
        demo_key: Episode/demo key (e.g., "demo_0")
        image_key: Image observation key (e.g., "obs/agentview_rgb")

    Returns:
        List of numpy arrays representing video frames
    """
    data_group = hdf5_file["data"][demo_key]

    # Handle different possible structures
    frames = None

    # Try with obs/ prefix handling
    if "obs" in data_group:
        # Check if key without "obs/" prefix exists in obs group
        key_without_prefix = image_key.replace("obs/", "")
        if key_without_prefix in data_group["obs"]:
            frames = np.array(data_group["obs"][key_without_prefix][:])
        # Try with full "obs/" prefix
        elif image_key in data_group["obs"]:
            frames = np.array(data_group["obs"][image_key][:])
    frames = frames[start_frame:]

    if frames is None:
        # Try to find the key by checking all available keys
        available_keys = list(data_group.keys())
        if "obs" in data_group:
            available_keys.extend([f"obs/{k}" for k in data_group["obs"].keys()])
        raise ValueError(
            f"Image key {image_key} not found in episode {demo_key}. "
            f"Available keys: {available_keys}"
        )

    # Ensure frames are in the right format (T, H, W, C)
    if len(frames.shape) == 4:
        # Already in (T, H, W, C) format
        pass
    elif len(frames.shape) == 3:
        # Single frame, add time dimension
        frames = frames[None, ...]
    else:
        raise ValueError(f"Unexpected frame shape: {frames.shape}")

    # Convert to uint8 if needed
    if frames.dtype != np.uint8:
        if frames.max() <= 1.0:
            frames = (frames * 255).astype(np.uint8)
        else:
            frames = frames.astype(np.uint8)
    if flip_image_channel:
        assert frames.shape[-1] == 3, "Frames must have 3 channels to flip the channel"
        frames = frames[..., ::-1] # BGR to RGB
    if max_frames > 0:
        frames = frames[:max_frames]

    # Convert to list of numpy arrays
    frame_list = [frames[i] for i in range(frames.shape[0])]
    return frame_list


def overlay_mask(frame, mask, color):
    """
    Overlay a mask on a frame with a given color.

    Args:
        frame: PIL Image or numpy array
        mask: Boolean or 0-1 mask (numpy array or torch tensor)
        color: RGB tuple for overlay color

    Returns:
        PIL Image with mask overlaid
    """
    # Ensure frame is a PIL Image
    if isinstance(frame, np.ndarray):
        frame = Image.fromarray(frame)
    frame = frame.convert("RGB")

    # Convert mask to numpy if needed
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    # Convert boolean/0-1 mask to a single-channel (L) image for use as an alpha mask
    mask_img = Image.fromarray(mask.astype(np.uint8) * 255).convert("L")

    # Create a solid color image to overlay
    color_img = Image.new("RGB", frame.size, color)

    # Paste the color image onto the frame using the mask as transparency
    frame.paste(color_img, mask=mask_img)
    return frame


def visualize_and_save_video(saved_outputs, video_frames, output_dir, hdf5_name, demo_key, image_key, fps=30, prompt_to_uniq_color_id={}):
    """
    Visualize video by overlaying masks on frames and save as MP4 video.

    Args:
        saved_outputs: Dictionary with organized outputs from organize_outputs_by_id
        video_frames: List of original video frames (PIL Images or numpy arrays)
        output_dir: Directory to save output video
        hdf5_name: Name of the original HDF5 file (without path)
        demo_key: Episode/demo key
        image_key: Image observation key
        fps: Frames per second for output video (default: 30)
        prompt_to_uniq_color_id: Dictionary mapping prompt strings to color IDs
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract bbox_key from image_key
    bbox_key = image_key.replace("image", "bbox").replace("rgb", "bbox")
    
    # Get the keys from saved_outputs
    bbox_mask_key = f"{bbox_key}_mask"
    bbox_names_key = f"{bbox_key}_names"
    mask_instances_key = f"{bbox_key.replace('bbox', 'mask_instances')}"
    
    # Verify required keys exist
    if bbox_mask_key not in saved_outputs or bbox_names_key not in saved_outputs or mask_instances_key not in saved_outputs:
        print(f"Warning: Required keys not found in saved_outputs for {demo_key}/{image_key}")
        print(f"Available keys: {list(saved_outputs.keys())}")
        return
    
    # Extract arrays
    bbox_mask = saved_outputs[bbox_mask_key]  # (T, N)
    bbox_names = saved_outputs[bbox_names_key]  # (T, N)
    mask_instances = saved_outputs[mask_instances_key]  # (T, N, H, W)
    
    T, N = bbox_mask.shape
    
    # Get all unique object names across all frames
    all_object_names = set()
    for t in range(T):
        for n in range(N):
            if bbox_mask[t, n]:
                name = bbox_names[t, n]
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                elif isinstance(name, np.ndarray):
                    name = name.tobytes().decode('utf-8')
                if name:
                    all_object_names.add(name)
    
    if len(all_object_names) == 0:
        print(f"Warning: No objects detected for {demo_key}/{image_key}")
        return
    
    print(f"    Unique object names: {sorted(all_object_names)}")

    # Get video dimensions from first frame
    first_frame = video_frames[0]
    if isinstance(first_frame, np.ndarray):
        height, width = first_frame.shape[:2]
    else:
        width, height = first_frame.size

    # Create output video path
    video_filename = f"{hdf5_name.replace('.hdf5', '')}_{demo_key}_{image_key.replace('/', '_')}_overlay.mp4"
    video_path = os.path.join(output_dir, video_filename)

    # Adjust fps if video is too short
    num_frames = len(video_frames)
    actual_fps = fps
    if num_frames < fps:
        actual_fps = max(1, num_frames)  # At least 1 fps, or number of frames if less

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, actual_fps, (width, height))

    if not video_writer.isOpened():
        print(f"Error: Could not open video writer for {video_path}")
        return

    # Process all frames and write to video
    for frame_idx in tqdm(range(min(num_frames, T)), desc=f"    Processing frames", leave=False):
        # Get original frame
        frame = video_frames[frame_idx]
        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(frame)
        frame = frame.convert("RGB")

        # Overlay all valid masks for this frame
        for n in range(N):
            if bbox_mask[frame_idx, n]:
                mask = mask_instances[frame_idx, n]  # (H, W)
                name = bbox_names[frame_idx, n]
                
                # Decode name if needed
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                elif isinstance(name, np.ndarray):
                    name = name.tobytes().decode('utf-8')
                
                # Get color for this object name
                if name in prompt_to_uniq_color_id:
                    color_id = prompt_to_uniq_color_id[name]
                    frame = overlay_mask(frame, mask, colors[color_id])
                else:
                    # Use a default color if name not found in mapping
                    print(f"Warning: Object name '{name}' not found in prompt_to_uniq_color_id, using default color")
                    frame = overlay_mask(frame, mask, colors[0])

        # Convert PIL Image to numpy array (RGB -> BGR for OpenCV)
        frame_np = np.array(frame)
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

        # Write frame to video
        video_writer.write(frame_bgr)

    # Release video writer
    video_writer.release()
    print(f"    Saved video to {video_path}")


def process_hdf5_file(hdf5_path, image_keys=None, text_prompts=["sink", "plate", "bowl"], output_dir=None, fps=30, args=None):
    """
    Process a single HDF5 file with SAM3 and visualize results.

    Args:
        hdf5_path: Path to HDF5 file (can be relative or absolute)
        image_keys: List of image observation keys. If None, will try to discover them.
        text_prompts: List of text prompts for SAM3
        output_dir: Directory to save output videos (default: same as input with _sam3_viz suffix)
        fps: Frames per second for output videos (default: 30)
        args: Command-line arguments object (must have min_mask_area_ratio attribute)
    """
    # assign each text prompt a unique id
    prompt_to_uniq_color_id = {
        prompt: i for i, prompt in enumerate(text_prompts)
    }
    # Get full path
    assert os.path.isabs(hdf5_path), "HDF5 path must be absolute"
    full_hdf5_path = hdf5_path

    if not os.path.exists(full_hdf5_path):
        raise FileNotFoundError(f"HDF5 file not found: {full_hdf5_path}")

    print(f"Processing: {full_hdf5_path}")
    hdf5_name = os.path.basename(hdf5_path)

    # Set output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(full_hdf5_path), "sam3_viz")
    os.makedirs(output_dir, exist_ok=True)

    # Open HDF5 file to discover image keys if needed
    with h5py.File(full_hdf5_path, "r") as f:
        if image_keys is None:
            first_demo = sorted(f["data"].keys(), key=lambda x: int(x.split('_')[-1]))[0]
            ep = f["data"][first_demo]
            image_keys = sorted(set([k for k in ep.keys() if any(t in k.lower() for t in ['image', 'rgb', 'camera'])] +
                                    [f"obs/{k}" for k in ep.get("obs", {}).keys() if any(t in k.lower() for t in ['image', 'rgb', 'camera'])]))
            if not image_keys:
                raise ValueError(f"Could not discover image keys. Available keys: {list(ep.keys())}. Please specify using --image_keys")

        if isinstance(image_keys, str):
            image_keys = [image_keys]

        print(f"Image keys: {image_keys}")
        print(f"Text prompts: {text_prompts}")

        demo_keys = sorted(f["data"].keys(), key=lambda x: int(x.split('_')[-1]))
        print(f"Found {len(demo_keys)} episodes")

    # Setup SAM3 model and processor
    device = Accelerator().device
    print(f"Loading SAM3 model on {device}...")
    model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)
    processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
    print("SAM3 model loaded successfully")

    all_demo_keys = []
    with h5py.File(full_hdf5_path, "r") as f:
        all_demo_keys = sorted(f["data"].keys(), key=lambda x: int(x.split('_')[-1]))

    for demo_key in tqdm(all_demo_keys, desc=f"Processing episodes in {hdf5_name}"):
        for image_key in image_keys:
            # Process each episode
            with h5py.File(full_hdf5_path, "r") as f:
                # Extract frames
                frames = extract_frames_from_hdf5(f, demo_key, image_key, max_frames=args.max_frames, start_frame=args.start_frame, flip_image_channel=args.flip_image_channel)
                print(f"  Processing {demo_key}/{image_key}: {len(frames)} frames")

            # Convert frames to PIL Images for visualization
            pil_frames = []
            for frame in frames:
                if isinstance(frame, np.ndarray):
                    if frame.dtype != np.uint8:
                        if frame.max() <= 1.0:
                            frame = (frame * 255).astype(np.uint8)
                        else:
                            frame = frame.astype(np.uint8)
                    pil_frames.append(Image.fromarray(frame))
                else:
                    pil_frames.append(frame)

            # # Process with SAM3
            outputs_per_frame = process_video_with_sam3(
                pil_frames, processor, model, device, text_prompts
            )

            # # dump it as /tmp/sam3.pkl
            # with open("/tmp/sam3.pkl", "wb") as f:
            #     pickle.dump(outputs_per_frame, f)
            # # load it from /tmp/sam3.pkl
            # with open("/tmp/sam3.pkl", "rb") as f:
            #     outputs_per_frame = pickle.load(f)

            saved_outputs = organize_outputs_by_id(outputs_per_frame, image_key=image_key, min_mask_area_ratio=args.min_mask_area_ratio, debug=args.debug)
            if not args.debug:
                save_outputs_to_hdf5(saved_outputs, full_hdf5_path, demo_key)


            # Get number of frames from saved_outputs
            bbox_key = image_key.replace("image", "bbox").replace("rgb", "bbox")
            bbox_mask_key = f"{bbox_key}_mask"
            if bbox_mask_key in saved_outputs:
                num_frames = saved_outputs[bbox_mask_key].shape[0]
                print(f"    Detected {num_frames} frames with objects")
            else:
                print(f"    Detected {len(outputs_per_frame)} frames with objects")


            # Visualize and save
            visualize_and_save_video(
                saved_outputs, pil_frames, output_dir,
                hdf5_name, demo_key, image_key, fps=fps, prompt_to_uniq_color_id=prompt_to_uniq_color_id
            )

        if args.debug:
            exit()

    print("\nProcessing complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize SAM3 masks on HDF5 dataset videos")
    parser.add_argument("--hdf5_path", type=str, required=True,
                        help="Path to HDF5 file (can be relative or absolute)")
    parser.add_argument("--image_keys", type=str, nargs="+", default=None,
                        help="Image observation keys (e.g., 'obs/agentview_rgb obs/eye_in_hand_rgb'). If not specified, will try to auto-discover.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for visualizations (default: same as input with _sam3_viz suffix)")
    parser.add_argument("--prompts", type=str, nargs="+", default=["sink", "plate", "bowl"],
                        help="Text prompts for SAM3 (default: sink plate bowl)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frames per second for output videos (default: 30)")
    parser.add_argument("--max_frames", type=int, default=-1,
                        help="Maximum number of frames to extract from the video (default: -1, all frames)")
    parser.add_argument("--start_frame", type=int, default=0,
                        help="Start frame to extract from the video (default: 0)")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Debug mode: only process the first episode and camera")
    parser.add_argument("--flip_image_channel", action="store_true", default=False,
                        help="Flip the image channel")
    parser.add_argument("--min_mask_area_ratio", type=float, default=0.012,
                        help="Minimum ratio of mask area to total image area (default: 0.012 = 1.2%%)")

    args = parser.parse_args()

    process_hdf5_file(
        args.hdf5_path,
        image_keys=args.image_keys,
        text_prompts=args.prompts,
        output_dir=args.output_dir,
        fps=args.fps,
        args=args
    )
