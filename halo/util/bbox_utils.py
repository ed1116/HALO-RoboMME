import numpy as np
import fnmatch
import torch
from typing import Dict


def pad_bboxes_to_max_length(subsequence, bbox_ids, bbox_names, max_length, use_pbar=False):
    """
    Truncate or pad each list of boxes in `subsequence` up to max_length,
    pad bbox_ids with -1, and return a base_mask indicating real vs. padded slots.

    Args:
        subsequence    List[List[4‐tuples or arrays]] of bboxes, length B.
        bbox_ids       List[torch.Tensor] of shape (Ni,) for each i in B.
        max_length     int, target number of boxes per entry.
        use_pbar      bool, if True, show progress bar.

    Returns:
        padded_boxes   np.ndarray, shape (B, max_length, 4)
        padded_ids     torch.LongTensor, shape (B, max_length)
        base_mask      torch.IntTensor, shape (B, max_length), 1=real, 0=padded
    """
    B = len(subsequence)
    base_mask = torch.ones((B, max_length), dtype=torch.int32)
    padded_boxes = np.zeros((B, max_length, 4), dtype=float)
    padded_ids   = torch.full((B, max_length), -1, dtype=torch.long)
    padded_names = np.full((B, max_length), "", dtype=object)
    padded_coords_cc = np.zeros((B, max_length, 2), dtype=float)
    if use_pbar:
        pbar = tqdm(total=B)
    for i, (boxes, ids) in enumerate(zip(subsequence, bbox_ids)):
        # — Truncate if too many
        if len(boxes) > max_length:
            # sort by area (smallest first), drop extras
            areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
            order = np.argsort(areas)[:max_length]
            boxes = [boxes[j] for j in order]
            ids   = ids[order]
            bbox_names = bbox_names[order]
        # calculate cc
        cc = [np.array([(b[0] + b[2])/2, (b[1] + b[3])/2]) for b in boxes]

        n = len(boxes)
        # — Copy real boxes & ids
        if n > 0:
            padded_boxes[i, :n] = np.array(boxes)
            padded_ids[i, :n]   = ids
            padded_names[i, :n] = bbox_names[i]
            padded_coords_cc[i, :n] = np.array(cc)
        # — For the padded slots, mask them out
        if n < max_length:
            base_mask[i, n:] = 0
        if use_pbar:
            pbar.update(1)

    return padded_boxes, padded_ids, padded_names, padded_coords_cc, base_mask

def get_all_bbox_observations(obs: Dict, id2name: Dict) -> Dict:
    segm_keys = [k for k in obs.keys() if "segmentation" in k]
    bbox_keys = [k.replace("segmentation_instance", "bbox") for k in segm_keys]
    bbox_obs = {}
    for segm_key, bbox_key in zip(segm_keys, bbox_keys):
        # robocasa's robomimic wrapper already flips the image for us.
        segm = obs[segm_key]
        bbox = only_bbox_from_segm(segm, id2name=id2name, exclude_geom_regex=[r'*gripper*', r'*stack_*'])
        bbox = list(bbox.values())
        sorted_bbox = sorted(bbox, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))

        # unpack coords and ids
        coords = np.array([b[:4] for b in sorted_bbox], dtype=np.int32)   # (N,4)
        ids    = torch.tensor([b[4][-1] for b in sorted_bbox], dtype=torch.long)  # (N,)
        names = [b[4][-2] for b in sorted_bbox]

        # pad / truncate to exactly 32 boxes
        padded_coords, padded_ids, padded_names, padded_coords_cc, mask = pad_bboxes_to_max_length(
            [coords],            # list of length B=1
            [ids],               # list of length B=1
            [names],             # list of length B=1
            max_length=32
        )
        bbox_obs[bbox_key] = np.array(padded_coords[0])
        bbox_obs[bbox_key + '_ids'] = np.array(padded_ids[0])
        bbox_obs[bbox_key + '_mask'] = np.array(mask[0])
        bbox_obs[bbox_key + '_names'] = np.array(padded_names[0])
        bbox_obs[bbox_key + '_cc'] = np.array(padded_coords_cc[0])
    return bbox_obs

def match_regex(name, regex_list):
    """
    Return True if name matches any regex in regex_list (glob-style).
    Convert shell-style wildcards to valid regex patterns using fnmatch.
    """
    for pattern in regex_list:
        if fnmatch.fnmatchcase(name, pattern):
            return True
    return False


def only_bbox_from_segm(
        seg_im,
        obj_id=None,
        area_th=0.01,
        id2name=None,
        exclude_geom_regex=[r'*gripper*', r'*stack_*']
    ):
    if seg_im.ndim == 3:
        seg_im = seg_im.squeeze(-1)
    if obj_id is None:
        obj_id = np.unique(seg_im)
    bbox_all = {}
    for val in obj_id:
        if val == 0:
            continue
        y, x = np.where(seg_im == val)
        if len(x) == 0 or len(y) == 0:
            print(f"No object found for {val}")
            continue
        xmin, ymin, xmax, ymax = np.min(x), np.min(y), np.max(x), np.max(y)
        area = (xmax - xmin) * (ymax - ymin)
        if xmax - xmin < 0.05 * seg_im.shape[0] or ymax - ymin < 0.05 * seg_im.shape[1]:
            continue
        # value stored is wrt to segmentation value
        # val-1 is wrt instance to id naming
        name = id2name[val-1] if id2name is not None else ""
        if match_regex(name.lower(), exclude_geom_regex):
            continue
        bbox_all[val-1] = [xmin, ymin, xmax, ymax, (None, id2name[val-1], val)]
    return bbox_all
