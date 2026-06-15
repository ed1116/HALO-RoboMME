from typing import List
from dataclasses import dataclass
import torch
from collections import OrderedDict

import halo.util.tensor_utils as TU

@dataclass
class LangTrajSequence:
    image_keys: List[str]
    image_token_ids: List[int]
    tokens_per_frame: int
    action_token_ids: List[int]
    tokens_per_action: int
    action_in_inputs: bool

    pad_inst_tokens: bool = False
    inst_token_pad_value: int = 0
    max_inst_tokens: int = 32
    mode: str = "lang-vision-action" # only mode supported

    def __post_init__(self):
        assert self.mode == "lang-vision-action", "Only lang-vision-action mode is supported"
        assert isinstance(self.image_keys, list) and len(self.image_keys) > 0, "image_keys must be a non-empty list"
        assert isinstance(self.action_token_ids, list) and len(self.action_token_ids) > 0, "action_token_ids must be a non-empty list"

    def __call__(self, inst_ids: list[int], num_frames: int, device: torch.device = 'cpu', skip_language_tokens: bool = False):
        if self.pad_inst_tokens:
            if len(inst_ids) > self.max_inst_tokens:
                # cut it to the max length
                inst_ids = inst_ids[:self.max_inst_tokens]
            inst_ids = inst_ids + [self.inst_token_pad_value] * (self.max_inst_tokens - len(inst_ids))
        if skip_language_tokens:
            inst_ids = [] # empty list means no language tokens
        num_action_tokens = self.tokens_per_action
        base_img_action_tokens = [tok for tok in self.image_token_ids for _ in range(self.tokens_per_frame)]
        if self.action_in_inputs:
            base_img_action_tokens += self.tokens_per_action * self.action_token_ids
        
        # get the chunk length
        chunk_len = len(base_img_action_tokens)

        # repeat the base img action tokens for each frame
        img_action_ids = base_img_action_tokens * num_frames

        # combine the instruction and img action tokens
        full_ids = inst_ids + img_action_ids

        img_token_positions = OrderedDict({
            img_key: [] for img_key in self.image_keys
        })
        action_inp_token_pos = []
        action_out_token_pos = []
        # the last image token position is the position of the action output token
        for ind, token in enumerate(full_ids):
            if token in self.image_token_ids:
                img_key = self.image_keys[self.image_token_ids.index(token)] # order is preserved
                img_token_positions[img_key].append(ind)
            elif token in self.action_token_ids:
                action_inp_token_pos.append(ind)

        # last token in the observation is the action output token
        # first action positions: len(inst_ids) + chunk_len - 1 if not remove_action else -2
        first_action_positions = len(inst_ids) + chunk_len - 1
        if self.action_in_inputs:
            first_action_positions -= num_action_tokens

        # increase the first action positions by chunk_len
        action_out_token_pos = torch.arange(len(full_ids), dtype=torch.long)
        action_out_token_pos = action_out_token_pos[first_action_positions::chunk_len]

        return (
            TU.to_torch_long(full_ids).to(device),
            OrderedDict({img_key: TU.to_torch_long(positions).to(device) for img_key, positions in img_token_positions.items()}),
            TU.to_torch_long(action_inp_token_pos).to(device),
            TU.to_torch_long(action_out_token_pos).to(device),
        )
