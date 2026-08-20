import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, Union, Tuple, Literal, List
from collections import OrderedDict
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from timm.layers import Mlp
from tqdm import tqdm, trange
from termcolor import colored

from halo.models.policy.action_head import (
    MLPHead,
    CEHead,
    FlowMatchingHead,
    resolve_physical_action_dim,
    select_action_targets,
)
import halo.data.utils as data_utils
from halo.models.policy.llama import Transformer, ModelArgs
from halo.models.policy.retriever_topk import TopKTransformerBlock
from halo.models.backbones.encoders import VisionEncoder, VisionEncoderCNN, AttentionPool, MultiKVAttentionPool
from halo.util.args import SharedConfig, PolicyConfig
import halo.util.misc as misc_utils
import halo.models.backbones.token_sequence_gen as TokenSequenceGen


def check_params(params: dict):
    assert 'vocab_size' in params, "vocab_size is required"
    assert 'dim' in params or 'head_dim' in params, "dim or head_dim is required"
    assert 'n_layers' in params or 'num_hidden_layers' in params, "n_layers or num_hidden_layers is required"
    return True

class LanguageModel:
    def _get_default_model_args(self, params, shared_config: SharedConfig, policy_config: PolicyConfig, train: bool, output_modes: List[Literal["text", "action", "both"]]):
        max_language_tokens = 1024
        compute_dtype = misc_utils.convert_str_to_torch_dtype(shared_config.compute_dtype)
        if 'compute_dtype' in params:
            compute_dtype = misc_utils.convert_str_to_torch_dtype(params.pop('compute_dtype'))
        model_args = ModelArgs(
            max_seq_len=self.latent_len*(shared_config.seq_length//shared_config.downsample_obs) + max_language_tokens,
            max_batch_size=shared_config.batch_size if train else 1,
            w_bias=False,
            is_causal=True,
            text_mode=True if 'text' in output_modes else False,
            enable_gradient_checkpointing=self.enable_gradient_checkpointing,
            kv_dim=params['dim'],
            is_train=train,
            # block attention related parameters
            block_attn_layer_ind=policy_config.block_attn_idx,
            local_attn_layer_ind=policy_config.local_attn_idx,
            # memory related parameters
            mem_attn_layer_ind=policy_config.mem_attn_idx,
            mem_chunk_len=policy_config.mem_chunk_ts_len * self.latent_len,
            mem_threshold=policy_config.mem_threshold,
            mem_store_pre_rope=True, # always true
            # block attention related parameters
            block_len=policy_config.block_chunk_ts_len * self.latent_len,
            block_pattern_start_offset=policy_config.block_pattern_start_offset,
            # maintain the compute dtype
            compute_dtype=compute_dtype,
            # topk attention related parameters
            ret_topk=policy_config.ret_topk,
            ret_chunk_len=policy_config.ret_chunk_len * self.latent_len,
            ret_tau_init=policy_config.ret_tau_init,
            ret_straight_through=policy_config.ret_straight_through,
            ret_recursions=policy_config.ret_recursions,
            ret_multikv=policy_config.ret_multikv,
            ret_add_time_aware=policy_config.ret_add_time_aware,
            ret_relative_time=policy_config.ret_relative_time,
            ret_topk_attn_idx=policy_config.ret_topk_attn_idx,
            ret_bank_causal=policy_config.ret_bank_causal,
            # strided attention related parameters
            strided_block_len=policy_config.strided_len * self.latent_len,
            strided_attn_layer_ind=policy_config.strided_attn_idx,
            gated_attn_idx=policy_config.gated_attn_idx,
            tokme_attn_layer_ind=policy_config.tokme_attn_idx,
            **params
        )
        model_args.cache = self.eval_caching
        model_args.latent_len = self.latent_len
        return model_args

    def _load_llama_model(self, params, shared_config: SharedConfig, policy_config: PolicyConfig, train: bool, output_modes: List[Literal["text", "action", "both"]]):
        model_args = self._get_default_model_args(params, shared_config, policy_config, train, output_modes)
        if self.scratch_llama_config is None and policy_config.load_llama:
            if 'Llama3' in policy_config.llama_ckpt_dir:
                torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
            else:
                torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)

        print("initializing main transformer ...")
        llama = Transformer(model_args)
        torch.set_default_tensor_type(torch.FloatTensor)
        return llama


class MemModel(nn.Module, LanguageModel):
    def __init__(
        self,
        shared_config: SharedConfig,
        policy_config: PolicyConfig,
        vision_encoder: Union[VisionEncoder, VisionEncoderCNN],
        train: bool = True,
        extra_kwargs: Optional[dict] = None,
        output_modes: List[Literal["text", "action", "both"]] = ["text", "action"],
    ):
        super().__init__()

        extra_kwargs = extra_kwargs or {}
        self.policy_config = policy_config
        self.shared_config = shared_config
        self.coeff_text_loss = 0.01
        self.coeff_state_supervision_loss = shared_config.coeff_state_supervision_loss
        self.eval_caching = False
        self.phase = policy_config.phase
        if self.phase == "pretrain_lora":
             assert policy_config.use_lora, "use_lora must be True for pretrain_lora phase"
        self.enable_gradient_checkpointing = shared_config.enable_gradient_checkpointing
        self.separate_camera_adapter = policy_config.separate_camera_adapter and shared_config.num_cameras > 1
        self.output_modes = output_modes
        self.pretrained_path = policy_config.pretrained_path
        self.scratch_llama_config = policy_config.scratch_llama_config
        self.seq_length = shared_config.seq_length
        self.per_cam_latent_len = shared_config.attn_latent_len
        self.has_base_action = shared_config.has_base_action
        if self.scratch_llama_config is not None:
            llama_config_path = self.scratch_llama_config
        else:
            llama_config_path = os.path.join(policy_config.llama_ckpt_dir, "params.json")
        with open(llama_config_path, "r") as f:
            params = json.loads(f.read())
        check_params(params)
        if 'dim' in params:
            self.latent_dim = params['dim']
        elif 'head_dim' in params:
            self.latent_dim = params['head_dim'] * params['num_key_value_heads']
        self.vision_encoder = vision_encoder
        self.vision_out_dim = self.vision_encoder.out_dim()
        self.vision_finetune = vision_encoder.finetune
        self.tt_window_obs = policy_config.tt_window_obs
        self.tt_obs_window_len = self.seq_length // shared_config.downsample_obs
        self.tt_action_window_len = self.seq_length // shared_config.downsample_obs
        assert shared_config.seq_length % shared_config.downsample_obs == 0, f"seq_length must be divisible by downsample_obs: {shared_config.seq_length=} {shared_config.downsample_obs=}"
        if vision_encoder.finetune == 'all':
            self.vision_finetune = True

        self.num_cameras = shared_config.num_cameras
        assert self.separate_camera_adapter, "separate_camera_adapter is not supported for MemModel"
        attn_pool_cls = MultiKVAttentionPool if policy_config.multikv_attn_pool else AttentionPool
        common_kwargs = {
            "in_features": self.vision_out_dim,
            "out_features": self.latent_dim,
            "mlp_ratio": policy_config.adapter_mlp_ratio,
            "num_heads": policy_config.adapter_num_heads,
            "latent_len": self.per_cam_latent_len
        }
        self.icrt_attn_pooling = nn.ModuleList([
            attn_pool_cls(
                **common_kwargs,
                mini_batch_size=extra_kwargs.get('mini_batch_size', 1),
                img_seq_len=extra_kwargs.get('img_seq_len'),
            ) for _ in range(self.num_cameras)
        ])
        self.padding = 0
        self.attn_latent_len = self.per_cam_latent_len * self.num_cameras
        self.latent_len = self.attn_latent_len + 1 if not shared_config.remove_action else self.attn_latent_len
        # assert that the max_inst_tokens is divisible by the block_len or vice versa
        assert self.shared_config.max_inst_tokens % self.latent_len == 0, f"block_len must be divisible by latent_len: {block_len=} {self.latent_len=}"

        self.downsample_obs = shared_config.downsample_obs
        self.icrt_vision_norm = nn.LayerNorm(
            normalized_shape=self.vision_out_dim, eps=1e-6
        ) if not self.vision_encoder._apply_post_norm else nn.Identity()

        self.tokenizer = data_utils.build_tokenizer(shared_config.tokenizer_name, extra_kwargs.get('image_keys'))
        self.image_keys = extra_kwargs.get('image_keys')
        self.proprio_keys = extra_kwargs.get('proprio_keys')
        self.action_keys = extra_kwargs.get('action_keys')
        self.image_token_ids = self.tokenizer.convert_tokens_to_ids(data_utils.get_img_token_str_list(self.image_keys))
        self.action_token_ids = self.tokenizer.convert_tokens_to_ids(data_utils.get_action_token_str_list())
        params['vocab_size'] = len(self.tokenizer) # if 'text' in output_modes else params['vocab_size']
        # check either the n_layers or num_hidden_layers is present in the params

        # load whatever model
        self.llama = self._load_llama_model(params=params, shared_config=shared_config, policy_config=policy_config, train=train, output_modes=output_modes)

        max_model_layers = params.get('n_layers', params.get('num_hidden_layers', None))
        if max_model_layers > 0:
            self.llama.keep_first_n_layers(max_model_layers)

        if policy_config.use_lora:
            self.llama.add_lora_layers(policy_config.lora_rank, policy_config.lora_alpha, policy_config.lora_dropout)

        if True:
            self.llama.eval()
            for param in self.llama.parameters():
                param.requires_grad = False # freeze the model parameters

        if self.scratch_llama_config is None and policy_config.load_llama:
            ckpts = sorted(Path(policy_config.llama_ckpt_dir).glob("*.pth"))
            for ckpt in tqdm(ckpts, desc="Loading LLaMA ckpt"):
                ckpt = torch.load(ckpt, map_location='cpu', weights_only=True)
                names = self.llama.state_dict().keys()
                ckpt_names = ckpt.keys()
                for n in ckpt_names:
                    if n not in names:
                        print(f"Warning: {n} not in llama model")
                for n in names:
                    if n not in ckpt_names:
                        print(f"Warning: {n} not in ckpt")
                self.llama.load_state_dict(ckpt, strict=False)

        self.text_output_head = None
        if "text" in output_modes:
            self.text_output_head = nn.Linear(self.latent_dim, params['vocab_size'], bias=False)

        # action output head
        self.add_state_supervision = shared_config.add_state_supervision or shared_config.add_gpt_state_supervision or shared_config.add_fake_state_supervision
        self.state_supervision_mode = shared_config.state_supervision_mode
        self.max_state_supervision_len = shared_config.max_state_supervision_len
        self.num_pred_steps = shared_config.num_pred_steps
        self.serialized_action_dim = extra_kwargs.get("action_dim")
        self.action_eos_dim = extra_kwargs.get("action_eos_dim", 1)
        expected_action_dim = extra_kwargs.get("physical_action_dim")
        self._action_dim = resolve_physical_action_dim(
            self.serialized_action_dim,
            self.action_eos_dim,
            expected_action_dim,
        )
        self.remove_action = shared_config.remove_action
        self.is_bimanual = shared_config.is_bimanual
        assert not self.is_bimanual, "Bimanual is not supported for MemModel"
        if "action" in output_modes:
            pred_head = policy_config.decoder_pred_head
            if pred_head == "fm":
                self.action_output_head = FlowMatchingHead(
                    input_dim=self.latent_dim,
                    hidden_features=policy_config.decoder_hidden_features,
                    output_dim=self.action_dim * self.num_pred_steps,
                    loss_fn=nn.L1Loss(reduction="none"),
                    action_chunk_len=self.num_pred_steps // 2,
                )
            else:
                self.action_output_head = MLPHead(
                    input_dim=self.latent_dim,
                    hidden_features=policy_config.decoder_hidden_features,
                    output_dim=self.action_dim * self.num_pred_steps,
                    loss_fn=nn.L1Loss(reduction="none"),
                    action_chunk_len=self.num_pred_steps // 2,
                )
            self.state_supervision_output_head = None
            if self.add_state_supervision:
                if '_str' in self.state_supervision_mode:
                    self.state_supervision_output_head = CEHead(
                        input_dim=self.latent_dim,
                        vocab_size=len(self.tokenizer),
                        max_seq_len=self.max_state_supervision_len,
                        loss_fn=nn.CrossEntropyLoss(reduction="none", ignore_index=self.tokenizer.pad_token_id),
                        ignore_index=self.tokenizer.pad_token_id,
                        hidden_features=policy_config.ce_hidden_features,
                    )
                else:
                    dims = 4*self.num_cameras
                    if self.shared_config.ss_create_mode == "time":
                        dims = 1
                    self.state_supervision_output_head = MLPHead(
                        input_dim=self.latent_dim,
                        hidden_features=policy_config.decoder_hidden_features,
                        output_dim=dims,
                        loss_fn=nn.L1Loss(reduction="none"),
                        action_chunk_len=self.num_pred_steps//2, # does not matter
                    )
        else:
            self.action_output_head = None

        # this is about input and not output
        if not self.remove_action:
            self.icrt_action_encoder = Mlp(in_features=self.action_dim, out_features=self.latent_dim, bias=False)

        # proprioception encoder
        self.proprio_dim = extra_kwargs.get("proprio_dim")
        self.remove_proprio = shared_config.remove_proprio
        if not self.remove_proprio:
            self.icrt_proprio_encoder = Mlp(in_features=self.proprio_dim, out_features=self.vision_out_dim) # backward compatibility

        if policy_config.use_topk_attention:
            # since we are using the topk inside preprocessing, we need to use the
            params['compute_dtype'] = 'float32'
            args = self._get_default_model_args(params=params, shared_config=shared_config, policy_config=policy_config, train=train, output_modes=output_modes)

            n_topk_blocks = policy_config.ret_n_topk_blocks
            self.topk_transformer_block = nn.ModuleList([TopKTransformerBlock(args=args, layer_id=0) for _ in range(n_topk_blocks)])
            if n_topk_blocks == 1: # backward compatibility
                self.topk_transformer_block = self.topk_transformer_block[0]
            print(f"Using {n_topk_blocks} TopK blocks")

        if train:
            self.set_default_trainability(self.phase)
        else:
            self.eval()

    def preprocessing(
        self,
        input_ids: torch.Tensor,
        proprio: torch.Tensor,
        action : Optional[torch.Tensor],
        action_inp_token_pos: Optional[torch.Tensor],
        images: Dict[str, torch.Tensor],
        image_token_positions: Dict[str, torch.Tensor],
        topk: Optional[int] = None,
    ) -> torch.Tensor:
        B, L = input_ids.shape
        # assert input_ids.min() >= 0 and input_ids.max() < self.llama.vocab_size, \
        #     f"Input IDs are not within valid range: {input_ids.min()=} {input_ids.max()=} {self.llama.vocab_size=}"
        embeddings = self.llama.get_token_embeds(input_ids)
        if self.training and (self.shared_config.embeddings_noise > 0):
            noise = torch.randn_like(embeddings) * self.shared_config.embeddings_noise
            embeddings = embeddings + noise
        # add embedding RMS Normalization and Gaussian Noise
        if not self.remove_proprio:
            assert proprio.ndim == 4, f"Proprio should be of shape (B, T, N, D): {proprio.shape=}"
            proprio = proprio[:, :, 0, :]
            f_prop = self.icrt_proprio_encoder(proprio) # B, T, self.vision_out_dim
        
        # create non-zero image positions tuple here and save it as a dictionary. Make it a non-blocking operation
        non_zero_positions = {}
        for img_key, img_positions in image_token_positions.items():
            img_positions_non_zero = (img_positions.view(-1) != -1)
            # B_img, T_img = img_positions.shape[:2]
            B_img, T_img = images[img_key].shape[:2]
            size = img_positions_non_zero.sum()
            non_zero_positions[img_key] = torch.nonzero_static(img_positions_non_zero, size=size)

        action_inp_token_pos_non_zero = (action_inp_token_pos.view(-1) != -1)
        n_actions = action_inp_token_pos_non_zero.sum()
        non_zero_positions['action'] = torch.nonzero_static(action_inp_token_pos_non_zero, size=n_actions)

        # Batch all cameras through vision encoder in a single forward pass
        img_keys = list(image_token_positions.keys())
        all_img_tensors = [images[k] for k in img_keys]
        B_img, T_img = all_img_tensors[0].shape[:2]
        all_f_obs = self.vision_encoder(torch.cat(all_img_tensors, dim=0))  # (num_cams*B, T, N, vision_out_dim)
        all_f_obs = self.icrt_vision_norm(all_f_obs)
        f_obs_per_cam = all_f_obs.split(B_img, dim=0)  # list of (B, T, N, vision_out_dim)

        cam_idx = 0
        # [0,0,0,1,1,1,2,2,2,...]
        for img_key, img_positions in image_token_positions.items():
            img_tensor = images[img_key]
            B_img, T_img = img_tensor.shape[:2]

            f_obs = f_obs_per_cam[cam_idx]
            if self.remove_proprio:
                f_obs = self.icrt_attn_pooling[cam_idx].forward_visual(f_obs)
            else:
                f_obs = self.icrt_attn_pooling[cam_idx].combine_forward(f_obs, f_prop)
            if self.padding > 0:
                f_obs = F.pad(f_obs, (0, self.padding), "constant", 0)

            num_patches = f_obs.shape[-2]
            f_obs = f_obs.view(B_img, T_img * num_patches, self.latent_dim)

            # filter out image_positions that are -1
            device = embeddings.device
            img_positions = img_positions.view(-1)                      # [B*T*P]
            f_obs_flat = f_obs.view(-1, self.latent_dim)                # [B*T*P, D]

            # batch indices without repeat(): cheaper and cleaner
            batch_idx = torch.arange(B_img, device=device).repeat_interleave(T_img * num_patches)  # [B*T*P]

            # valid = (img_positions != -1).nonzero(as_tuple=True)[0]     # [Nv]
            valid = non_zero_positions[img_key] # [Nv, 1]
            valid = valid[:,0].flatten()
            b = batch_idx.index_select(0, valid)
            p = img_positions.index_select(0, valid)
            v = f_obs_flat.index_select(0, valid).to(dtype=embeddings.dtype)

            # one kernel to write all updates
            embeddings.index_put_((b, p), v, accumulate=False)
            cam_idx += 1

        # action processing
        if not self.remove_action:
            assert action.ndim == 4, f"Action should be of shape (B, T, N, D): {action.shape=}"
            action = action[:, :, 0, :self.action_dim]
            B_act, T_act = action.shape[:2]
            f_a = self.icrt_action_encoder(action) # B, T, D
            f_a = f_a.view(-1, f_a.shape[-1])
            action_inp_token_pos = action_inp_token_pos.view(-1)
            batch_idx = torch.arange(B_act, dtype=torch.long, device=embeddings.device).repeat_interleave(T_act)
            valid = non_zero_positions['action'] # [Nv, 1]
            valid = valid[:,0].flatten()
            b = batch_idx.index_select(0, valid)
            p = action_inp_token_pos.index_select(0, valid)
            v = f_a.index_select(0, valid).to(dtype=embeddings.dtype)
            embeddings.index_put_((b, p), v, accumulate=False)

        # maybe here's a good place to apply the topk attention :D
        retriever_info = {}
        if self.policy_config.use_topk_attention:
            L = embeddings.shape[1]
            # Ensure freqs_cis are on the same device as embeddings
            freqs_cis = (self.llama.freqs_cos[:L].to(embeddings.device), self.llama.freqs_sin[:L].to(embeddings.device)) # assumes the freqs_cis are precomputed
            if isinstance(self.topk_transformer_block, nn.ModuleList):
                for block in self.topk_transformer_block:
                    embeddings, retriever_info = block(embeddings, start_pos=0, freqs_cis=freqs_cis, mask=None, topk=topk)
            else:
                embeddings, retriever_info = self.topk_transformer_block(embeddings, start_pos=0, freqs_cis=freqs_cis, mask=None, topk=topk)
            retriever_info = {} # empty it since we are not using the retriever info

        return embeddings, retriever_info

    def _convert_to_action_len(self, len: int):
        return (len - self.shared_config.max_inst_tokens) // self.latent_len

    def forward(
        self,
        sequences: Dict[str, Any],
        log_dict: Optional[Dict[str, Any]] = None,
        log_writer: Optional[Any] = None,
    ) -> tuple:
        # only used in preprocessing
        images = sequences['observation']
        image_token_positions = sequences['image_token_positions']
        proprio = sequences.get('proprio', None)
        action_inp_token_pos = sequences.get('action_inp_token_pos', None)

        # used after preprocessing as well
        input_ids = sequences['input_ids']
        text_prompt_mask = sequences['text_prompt_mask']
        action_prompt_mask = sequences['action_prompt_mask']
        action = sequences.get('action', None)
        state_supervision_tar = sequences.get('state_supervision', None)
        action_out_token_pos = sequences.get('action_out_token_pos', None)
        action_tar_pad_mask = sequences.get('action_tar_pad_mask', None)
        ss_out_token_pos = sequences.get('state_supervision_out_token_pos', None)
        B, L = input_ids.shape

        embeddings, additional_info = self.preprocessing(input_ids, proprio, action, action_inp_token_pos, images, image_token_positions, topk=sequences.get('topk', None))

        # if EMDR2-loss, then concatenate output values with the embeddings
        if self.policy_config.ret_emdr2_loss:
            raise NotImplementedError("Ret EMDR2 loss is not implemented for MemModel")

        outputs, _ = self.llama(embeddings, mask=None)
        total_loss, loss_dict, loss_info = self.compute_all_losses(
            outputs=outputs,
            input_ids=input_ids,
            text_prompt_mask=text_prompt_mask,
            action_prompt_mask=action_prompt_mask,
            action=action,
            action_out_token_pos=action_out_token_pos,
            state_supervision_tar=state_supervision_tar,
            ss_out_token_pos=ss_out_token_pos,
            additional_info=additional_info,
        )

        return total_loss, loss_dict, outputs, {'pred_actions': None, 'action_prompt_mask': None}


    def compute_all_losses(
        self,
        outputs: torch.Tensor,
        input_ids: torch.Tensor,
        text_prompt_mask: torch.Tensor,
        action_prompt_mask: torch.Tensor,
        action: torch.Tensor,
        action_out_token_pos: torch.Tensor,
        state_supervision_tar: torch.Tensor,
        ss_out_token_pos: torch.Tensor,
        additional_info: Dict[str, torch.Tensor] = {},
    ):
        # Initialize all losses as zero tensors to ensure consistent keys across ranks for distributed reduction
        # Using requires_grad=True ensures all losses are part of the computation graph, even when zero
        device = input_ids.device
        text_loss = torch.tensor(0.0, requires_grad=True, device=device)
        text_accuracy = torch.tensor(0.0, requires_grad=True, device=device)
        ret_text_loss = torch.tensor(0.0, requires_grad=True, device=device)
        action_loss = torch.tensor(0.0, requires_grad=True, device=device)
        ret_action_loss = torch.tensor(0.0, requires_grad=True, device=device)
        state_supervision_loss = torch.tensor(0.0, requires_grad=True, device=device)
        ret_state_supervision_loss = torch.tensor(0.0, requires_grad=True, device=device)
        if torch.any(text_prompt_mask > 0):
            text_loss, text_accuracy, ret_text_loss = self._compute_text_loss(
                outputs=outputs,
                input_ids=input_ids,
                text_prompt_mask=text_prompt_mask,
                additional_info=additional_info,
            )

        pred_actions = None
        if torch.any(action_prompt_mask > 0):
            action_loss, ret_action_loss, _ = self._compute_action_loss(
                outputs=outputs,
                action=action,
                action_out_token_pos=action_out_token_pos,
                action_prompt_mask=action_prompt_mask,
                additional_info=additional_info,
            )
            # B, T, 1, 1 same dimensions as action_pred: B, T, self.num_pred_steps, self.action_dim
            action_prompt_mask = action_prompt_mask.unsqueeze(-1)

        # if any of the action_out_token_pos are not -1, then compute the state supervision loss
        if self.add_state_supervision and torch.any(ss_out_token_pos != -1):
            # assert action.shape[:2] == action_out_token_pos.shape[:2]
            state_supervision_loss, ret_state_supervision_loss = self._compute_state_supervision_loss(
                outputs=outputs,
                state_supervision_tar=state_supervision_tar,
                ss_out_token_pos=ss_out_token_pos,
                additional_info=additional_info,
            )

        # Build total_loss and loss_dict
        # All losses are now guaranteed to be tensors (either computed or zero), ensuring consistent keys across ranks
        loss_items = [
            ("action_loss", action_loss, 1.0),
            ("ret_action_loss", ret_action_loss, 1.0),
            ("text_loss", text_loss, self.coeff_text_loss),
            ("ret_text_loss", ret_text_loss, self.coeff_text_loss),
            ("state_supervision_loss", state_supervision_loss, self.coeff_state_supervision_loss),
            ("ret_state_supervision_loss", ret_state_supervision_loss, self.coeff_state_supervision_loss),
        ]
        total_loss = torch.tensor(0.0, requires_grad=True, device=device)
        loss_dict = {}
        for name, loss_val, coeff in loss_items:
            total_loss = total_loss + coeff * loss_val
            loss_dict[name] = loss_val
        loss_dict["text_accuracy"] = text_accuracy
        loss_dict["loss"] = total_loss
        return total_loss, loss_dict, {'pred_actions': None, 'action_prompt_mask': None}

    def _compute_state_supervision_loss(
        self,
        outputs: torch.Tensor,
        state_supervision_tar: torch.Tensor,
        ss_out_token_pos: torch.Tensor,
        additional_info: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B_act, T_act = state_supervision_tar.shape[:2]
        ss_out_token_pos = ss_out_token_pos.view(-1)
        mask = ss_out_token_pos != -1
        # batch_idx = torch.arange(B_act, dtype=torch.long, device=outputs.device).unsqueeze(1).repeat(1, T_act).view(-1)
        # outputs_og: torch.Tensor = outputs[batch_idx[mask], ss_out_token_pos[mask]]
        if not mask.any():
            # Early exit if no valid positions
            state_supervision_loss = torch.tensor(0.0, requires_grad=True, device=outputs.device)
            ret_state_supervision_loss = torch.tensor(0.0, requires_grad=True, device=outputs.device)
            return state_supervision_loss, ret_state_supervision_loss
        
        # Optimized: use more efficient indexing pattern
        batch_idx_flat = torch.arange(B_act, dtype=torch.long, device=outputs.device).repeat_interleave(T_act)
        
        # Optimized: use advanced indexing more efficiently
        # Flatten outputs for easier indexing: (B, L, D) -> (B*L, D)
        B_out, L_out, D_out = outputs.shape
        outputs_flat = outputs.view(B_out * L_out, D_out)
        # Convert 2D indices to 1D for flat indexing
        indices_flat = batch_idx_flat[mask] * L_out + ss_out_token_pos[mask]
        outputs_selected = outputs_flat[indices_flat]
        # assert torch.allclose(outputs_og, outputs_selected), f"Outputs og and outputs selected are not close: {outputs_og.shape=}, {outputs_selected.shape=}"

        assert state_supervision_tar.ndim == 3, f"State supervision should be of shape (B, T, D): {state_supervision_tar.shape=}"
        if '_str' in self.state_supervision_mode: # can chunk this if memory is an issue?
            state_supervision_tar = state_supervision_tar.view(-1, self.max_state_supervision_len)[mask]
            gt_latent_embeds = self.llama.get_token_embeds(state_supervision_tar)
            state_supervision_loss = self.state_supervision_output_head.loss(
                outputs_selected, gt_latent_embeds=gt_latent_embeds, gt_token_indices=state_supervision_tar, return_pred=False
            )
            del gt_latent_embeds  # Free memory after use
        else:
            # Memory optimization: Combine reshape and masking
            dim = state_supervision_tar.shape[-1]
            state_supervision_tar = state_supervision_tar.view(-1, dim)[mask]
            state_supervision_loss = self.state_supervision_output_head.loss(outputs_selected, state_supervision_tar, return_pred=False)

        ret_state_supervision_loss = torch.tensor(0.0, requires_grad=True, device=outputs.device)
        if self.policy_config.ret_emdr2_loss:
            raise NotImplementedError("Ret EMDR2 loss is not implemented for state supervision loss")

        # Memory optimization: Use in-place division where possible
        numel = state_supervision_loss.numel()
        state_supervision_loss = state_supervision_loss.sum() / (numel + 1e-6)
        # print(f"{ret_state_supervision_loss.item()=}, {state_supervision_loss.item()=}")
        return state_supervision_loss, ret_state_supervision_loss

    def _compute_action_loss(
            self,
            outputs: torch.Tensor,
            action: torch.Tensor,
            action_out_token_pos: torch.Tensor,
            action_prompt_mask: torch.Tensor,
            additional_info: Dict[str, torch.Tensor],
        ) -> Tuple[torch.Tensor, torch.Tensor]:

        # action-target is the ground truth action indices
        # gather the relevant latent tokens for the action
        B_act, T_act = action.shape[:2]
        _, seqlen = outputs.shape[:2]
        action_out_token_pos = action_out_token_pos.view(-1) # (B_act * T_act,)
        mask = action_out_token_pos != -1
        # # get the outputs for the action
        # outputs: torch.Tensor = outputs[batch_idx[mask], action_out_token_pos[mask]]
        
        if not mask.any():
            # Early exit if no valid positions
            action_loss = torch.tensor(0.0, requires_grad=True, device=outputs.device)
            ret_action_loss = torch.tensor(0.0, requires_grad=True, device=outputs.device)
            return action_loss, ret_action_loss, None

        # batch_idx = torch.arange(B_act, dtype=torch.long, device=outputs.device).unsqueeze(1).repeat(1, T_act).view(-1)
        # Instead of creating full batch_idx, use advanced indexing directly
        batch_idx_flat = torch.arange(B_act, dtype=torch.long, device=outputs.device).repeat_interleave(T_act)
        
        # calculate the action target
        act_dim = self.action_dim * self.num_pred_steps
        action_tar = select_action_targets(
            action,
            mask.view(B_act, T_act),
            num_pred_steps=self.num_pred_steps,
            action_dim=self.action_dim,
        )

        # Optimized: use advanced indexing more efficiently
        # Flatten outputs for easier indexing: (B, L, D) -> (B*L, D)
        B_out, L_out, D_out = outputs.shape
        outputs_flat = outputs.view(B_out * L_out, D_out)
        # Convert 2D indices to 1D for flat indexing
        indices_flat = batch_idx_flat[mask] * L_out + action_out_token_pos[mask]
        outputs_selected = outputs_flat[indices_flat]
        # calculate the action target

        # output_og = outputs[batch_idx[mask], action_out_token_pos[mask]]
        # assert torch.allclose(output_og, outputs_selected), f"Output og and outputs selected are not close: {output_og.shape=}, {outputs_selected.shape=}"
        # reduce using the action_prompt_mask
        action_prompt_mask_selected = action_prompt_mask.view(-1)[indices_flat].unsqueeze(-1)
        # action_prompt_mask_og = action_prompt_mask[batch_idx[mask], action_out_token_pos[mask]].unsqueeze(-1)
        # assert torch.allclose(action_prompt_mask_og, action_prompt_mask_selected), f"Action prompt mask og and action prompt mask selected are not close: {action_prompt_mask_og.shape=}, {action_prompt_mask_selected.shape=}"
        
        # calculate the action loss
        action_loss, pred_actions = self.action_output_head.loss(outputs_selected, action_tar, return_pred=True)
        # pred_actions = pred_actions.view(B_act, T_act, pred_actions.shape[-1])
        pred_actions = pred_actions.view(pred_actions.shape[0], self.num_pred_steps, self.action_dim)

        ret_action_loss = torch.tensor(0.0, requires_grad=True, device=outputs.device)
        if self.policy_config.ret_emdr2_loss:
            raise NotImplementedError("Ret EMDR2 loss is not implemented for action loss")

        action_loss = (action_loss * action_prompt_mask_selected).sum() / (action_prompt_mask_selected.sum() * act_dim + 1e-6)
        # print(f"{ret_action_loss.item()=}, {action_loss.item()=}")
        return action_loss, ret_action_loss, None

    def _compute_text_loss(
        self,
        outputs: torch.Tensor,
        input_ids: torch.Tensor,
        text_prompt_mask: torch.Tensor,
        additional_info: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Optimized: avoid unnecessary contiguous() calls - slicing already returns views
        shift_outputs = outputs[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        shift_mask = text_prompt_mask[:, 1:]

        # flatten the outputs, labels, and only use the masked positions
        flat_mask = shift_mask.view(-1).bool()
        if not flat_mask.any():
            # Early exit if no valid positions
            text_loss = torch.tensor(0.0, requires_grad=True, device=outputs.device)
            text_accuracy = torch.tensor(0.0, requires_grad=True, device=outputs.device)
            ret_text_loss = torch.tensor(0.0, requires_grad=True, device=outputs.device)
            return text_loss, text_accuracy, ret_text_loss
        
        # Optimized: use advanced indexing to avoid intermediate views
        B, L_minus_1, D = shift_outputs.shape
        flat_outputs = shift_outputs.view(B * L_minus_1, D)[flat_mask]
        flat_labels = shift_labels.view(-1)[flat_mask]

        flat_logits = self.text_output_head(flat_outputs)

        loss_fct = nn.CrossEntropyLoss(reduction='none')
        flat_loss = loss_fct(flat_logits, flat_labels)

        ret_text_loss = torch.tensor(0.0, requires_grad=True, device=flat_outputs.device)
        if self.policy_config.ret_emdr2_loss:
            raise NotImplementedError("Ret EMDR2 loss is not implemented for text loss")

        text_loss = flat_loss.mean()

        # Optimized: only compute accuracy if needed (e.g., during validation or logging)
        # During training, this is expensive and often not needed
        # If you need accuracy, consider computing it less frequently
        if self.training:
            # Skip expensive argmax during training unless explicitly needed
            text_accuracy = torch.tensor(0.0, requires_grad=True, device=flat_outputs.device)
        else:
            predictions = torch.argmax(flat_logits, dim=-1)
            correct = predictions == flat_labels
            text_accuracy = correct.float().mean()
        # print(f"{ret_text_loss.item()=}, {text_loss.item()=}")
        return text_loss, text_accuracy, ret_text_loss

    def add_sequence(self, sequences: Dict[str, Any]):

        for i in range(len(self.sequences)):
            # convert each batch's action from T, N, self.action_dim to T+1, N, self.action_dim
            self.sequences[i]['action'] = torch.cat([self.sequences[i]['action'], sequences[i]['action']], dim=0)

            # convert each batch's proprio from T, N, D to T+1, N, D
            self.sequences[i]['proprio'] = torch.cat([self.sequences[i]['proprio'], sequences[i]['proprio']], dim=0)

            # convert each batch's observation from T, C, H, W to T+1, C, H, W
            self.sequences[i]['observation'] = OrderedDict({k: torch.cat([self.sequences[i]['observation'][k], sequences[i]['observation'][k]], dim=0) for k in self.image_keys})

            # remove access
            if self.tt_window_obs:
                self.sequences[i]['observation'] = OrderedDict({k: self.sequences[i]['observation'][k][-self.tt_obs_window_len:] for k in self.image_keys})
                self.sequences[i]['action'] = self.sequences[i]['action'][-self.tt_action_window_len:]
                self.sequences[i]['proprio'] = self.sequences[i]['proprio'][-self.tt_obs_window_len:]

        return

    def _binarize_gripper(self, action_pred: torch.Tensor) -> torch.Tensor:
        n_index = 2 if self.is_bimanual else 1
        gripper_vals = action_pred[..., -n_index:]
        gripper_vals = gripper_vals.sign() * (gripper_vals.abs() > 0.5).float()
        action_pred[..., -n_index:] = gripper_vals
        if self.has_base_action:
            new_pos = action_pred.shape[-1] - 6
            gripper_vals = action_pred[..., new_pos:new_pos+1]
            gripper_vals = gripper_vals.sign() * (gripper_vals.abs() > 0.5).float()
            action_pred[..., new_pos:new_pos+1] = gripper_vals
        return action_pred

    @torch.inference_mode()
    def predict_raw_action_chunk(self, action_latents: torch.Tensor) -> torch.Tensor:
        """Decode every future physical action without using the rollout queue."""
        action_chunk = self.action_output_head.predict_chunk(
            action_latents,
            num_pred_steps=self.num_pred_steps,
            action_dim=self.action_dim,
        )
        expected_shape = (
            action_latents.shape[0],
            self.num_pred_steps,
            self.action_dim,
        )
        if action_chunk.shape != expected_shape:
            raise RuntimeError(
                f"Action head returned {tuple(action_chunk.shape)}; expected {expected_shape}"
            )
        return action_chunk

    @torch.inference_mode()
    def forward_inference(
        self,
        sequences: Dict[str, Any],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, Any]]:
        extra_info = {}
        # output shape: B, T=1, self.action_dim
        use_cached_action =  (self.start_pos % self.downsample_obs) != 0
        if use_cached_action:
            self.start_pos += 1
            action_pred = self.action_output_head.get_cached_action(pop=True)
            action_pred = self._binarize_gripper(action_pred)
            return action_pred, extra_info
        # assuming augmentations are already done outside
        # images is list of shape (T, C, H, W)
        # proprio is list of shape (T, N, D)
        # action is list of shape (T-1, N, D)
        # update sequence with the original data
        input_seq = () # tuple to make sure it is immutable
        for i in range(len(self.sequences)): # batch size
            # pad action with zeros to match the proprio length
            device = sequences[i]['proprio'].device
            each_batch_dict = {}
            if (not self.eval_caching) or (self.cache_pos == 0): # if we are not caching or we are at the beginning of the sequence
                each_batch_dict['observation'] = OrderedDict({k: torch.cat([self.sequences[i]['observation'][k], sequences[i]['observation'][k]], dim=0) for k in self.image_keys})
                each_batch_dict['proprio'] = torch.cat([self.sequences[i]['proprio'], sequences[i]['proprio']], dim=0)
                each_batch_dict['action'] = torch.cat([self.sequences[i]['action'], torch.zeros(1, 1, self.action_dim).to(device)], dim=0)
                if self.tt_window_obs:
                    each_batch_dict['observation'] = OrderedDict({k: each_batch_dict['observation'][k][-self.tt_obs_window_len:] for k in self.image_keys})
                    each_batch_dict['action'] = each_batch_dict['action'][-self.tt_action_window_len:]
                    each_batch_dict['proprio'] = each_batch_dict['proprio'][-self.tt_obs_window_len:]
            else:
                # set each_batch_dict to be previous one in the sequence and current one in the sequence
                each_batch_dict['observation'] = OrderedDict({k: torch.cat([self.sequences[i]['observation'][k][-1:], sequences[i]['observation'][k]], dim=0) for k in self.image_keys})
                each_batch_dict['proprio'] = torch.cat([self.sequences[i]['proprio'][-1:], sequences[i]['proprio']], dim=0)
                each_batch_dict['action'] = torch.cat([self.sequences[i]['action'][-1:], torch.zeros(1, 1, self.action_dim).to(device)], dim=0)

            assert each_batch_dict['action'].shape[0] == each_batch_dict['proprio'].shape[0], f"Action should have equal frames as proprio (last action is padded to be zero): {each_batch_dict['action'].shape[0]=} {each_batch_dict['proprio'].shape[0]=}"
            num_frames = each_batch_dict['proprio'].shape[0]

            # fill out token positions
            full_ids, img_token_positions, action_inp_token_pos, action_out_token_pos = \
                self.token_seq_gen(self.sequences[i]['inst_ids'].cpu().numpy().tolist(), num_frames, skip_language_tokens=(self.eval_caching and (self.cache_pos > 0)))
            each_batch_dict['input_ids'] = full_ids
            each_batch_dict['image_token_positions'] = img_token_positions
            each_batch_dict['action_inp_token_pos'] = action_inp_token_pos
            each_batch_dict['action_out_token_pos'] = action_out_token_pos

            # fill out masks: not used during inference
            each_batch_dict['text_prompt_mask'] = torch.zeros(len(full_ids), dtype=torch.bool)
            each_batch_dict['action_prompt_mask'] = torch.zeros(len(full_ids), dtype=torch.bool)

            # add to the input sequence
            input_seq += (each_batch_dict,)

        batch = data_utils.collate_fn_tokenizer(input_seq)
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device='cuda', non_blocking=True)
            elif isinstance(v, dict):
                batch[k] = {k2: v2.to(device='cuda', non_blocking=True) if isinstance(v2, torch.Tensor) else v2
                           for k2, v2 in v.items()}
        B, L = batch['input_ids'].shape
        embeddings, additional_info = self.preprocessing(
            input_ids=batch['input_ids'],
            proprio=batch['proprio'],
            action=batch['action'],
            action_inp_token_pos=batch['action_inp_token_pos'],
            images=batch['observation'],
            image_token_positions=batch['image_token_positions'],
        )
        cache_start_pos = self.cache_pos if self.eval_caching else 0
        outputs, _ = self.llama(embeddings, mask=None, start_pos=cache_start_pos)
        # get the last one output; and use it to generate the action
        # using action_out_token_pos to get the last output
        action_out_token_pos = batch['action_out_token_pos'][:, -1:]
        B_act, T_act = action_out_token_pos.shape[:2]
        action_out_token_pos = action_out_token_pos.view(-1)
        batch_idx = torch.arange(B_act, dtype=torch.long, device=outputs.device).unsqueeze(1).repeat(T_act, 1).view(-1)
        mask = action_out_token_pos != -1
        outputs = outputs[batch_idx[mask], action_out_token_pos[mask]]

        # B, T=1, self.action_dim
        action_pred = self.action_output_head.pred(outputs, num_pred_steps=self.num_pred_steps, action_dim=self.action_dim)
        # print(colored(f"{action_pred.shape=}", "green"))

        # binarize the gripper values
        if True:
            action_pred = self._binarize_gripper(action_pred)

        for i in range(len(self.sequences)):
            # convert each batch's action_pred from T=1, self.action_dim to T, 1, self.action_dim
            sequences[i]['action'] = action_pred[i][:, None, :]

        # save the predicted action, observation, and proprio to self.sequences
        self.add_sequence(sequences)

        self.start_pos += 1
        self.cache_pos += (L - self.latent_len) # increase by L tokens - latent_len tokens
        return action_pred, extra_info

    def state_dict(self, destination=None, prefix='', keep_vars=False, mode=''):
        state_dict = super().state_dict(destination, prefix, keep_vars)
        # only returns trainable parameters
        trainable_params = self.get_trainable_params(self.phase)
        # add any parameters with batch norm to the trainable parameters
        for n, b in self.named_buffers():
            if "running_" in n or "num_batches_tracked" in n or "tau" in n:
                trainable_params[n] = b

        new_state_dict = OrderedDict()
        for k in trainable_params:
            if k in state_dict:
                new_state_dict[k] = state_dict[k]
        return new_state_dict

    def get_trainable_params(self, phase='pretrain'):
        trainable = {}
        def handle_vision_encoder(name, para):
            if self.vision_finetune and name.startswith("vision_encoder.") and para.requires_grad:
                trainable[name] = para
            elif name.startswith("vision_encoder."):
                pass
            else:
                return False
            return True
        if phase == 'pretrain':
            for name, para in self.named_parameters():
                param_vision = handle_vision_encoder(name, para)
                if not param_vision:
                    trainable[name] = para
        elif (phase == 'pretrain_lora'):
            for name, para in self.named_parameters():
                param_vision = handle_vision_encoder(name, para)
                if not param_vision:
                    if (phase == 'pretrain_lora') and (name.startswith("llama.")):
                        if ('lora' in name) or ('lm_head' in name) or ('norm' in name):
                            trainable[name] = para
                    else:
                        trainable[name] = para
        else:
            raise ValueError(f"Unknown model phase: {phase}")
        return trainable

    def set_default_trainability(self, phase='pretrain'):
        for key, value in self.named_parameters():
            if key.startswith("vision_encoder."):
                continue
            value.requires_grad = False
        for key, value in self.get_trainable_params(phase).items():
            value.data = value.data.float()
            value.requires_grad = True
        return

    def reset_inference_state(self, batch_size: int = 1):
        self.start_pos = 0
        self.cache_pos = 0
        if hasattr(self.llama, 'reset'):
            self.llama.reset()
        self.action_output_head.reset()
        self.input_ids = []
        # each one will be of shape (B, T, ...)
        self.sequences = [{
            'observation': OrderedDict({img_key: torch.Tensor().to(device='cuda') for img_key in self.image_keys}),
            'proprio': torch.Tensor().to(device='cuda'),
            'action': torch.Tensor().to(device='cuda'),
        } for _ in range(batch_size)]
        self.token_seq_gen = TokenSequenceGen.LangTrajSequence(
            image_keys=self.image_keys,
            image_token_ids=self.image_token_ids,
            tokens_per_frame=self.per_cam_latent_len,
            action_token_ids=self.action_token_ids,
            tokens_per_action=1,
            action_in_inputs=not self.remove_action,
            pad_inst_tokens=self.policy_config.model_version != "v2",
            max_inst_tokens=self.shared_config.max_inst_tokens,
            inst_token_pad_value=self.tokenizer.pad_token_id,
        )
        return

    def set_inst_ids(self, inst_ids: List[int]):
        assert len(inst_ids) == len(self.sequences), f"Expected {len(self.sequences)} inst_ids, got {len(inst_ids)}"
        for i in range(len(self.sequences)):
            self.sequences[i]['inst_ids'] = torch.tensor(inst_ids[i], dtype=torch.long)
        return

    def prompt(self, languages: Union[str, List[str]]):
        if isinstance(languages, str):
            languages = [languages]
        inst_ids = []
        for language in languages:
            inst_ids.append(self.tokenizer.encode(language, add_special_tokens=False))
        self.set_inst_ids(inst_ids)
        return

    def reset(self, *args, **kwargs): # backward compatibility
        return self.reset_inference_state(*args, **kwargs)

    def get_logging_values(self):
        logging_values = {}
        if hasattr(self.llama, 'get_logging_values'):
            logging_values['model'] = self.llama.get_logging_values()
        if hasattr(self, 'topk_transformer_block'):
            if isinstance(self.topk_transformer_block, nn.ModuleList):
                logging_values['topk_transformer_block'] = {f'block_{i}': block.get_logging_values() for i, block in enumerate(self.topk_transformer_block)}
            else:
                logging_values['topk_transformer_block'] = self.topk_transformer_block.get_logging_values()
        return logging_values

    def rename_state_dict_keys(self, state_dict, phase):
        if hasattr(self.llama, 'rename_state_dict_keys'):
            state_dict = self.llama.rename_state_dict_keys(state_dict, phase)
        return state_dict

    def get_total_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def action_dim(self) -> int:
        return self._action_dim
    
    def use_cache(self, use_cache: bool):
        self.eval_caching = use_cache
        self.llama.use_cache(use_cache)
        return
