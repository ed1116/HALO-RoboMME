import json
from typing import Union, Optional
import torch.nn as nn
from halo.util.args import ModelConfig, SharedConfig, VisionEncoderConfig, PolicyConfig
from halo.models.backbones.encoders import VisionEncoder, VisionEncoderCNN, VisionPatchEncoder, VisionEncoderCNNNoPool
from halo.models.policy.mem_model import MemModel

def vision_encoder_constructor(
    vision_encoder_cfg : VisionEncoderConfig,
    extra_kwargs: Optional[dict] = None,
) -> Union[VisionEncoder, VisionEncoderCNN, VisionEncoderCNNNoPool]:
    """Instantiate a vision encoder based on the vision_encoder_cfg

    Return the vision encoder instance
    """
    if vision_encoder_cfg.vision_unfreeze_all:
        vision_finetune = "all"
    elif vision_encoder_cfg.vision_unfreeze_last_n > 0:
        vision_finetune = vision_encoder_cfg.vision_unfreeze_last_n
    elif vision_encoder_cfg.vision_lora:
        vision_finetune = "lora"
    elif vision_encoder_cfg.freeze_all:
        assert not vision_encoder_cfg.vision_nonpretrained, "if you are freezing all, you need to use a pretrained model"
        vision_finetune = False
    else:
        assert not vision_encoder_cfg.vision_nonpretrained, "if you are freezing all, you need to use a pretrained model"
        vision_encoder_cfg.freeze_all = True # default to freeze all
        vision_finetune = False

    vision_encoder_name = vision_encoder_cfg.vision_encoder
    vision_pretrained = not vision_encoder_cfg.vision_nonpretrained
    print(f"using vision encoder {vision_encoder_name}")
    extra_kwargs = extra_kwargs or {}
    ### print all the properties of the vision encoder config
    vision_encoder_cls = VisionEncoder
    if "bbox" in vision_encoder_name.lower():
        raise ValueError("bbox vision encoder is not supported for vision encoder constructor")
    else:
        global_pool = ""
        if "resnet" in vision_encoder_name.lower() and "nopool" in vision_encoder_name.lower():
            vision_encoder_cls = VisionEncoderCNNNoPool
        elif "resnet" in vision_encoder_name.lower():
            vision_encoder_cls = VisionEncoderCNN
        elif "patch-only" in vision_encoder_name.lower():
            # name is of the format patch-only-32-128
            vision_encoder_cfg.vision_nonpretrained = True
            vision_pretrained = False
            vision_finetune = "all"
            vision_encoder_cls = VisionPatchEncoder 
            extra_kwargs["patch_size"] = int(vision_encoder_name.split("-")[-2])
            extra_kwargs["img_size"] = int(vision_encoder_name.split("-")[-1])
        else:
            vision_encoder_cls = VisionEncoder
        # print all the properties of the vision encoder config
        print("*"*20)
        for k, v in vision_encoder_cfg.__dict__.items():
            print(f"{k}: {v}")
        print(f"vision_encoder_name: {vision_encoder_name}")
        print(f"vision_pretrained: {vision_pretrained}")
        print(f"vision_finetune: {vision_finetune}")
        print(f"vision_encoder_cls: {vision_encoder_cls}")
        print(f"extra_kwargs: {extra_kwargs}")
        print("*"*20)
        assert "mini_batch_size" in extra_kwargs, "mini_batch_size is required for vision encoder"
        assert "img_seq_len" in extra_kwargs, "img_seq_len is required for vision encoder"
        assert "num_cameras" in extra_kwargs, "num_cameras is required for vision encoder"
        vision_encoder = vision_encoder_cls(
            vision_encoder_name,
            pretrained=vision_pretrained,
            global_pool=global_pool,
            finetune=vision_finetune,
            lora_rank=vision_encoder_cfg.vision_lora_rank,
            **extra_kwargs,
        )

    return vision_encoder

def policy_constructor(
    policy_cfg : PolicyConfig,
    shared_config : SharedConfig,
    vision_encoder : Union[VisionEncoder, VisionEncoderCNN],
    train : bool = True,
    extra_kwargs: Optional[dict] = {},
) -> nn.Module:
    """
    Instantiate a policy based on the policy_cfg
    """
    # proprio and action dim
    proprio_dim = 10 if shared_config.rot_6d else 8 # OR 8 for robocasa quaternion prediction
    action_dim = 3+6+1 if shared_config.rot_6d else 7+1 # actions for robocasa are in axis-angle format
    if shared_config.rot_euler:
        proprio_dim = 7
    if shared_config.rot_euler:
        action_dim = 8
    if shared_config.no_img:
        shared_config.num_cameras = 0
    if shared_config.use_lerobot:
        proprio_dim = 16 # 7 + 7 + 2
        
    if shared_config.is_bimanual:
        proprio_dim *= 2
        action_dim *= 2
        action_dim -= 1
    if shared_config.has_base_action:
        action_dim += 5
        
    if shared_config.use_tokenizer_dataset:
        extra_kwargs.update({
            'action_dim': action_dim,
            'proprio_dim': proprio_dim,
        })
        assert "image_keys" in extra_kwargs, "image_keys are required for tokenizer dataset"
        model = MemModel(
            shared_config=shared_config,
            policy_config=policy_cfg,
            vision_encoder=vision_encoder,
            train=train,
            extra_kwargs=extra_kwargs,
            output_modes=["action", "text"],
        )
    else:
        raise ValueError(f"Set use_tokenizer_dataset to True to use the tokenizer dataset")
    return model

def model_constructor(
    model_config: ModelConfig,
    shared_config: SharedConfig,
    train: bool = True,
    extra_kwargs: Optional[dict] = None,
) -> Union[nn.Module]:
    """
    Main model constructor that creates an ICRT policy
    """
    extra_kwargs = extra_kwargs or {}

    vision_encoder = None
    extra_kwargs['mini_batch_size'] = 1  # shared_config.batch_size
    extra_kwargs['img_seq_len'] = shared_config.seq_length // shared_config.downsample_obs
    extra_kwargs['num_cameras'] = shared_config.num_cameras
    if not shared_config.no_img:
        vision_encoder = vision_encoder_constructor(model_config.vision_encoder_cfg, extra_kwargs=extra_kwargs)
    policy = policy_constructor(model_config.policy_cfg, shared_config, vision_encoder, train=train, extra_kwargs=extra_kwargs)
    return policy
