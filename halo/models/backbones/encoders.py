from typing import Optional

from functools import partial
from typing import Optional
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
import timm.models.vision_transformer
from timm.layers import Mlp
from timm.layers.config import use_fused_attn
from timm.layers.mlp import Mlp
from timm.layers.weight_init import trunc_normal_tf_, init_weight_vit
from timm.models.vision_transformer import checkpoint_filter_fn, build_model_with_cfg, VisionTransformer
from torch.jit import Final
import loralib as lora
import numpy as np
from timm.models.layers import PatchEmbed, RotaryEmbedding

def vit_get_latent_len(global_pool, model):
    global_pool = model.global_pool
    if global_pool in ['']:
        latent_len = model.patch_embed.num_patches
        if model.cls_token is not None:
            latent_len += 1
        if model.reg_token is not None:
            latent_len += 1
    elif global_pool in ['avg', 'cls']:
        latent_len = 1
    else:
        raise ValueError(f"global_pool {global_pool} is not supported")
    return latent_len

class SpatialSoftmax(torch.nn.Module):
    """
    Spatial Softmax Layer.

    Based on Deep Spatial Autoencoders for Visuomotor Learning by Finn et al.
    https://rll.berkeley.edu/dsae/dsae.pdf
    """
    def __init__(
        self,
        input_shape,
        num_kp=32,
        temperature=1.,
        learnable_temperature=False,
        output_variance=False,
        noise_std=0.0,
    ):
        """
        Args:
            input_shape (list): shape of the input feature (C, H, W)
            num_kp (int): number of keypoints (None for not using spatialsoftmax)
            temperature (float): temperature term for the softmax.
            learnable_temperature (bool): whether to learn the temperature
            output_variance (bool): treat attention as a distribution, and compute second-order statistics to return
            noise_std (float): add random spatial noise to the predicted keypoints
        """
        super(SpatialSoftmax, self).__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape # (C, H, W)

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._num_kp = num_kp
        else:
            self.nets = None
            self._num_kp = self._in_c
        self.learnable_temperature = learnable_temperature
        self.output_variance = output_variance
        self.noise_std = noise_std

        if self.learnable_temperature:
            # temperature will be learned
            temperature = torch.nn.Parameter(torch.ones(1) * temperature, requires_grad=True)
            self.register_parameter('temperature', temperature)
        else:
            # temperature held constant after initialization
            temperature = torch.nn.Parameter(torch.ones(1) * temperature, requires_grad=False)
            self.register_buffer('temperature', temperature)

        pos_x, pos_y = np.meshgrid(
                np.linspace(-1., 1., self._in_w),
                np.linspace(-1., 1., self._in_h)
                )
        pos_x = torch.from_numpy(pos_x.reshape(1, self._in_h * self._in_w)).float()
        pos_y = torch.from_numpy(pos_y.reshape(1, self._in_h * self._in_w)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

        self.kps = None

    def __repr__(self):
        """Pretty print network."""
        header = format(str(self.__class__.__name__))
        return header + '(num_kp={}, temperature={}, noise={})'.format(
            self._num_kp, self.temperature.item(), self.noise_std)

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        assert(len(input_shape) == 3)
        assert(input_shape[0] == self._in_c)
        return [self._num_kp, 2]

    def forward(self, feature):
        """
        Forward pass through spatial softmax layer. For each keypoint, a 2D spatial 
        probability distribution is created using a softmax, where the support is the 
        pixel locations. This distribution is used to compute the expected value of 
        the pixel location, which becomes a keypoint of dimension 2. K such keypoints
        are created.

        Returns:
            out (torch.Tensor or tuple): mean keypoints of shape [B, K, 2], and possibly
                keypoint variance of shape [B, K, 2, 2] corresponding to the covariance
                under the 2D spatial softmax distribution
        """
        assert(feature.shape[1] == self._in_c)
        assert(feature.shape[2] == self._in_h)
        assert(feature.shape[3] == self._in_w)
        if self.nets is not None:
            feature = self.nets(feature)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        feature = feature.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(feature / self.temperature, dim=-1)
        # [1, H * W] x [B * K, H * W] -> [B * K, 1] for spatial coordinate mean in x and y dimensions
        expected_x = torch.sum(self.pos_x * attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * attention, dim=1, keepdim=True)
        # stack to [B * K, 2]
        expected_xy = torch.cat([expected_x, expected_y], 1)
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._num_kp, 2)

        if self.training:
            noise = torch.randn_like(feature_keypoints) * self.noise_std
            feature_keypoints += noise

        if self.output_variance:
            # treat attention as a distribution, and compute second-order statistics to return
            expected_xx = torch.sum(self.pos_x * self.pos_x * attention, dim=1, keepdim=True)
            expected_yy = torch.sum(self.pos_y * self.pos_y * attention, dim=1, keepdim=True)
            expected_xy = torch.sum(self.pos_x * self.pos_y * attention, dim=1, keepdim=True)
            var_x = expected_xx - expected_x * expected_x
            var_y = expected_yy - expected_y * expected_y
            var_xy = expected_xy - expected_x * expected_y
            # stack to [B * K, 4] and then reshape to [B, K, 2, 2] where last 2 dims are covariance matrix
            feature_covar = torch.cat([var_x, var_xy, var_xy, var_y], 1).reshape(-1, self._num_kp, 2, 2)
            feature_keypoints = (feature_keypoints, feature_covar)

        if isinstance(feature_keypoints, tuple):
            self.kps = (feature_keypoints[0].detach(), feature_keypoints[1].detach())
        else:
            self.kps = feature_keypoints.detach()
        return feature_keypoints

class Attention(nn.Module):
    fused_attn: Final[bool]
    lora_rank = 8
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = lora.Linear(dim, dim * 3, r=self.lora_rank, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def _create_vision_transformer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    if 'flexi' in variant:
        # FIXME Google FlexiViT pretrained models have a strong preference for bilinear patch / embed
        # interpolation, other pretrained models resize better w/ anti-aliased bicubic interpolation.
        _filter_fn = partial(checkpoint_filter_fn, interpolation='bilinear', antialias=False)
    else:
        _filter_fn = checkpoint_filter_fn

    # FIXME attn pool (currently only in siglip) params removed if pool disabled, is there a better soln?
    if "pretrained_strict" in kwargs:
        strict = kwargs["pretrained_strict"]
        kwargs.pop("pretrained_strict")
    else:
        strict = True
    if 'siglip' in variant and kwargs.get('global_pool', None) != 'map':
        strict = False

    return build_model_with_cfg(
        VisionTransformer,
        variant,
        pretrained,
        pretrained_filter_fn=_filter_fn,
        pretrained_strict=strict,
        **kwargs,
    )

class VisionEncoder(nn.Module):
    def __init__(
            self, 
            name, 
            pretrained=True,
            global_pool='',
            finetune=False,
            lora_rank=8,
            mini_batch_size=None,
            img_seq_len=None,
            num_cameras=None,
            **kwargs,
        ):
        """
        Using timm vision encoder
        by default, we do not pool visual features at this stage
        currently it only works with a single camera

        Params:
            finetune has a few options
                1. False: we freeze the model
                2. (int) > 0: we unfreeze the last n blocks
                3. "all": we finetune the entire model
                4. "lora": lora finetuning
        """
        super().__init__()
        self.name = name
        self.pretrained = pretrained
        self.finetune = finetune
        self.mini_batch_size = mini_batch_size # used for padding during evaluation
        self.img_seq_len = img_seq_len
        self.num_cameras = num_cameras
        if finetune == "lora":
            Attention.lora_rank = lora_rank
            timm.models.vision_transformer.Attention = Attention
            timm.models.vision_transformer._create_vision_transformer = _create_vision_transformer
            kwargs = {"pretrained_strict": False}
        else:
            kwargs = {}
        if "cross-mae-rtx" in name:
            self.model = timm.create_model("vit_base_patch16_224.mae", pretrained=pretrained, global_pool=global_pool, **kwargs)
            timm.models.load_checkpoint(self.model, name, strict=False)
        elif "dust3r" in name.lower():
            self.model = timm.create_model("vit_large_patch16_224", pretrained=False)
            ckpt = torch.load(name, map_location='cpu')
            # Extract encoder weights from the dust3r model
            encoder_weights = ckpt['model']['patch_embed.proj.weight']
            # Load encoder weights into the corresponding part
            self.model.patch_embed.proj.weight.data.copy_(encoder_weights)
        else:
            self.model = timm.create_model(name, pretrained=pretrained, global_pool=global_pool, **kwargs)
        if self.finetune:
            self.model.train()
            if isinstance(self.finetune, int):
                self.unfreeze_last_n_blocks(self.finetune)
            elif self.finetune == "all":
                self.unfreeze()
            elif self.finetune == "lora":
                # enable training of all biases to sequeze more performance out
                # https://github.com/microsoft/LoRA
                lora.mark_only_lora_as_trainable(self.model, bias='all')
        else:
            self.model.eval()
            self.freeze()

        # we need to extract the latent length from the name and if there's a cls token, add 1 to the latent length
        self._latent_len = vit_get_latent_len(global_pool, self.model)


        self.model.norm = nn.Identity() # we apply this outside to make it trainable
        self._apply_post_norm = False

        # if "cross-mae-rtx" in name:
        #     self.model.train()
        #     self.model.norm.weight.requires_grad = True
        #     self.model.norm.bias.requires_grad = True

    @property
    def latent_len(self):
        return self._latent_len

    def out_dim(self):
        return self.model.embed_dim

    def unfreeze_last_n_blocks(self, n):
        # we unfreeze the last n blocks
        assert isinstance(self.model, timm.models.vision_transformer.VisionTransformer), "unfreeze only works for vision transformer"
        if len(self.model.blocks) < n:
            print(f"can only unfreeze {len(self.model.blocks)} blocks instead of {n} blocks!")
            n = len(self.model.blocks)
        # first freeze everything
        self.freeze()
        for i in range(n):
            for param in self.model.blocks[-i].parameters():
                param.requires_grad = True
        print(f"unfreeze last {n} blocks")

    def unfreeze(self):
        # we unfreeze the model depends on finetune or not
        for param in self.model.parameters():
            param.requires_grad = True

    def freeze(self):
        # we freeze the model depends on finetune or not
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        x : B, (T), (N), 3, H, W
        remember to move all data to batch axis first
        output: B, (T), (N), L, D
        """
        initial_shape = x.shape[:-3]
        trailing_shape = x.shape[-3:]
        x = x.view(-1, *trailing_shape)
        new_B = x.shape[0]
        # ########################## TO MAINTAIN DETERMINISM #########################################
        # padding_length = self.mini_batch_size * self.img_seq_len * self.num_cameras # maybe so much padding is not needed
        # if (not self.training) and (new_B < padding_length):
        #     x = torch.cat([x, torch.zeros(padding_length - new_B, *x.shape[1:], device=x.device, dtype=x.dtype)], dim=0)
        # ##############################################################################################

        if self.finetune:
            feats = self.model.forward_features(x)
        else:
            with torch.no_grad():
                feats = self.model.forward_features(x)

        # ########################## TO MAINTAIN DETERMINISM #########################################
        # if (not self.training) and (new_B < padding_length):
        #     feats = feats[:new_B]
        # ##############################################################################################
        assert feats.shape[-2] == self.latent_len, f"feats.shape[-2]: {feats.shape[-2]}, self.latent_len: {self.latent_len}"

        return feats.view(*initial_shape, *feats.shape[1:])


class VisionEncoderCNN(nn.Module):
    def __init__(self, name, pretrained=True, global_pool='', finetune=False, lora_rank=8, **kwargs):
        """
        Using timm vision encoder
        by default, we do not pool visual features at this stage
        currently it only works with a single camera

        Params:
            finetune has a few options
                1. False: we freeze the model
                2. (int) > 0: we unfreeze the last n blocks
                3. "all": we finetune the entire model
                4. "lora": lora finetuning
        """
        super().__init__()
        self.name = name
        self.pretrained = pretrained
        self.finetune = finetune
        kwargs = {"num_classes": 0}
        # print all the default arguments
        print(f"[timm info] using vision encoder {name}, pretrained: {pretrained}, global_pool: {global_pool}, finetune: {finetune}, lora_rank: {lora_rank}, kwargs: {kwargs}")
        self.model = timm.create_model(name, pretrained=pretrained, global_pool=global_pool, **kwargs)
        if self.finetune:
            self.model.train()
            if self.finetune == "all":
                self.unfreeze()
            elif self.finetune == "lora":
                # enable training of all biases to sequeze more performance out
                # https://github.com/microsoft/LoRA
                lora.mark_only_lora_as_trainable(self.model, bias='all')
        else:
            self.model.eval()
            self.freeze()
        self._apply_post_norm = False
        self._latent_len = 1
        # self.pooling_layer = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.pooling_layer = SpatialSoftmax(input_shape=(self.model.num_features, 4, 4), num_kp=256)
        # change pooling layer to be a global learnable pooling layer

        assert finetune == "all", "only all finetuning is supported for CNN"


    @property   
    def latent_len(self):
        return self._latent_len

    def out_dim(self):
        return self.model.num_features

    def unfreeze_last_n_blocks(self, n):
        # we unfreeze the last n blocks
        raise NotImplementedError("Not supported for CNN")

    def unfreeze(self):
        # we unfreeze the model depends on finetune or not
        for param in self.model.parameters():
            param.requires_grad = True

    def freeze(self):
        # we freeze the model depends on finetune or not
        for param in self.model.parameters():
            param.requires_grad = False

    def combine_post_pooling(self, feats : torch.Tensor) -> torch.Tensor:
        """
        feats are of dimension: B, T, num_kp, 2
        output is B, T, L, D
        """
        return feats.flatten(-2, -1).unsqueeze(1)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        x : B, (T), (N), 3, H, W
        remember to move all data to batch axis first
        output: B, (T), (N), L, D
        """
        initial_shape = x.shape[:-3]
        trailing_shape = x.shape[-3:]
        x = x.view(-1, *trailing_shape)

        # if self.finetune == "all":
        feats = self.model.forward_features(x)
        feats = self.pooling_layer(feats)
        feats = self.combine_post_pooling(feats)
        # else:
        #     with torch.no_grad():
        #         feats = self.model.forward_features(x)
        #         feats = feats.permute(0, 2, 3, 1).flatten(1, 2)

        return feats.view(*initial_shape, *feats.shape[1:])

def make_resnet_output_stride_16(model: nn.Module) -> nn.Module:
    """
    For the ResNet shown (BasicBlock stages), remove the final downsampling
    in layer4 so overall output stride becomes 16 instead of 32.

    For 128x128 input:
      - before: layer4 output ~ 4x4
      - after:  layer4 output ~ 8x8
    """
    b0 = model.layer4[0]  # first block in layer4

    # Main path: BasicBlock downsampling happens in conv1
    b0.conv1.stride = (1, 1)

    # Skip path: downsample conv must match
    if b0.downsample is not None:
        # downsample is Sequential(Conv2d, BN)
        b0.downsample[0].stride = (1, 1)

    return model


class VisionEncoderCNNNoPool(VisionEncoderCNN):
    def __init__(self, name, *args, **kwargs):
        name = name.replace("_nopool", "") # resnet50_nopool -> resnet50
        super().__init__(name, *args, **kwargs)
        self.model = make_resnet_output_stride_16(self.model)
        self.pooling_layer = nn.Identity()
        self._latent_len = 8 * 8

    def combine_post_pooling(self, feats : torch.Tensor) -> torch.Tensor:
        """
        feats are of dimension: B, T, num_features, 8, 8
        output is B, T, L = 8 * 8, D = num_features
        """
        return feats.flatten(-2, -1).transpose(-1, -2)

class MlpProjection(Mlp):
    def combine_forward(self, visual_tokens : torch.Tensor, proprio_tokens : torch.Tensor) -> torch.Tensor:
        """
        visual_tokens are of dimension: B, T, num_tokens, C
        proprio_tokens are of dimension: B, T, C
        output is B x T x latent_len x C
        """
        B, T, num_tokens, C = visual_tokens.shape
        proprio_tokens = proprio_tokens.view(B, T, 1, C)
        all_tokens = torch.cat([visual_tokens, proprio_tokens], dim=2)
        all_tokens = self.forward(all_tokens)
        return all_tokens
    def combine_forward_visual(self, visual_tokens : torch.Tensor) -> torch.Tensor:
        """
        visual_tokens are of dimension: B, T, num_tokens, C
        output is B x T x latent_len x C
        """
        B, T, num_tokens, C = visual_tokens.shape
        return self.forward(visual_tokens)

class AttentionPool(nn.Module):
    """
    Attention pooling w/ latent query
    modified from
    https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/attention_pool.py
    """
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            in_features: int,
            out_features: int = None,
            embed_dim: int = None,
            num_heads: int = 8,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            latent_len: int = 1,
            latent_dim: int = None,
            pos_embed: str = '',
            pool_type: str = '',
            norm_layer: Optional[nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            drop: float = 0.0,
            spatial_len : int = 7,
            mini_batch_size=None,
            img_seq_len=None,
    ):
        super().__init__()
        self.in_features = in_features
        embed_dim = embed_dim or in_features
        out_features = out_features or in_features
        self.out_features = out_features
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.pool = pool_type
        self.fused_attn = use_fused_attn()

        self.latent_dim = latent_dim or embed_dim
        self.latent_len = latent_len
        self.latent = nn.Parameter(torch.zeros(1, self.latent_len, embed_dim))

        if (pos_embed == 'learned' or pos_embed == 'abs') and latent_len > 1:
            if pos_embed == "learned":
                self.pos_embed = nn.Parameter(torch.zeros(self.latent_len, in_features))
            self.pos_embed_type = pos_embed
        else:
            self.pos_embed = None
            self.pos_embed_type = None

        self.q = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.kv = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(embed_dim, out_features)
        self.proj_drop = nn.Dropout(drop)

        self.norm = norm_layer(out_features) if norm_layer is not None else nn.Identity()
        self.mlp = Mlp(
            in_features=out_features,
            hidden_features=int(out_features * mlp_ratio),
            out_features=out_features
        )
        self.mini_batch_size = mini_batch_size
        self.img_seq_len = img_seq_len

        self.init_weights()

    def init_weights(self):
        if self.pos_embed_type is not None:
            if self.pos_embed_type == 'learned':
                trunc_normal_tf_(self.pos_embed, std=self.pos_embed.shape[1] ** -0.5)
            elif self.pos_embed_type == 'abs':
                from .pos_embed import get_1d_sincos_pos_embed_from_grid
                self.pos_embed = nn.Parameter(torch.tensor(
                    get_1d_sincos_pos_embed_from_grid(self.in_features, np.arange(self.latent_len)),
                    requires_grad=False
                ))
        trunc_normal_tf_(self.latent, std=self.latent_dim ** -0.5)

    def forward(self, x):
        B, N, C = x.shape

        if self.pos_embed_type is not None:
            x = x.repeat(1, self.latent_len, 1)
            N = self.latent_len
            x = x + self.pos_embed.unsqueeze(0).to(x.dtype)

        q_latent = self.latent.expand(B, -1, -1)
        q = self.q(q_latent).reshape(B, self.latent_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, self.latent_len, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x + self.mlp(self.norm(x))

        # optional pool if latent seq_len > 1 and pooled output is desired
        if self.pool == 'token':
            x = x[:, 0]
        elif self.pool == 'avg':
            x = x.mean(1)
        elif self.pool == '':
            pass # we return the whole sequence
        return x

    def expand_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x is of dimension: B, T, (input_dim), C
        output is of dimension B, T, latent_len, out_features
        """
        if x.ndim == 3:
            x = x[:, :, None, :]
        B, T, input_dim, C = x.shape
        x = x.view(B*T, input_dim, C)
        out = self.forward(x)
        return out.view(B, T, self.latent_len, self.out_features)

    def combine_forward(self, visual_tokens : torch.Tensor, proprio_tokens : torch.Tensor) -> torch.Tensor:
        """
        visual_tokens are of dimension: B, T, num_tokens, C
        proprio_tokens are of dimension: B, T, C
        output is B x T x latent_len x C
        """
        B, T, num_tokens, C = visual_tokens.shape
        visual_tokens = visual_tokens.view(B*T, num_tokens, C)
        proprio_tokens = proprio_tokens.view(B*T, 1, C)
        tokens = torch.cat([visual_tokens, proprio_tokens], dim=1)
        orig_B = tokens.shape[0]
        # ########################## TO MAINTAIN DETERMINISM #########################################
        # padding_B = self.mini_batch_size * self.img_seq_len
        # if (not self.training) and (orig_B < padding_B):
        #     tokens = torch.cat([tokens, torch.zeros(padding_B - orig_B, *tokens.shape[1:], device=tokens.device, dtype=tokens.dtype)], dim=0)
        # ##############################################################################################
        input_B = tokens.shape[0]
        tokens =  self.forward(tokens)
        # ########################## TO MAINTAIN DETERMINISM #########################################
        # if (not self.training) and (orig_B < padding_B):
        #     tokens = tokens[:orig_B]
        # ##############################################################################################
        return tokens.view(B, T, self.latent_len, self.out_features)  

    def forward_visual(self, visual_tokens : torch.Tensor) -> torch.Tensor:
        """
        visual_tokens are of dimension: B, T, num_tokens, C
        output is B x T x latent_len x C
        """
        B, T, num_tokens, C = visual_tokens.shape
        visual_tokens = visual_tokens.view(B*T, num_tokens, C)
        return self.forward(visual_tokens).view(B, T, self.latent_len, self.out_features)

    def combine_forward_discrete(self, visual_tokens : torch.Tensor, proprio_tokens: torch.Tensor) -> torch.Tensor:
        """
        used for dicrt exclusively
        visual_tokens are of dimension: B, T, num_tokens, C
        proprio_tokens are of dimension: B, T, num_prop_tokens, C
        output is B x T x latent_len x C
        """
        B, T, num_tokens, C = visual_tokens.shape
        B, T, num_prop_tokens, C = proprio_tokens.shape
        visual_tokens = visual_tokens.view(B*T, num_tokens, C)
        proprio_tokens = proprio_tokens.view(B*T, num_prop_tokens, C)
        tokens = torch.cat([visual_tokens, proprio_tokens], dim=1)
        return self.forward(tokens).view(B, T, self.latent_len, self.out_features)


class MultiKVAttentionPool(nn.Module):
    """
    Attention pooling w/ latent query and different key-value projections for different data
    modified from
    https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/attention_pool.py
    """
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            in_features: int,
            out_features: int = None,
            num_modalities=2,
            embed_dim: int = None,
            num_heads: int = 8,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            latent_len: int = 1,
            latent_dim: int = None,
            pos_embed: str = '',
            pool_type: str = '',
            norm_layer: Optional[nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            drop: float = 0.0,
            spatial_len : int = 7,
    ):
        super().__init__()
        self.in_features = in_features
        embed_dim = embed_dim or in_features
        out_features = out_features or in_features
        self.out_features = out_features
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.pool = pool_type
        self.fused_attn = use_fused_attn()
        self.num_modalities = num_modalities

        self.latent_dim = latent_dim or embed_dim
        self.latent_len = latent_len
        self.latent = nn.Parameter(torch.zeros(1, self.latent_len, embed_dim))

        if (pos_embed == 'learned' or pos_embed == 'abs') and latent_len > 1:
            if pos_embed == "learned":
                self.pos_embed = nn.Parameter(torch.zeros(self.latent_len, in_features))
            self.pos_embed_type = pos_embed
        else:
            self.pos_embed = None
            self.pos_embed_type = None

        self.q = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.kv = nn.ModuleList([nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias) for _ in range(self.num_modalities)])
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(embed_dim, out_features)
        self.proj_drop = nn.Dropout(drop)

        self.norm = norm_layer(out_features) if norm_layer is not None else nn.Identity()
        self.mlp = Mlp(
            in_features=out_features,
            hidden_features=int(out_features * mlp_ratio),
            out_features=out_features
        )

        self.init_weights()

    def init_weights(self):
        if self.pos_embed_type is not None:
            if self.pos_embed_type == 'learned':
                trunc_normal_tf_(self.pos_embed, std=self.pos_embed.shape[1] ** -0.5)
            elif self.pos_embed_type == 'abs':
                from .pos_embed import get_1d_sincos_pos_embed_from_grid
                self.pos_embed = nn.Parameter(torch.tensor(
                    get_1d_sincos_pos_embed_from_grid(self.in_features, np.arange(self.latent_len)),
                    requires_grad=False
                ))
        trunc_normal_tf_(self.latent, std=self.latent_dim ** -0.5)

    def forward(self, *modality_tokens):
        kv = []
        for i in range(self.num_modalities):
            tokens = modality_tokens[i]
            if len(tokens.shape) == 3:
                num_tokens = 1
                B, T, C = tokens.shape
            else:
                B, T, num_tokens, C = tokens.shape
            tokens = tokens.view(B * T, num_tokens, C)
            kv.append(self.kv[i](tokens).reshape(B * T, num_tokens, 2, self.num_heads, self.head_dim))

        kv = torch.cat(kv, dim=1).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)


        q_latent = self.latent.expand(B * T, -1, -1)
        q = self.q(q_latent).reshape(B * T, self.latent_len, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B * T, self.latent_len, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x + self.mlp(self.norm(x))
        return x.view(B, T, self.latent_len, self.out_features)

    def forward_attention(self, *modality_tokens):
        kv = []
        for i in range(self.num_modalities):
            tokens = modality_tokens[i]
            if len(tokens.shape) == 3:
                num_tokens = 1
                B, T, C = tokens.shape
            else:
                B, T, num_tokens, C = tokens.shape
            tokens = tokens.view(B * T, num_tokens, C)
            kv.append(self.kv[i](tokens).reshape(B * T, num_tokens, 2, self.num_heads, self.head_dim))

        kv = torch.cat(kv, dim=1).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)


        q_latent = self.latent.expand(B * T, -1, -1)
        q = self.q(q_latent).reshape(B * T, self.latent_len, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        return attn


    def combine_forward(self, *modality_tokens):
        return self.forward(*modality_tokens)

class NoPoolConcat(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x
    def combine_forward(self, visual_tokens : torch.Tensor, proprio_tokens : torch.Tensor) -> torch.Tensor:
        '''
        visual_tokens are of dimension: B, T, num_tokens, C
        proprio_tokens are of dimension: B, T, C OR B, T, num_prop_tokens, C
        output is B x T x latent_len x C
        '''
        B, T, num_tokens, C = visual_tokens.shape
        if len(proprio_tokens.shape) == 3:
            proprio_tokens = proprio_tokens.unsqueeze(2)
        tokens = torch.cat([visual_tokens, proprio_tokens], dim=2)
        return tokens
    def forward_visual(self, visual_tokens : torch.Tensor) -> torch.Tensor:
        '''
        visual_tokens are of dimension: B, T, num_tokens, C
        output is B x T x num_tokens x C
        '''
        return visual_tokens

class VisionPatchEncoder(nn.Module):
    def __init__(
        self,
        name,
        img_size=128,
        patch_size=32,
        in_chans=3,
        embed_dim=768,
        norm_layer=nn.LayerNorm,
        pretrained=False, 
        global_pool='', 
        finetune="all",
        lora_rank=8,
        use_rope=True,  # New parameter to control RoPE usage
    ):
        """
        Simple patch-based vision encoder using timm's PatchEmbed
        
        Args:
            img_size (int): Input image size
            patch_size (int): Size of each patch
            in_chans (int): Number of input channels
            embed_dim (int): Embedding dimension for each patch
            norm_layer (nn.Module): Normalization layer
        """
        super().__init__()
        # assert not pretrained, "pretrained is not supported for VisionPatchEncoder"

        self.finetune = finetune
        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.embed_dim = embed_dim
        self._apply_post_norm = norm_layer is not None
        self.latent_len = self.num_patches
        
        # patch_embed here is just a tokenizer. we will use it to tokenize the image.
        # if patch_embed is pretrained, we will use a model for getting the image patch features.
        if pretrained:
            kwargs = {"num_classes": 0} # we don't need the classification head
            self.patch_embed = timm.create_model(name, pretrained=pretrained, global_pool=global_pool, **kwargs)
            if finetune == "all":
                self.unfreeze()
            elif finetune == False or finetune.lower() == "none":
                self.freeze()
                self.eval()
            else:
                raise ValueError(f"Invalid finetune value: {finetune}")
        else:
            assert finetune == "all", "only all is supported for VisionPatchEncoder."
            # Use timm's PatchEmbed
            self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
                norm_layer=norm_layer,
            )
        
        # Add Rotary Position Embedding
        self.use_rope = use_rope
        if use_rope:
            # Refer to the test case in dump_scripts/rope_timm_test.py
            self.rope = RotaryEmbedding(dim=embed_dim, in_pixels=False, feat_shape=self.grid_size)
        if finetune:
            print("finetuning patch embed")
            self.patch_embed.requires_grad = True
        else:
            raise NotImplementedError("not finetuning is not supported for VisionPatchEncoder")
        
        # Initialize weights
        self._init_weights(self.patch_embed.proj)
        if self.patch_embed.norm is not None:
            self._init_weights(self.patch_embed.norm)

    def unfreeze(self):
        self.patch_embed.requires_grad = True

    def freeze(self):
        self.patch_embed.requires_grad = False
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        else:
            raise ValueError(f"Unsupported module type: {type(m)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, N, C, H, W) 
                B: batch size
                T: number of frames
                N: number of cameras
                C: number of channels
                H: height
                W: width
        Returns:
            Tensor of shape (B, num_patches, embed_dim)
        """
        B, T, N, C, H, W = x.shape
        x = rearrange(x, "b t n c h w -> (b t n) c h w")
        # Get patch embeddings
        x = self.patch_embed(x)
        if self.use_rope:
            x = self.rope(x)     
        x = rearrange(x, "(b t n) p d -> b t n p d", b=B, t=T, n=N)
        return x
    
    def out_dim(self):
        return self.embed_dim

if __name__ == "__main__":
    print("testing vision encoder")
    # model = VisionEncoder('vit_small_patch16_224.dino', pretrained=True)
    # print("dim: ", model.out_dim())
    # x = torch.randn(2, 4, 5, 3, 224, 224)
    # y = model(x) # torch.Size([2, 4, 5, 197, 384])
    # print(y.shape)

    print("testing mlp")
    model = Mlp(in_features=7, out_features=768)
    x = torch.randn(2, 7)
    y = model(x)
    print(y.shape)

    print("testing attention pool")
    model = AttentionPool(768, out_features=1024, latent_len=4, pool_type='')
    x = torch.randn(2, 7, 768)
    y = model(x)
    print(y.shape)

    # robomimic
    print("testing robomimic")
    model = AttentionPool(768, out_features=1024, latent_len=1, pool_type='')
    x = torch.randn(2, 4, 768) # B, K, D
    y = model(x) # B, 1, C
    print(y.shape)

    x = torch.randn(2, 4, 768) # B, T, D
    y = model.expand_forward(x) # B, T, 1, 1024
    print(y.shape)

    print("testing combine forward")
    model = AttentionPool(768, latent_len=1, pool_type='')
    x = torch.randn(2, 4, 5, 768)
    y = torch.randn(2, 4, 768)
    z = model.combine_forward(x, y)
    print(z.shape)

    print("\ntesting VisionPatchEncoder")
    # Test with different batch sizes and image sizes
    test_cases = [
        (1, 224, 224),  # single image
        (2, 224, 224),  # batch of 2
        (4, 224, 224),  # batch of 4
    ]
    
    model = VisionPatchEncoder(
        name="vit_small_patch16_224.dino",
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=nn.LayerNorm,
        finetune="all"
    )
    
    for batch_size, height, width in test_cases:
        x = torch.randn(batch_size, 256, 3, 3, height, width)
        y = model(x)
        
        # Calculate expected number of patches
        num_patches = (height // model.patch_size) * (width // model.patch_size)
        expected_shape = (batch_size, num_patches, model.embed_dim)
        
        print(f"\nInput shape: {x.shape}")
        print(f"Output shape: {y.shape}")
        print(f"Expected shape: {expected_shape}")
        print(f"Test passed: {y.shape == expected_shape}")
        
        # Test that output values are reasonable
        print(f"Output stats - min: {y.min().item():.3f}, max: {y.max().item():.3f}, mean: {y.mean().item():.3f}")
