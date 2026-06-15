import torch
import torch.nn as nn
from collections import OrderedDict
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomResizedCrop

from halo.data.utils import IMAGENET_MEAN, IMAGENET_STD


class BatchResize(nn.Module):
    """Batch-wise resize transform."""
    def __init__(self, size: int, antialias: bool = True):
        super().__init__()
        self.size = size
        self.antialias = antialias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) or (B*T, C, H, W)
        return TF.resize(x, (self.size, self.size), antialias=self.antialias)


class BatchCenterCrop(nn.Module):
    """Batch-wise center crop transform."""
    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) or (B*T, C, H, W)
        return TF.center_crop(x, (self.size, self.size))


class BatchRandomResizedCrop(nn.Module):
    """
    Batch-wise RandomResizedCrop that applies same crop parameters across frames.
    Uses torchvision's RandomResizedCrop.get_params() for standard parameter computation.
    Input: (B*T, C, H, W) where B is batch size and T is number of frames
    Output: (B*T, C, size, size)
    For each batch sample, same crop parameters are applied to all T frames.
    """
    def __init__(self, size: int, scale=(0.65, 1.0), ratio=(1.0, 1.0), antialias: bool = True):
        super().__init__()
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.antialias = antialias

    def forward(self, x: torch.Tensor, num_frames: int = 1) -> torch.Tensor:
        """
        x: (B*T, C, H, W) where T is num_frames
        Returns: (B*T, C, size, size)
        """
        BT, C, H, W = x.shape
        B = BT // num_frames

        # Get crop parameters using torchvision's standard get_params for each batch sample
        # We use the first frame of each batch sample to get parameters, then apply to all frames
        params_list = []
        for b in range(B):
            # Use first frame to get parameters (same params will be applied to all T frames)
            first_frame = x[b * num_frames]  # (C, H, W)
            # Call torchvision's get_params method
            assert first_frame.ndim == 3 and first_frame.shape[0] == 3, f"Expected (C, H, W), got {first_frame.shape}"
            i, j, h, w = RandomResizedCrop.get_params(first_frame, scale=self.scale, ratio=self.ratio)
            params_list.append((i, j, h, w))

        # Apply crop and resize per batch sample (batched across T frames)
        result_list = []
        for b, (i, j, h, w) in enumerate(params_list):
            # Get all T frames for this batch sample: (T, C, H, W)
            frames = x[b * num_frames:(b + 1) * num_frames]
            # Crop all T frames with same parameters (batch operation)
            cropped = TF.crop(frames, i, j, h, w) # first crop
            # Batch resize all T frames at once (second resize)
            resized = TF.resize(cropped, (self.size, self.size), antialias=self.antialias)
            result_list.append(resized)

        # Concatenate: (B*T, C, size, size)
        return torch.cat(result_list, dim=0)


class MultiViewTorchVideoTransform(nn.Module):
    """
    Torchvision transform for observations shaped (B, T, N, C, H, W).
    Optimized with batch-wise transforms, Sequential, and JIT scriptable.
    `gripper_flags`: list[bool] length N indicating which views are 'gripper'.
    - Non-gripper: RandomResizedCrop(224, scale=(0.65,1.0), ratio=(1.0,1.0))
    - Gripper: Resize(248)->CenterCrop(224)
    - Shared per-sample brightness/contrast across ALL frames & ALL views:
        x' = clamp(x * c + s, 0, 1), c~U[0.8,1.2], s~U[-0.1,0.1]
    - ImageNet normalize at the end.
    """
    def __init__(
        self,
        gripper_flags,
        final_image_size: int = 224,
        resize: int = 248,
        static_scale=(0.65, 1.0),
        share_bc_across_frames_and_views: bool = True,
        mean: torch.Tensor = IMAGENET_MEAN,
        std: torch.Tensor = IMAGENET_STD,
    ):
        super().__init__()
        assert isinstance(gripper_flags, (list, tuple))
        self.gripper_flags = list(gripper_flags)
        self.N = len(gripper_flags)
        self.final = final_image_size
        self.resize = resize
        self.static_scale = static_scale
        self.share_bc = share_bc_across_frames_and_views

        # Eval mode transforms (deterministic, can use Sequential)
        self.gripper_geom_eval = nn.Sequential(
            BatchResize(resize, antialias=True),
            BatchCenterCrop(final_image_size),
        )
        self.static_geom_eval = nn.Sequential(
            BatchResize(resize, antialias=True),
            BatchCenterCrop(final_image_size),
        )

        # Train mode transform for static views
        self.static_geom_train = BatchRandomResizedCrop(
            size=final_image_size,
            scale=static_scale,
            ratio=(1.0, 1.0),  # fixed ratio as per original spec
            antialias=True,
        )

        # register mean/std buffers broadcastable to (B,T,N,C,H,W)
        mean = torch.tensor(mean)
        std  = torch.tensor(std)
        self.register_buffer("mean", mean.view(1, 1, 1, 3, 1, 1))
        self.register_buffer("std",  std.view(1, 1, 1, 3, 1, 1))
        self.register_buffer("_training", torch.tensor(True, dtype=torch.bool))

    def train(self, mode: bool = True):
        self._training = torch.tensor(mode, dtype=torch.bool)
        return super().train(mode)

    def eval(self):
        self._training = torch.tensor(False, dtype=torch.bool)
        return super().eval()

    @staticmethod
    def _to_float01(x: torch.Tensor) -> torch.Tensor:
        # Accept uint8 or float; return float in [0,1] (clamped).
        if x.dtype == torch.uint8:
            return x.float().div_(255.0)
        if x.dtype.is_floating_point:
            if x.max() > 10.0:  # likely 0..255
                x = x / 255.0
            return x.clamp_(0.0, 1.0)
        return x.float().clamp_(0.0, 1.0)

    def _shared_bc(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (B,T,N,C,H,W). Apply SAME (per-sample) brightness/contrast
        across all frames & views if share_bc=True, else per-frame.
        """
        B, T, N, C, H, W = obs.shape
        if self.share_bc:
            shape = (B, 1, 1, 1, 1, 1)  # one (c,s) per sample across all T,N
        else:
            shape = (B, T, 1, 1, 1, 1)  # per frame, shared across views

        c = torch.empty(shape, device=obs.device, dtype=obs.dtype).uniform_(0.8, 1.2)
        s = torch.empty(shape, device=obs.device, dtype=obs.dtype).uniform_(-0.1, 0.1)
        return (obs * c + s).clamp_(0.0, 1.0)

    def _apply_static_geom(self, xt: torch.Tensor) -> torch.Tensor:
        """
        Apply RandomResizedCrop consistently across frames (train) or resize+center_crop (eval).
        xt: (B, T, C, H, W)
        """
        B, T, C, H, W = xt.shape
        xt_flat = xt.view(B * T, C, H, W)  # (B*T, C, H, W)

        if not self._training:
            # Eval mode: batch-wise resize + center crop
            result = self.static_geom_eval(xt_flat)  # (B*T, C, final, final)
        else:
            # Train mode: use BatchRandomResizedCrop module
            result = self.static_geom_train(xt_flat, num_frames=T)  # (B*T, C, final, final)

        return result.view(B, T, C, self.final, self.final)

    def _apply_gripper_geom(self, xt: torch.Tensor) -> torch.Tensor:
        """
        Apply Resize -> CenterCrop consistently across frames.
        xt: (B, T, C, H, W)
        """
        B, T, C, H, W = xt.shape
        # Reshape to (B*T, C, H, W) for batch processing
        xt_flat = xt.view(B * T, C, H, W)
        # Apply batch-wise transforms
        result = self.gripper_geom_eval(xt_flat)  # (B*T, C, final, final)
        return result.view(B, T, C, self.final, self.final)

    def forward(self, obs_btnchw: torch.Tensor) -> torch.Tensor:
        """
        Input:  (B,T,N,C,H,W) or (T,N,C,H,W), float in [0,1] or uint8 
        Output: (B,T,N,C,H',W') or (T,N,C,H',W'), normalized
        """
        is_batch = obs_btnchw.ndim == 6
        if obs_btnchw.ndim == 5:
            is_batch = False
            obs_btnchw = obs_btnchw.unsqueeze(0)
        assert obs_btnchw.ndim == 6, "Expect (B,T,N,C,H,W)"
        assert obs_btnchw.shape[-3] == 3, f"Expected C=3, got {obs_btnchw.shape[-3]=} with shape {obs_btnchw.shape=}"
        B, T, N, C, H, W = obs_btnchw.shape
        assert N == self.N, f"Expected N={self.N}, got {N}"

        # to float in [0,1]
        x = self._to_float01(obs_btnchw)

        # shared brightness/contrast across frames & views (per sample)
        x = self._shared_bc(x)

        # per-view geometry: process each n as (B,T,C,H,W)
        out_views = []
        for n in range(N):
            xt = x[:, :, n, :, :, :]  # (B,T,C,H,W)
            if self.gripper_flags[n]:
                xt = self._apply_gripper_geom(xt)
            else:
                xt = self._apply_static_geom(xt)
            out_views.append(xt)
        x = torch.stack(out_views, dim=2)  # (B,T,N,C,final,final)

        # normalize
        x = (x - self.mean) / self.std
        return x if is_batch else x.squeeze(0)

class MultiViewTorchDictTransform(MultiViewTorchVideoTransform):
    def forward(self, obs_dict: dict) -> dict:
        n_imgs = len(obs_dict)
        keys = list(obs_dict.keys())
        # stack the images along the dim = 2 and call super().forward
        dim = -4
        obs = torch.stack([obs_dict[k] for k in obs_dict], dim=dim)
        obs = super().forward(obs)
        # convert it to dict again
        obs_dict = OrderedDict({k: obs[..., i, :, :, :] for i, k in enumerate(keys)})
        return obs_dict
