import torch
import torch.nn as nn
from collections import OrderedDict
import kornia.augmentation as KA
import kornia.geometry.transform as KG

from halo.data.utils import IMAGENET_MEAN, IMAGENET_STD

class MultiViewVideoTransform(nn.Module):
    """
    Kornia transform for observations shaped (B, T, N, C, H, W).
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
        self.share_bc = share_bc_across_frames_and_views

        # geometry transforms (video-aware)
        self.static_geom = KA.VideoSequential(
            KA.RandomResizedCrop(
                (final_image_size, final_image_size),
                scale=static_scale, 
                ratio=(1.0, 1.0),
                p=1.0
            ),
            data_format="BTCHW",
            same_on_frame=True,
        )
        self.gripper_geom = KA.VideoSequential(
            KA.Resize(  # Changed from KG.Resize to KA.Resize for better compatibility within VideoSequential
                (resize, resize),
                antialias=True,           # Enable antialiasing for better quality
                align_corners=None,
                p=1.0                     # Ensure this transformation is always applied
            ),
            KA.CenterCrop((final_image_size, final_image_size)),
            data_format="BTCHW",
            same_on_frame=True,
        )

        # register mean/std buffers broadcastable to (B,T,N,C,H,W)
        mean = torch.tensor(mean)
        std  = torch.tensor(std)
        self.register_buffer("mean", mean.view(1, 1, 1, 3, 1, 1))
        self.register_buffer("std",  std.view(1, 1, 1, 3, 1, 1))
        self.mode = 'train'

    def train(self, mode: bool = True):
        self.mode = 'train' if mode else 'eval'
        if self.mode == 'train':
            self.static_geom.train()
            self.gripper_geom.train()
        else:
            self.static_geom.eval()
            self.gripper_geom.eval()
        return self
    
    def eval(self, mode: bool = True):
        return self.train(not mode)

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

        c = torch.empty(shape, device=obs.device).uniform_(0.8, 1.2)
        s = torch.empty(shape, device=obs.device).uniform_(-0.1, 0.1)
        return (obs * c + s).clamp_(0.0, 1.0)

    def forward(self, obs_btnchw: torch.Tensor) -> torch.Tensor:
        """
        Input:  (B,T,N,C,H,W), float in [0,1] or uint8
        Output: (B,T,N,C,H',W'), normalized
        """
        assert obs_btnchw.ndim == 6, "Expect (B,T,N,C,H,W)"
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
                xt = self.gripper_geom(xt)
            else:
                xt = self.static_geom(xt)
            out_views.append(xt)
        x = torch.stack(out_views, dim=2)  # (B,T,N,C,final,final)

        # normalize
        x = (x - self.mean) / self.std
        return x

class MultiViewDictTransform(MultiViewVideoTransform):
    def forward(self, obs_dict: dict) -> dict:
        n_imgs = len(obs_dict)
        keys = list(obs_dict.keys())
        # stack the images along the dim = 2 and call super().forward
        obs = torch.stack([obs_dict[k] for k in obs_dict], dim=2)
        obs = super().forward(obs)
        # convert it to dict again
        obs_dict = OrderedDict({k: obs[:, :, i, :, :, :] for i, k in enumerate(keys)})
        return obs_dict