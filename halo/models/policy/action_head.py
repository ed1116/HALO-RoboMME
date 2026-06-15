import abc
import queue
from typing import Dict, Union, Optional, Literal
import torch
import numpy as np
import torch.nn as nn
import torch.distributions as D
from timm.layers import Mlp
from torch.nn import functional as F
from diffusers import DDIMScheduler, DDPMScheduler
from halo.models.policy.utils import ConditionedDiffusion
from halo.models.policy.llama import ModelArgs, Transformer

class PredHead(abc.ABC, nn.Module):
    """
    Abstract class for prediction head
    """
    @abc.abstractmethod
    def forward(self, x : torch.Tensor) -> Union[torch.Tensor, D.Distribution]:
        """
        Forward pass of the prediction head
        Args:
            x: torch.Tensor, (B, input_dim)
        Returns:
            Union[torch.Tensor, D.Distribution], (B, output_dim) or D.Distribution
        """
        raise NotImplementedError

    @abc.abstractmethod
    def pred(self, x : torch.Tensor) -> torch.Tensor:
        """
        Predict the output of the prediction head
        Args:
            x: torch.Tensor, (B, input_dim)
        Returns:
            torch.Tensor, (B, output_dim)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def loss(self, x : torch.Tensor, y : torch.Tensor) -> torch.Tensor:
        """
        Compute the loss of the prediction head
        Args:
            x: torch.Tensor, (B, input_dim)
            y: torch.Tensor, (B, output_dim)
        Returns:
            torch.Tensor, scalar
        """
        raise NotImplementedError

    def get_cached_action(self, pop: bool = False):
        # retrieve the first action in the queue but do not pop it
        action = self.action_queue.queue[0]
        if pop:
            tmp = self.action_queue.get()
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)
        return action.unsqueeze(1)

class MLPHead(PredHead):
    """
    a 2 layer mlp
    """
    def __init__(
        self,
        input_dim : int,
        hidden_features : int,
        output_dim : int,
        loss_fn : nn.Module,
        use_action_chunking: bool = True,
        action_chunk_len: int = 16,
    ):
        super(MLPHead, self).__init__()
        self.input_dim = input_dim
        self.hidden_features = hidden_features
        self.output_dim = output_dim
        self.use_action_chunking = use_action_chunking
        self.action_chunk_len = action_chunk_len
        # create a queue for the action chunking prediction
        self.action_queue = queue.Queue(maxsize=self.action_chunk_len)
        self.mlp = Mlp(
            in_features=input_dim,
            hidden_features=hidden_features,
            out_features=output_dim
        )
        self.loss_fn = loss_fn

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP
        Args:
            x: torch.Tensor, (B, input_dim)
        Returns:
            torch.Tensor, (B, output_dim)
        """
        return self.mlp(x)

    def pred(self, x : torch.Tensor, num_pred_steps: int, action_dim: int, skip_n_pred_actions: int = 0) -> torch.Tensor:
        """
        Predict the output of the MLP
        Args:
            x: torch.Tensor, (B, input_dim)
            num_pred_steps: int, number of future prediction steps
            action_dim: int, action dimension
            skip_n_pred_actions: int, number of actions to skip in the prediction consisting of past actions (for ptp)
        Returns:
            torch.Tensor, (B, output_dim)
        """
        bs = x.shape[0]
        if (self.use_action_chunking) and (not self.action_queue.empty()):
            action = self.action_queue.get()
            return action.unsqueeze(1).to(x.device)
        action = self.forward(x).view(bs, num_pred_steps + skip_n_pred_actions, action_dim)
        if self.use_action_chunking:
            for ind in range(skip_n_pred_actions, self.action_chunk_len + skip_n_pred_actions, 1):
                a = action[:,ind] # (B, action_dim)
                self.action_queue.put(a)
            action = action[:, :1, :]
            # pop the last action
            self.action_queue.get()
        else:
            action = action[:, :1, :] # only returning first action to execute.
        return action.to(x.device)

    def reset(self):
        self.action_queue = queue.Queue(maxsize=self.action_chunk_len)

    def loss(self, latent : torch.Tensor, gt_action : torch.Tensor, return_pred = False) -> torch.Tensor:
        """
        Compute the loss of the MLP
        Args:
            latent: torch.Tensor, (B, input_dim)
            gt_action: torch.Tensor, (B, output_dim)
        Returns:
            torch.Tensor, (B, output_dim)
        """
        y_pred = self.forward(latent)
        if return_pred:
            return self.loss_fn(y_pred, gt_action) , y_pred
        else:
            return self.loss_fn(y_pred, gt_action)

class GMMHead(PredHead):
    """
    A 2 layer MLP that outputs the mean and log_std of a GMM distribution
    Reference: https://github.com/ARISE-Initiative/robomimic/blob/5dee58f9cc1235010d0877142b54d0e82dd23986/robomimic/models/policy_nets.py#L397
    Args:
        input_dim: int, input dimension
        output_dim: int, output dimension
        num_components: int, number of components in the GMM
        activation: nn.Module, activation function
    """
    def __init__(
        self,
        input_dim : int,
        hidden_features : int,
        output_dim : int,
        num_components : int = 5,
        min_std : float = 0.01,
        activation : nn.Module = nn.ReLU,
        num_layers=2,
        fixed_scale: bool = True,
        std_activation: str = "softplus",
        use_tanh: bool = True,
    ):
        super(GMMHead, self).__init__()
        self.input_dim = input_dim
        self.hidden_features = hidden_features
        self.output_dim = output_dim
        self.num_components = num_components

        if num_layers > 0:
            sizes = [input_dim] + [hidden_features] * num_layers
            layers = []
            for i in range(num_layers):
                layers += [nn.Linear(sizes[i], sizes[i + 1]), activation()]
            layers += [nn.Linear(sizes[-2], sizes[-1])]
            self.share = nn.Sequential(*layers)
        else:
            self.share = nn.Identity()

        self.min_std = min_std

        # generate three heads
        self.mean_fc = nn.Linear(hidden_features, output_dim * num_components)
        self.logits_fc = nn.Linear(hidden_features, num_components)

        self.fixed_scale = fixed_scale
        if fixed_scale:
            self.scale = nn.Parameter(torch.zeros(1, num_components, output_dim), requires_grad=True)
        else:
            self.scale_fc = nn.Linear(hidden_features, output_dim * num_components)

        if std_activation == "softplus":
            self.std_actv = F.softplus
        else:
            self.std_actv = torch.exp

        self.use_tanh = use_tanh

    def _forward_model(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate the mean, scale and logits of the GMM distribution
        Args:
            x: torch.Tensor, (B, input_dim)
        Returns:
            Dict[str, torch.Tensor], dictionary containing mean, scale and logits
                mean: torch.Tensor, (B, num_components, output_dim)
                scale: torch.Tensor, (B, num_components, output_dim)
                logits: torch.Tensor, (B, num_components)
        """
        x = self.share(x)
        mean = self.mean_fc(x).reshape(-1, self.num_components, self.output_dim)
        if self.fixed_scale:
            scale = self.scale.repeat(x.size(0), 1, 1)
        else:
            scale = self.scale_fc(x).reshape(-1, self.num_components, self.output_dim)
        logits = self.logits_fc(x)
        return {"mean": mean, "scale": scale, "logits": logits}

    def forward(self, x : torch.Tensor, low_eval_noise=False) -> D.Distribution:
        """
        Generate the GMM distribution
        Args:
            x: torch.Tensor, (B, input_dim)
        Returns:
            D.Distribution, GMM distribution
        """
        gmm_stats = self._forward_model(x)
        mean, scale, logits = gmm_stats["mean"], gmm_stats["scale"], gmm_stats["logits"]

        if self.use_tanh:
            mean = torch.tanh(mean)

        # generate gmm distribution
        # libero has been using low-eval noise for the scale
        if (self.training) or (not low_eval_noise):
            scales = self.std_actv(scale) + self.min_std
        else:
            scales = torch.ones_like(mean) * 1e-4

        component_distribution = D.Normal(loc=mean, scale=scales)
        component_distribution = D.Independent(component_distribution, 1)

        # unnormalized logits to categorical distribution for mixing the modes
        mixture_distribution = D.Categorical(logits=logits)

        dist = D.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution,
        )

        return dist

    def pred(self, x : torch.Tensor) -> torch.Tensor:
        """
        Sample from the GMM distribution
        Args:
            x: torch.Tensor, (B, input_dim)
        Returns:
            torch.Tensor, (B, output_dim)
        """
        dist = self.forward(x, low_eval_noise=True)
        # action = dist.sample()
        action = dist.mean
        return action


    def loss(self, latent : torch.Tensor, gt_action : torch.Tensor) -> torch.Tensor:
        """
        Compute the loss of the GMM distribution
        Args:
            latent: torch.Tensor, (B, input_dim)
            gt_action: torch.Tensor, (B, output_dim)
        Returns:
            torch.Tensor, (B, output_dim)
        """
        dist = self.forward(latent)
        # this log_prob_x = self.component_distribution.log_prob(x) returns very large values
        log_prob = dist.log_prob(gt_action)
        loss = -log_prob
        return loss

class DiffusionHead(PredHead):
    """Using a diffusion model as the prediction head

    Args:
        input_dim: int, input dimension
        hidden_features: int, hidden dimension
        output_dim: int, output dimension
        noise_scheduler: DDIMScheduler, noise scheduler
        inference_steps: int, number of inference steps
    """
    def __init__(
        self,
        input_dim : int,
        hidden_features : int,
        output_dim : int,
        train_noise_scheduler : DDPMScheduler,
        inference_noise_scheduler : DDIMScheduler,
        inference_steps=None,
    ):
        super(DiffusionHead, self).__init__()
        self.model = ConditionedDiffusion(
            output_dim=output_dim,
            hidden_dim=hidden_features,
            cond_dim=input_dim,
            train_noise_scheduler=train_noise_scheduler,
            inference_noise_scheduler=inference_noise_scheduler,
            inference_steps=inference_steps
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the diffusion model
        Args:
            x: torch.Tensor, (B, input_dim)
        Returns:
            torch.Tensor, (B, output_dim)
        """
        return self.model.conditional_sample(x)

    def pred(self, x : torch.Tensor) -> torch.Tensor:
        """
        Predict the output of the diffusion model
        Args:
            x: torch.Tensor, (B, input_dim)
        Returns:
            torch.Tensor, (B, output_dim)
        """
        return self.forward(x)

    def loss(self, latent : torch.Tensor, gt_action : torch.Tensor) -> torch.Tensor:
        """
        Compute the loss of the diffusion model
        Args:
            latent: torch.Tensor, (B, input_dim)
            gt_action: torch.Tensor, (B, output_dim)
        Returns:
            torch.Tensor, scalar
        """
        return self.model.calculate_loss(latent, gt_action)

class FlowMatchingHead(PredHead):
    """
    Conditional Flow Matching action head.

    Learns a velocity field v(x_t, t, cond) that transforms N(0,I) noise into
    actions via Euler integration over [0, 1].

    Training loss: MSE( v_pred, x_1 - x_0 ) with element-wise reduction so the
    caller can apply masking identical to MLPHead.

    __init__ signature is identical to MLPHead for drop-in compatibility.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_features: int,
        output_dim: int,
        loss_fn: nn.Module,           # kept for interface parity; not used
        use_action_chunking: bool = True,
        action_chunk_len: int = 16,
        num_inference_steps: int = 10,
        num_velocity_layers: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_features = hidden_features
        self.output_dim = output_dim
        self.use_action_chunking = use_action_chunking
        self.action_chunk_len = action_chunk_len
        self.num_inference_steps = num_inference_steps
        self.action_queue = queue.Queue(maxsize=action_chunk_len)

        # Sinusoidal-style time embedding: scalar t → hidden_features
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, hidden_features),
        )

        # Velocity network: [x_t | t_emb | cond] → velocity (same shape as x_t)
        in_dim = output_dim + hidden_features + input_dim
        layers: list[nn.Module] = []
        for i in range(num_velocity_layers):
            out_dim = hidden_features if i < num_velocity_layers - 1 else output_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_velocity_layers - 1:
                layers.append(nn.SiLU())
            in_dim = hidden_features
        self.velocity_net = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _velocity(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Predict velocity.  x_t/cond: (B, D), t: (B,) or (B,1)."""
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        t_emb = self.time_mlp(t.to(dtype=x_t.dtype))
        return self.velocity_net(torch.cat([x_t, t_emb, cond], dim=-1))

    def _euler_integrate(self, cond: torch.Tensor) -> torch.Tensor:
        """Euler ODE integration from t=0 (noise) to t=1 (action). Returns (B, output_dim)."""
        B = cond.shape[0]
        x = torch.randn(B, self.output_dim, device=cond.device, dtype=cond.dtype)
        dt = 1.0 / self.num_inference_steps
        for i in range(self.num_inference_steps):
            t = torch.full((B,), i * dt, device=cond.device, dtype=cond.dtype)
            x = x + self._velocity(x, t, cond) * dt
        return x

    # ------------------------------------------------------------------
    # PredHead interface
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full inference: noise → action via Euler integration.  x: (B, input_dim)."""
        return self._euler_integrate(x)

    def pred(self, x: torch.Tensor, num_pred_steps: int, action_dim: int, skip_n_pred_actions: int = 0) -> torch.Tensor:
        """Same chunking logic as MLPHead.pred."""
        bs = x.shape[0]
        if self.use_action_chunking and not self.action_queue.empty():
            action = self.action_queue.get()
            return action.unsqueeze(1).to(x.device)
        action = self.forward(x).view(bs, num_pred_steps + skip_n_pred_actions, action_dim)
        if self.use_action_chunking:
            for ind in range(skip_n_pred_actions, self.action_chunk_len + skip_n_pred_actions):
                self.action_queue.put(action[:, ind])
            action = action[:, :1, :]
            self.action_queue.get()
        else:
            action = action[:, :1, :]
        return action.to(x.device)

    def reset(self):
        self.action_queue = queue.Queue(maxsize=self.action_chunk_len)

    def loss(self, latent: torch.Tensor, gt_action: torch.Tensor, return_pred: bool = False) -> torch.Tensor:
        """
        Flow matching loss (element-wise MSE, no reduction) so the caller can
        apply action_prompt_mask in the same way as with MLPHead.

        latent:    (N, input_dim)
        gt_action: (N, output_dim)   — flattened [num_pred_steps * action_dim]
        returns:   loss (N, output_dim) [, pred (N, output_dim)]
        """
        x_1 = gt_action
        x_0 = torch.randn_like(x_1)
        t = torch.rand(x_1.shape[0], device=latent.device, dtype=latent.dtype)
        t_exp = t.unsqueeze(-1)
        x_t = (1.0 - t_exp) * x_0 + t_exp * x_1      # linear interpolation
        v_target = x_1 - x_0                           # constant velocity field

        v_pred = self._velocity(x_t.detach(), t, latent)
        loss = F.mse_loss(v_pred, v_target, reduction="none")  # (N, output_dim)

        if return_pred:
            # Cheap 1-step estimate for logging (same shape as MLPHead's y_pred)
            with torch.no_grad():
                x0_sample = torch.randn_like(x_1)
                t_zero = torch.zeros(x_1.shape[0], device=latent.device, dtype=latent.dtype)
                pred = x0_sample + self._velocity(x0_sample, t_zero, latent)
            return loss, pred
        return loss


class CEHead(PredHead):
    '''
    A cross entropy head that predicts the action using a cross entropy loss
    Args:
        Uses a simplem trasnformer to predict the action with the action head
    '''
    def __init__(
        self,
        input_dim : int,
        vocab_size : int,
        max_seq_len : int,
        loss_fn : nn.Module, # not relevant
        n_layers : int = 1, # number of transformer layers
        use_action_chunking: bool = True, # not relevant
        action_chunk_len: int = 16, # not relevant
        ignore_index: int = -100,
        hidden_features: int = -1,
    ):
        super(CEHead, self).__init__()
        self.input_dim = input_dim
        self.vocab_size = vocab_size
        self.use_action_chunking = use_action_chunking
        model_args = ModelArgs(
            dim=input_dim if hidden_features <= 0 else hidden_features,
            n_layers=n_layers,
            n_heads=8,
            vocab_size=-1,
            lang_offset_in_max_seq=1,
            is_causal=True,
        )
        self.input_proj = None
        if hidden_features > 0:
            self.input_proj = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_features),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
        self.transformer = Transformer(model_args)
        self.lm_head = nn.Linear(
            in_features=input_dim if hidden_features <= 0 else hidden_features,
            out_features=vocab_size
        )
        self.loss_fn = loss_fn
        self.ignore_index = ignore_index

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the transformer
        Args:
            x: torch.Tensor, (B, T, input_dim)
        Returns:
            torch.Tensor, (B, T, output_dim)
        """
        if self.input_proj is not None:
            x = self.input_proj(x)
        out = self.transformer(x)
        if isinstance(out, tuple):
            out = out[0]
        return self.lm_head(out)

    def pred(self, x : torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def loss(self, latent : torch.Tensor, gt_latent_embeds: torch.Tensor, gt_token_indices : torch.Tensor, return_pred = False) -> torch.Tensor:
        '''
        Compute the loss of the transformer
        Args:
            latent: torch.Tensor, (B, input_dim)
            gt_latent_embeds: torch.Tensor, (B, T, input_dim) / (B, input_dim)
            ground token indices: torch.Tensor, (B, T) / (B,)
        Returns:
            torch.Tensor, scalar
        '''
        if latent.ndim == 2:
            latent = latent.unsqueeze(1)
        if gt_latent_embeds.ndim == 2:
            gt_latent_embeds = gt_latent_embeds.unsqueeze(1)
        if gt_token_indices.ndim == 1:
            gt_token_indices = gt_token_indices.unsqueeze(1)
        B, T, D = gt_latent_embeds.shape
        latent = torch.cat([latent, gt_latent_embeds], dim=1)
        logits = self.forward(latent)
        logits = logits[:, :-1]
        logits = logits.reshape(-1, logits.size(-1))
        gt_token_indices = gt_token_indices.reshape(-1)
        loss = self.loss_fn(logits, gt_token_indices)
        return loss if not return_pred else (loss, logits)
