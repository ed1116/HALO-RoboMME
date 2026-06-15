import torch
from torch import nn
from timm.layers import Mlp

class GatingFunction(nn.Module):
    """
    Base class for gating functions.
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, gating_feature: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement forward method")

    def logging_data(self):
        return {}

class IdentityGatingFunction(GatingFunction):
    """
    Simply returns the prediction.
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, gating_feature: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        return prediction

class SigmoidBlindGatingFunction(GatingFunction):
    """
    This will be a simple scalar gate that will be applied to the softmaxed output of the attention. 
    """
    def __init__(self, gating_dim: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(1), requires_grad=True)  # Learnable scalar gate, initialized at 0
    
    def forward(self, gating_feature: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.scale) * prediction
    
    def logging_data(self):
        return {"scale": self.scale.item()}

class TanhGatingFunction(GatingFunction):
    """
    Applies a learnable tanh gate to modulate a prediction. The gating is blind to the task itself.
    Given:
        - prediction: (..., D)
    Output:
        - tanh(scale) * prediction
        - scale is a learnable scalar
    """
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(1), requires_grad=True)  # Learnable scalar gate, initialized at 0
    
    def forward(self, gating_feature: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        # gating_feature is not used in this implementation
        return torch.tanh(self.scale) * prediction
    
    def logging_data(self):
        return {"scale": self.scale.item()}

class TanhGatedFeatureModulation(GatingFunction):
    """
    Applies a learnable tanh gate to modulate a prediction using a separate gating feature.
    
    Given:
        - gating_feature: used to compute a scalar gate (..., 1)
        - prediction: feature-conditioned prediction to be modulated
    
    Output:
        - tanh(gate * scale) * prediction
    """

    def __init__(self, gating_dim: int) -> None:
        """
        Args:
            gating_dim (int): Dimensionality of the gating feature vector.
        """
        super().__init__()
        self.to_gate = nn.Linear(gating_dim, 1)  # (..., gating_dim) → (..., 1)
        self.scale = nn.Parameter(torch.zeros(1), requires_grad=True)  # Learnable scalar gate, initialized at 0
        self.apply(self._init_weights)
        self._mean_pred_scale = 0.0
        self.register_buffer("_raw_gate_mean_buf", torch.tensor(0.0), persistent=False)

    def forward(self, gating_feature: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gating_feature (torch.Tensor): Tensor of shape (..., gating_dim)
            prediction (torch.Tensor): Tensor of shape (..., D) to be modulated
        
        Returns:
            torch.Tensor: Modulated prediction, same shape as input prediction
        """
        gate = self.to_gate(gating_feature)  # (..., 1)
        modulation = torch.tanh(gate * self.scale)  # (..., 1)
        self._raw_gate_mean_buf.copy_(gate.mean().detach())
        assert modulation.ndim == prediction.ndim, f"Modulation and prediction must have the same number of dimensions, but got {modulation.ndim} and {prediction.ndim}"
        return modulation * prediction  # Broadcasting over last dim of prediction

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def logging_data(self):
        return {
            "scale": self.scale.item(), 
            "mean_pred_scale": float(self._raw_gate_mean_buf), 
            "final_scale_value": self.scale.item() * float(self._raw_gate_mean_buf)
        }

class TanhGatedBiasFeatureModulation(GatingFunction):
    """
    Applies a learnable tanh gate to modulate a prediction using a separate gating feature.
    
    Given:
        - gating_feature: used to compute a scalar gate (..., 1)
        - prediction: feature-conditioned prediction to be modulated
    
    Output:
        - tanh(gate * scale) * prediction
    """

    def __init__(self, gating_dim: int) -> None:
        """
        Args:
            gating_dim (int): Dimensionality of the gating feature vector.
        """
        super().__init__()
        self.to_gate = nn.Linear(gating_dim, 1)  # (..., gating_dim) → (..., 1)
        self.scale = nn.Parameter(torch.zeros(1), requires_grad=True)  # Learnable scalar gate, initialized at 0
        self.bias = nn.Parameter(torch.zeros(1), requires_grad=True)  # Learnable bias, initialized at 0
        self.apply(self._init_weights)
        self._mean_pred_scale = 0.0

        self.register_buffer("_raw_gate_mean_buf", torch.tensor(0.0), persistent=False)
    
    def forward(self, gating_feature: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gating_feature (torch.Tensor): Tensor of shape (..., gating_dim)
            prediction (torch.Tensor): Tensor of shape (..., D) to be modulated
        
        Returns:
            torch.Tensor: Modulated prediction, same shape as input prediction
        """
        gate = self.to_gate(gating_feature)  # (..., 1)
        self._raw_gate_mean_buf.copy_(gate.mean().detach())
        modulation = torch.tanh(gate * self.scale + self.bias)  # (..., 1)
        assert modulation.ndim == prediction.ndim, f"Modulation and prediction must have the same number of dimensions, but got {modulation.ndim} and {prediction.ndim}"
        return modulation * prediction  # Broadcasting over last dim of prediction

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def logging_data(self):
        return {
            "scale": self.scale.item(), 
            "mean_pred_scale": float(self._raw_gate_mean_buf), 
            "pred_bias": self.bias.item(), 
            "final_scale_value": self.scale.item() * float(self._raw_gate_mean_buf) + self.bias.item()
        }

class TanhGatedMLPFeatureModulation(GatingFunction):
    """
    Applies a learnable tanh gate to modulate a prediction using a separate gating feature.
    Efficiently logs raw gate and final scale means via a hook on the mlp_block.
    """
    def __init__(self, gating_dim: int) -> None:
        super().__init__()
        self.mlp_block = Mlp(
            in_features=gating_dim,
            hidden_features=int(gating_dim * 2),
            out_features=1,
            act_layer=nn.ReLU,
        )
        # storage for logging
        self._raw_gate_mean = 0.0
        self._final_scale_mean = 0.0

        self.register_buffer("_raw_gate_mean_buf", torch.tensor(0.0), persistent=False)

    def forward(self, gating_feature: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        gate = self.mlp_block(gating_feature)  # (..., 1)
        self._raw_gate_mean_buf.copy_(gate.mean().detach())

        assert gate.ndim == prediction.ndim, (
            f"Gate and prediction must have the same number of dimensions, "
            f"but got {gate.ndim} and {prediction.ndim}"
        )
        return torch.tanh(gate) * prediction  # Broadcasting over last dim of prediction

    def logging_data(self):
        return {
            "mean_pred_scale": float(self._raw_gate_mean_buf),
            "final_scale_value": float(self._raw_gate_mean_buf)
        }

class SigmoidMLPFeatureModulation(GatingFunction):
    """
    This will be an MLP that will be applied to the softmaxed output of the attention. 
    """
    def __init__(self, gating_dim: int) -> None:
        super().__init__()
        # we will have a separate MLP for each head
        self.mlp = Mlp(
            in_features=gating_dim,
            out_features=gating_dim,
        )
        self.register_buffer("_raw_gate_mean_buf", torch.tensor(0.0), persistent=False)
    def forward(self, x: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        # extract the sigmoid output for each head; make it sigmoid and then do multiplicative gating
        gate = self.mlp(x)
        gate = torch.sigmoid(gate)
        self._raw_gate_mean_buf.copy_(gate.mean().detach())
        assert gate.shape == output.shape, f"Gate and output must have the same shape, but got {gate.shape} and {output.shape}"
        return gate * output
    
    def logging_data(self):
        return {
            "mean_pred_scale": float(self._raw_gate_mean_buf),
            "final_scale_value": float(self._raw_gate_mean_buf)
        }