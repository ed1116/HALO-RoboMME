from typing import Any
import numpy as np
import torch

def to_torch_long(input: Any) -> torch.Tensor:
    """
    Convert input to a torch.Tensor of dtype torch.long.
    """
    if isinstance(input, list):
        return torch.tensor(np.array(input), dtype=torch.long)
    elif isinstance(input, np.ndarray):
        return torch.from_numpy(input).to(torch.long)
    elif isinstance(input, torch.Tensor):
        return input.to(torch.long)
    else:
        raise ValueError(f"Unsupported type: {type(input)}")
