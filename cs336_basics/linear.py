import torch
from einops import rearrange, einsum

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        std = (2 / (in_features + out_features)) ** 0.5
        self.weight = torch.nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
            )
        )
        torch.nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a = -3 * std,
            b = 3 * std,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight,"... in_features, out_features in_features -> ... out_features")