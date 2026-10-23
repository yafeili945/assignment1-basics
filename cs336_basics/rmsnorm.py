import torch
from einops import rearrange, einsum

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(
            torch.empty(
                d_model,
                device=device,
                dtype=dtype,
            )
        )
        torch.nn.init.ones_(self.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x -> (batch_size, sequence_length, d_model)

        in_dtype = x.dtype
        x = x.to(torch.float32)

        mean_val = x.square().mean(dim=-1, keepdim=True)

        rms = torch.sqrt(mean_val + self.eps)
        return ((x / rms) * self.weight).to(in_dtype)