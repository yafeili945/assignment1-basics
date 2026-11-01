import torch
from einops import rearrange, einsum
from .linear import Linear

class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = max(64, 64 * round((8 * d_model / 3) / 64))

        self.weight1 = Linear(self.d_model, self.d_ff)
        self.weight3 = Linear(self.d_model, self.d_ff)
        self.weight2 = Linear(self.d_ff, self.d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x -> (batch_size, sequence_length, d_model)
        a = self.SiLU(self.weight1.forward(x))
        b = self.weight3.forward(x)
        return self.weight2.forward(a * b)

    def SiLU(self, x: torch.Tensor):
        return x * torch.sigmoid(x)