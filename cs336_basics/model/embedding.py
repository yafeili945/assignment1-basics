import torch
from einops import rearrange, einsum

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        std = 1
        self.weight = torch.nn.Parameter(
            torch.empty(
                num_embeddings,
                embedding_dim,
                device=device,
                dtype=dtype,
            )
        )
        torch.nn.init.trunc_normal_(
            self.weight,
            mean = 0.0,
            std = std,
            a = -3 * std,
            b = 3 * std,
        )
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]