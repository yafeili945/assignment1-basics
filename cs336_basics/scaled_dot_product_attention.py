import torch
from .softmax import softmax
import math

def scaled_dot_product_attention(queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    # keys, queries -> (batch_size, ..., seq_len, d_k)
    # values -> (batch_size, ..., seq_len, d_v)
    # (optional)mask -> (seq_len, seq_len)
    # output -> (batch_size, ..., seq_len, d_v)
    
    d_v = keys.shape[-1]
    scores = queries @ keys.transpose(-2, -1) / math.sqrt(d_v)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    return softmax(scores, -1) @ values
