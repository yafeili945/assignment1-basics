import torch
from .multihead_self_attention import MultiHeadSelfAttention
from .positionwise_feedforward import SwiGLU
from .rmsnorm import RMSNorm

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float | None = None, max_seq_len: int | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        
        self.rmsnorm1 = RMSNorm(d_model)
        self.rmsnorm2 = RMSNorm(d_model)

        self.multiHeadAttention = MultiHeadSelfAttention(d_model, num_heads, theta=theta, max_seq_len=max_seq_len)
        self.swiGLU = SwiGLU(d_model, d_ff)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x -> (batch_size ... sequence_length, d_model)
        normalized_x = self.rmsnorm1.forward(x)
        y = x + self.multiHeadAttention.forward(normalized_x)
        return y + self.swiGLU.forward(self.rmsnorm2.forward(y))