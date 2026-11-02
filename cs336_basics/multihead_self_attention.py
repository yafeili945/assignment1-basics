import torch
from .linear import Linear
from .rope import RotaryPositionalEmbedding
from .scaled_dot_product_attention import scaled_dot_product_attention
from einops import rearrange

class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float | None = None, max_seq_len: int | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = self.d_v = d_model / num_heads
        
        self.wq = Linear(d_model, d_model)
        self.wk = Linear(d_model, d_model)
        self.wv = Linear(d_model, d_model)
        self.wo = Linear(d_model, d_model)

        self.rope = None
        if theta is not None:
            self.rope = RotaryPositionalEmbedding(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # x -> (batch_size ... sequence_length, d_model)

        Q = self.wq.forward(x)
        K = self.wk.forward(x)
        V = self.wv.forward(x)
        q_heads = rearrange(Q, "b ... l (h d) -> b ... h l d", h=self.num_heads)
        k_heads = rearrange(K, "b ... l (h d) -> b ... h l d", h=self.num_heads)
        v_heads = rearrange(V, "b ... l (h d) -> b ... h l d", h=self.num_heads)

        if self.rope is not None:
            q_heads = self.rope.forward(q_heads, token_positions)
            k_heads = self.rope.forward(k_heads, token_positions)
        
        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))

        head_outputs = scaled_dot_product_attention(q_heads, k_heads, v_heads, mask)
        combined = rearrange(head_outputs, "b ... h l d -> b ... l (h d)", h=self.num_heads)
        return self.wo.forward(combined)