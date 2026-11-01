import torch
from .linear import Linear

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        assert d_k % 2 == 0  # 最后一维必须能两两配对

        # 位置编号：0 到 max_seq_len - 1
        positions = torch.arange(
            max_seq_len, device=device, dtype=torch.float32
        )  # (max_seq_len,)

        # 配对编号：Python 下标从 0 开始
        pair_indices = torch.arange(
            d_k // 2, device=device, dtype=torch.float32
        )  # (d_k // 2,)

        # 每一对的旋转速度：Theta^(-2m/d_k)
        frequencies = theta ** (-2 * pair_indices / d_k)

        # 每个位置、每一对的角度
        angles = positions[:, None] * frequencies[None, :]
        # angles.shape == (max_seq_len, d_k // 2)

        self.register_buffer("cos_cache", angles.cos(), persistent=False)
        self.register_buffer("sin_cache", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x -> (... sequence_length, d_model)
        # token_positions -> (... sequence_length, d_model)
        cos = self.cos_cache[token_positions].to(dtype=x.dtype)
        sin = self.sin_cache[token_positions].to(dtype=x.dtype)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)