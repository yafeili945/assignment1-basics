import torch
from .transformer_block import TransformerBlock
from .linear import Linear
from .rmsnorm import RMSNorm
from .softmax import softmax
from .embedding import Embedding

class Transformer(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, vocab_size: int, context_length: int, num_layers: int, theta: float | None = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.blocks = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(TransformerBlock(d_model, num_heads, d_ff, theta, max_seq_len = context_length))
        self.embedding = Embedding(vocab_size, d_model)
        self.rmsnorm = RMSNorm(d_model)
        self.linear = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x -> (batch_size ... sequence_length, d_model)
        
        x = self.embedding.forward(x)
        for block in self.blocks:
            x = block.forward(x)

        norm_x = self.rmsnorm.forward(x)
        output = self.linear.forward(norm_x)
        return output