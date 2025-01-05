import torch
import torch.nn as nn

class AbsolutePositionEmbedding(nn.Embedding):
    """
    A class to implement Absolute Position Embedding for tensors.
    Attributes:
        context_length (int): Sequence length.
        dim (int): Hidden dimension size (must be even).
    """

    def __init__(self, context_length: int, dim: int):
        super().__init__(context_length, dim)

    def forward(self, x :torch.Tensor) -> torch.Tensor:
        """
        generate pisition encoding base on the shape of x 
        x.shape (batch_size, seq_len, dim)
        shape[-2]: for seq_len
        """
        positions = torch.arange(x.shape[-2], device=x.device)
        return super().forward(positions)

class RotaryPositionEmbedding(nn.Module):
    """
    A class to implement Rotary Position Embedding (RoPE) for tensors.

    Attributes:
        context_length (int): Sequence length.
        dim (int): Hidden dimension size (must be even).
    """

    def __init__(self, context_length: int, dim: int, theta=10_000):
        """
        Initialize the RoPE class.

        Args:
            context_length (int): Sequence length.
            dim (int): Hidden dimension size (must be even).
        """
        super().__init__()

        assert dim % 2 == 0, f"The hidden dimension {dim} must be even."
        self.context_length =  context_length
        self.dim = dim

        #precompute sin and cos for all positions
        #(context_length, 1)
        position_ids = torch.arange(context_length, dtype=torch.float32).unsqueeze(1) #(context_length, 1)
        indices = torch.arange(dim // 2, dtype=torch.float32) # (dim // 2,)
        theta = position_ids / (theta ** (2 * indices / dim)) # (context_length, dim // 2)
        self.sin = torch.sin(theta).unsqueeze(0).unsqueeze(0) # (1, 1, context_length, dim // 2)
        self.cos = torch.cos(theta).unsqueeze(0).unsqueeze(0) # (1, 1, context_length, dim // 2)

    def forward(self, x :torch.Tensor) -> torch.Tensor:
        assert x.shape[-2] == self.context_length, f"Mismatch in context_length {x.shape[-2]} vs {self.context_length}."
        assert x.shape[-1] == self.dim, f"Mismatch in dimension {x.shape[-1]} vs {self.dim}."

        x1, x2 = x[..., ::2], x[..., 1::2] # (batch_size, num_heads, context_length, dim // 2)
        return torch.cat([x1 * self.cos - x2 * self.sin, x1 * self.sin + x2 * self.cos], dim=-1)



