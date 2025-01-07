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
        head_dim (int): Head dimension size (must be even).
    """

    def __init__(self, context_length: int, head_dim: int, theta=10_000):
        """
        Initialize the RoPE class.

        Args:
            context_length (int): Sequence length.
            head_dim (int): Head_dim dimension size (must be even).
            theta (int): default 10_000
        """
        super().__init__()

        assert head_dim % 2 == 0, f"The head_dim dimension {head_dim} must be even."
        self.context_length =  context_length
        self.head_dim = head_dim

        #precompute sin and cos for all positions
        #(context_length, 1)
        position_ids = torch.arange(context_length, dtype=torch.float32).unsqueeze(1) #(context_length, 1)
        indices = torch.arange(head_dim // 2, dtype=torch.float32) # (head_dimi // 2,)
        theta = position_ids / (theta ** (2 * indices / head_dim)) # (context_length, head_dim // 2)
        self.sin = torch.sin(theta).unsqueeze(0).unsqueeze(0) # (1, 1, context_length, head_dim // 2)
        self.cos = torch.cos(theta).unsqueeze(0).unsqueeze(0) # (1, 1, context_length, head_dim)// 2

    def forward(self, x :torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = x.shape
        assert head_dim % 2 == 0, "Head dimension {head_dim} must be even."
        assert seq_len == self.context_length, f"Mismatch in context_length {seq_len} vs {self.context_length}."
        assert head_dim == self.head_dim, f"Mismatch in dimension {head_dim} vs {self.head_dim}."

        x1, x2 = x[..., ::2], x[..., 1::2] # (batch_size, num_heads, context_length, head_dim // 2)
        return torch.cat([x1 * self.cos - x2 * self.sin, x1 * self.sin + x2 * self.cos], dim=-1)



