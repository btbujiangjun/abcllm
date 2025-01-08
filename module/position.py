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

    def __init__(self, 
            context_length: int, 
            head_dim: int, 
            theta=10_000, 
            dtype=torch.float32):
        """
        Initialize the RoPE class.

        Args:
            context_length (int): Sequence length.
            head_dim (int): Head_dim dimension size (must be even).
            theta (int): default 10_000
        """
        super().__init__()

        assert head_dim % 2 == 0, f"The head_dim dimension {head_dim} must be even."
        
        self.head_dim = head_dim
        self.theta = theta
        if dtype is None:
            dtype = torch.float32
        self.dtype = dtype
        self._precompute(context_length)

    def _precompute(self, seq_len, device="cpu"):
        self.seq_len = seq_len
        #precompute sin and cos for all positions
        position_ids = torch.arange(seq_len, dtype=self.dtype).unsqueeze(1) #(seq_len, 1)
        indices = torch.arange(self.head_dim // 2, dtype=self.dtype) # (head_dim // 2,)
        theta = position_ids / (self.theta ** (2 * indices / self.head_dim)) # (seq_len, head_dim // 2)
        self.sin = torch.sin(theta).unsqueeze(0).unsqueeze(0).to(device) # (1, 1, seq_len, head_dim // 2)
        self.cos = torch.cos(theta).unsqueeze(0).unsqueeze(0).to(device) # (1, 1, seq_len, head_dim)// 2

    def forward(self, x :torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = x.shape
        assert head_dim % 2 == 0, "Head dimension {head_dim} must be even."
        assert head_dim == self.head_dim, f"Mismatch in dimension {head_dim} vs {self.head_dim}."
        
        if(seq_len != self.seq_len):
            self._precompute(seq_len, x.device)

        if self.sin.device != x.device:
            self.sin = self.sin.to(x.device)
            self.cos = self.cos.to(x.device)

        x1, x2 = x[..., ::2], x[..., 1::2] # (batch_size, num_heads, context_length, head_dim // 2)

        return torch.cat([x1 * self.cos - x2 * self.sin, x1 * self.sin + x2 * self.cos], dim=-1)



