# normalization.py
# Author: Jiang Jun
# Date: 2025.01.05
# Description: Implementation of custom normalization layers: LayerNorm and RMSNorm.

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Custom Layer Normalization implementation.

    This normalization method normalizes the input tensor to have zero mean
    and unit variance along the last dimension, with optional learnable
    scale and shift parameters.

    Args:
        emb_dim (int): Dimension of the input embedding.
        eps (float): Small constant to avoid division by zero during variance computation. Default: 1e-3.
    """
    def __init__(self, emb_dim :int, eps=1e-3):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x :torch.Tensor) -> torch.Tensor:
        """
        Normalize the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    This normalization method scales the input using the root mean square
    (RMS) of the input features instead of directly normalizing to a
    zero-mean, unit-variance distribution like traditional LayerNorm.

    Args:
        emb_dim (int): Dimension of the embedding.
        eps (float): Small constant to avoid division by zero. Default: 1e-5.
    """
    def __init__(self, emb_dim: int, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(emb_dim)).float()

    def forward(self, x :torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor of shape (..., emb_dim).

        Returns:
            torch.Tensor: Normalized tensor of the same shape as input.
        """
        # Compute the RMS (Root Mean Square) of the input tensor
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return (x / rms * self.weight).to(x.dtype)

