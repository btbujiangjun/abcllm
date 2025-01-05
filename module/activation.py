# activation_functions.py
# Author: Jiang Jun
# Date: 2025.01.05
# Description: Implementation of SiLU and GELU activation functions.

import torch
import torch.nn as nn

class SiLU(nn.Module):
    """
    Implements the SiLU (Sigmoid Linear Unit) activation function.

    Formula:
        SiLU(x) = x * sigmoid(x)
    """
    def forward(self, x :torch.Tensor) -> torch.Tensor:
        """
        Forward pass for SiLU activation.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Activated output.
        """
        return x * torch.sigmoid(x)

class GELU(nn.Module):
    """
    Implements the GELU (Gaussian Error Linear Unit) activation function.
    
    Approximation formula:
        GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    def forward(self, x :torch.Tensor) -> torch.Tensor:
        """
        Forward pass for GELU activation.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Activated output.
        """
        coeff = torch.sqrt(torch.tensor(2.0 / torch.pi))
        return 0.5 * x * (1 + torch.tanh(coeff * (x + 0.044715 * x ** 3)))

