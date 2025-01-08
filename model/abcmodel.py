# -*- coding: utf-8 -*-
"""
Author: JiangJun
Date: 2025-01-06
"""

import torch
from abc import ABC, abstractmethod
from torch.nn.parallel import DistributedDataParallel as DDP

class ABCModel(ABC, torch.nn.Module):
    """
    Abstract class, define a LLM model
    """
    def __init__(self, cfg):
        super().__init__()
        self._name = "ABCModel"
        self._version = "1.0"
        self._cfg = None
        self.cfg = cfg

    @property
    def name(self):
        return f"{self._name} {self._version}({self.param_size})"

    @property
    def cfg(self):
        if isinstance(self, DDP):
            return self.module._cfg
        else:
            return self._cfg

    @cfg.setter
    def cfg(self, value):
        if isinstance(self, DDP):
            self.module._cfg = value
        else:
            self._cfg = value

    @property
    def device(self)->str:
        return next(self.parameters()).device

    @property
    def param_size(self)->int:
        return sum(p.numel() for p in self.parameters())

    @property
    def size_byte(self)->float:
        return sum(p.numel() * torch.tensor([], dtype=p.dtype).element_size() for p in self.parameters())

    @property
    def size_mb(self)->float:
        return float(f"{self.size_byte / (1024 * 1024):.2f}")

    def reset_optimizer(self):
        """Reset the optimizer with the current configuration."""
        self.optimizer = torch.optim.AdamW(
            self.parameters()
            ,lr=self.cfg["lr"]
            ,weight_decay=self.cfg["decay"]
        )
   
    @abstractmethod
    def init(self, cfg):
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor)->torch.Tensor:
        pass


