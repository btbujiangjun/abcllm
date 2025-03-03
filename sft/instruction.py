# -*- coding: utf-8 -*-
"""
Instruction Fine-Tuning Module

This module defines the `InstructionFinetune` class, which is designed to fine-tune 
a transformer-based model for instruction-following tasks. It supports loss computation 
with an ignore index and provides a method for generating responses based on a given prompt.

Author: Jiang Jun
Date: 2025-02-18
"""

import torch
from model.trainer import Trainer

class InstructionFinetune(Trainer):
    """
    A class for fine-tuning a transformer model to follow instructions.

    Inherits from the Trainer class and enables:
    - Configurable sequence length.
    - Loss computation with an ignore index for masked training.
    - Text generation tailored for instruction-based response generation.

    Attributes:
        max_length (int): Maximum sequence length for generation.
        ignore_index (int): Token index to ignore during loss computation.
    """

    def __init__(self, model, tokenizer, max_length=0):
        super().__init__(model, tokenizer)
        self.max_length = max_length
        self.ignore_index = -999

    def ignore_index(self, value):
        self.ignore_index = value

    def loss_function(self, logits, target):
        return torch.nn.functional.cross_entropy(
            logits.flatten(0, 1),
            target.flatten().long(),
            ignore_index = self.ignore_index
        )

    def generate(self, 
            start_context: str, 
            max_length: int = None, 
            temperature:float = 0.0, 
            top_k: int = None, 
            eos_id:  int = None):
        max_length = max_length or self.max_length or self.model.cfg["context_length"]
        response_text = super().generate(
            start_context = start_context, 
            max_length = max(max_length, len(start_context)), 
            temperature = temperature, 
            top_k = top_k, 
            eos_id = eos_id
        )
        
        response_text = response_text[len(start_context):].replace("\n### Response:", "").strip()
        return {
            "Start_Context":start_context,
            "Generate_Text":response_text,
        }

