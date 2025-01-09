# -*- coding: utf-8 -*-
"""
generator.py

This module provides a Generator class for handling text generation with an autoregressive GPT model.

Author: JiangJun
Date: 2024-12-16
"""

import torch
from model.abcmodel import ABCModel
from tokenizer.tokenizer import ABCTokenizer

class Generator:
    """
    Wrapper class for handling text generation with a GPTModel.
    """

    @staticmethod
    @torch.no_grad()
    def generate(
            model: ABCModel,
            start_context: str,
            tokenizer: ABCTokenizer,
            max_length: int = None,
            context_length: int = None,
            temperature: float = 0.0,
            top_k: int = None,
            eos_id: int = None,
        ) -> str:
        """
        Generate text using the model by autoregressive sampling.

        Args:
            model (ABCModel): The language model instance.
            start_context (str): Starting context for generation.
            tokenizer (ABCTokenizer): Tokenizer for encoding and decoding text.
            max_length (int, optional): Maximum number of tokens to generate. Defaults to model's context length.
            context_length (int, optional): Maximum context length. Defaults to model's context length.
            temperature (float, optional): Sampling temperature. Defaults to 0.0 (deterministic).
            top_k (int, optional): Top-k sampling for nucleus filtering. Defaults to None (no filtering).
            eos_id (int, optional): End-of-sequence token ID. Defaults to None (no early stopping).

        Returns:
            str: Generated text(seq_len + max_length).
        """
        eos_id = eos_id or tokenizer.eos_id
        context_length = context_length or model.cfg["context_length"]
        max_length = max_length or context_length
        
        generate_tensor = torch.tensor(
            tokenizer.encode(start_context), 
            dtype=torch.long
        ).unsqueeze(0).to(model.device)

        for _ in range(max_length):
            #Truncate to the last `context_length` tokens
            logits = model(generate_tensor[:, -context_length:])
            #Consider only the last timestep's logits
            #-1: (batch, num_tokens, vacab_size) -> (batch, vocab_size)
            logits = logits[:, -1, :] #

            if top_k is not None:
                # Apply top-k filtering
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(
                    logits < min_val
                    ,torch.tensor(float("-inf")).to(logits.device)
                    ,logits
                )

            # Apply temperature scaling and sample from probabilities
            if temperature > 0.0:
                probs = torch.softmax(logits / temperature, dim=-1)
                generate_next = torch.multinomial(probs, num_samples=1)
            else:
                # Deterministic greedy decoding
                generate_next = torch.argmax(logits, dim=-1, keepdim=True)

            # Stop generation if the end-of-sequence token is generated
            if eos_id is not None and generate_next.item() == eos_id:
                break

            # Append the generated token to the sequence
            generate_tensor = torch.cat((generate_tensor, generate_next), dim=1)

        return tokenizer.decode(generate_tensor.squeeze(0).tolist()).strip()

