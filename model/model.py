# -*- coding: utf-8 -*-
"""
gpt_model.py

Implementation of a GPT-based model including transformer blocks, attention mechanisms, and 
feed-forward networks. This script supports model creation, training, and text generation.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Author: JiangJun
Date: 2024-12-16
"""



import torch
import torch.nn as nn
from attention.attention import MultiHeadAttention

# Default configuration for a GPTModel
GPT_CONFIG_124M = {
    "vocab_size": 64789,  # Number of tokens in the vocabulary
    "context_length": 256,  # Maximum sequence length
    "emb_dim": 768,  # Embedding dimension
    "n_heads": 12,  # Number of attention heads
    "n_layers": 12,  # Number of transformer layers
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": False,  # Whether to use bias in QKV projections
    "lr": 4e-5,  # Learning rate
    "decay": 0.1,  # Weight decay
    "device": torch.device(
        "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu")
    )
}

class GPTModel(nn.Module):
    """
    GPTModel defines a GPT-based transformer with embeddings, transformer layers,
    and output head for generating logits.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        device=cfg["device"]

        # Embedding layers
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"]).to(device)
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"]).to(device)
        self.drop_emb = nn.Dropout(cfg["drop_rate"]).to(device)
        
        # Transformer layers
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        ).to(device)
        
        # Final layers
        self.final_norm = LayerNorm(cfg["emb_dim"]).to(device)
        self.out_head = nn.Linear(
            cfg["emb_dim"], 
            cfg["vocab_size"], 
            bias=False
        ).to(device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=cfg["lr"], 
            weight_decay=cfg["decay"]
        )

        print(f"Initialized model with configuration:{cfg}")

    def reset_optimizer(self):
        """Reset the optimizer with the current configuration."""
        self.optimizer = torch.optim.AdamW(
            self.parameters()
            ,lr=self.cfg["lr"]
            ,weight_decay=self.cfg["decay"]
        )

    @property        
    def device(self)->str:
        """Return the device on which the model resides."""
        return next(self.parameters()).device

    @property
    def param_size(self)->int:
        """Calculate the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    @property
    def size_byte(self)->float:
        """Calculate the size of the model."""
        return sum(p.numel() * torch.tensor([], dtype=p.dtype).element_size() for p in self.parameters())

    @property
    def size_mb(self)->float:
        """Calculate the size of the model in megabytes."""
        return f"{self.size_byte / (1024 * 1024):.2f}"

    def forward(self, batch):
        """
        Forward pass of the GPT model.

        Args:
            batch (torch.Tensor): Input token IDs of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Logits of shape (batch_size, seq_len, vocab_size).
        """
        batch_size, seq_len = batch.shape
        batch = batch.to(self.device)

        # Token and position embeddings
        tok_embeds = self.tok_emb(batch)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=batch.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)

        # Transformer blocks
        x = self.trf_blocks(x)

        # Output
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

class TransformerBlock(nn.Module):
    """
    Transformer block consisting of multi-head attention, feed-forward layers,
    residual connections, and normalization.
    """
    def __init__(self, cfg):
        super().__init__()
        self.multi_attention = MultiHeadAttention(
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            context_length = cfg["context_length"],
            num_heads = cfg["n_heads"],
            dropout = cfg["drop_rate"],
            qkv_bias = cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        """
        Forward pass through a transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, emb_dim).

        Returns:
            torch.Tensor: Output tensor of the same shape as input.
        """
        # Multi-head attention + residual connection
        short_cut = x
        x = self.norm1(x)
        x = self.multi_attention(x)
        x = self.dropout(x)
        x = x + short_cut

        # Feed-forward network + residual connection
        short_cut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + short_cut #residual connection

        return x

class LayerNorm(nn.Module):
    """
    Custom Layer Normalization implementation.
    """
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
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

class FeedForward(nn.Module):
    """
    Feed-forward layer with GELU activation and dropout.
    """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"])
            ,GELU()
            ,nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )

    def forward(self, x):
        return self.layers(x)

class GELU(nn.Module):
    """
    Approximation of Gaussian Error Linear Unit (GELU) activation.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 +
            torch.tanh(
                torch.sqrt(torch.tensor(2.0 / torch.pi))
                * (x + 0.044715 * torch.pow(x, 3))
            )
        )

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
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.emb_dim = emb_dim
        self.weight = nn.Parameter(torch.ones(emb_dim)).float()

    def forward(self, x):
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor of shape (..., emb_dim).

        Returns:
            torch.Tensor: Normalized tensor of the same shape as input.
        """
        # Compute the mean of squared elements along the last dimension
        mean = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(means + self.eps)
        # Normalize using RMS and apply the learned scale
        return (x_normed * self.weight).to(dtype=x.dtype)

class ModelWrapper:
    """
    Wrapper class for handling text generation with a GPTModel.
    """
    def __init__(self):
        pass

    def __generate(self
            ,model
            ,ids
            ,max_generate_tokens=None
            ,context_length=None
            ,temperature=0.0
            ,top_k=None
            ,eos_id=None):
        """
        Internal method for token generation using autoregressive sampling.

        Args:
            model: GPT model instance.
            ids (torch.Tensor): Input token IDs of shape (1, seq_len).
            max_generate_tokens (int): Maximum number of tokens to generate.
            context_length (int, optional): Maximum context length.
            temperature (float): Sampling temperature.
            top_k (int, optional): Top-k sampling.
            eos_id (int, optional): End-of-sequence token ID.

        Returns:
            torch.Tensor: Generated token IDs of shape (1, seq_len + max_generate_tokens).
        """
        if context_length is None:
            context_length = model.cfg["context_length"]
        if max_generate_tokens is None:
            max_generate_tokens = context_length
        ids = ids.to(model.device)

        for _ in range(max_generate_tokens):
            with torch.no_grad():
                #Truncate to the last `context_length` tokens
                logits = model(ids[:, -context_length:])
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
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                ids_next = torch.multinomial(probs, num_samples=1)
            else:
                # Deterministic greedy decoding
                ids_next = torch.argmax(logits, dim=-1, keepdim=True)

            # Stop generation if the end-of-sequence token is generated
            if eos_id is not None and (ids_next == eos_id).item():
                break

            # Append the generated token to the sequence
            ids = torch.cat((ids, ids_next), dim=1)

        return ids

    def generate(self
            ,model
            ,start_context
            ,tokenizer
            ,max_generate_tokens
            ,context_length=None
            ,temperature=0.0
            ,top_k=None
            ,eos_id=None):
        """
        Generate text using the model.

        Args:
            model: GPT model instance.
            start_context (str): Starting context for generation.
            tokenizer: Tokenizer to encode and decode text.
            max_generate_tokens (int): Maximum number of tokens to generate.
            context_length (int, optional): Maximum context length.
            temperature (float): Sampling temperature.
            top_k (int, optional): Top-k sampling.
            eos_id (int, optional): End-of-sequence token ID.

        Returns:
            str: Generated text.
        """
        encode_ids = tokenizer.encode(start_context)
        encode_tensor = torch.tensor(encode_ids).unsqueeze(0)

        if eos_id is None:
            eos_id = tokenizer.eos_id

        out_ids = self.__generate(
            model
            ,encode_tensor
            ,max_generate_tokens
            ,context_length=context_length
            ,temperature=temperature
            ,top_k=top_k
            ,eos_id=eos_id
        )

        return tokenizer.decode(out_ids.squeeze(0).tolist()).strip()

