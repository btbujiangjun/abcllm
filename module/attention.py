# -*- coding: utf-8 -*-
"""
attention.py

Key modules for LLM tasks including SelfAttention, CausalAttention, and MultiHeadAttention.

Author: JiangJun
Date: 2024-12-16
"""


import torch
import torch.nn as nn
from module.position import RotaryPositionEmbedding

class SelfAttention(nn.Module):
    """
    Implements single-head self-attention mechanism.
    """

    def __init__(self, d_in: int, d_out :int, qkv_bias=False):
        """
        Initialize layers for query, key, and value projections.

        Args:
            d_in (int): Input feature dimension.
            d_out (int): Output feature dimension.
            qkv_bias (bool): Whether to use bias in linear projections.
        """
        super().__init__()
        self.w_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """
        Forward pass for self-attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_in).

        Returns:
            torch.Tensor: Output tensor after self-attention.
        """
        queries = self.w_query(x)
        keys = self.w_key(x)
        values = self.w_value(x)

        attn_scores = torch.matmul(queries, keys.transpose(-2, -1))
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5,
            dim=-1
        )

        return torch.matmul(attn_weights, values)

class CausalAttention(nn.Module):
    """
    Implements single-head causal (masked) attention mechanism for autoregressive tasks.
    """

    def __init__(self
            ,d_in: int
            ,d_out: int
            ,context_length: int
            ,dropout: float
            ,qkv_bias=False):
        """
        Initialize layers for query, key, and value projections, and the causal mask.

        Args:
            d_in (int): Input feature dimension.
            d_out (int): Output feature dimension.
            context_length (int): Maximum sequence length (context size).
            dropout (float): Dropout rate for attention weights.
            qkv_bias (bool): Whether to use bias in linear projections.
        """
        super().__init__()
        self.d_out = d_out
        self.w_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask'
            ,torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for causal attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_in).

        Returns:
            torch.Tensor: Output tensor after causal attention.
        """
        batch_size, n_tokens, d_in = x.shape
        queries = self.w_query(x)
        keys = self.w_key(x)
        values = self.w_value(x)

        #(batch, n_tokens, dim_in) -> (batch, dim_in, n_tokens)
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1)) 
        attn_scores.masked_fill_(
            self.mask.bool()[:n_tokens, :n_tokens]
            ,-torch.inf
        )

        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5
            ,dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        return torch.matmul(attn_weights, values)


class MultiHeadAttention(nn.Module):
    """
    Implements multi-head attention mechanism with causal masking.
    """
    def __init__(self
            ,d_in: int
            ,d_out: int
            ,context_length: int
            ,dropout: float
            ,num_heads: int
            ,qkv_bias=False):
        """
        Initialize multi-head attention layers and causal mask.

        Args:
            d_in (int): Input feature dimension.
            d_out (int): Output feature dimension.
            context_length (int): Maximum sequence length (context size).
            dropout (float): Dropout rate for attention weights.
            num_heads (int): Number of attention heads.
            qkv_bias (bool): Whether to use bias in linear projections.
        """
        super().__init__()
        assert(d_out % num_heads == 0), "d_out must be diviasible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.w_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask"
            ,torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        batch_size, n_tokens, _ = x.shape
        queries = self.w_query(x)
        keys = self.w_key(x)
        values = self.w_value(x)

        #unroll last dim:
        # (batch_size, n_tokens, d_out) -> (batch_size, n_tokens, num_heads, head_dim)
        queries = queries.view(batch_size, n_tokens, self.num_heads, self.head_dim)
        keys = keys.view(batch_size, n_tokens, self.num_heads, self.head_dim)
        values = values.view(batch_size, n_tokens, self.num_heads, self.head_dim)

        #transpose:
        #(batch_size, n_tokens, num_heads, head_dim)->(batch_size, num_heads, n_tokens, head_dim)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        #scaled dot-product attention with a causal mask
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1)) # dot prodcut for each head

        #original mask truncated to the number of tokens and convert to boolean
        mask_bool = self.mask.bool()[:n_tokens, :n_tokens]
        #use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        #shape:(batch_size, n_tokens, num_heads, head_dim)
        context = torch.matmul(attn_weights, values).transpose(1, 2).reshape(batch_size, n_tokens, -1)

        return self.out_proj(context)

class GroupedQueryAttention(nn.Module):
    def __init__(self, 
            d_in: int,
            d_out: int,
            context_length: int,
            num_heads: int,
            num_kv_groups: int,
            rope_base: int =10_000,
            dtype=None
        ):
        super().__init__()
        assert d_out % num_heads == 0, f"d_out {d_out} must be divisible by num_heads {num_heads}."
        assert num_heads % num_kv_groups == 0, f"num_heads {num_heads} must be divisible by num_kv_groups {num_kv_groups}."

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        self.w_query = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.w_key = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.w_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)

        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        self.register_buffer("mask", mask)

        self.position_embedding = RotaryPositionEmbedding(context_length, self.head_dim, dtype=dtype)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        batch_size, num_tokens, d_in = x.shape
        
        queries = self.w_query(x) #(batch_size, num_tokens, d_out)
        keys = self.w_key(x) #(batch_size, num_tokens, num_kv_groups * head_dim)
        values = self.w_value(x) #(batch_size, num_tokens, num_kv_groups * head_dim)

        #Reshape queries/keys/values
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        #Position embedding
        keys = self.position_embedding(keys)
        queries = self.position_embedding(queries)

        #expand keys and values to match the number of heads
        keys = keys.repeat_interleave(self.group_size, dim=1) #(batch_size, num_heads, num_tokens, head_dim)
        values = values.repeat_interleave(self.group_size, dim=1) #(batch_size, num_heads, num_tokens, head_dim)
        #Compute scaled dot-product attention with a causal mask
        # (batch_size, num_heads, num_tokens, num_tokens)
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1)) #Dot product for each head
        #Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weight = torch.softmax(attn_scores /keys.shape[-1] ** 0.5, dim=-1)

        #???
        assert keys.shape[-1] == self.head_dim

        # (batch_size, num_tokens, num_heads, head_dim)
        context = torch.matmul(attn_weight, values).transpose(-2, -1).reshape(batch_size, num_tokens, self.d_out)
        return self.out_proj(context)





