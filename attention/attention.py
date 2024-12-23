# -*- coding: utf-8 -*-
"""
attention.py

Key modules for LLM tasks including SelfAttention, CausalAttention, and MultiHeadAttention.

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

class SelfAttention(nn.Module):
    """
    Implements single-head self-attention mechanism.
    """

    def __init__(self, d_in, d_out, qkv_bias=False):
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

    def forward(self, x):
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

        attn_scores = queries @ keys.T

        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5,
            dim=-1
        )

        return attn_weights @ values

class CausalAttention(nn.Module):
    """
    Implements single-head causal (masked) attention mechanism for autoregressive tasks.
    """

    def __init__(self
            ,d_in
            ,d_out
            ,context_length
            ,dropout
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

    def forward(self, x):
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
        attn_scores = queries @ keys.transpose(1, 2) 
        attn_scores.masked_fill_(
            self.mask.bool()[:n_tokens, :n_tokens]
            ,-torch.inf
        )

        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5
            ,dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        return attn_weights @ values


class  MultiHeadAttention(nn.Module):
    """
    Implements multi-head attention mechanism with causal masking.
    """
    def __init__(self
            ,d_in
            ,d_out
            ,context_length
            ,dropout
            ,num_heads
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

    def forward(self, x):
        batch_size, n_tokens, d_in = x.shape

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
        attn_scores = queries @ keys.transpose(2, 3) # dot prodcut for each head

        #original mask truncated to the number of tokens and convert to boolean
        mask_bool = self.mask.bool()[:n_tokens, :n_tokens]
        #use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        #shape:(batch_size, n_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(batch_size, n_tokens, self.d_out)

        return self.out_proj(context_vec)




