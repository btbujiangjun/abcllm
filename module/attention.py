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
            x (torch.Tensor): Input tensor of shape (batch_size, context_length, d_in).

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
            x (torch.Tensor): Input tensor of shape (batch_size, context_length, d_in).

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

class SlidingWindowAttention(nn.Module):
    """
    ✅ 滑动窗口注意力：仅计算局部窗口，提高计算效率
    """
    def __init__(self, window_size, context_len):
        super().__init__()
        assert window_size <= context_len, "Error:window_size must be <= context_len"

        self.window_size = window_size
        mask = torch.triu(torch.ones(context_len, context_len), diagonal=1) #上三角
        self.register_buffer("mask", mask)

    def forward(self, Q, K, V) -> torch.Tensor:
        mask = self.mask[:self.window_size, :self.window_size] #仅保留滑动窗口

        scores = torch.matmul(Q, K.transpose(-2, -1)) / Q.shape[-1] ** 0.5
        print(f"scores:{scores.shape}")
        print(f"mask:{mask.shape}")
        scores = scores.masked_fill(mask.bool(), float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, V)

class CompressedAttention(nn.Module):
    """✅ 块级压缩注意力：降低计算量，同时保留全局信息"""
    def __init__(self, compression_ratio:int = 1):
        super().__init__()
        assert isinstance(compression_ratio, int) and compression_ratio >= 1, "compression_ratio must be int and >= 1"
        self.compression_ratio = compression_ratio

    def forward(self, Q, K, V) -> torch.Tensor:
        batch_size, context_len, dim = q.shape
        new_context_len = context_len // self.compression_ratio

        K_compressed = K.reshape(batch_size, new_context_len, self.compression_ratio, dim).mean(dim=2)
        V_compressed = V.reshape(batch_size, new_context_len, self.compression_ratio, dim).mean(dim=2)

        scores = torch.matmul(Q, K_compressed.transpose(-2, -1)) / dim**0.5
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, V_compressed)

class SelectedAttention(nn.Module):
    """✅ 选择性注意力：动态选择最重要的 token 进行注意力计算"""
    def __init__(self, top_k:int):
        super().__init__()
        self.top_k = top_k

    def forward(self, Q, K, V) -> torch.Tensor:
        batch_size, context_len, dim = Q.shape

        # 计算 Q 与 K 的相似性得
        scores = torch.matmul(Q, K.transpose(-2, -1)) / dim**0.5
        #(batch, context_len, top_k)
        top_values, top_indices = torch.topk(scores, self.top_k, dim=-1)
        
        sparse_weights = torch.zeros_like(scores)
        sparse_weights.scatter_(-1, top_indices, top_values)

        attn = torch.softmax(sparse_weights, dim=-1)
        return torch.matmul(attn, V)

class NativeSparseAttention(nn.Module):
    """✅ NSA 结合三种注意力，提高长序列任务的效率"""
    def __init__(self,
            embed_dim: int,
            context_len: int,
            window_size: int = 128,
            compression_ratio: int = 4,
            top_k : int = 64):
        super().__init__()
        self.window_attn = SlidingWindowAttention(window_size, context_len)
        self.compressed_attn = CompressedAttention(compression_ratio)
        self.selected_attn = SelectedAttention(top_k)

        self.q_w = nn.Linear(embed_dim, embed_dim)
        self.k_w = nn.Linear(embed_dim, embed_dim)
        self.v_w = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x) -> torch.Tensor:
        Q, K, V = self.q_w(x), self.k_w(x), self.v_w(x)

        w_attn = self.window_attn(Q, K, V)
        c_attn = self.compressed_attn(Q, K, V)
        s_attn = self.selected_attn(Q, K, V)

        return self.out_proj((w_attn + c_attn + s_attn)/3)


