# 
# 
# the key module of LLM task
# include SelfAttention, CausalAttention
#

import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.w_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
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
    def __init__(self
            ,d_in
            ,d_out
            ,context_length
            ,dropout
            ,qkv_bias=False):
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
    def __init__(self
            ,d_in
            ,d_out
            ,context_length
            ,dropout
            ,num_heads
            ,qkv_bias=False):
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




