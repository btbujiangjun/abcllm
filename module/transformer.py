

import torch
import torch.nn as nn
from module.activation import GELU
from module.normalization import LayerNorm
from module.position import AbsolutePositionEmbedding
from module.attention import MultiHeadAttention, NativeSparseAttention

class Gpt2Transformer(nn.Module):
    """✅ GPT2 LLM Transformer"""
    def __init__(self, 
            vocab_size:int,
            emb_dim:int,
            context_len:int,
            num_heads:int,
            num_layers:int,
            dropout:float,
            qkv_bias:bool=False,
            device="cpu"):
        super().__init__()
        
        # Embedding layer
        self.tok_emb = nn.Embedding(vocab_size, emb_dim).to(device)
        self.pos_emb = AbsolutePositionEmbedding(context_len, emb_dim).to(device)
        self.drop_emb = nn.Dropout(dropout).to(device)

        # Transformer layers
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(
                d_in = emb_dim,
                d_out = emb_dim,
                context_length = context_len,
                num_heads = num_heads,
                dropout = dropout,
                qkv_bias = qkv_bias
            ) for _ in range(num_layers)]
        ).to(device)

        # Final layers
        self.final_norm = LayerNorm(emb_dim).to(device)
        self.out_head = nn.Linear(emb_dim, vocab_size, bias=False).to(device)

    def forward(self, x) -> torch.Tensor:
        x = x.to(self.device)

        # Token and position embedding
        token_embeds = self.tok_emb(x)
        pos_embeds = self.pos_emb(tok_embeds)
        x = self.dropout(token_embeds + pos_embeds)

        # Transformer
        x = self.trf_blocks(x)

        # Final layer
        return self.out_head(self.final_norm(x))


    """✅ Inner class"""
    class TransformerBlock(nn.Module):
        """
        Transformer block consisting of multi-head attention, feed-forward layers,
        residual connections, and normalization.
        """
        def __init__(self,
                emb_dim:int,
                context_len:int,
                num_heads:int,
                dropout:float,
                qkv_bias:bool=False):
            super().__init__()
            
            self.multi_attention = MultiHeadAttention(
                d_in = emb_dim,
                d_out = emb_dim,
                context_length = context_len,
                num_heads = num_heads,
                dropout = dropout,
                qkv_bias = qkv_bias
            )

            self.ff = FeedForward(emb_dim)
            self.norm1 = LayerNorm(emb_dim)
            self.norm2 = LayerNorm(emb_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x) -> torch.Tensor:
            # Multi-head attention + residual connection
            short_cut = x
            x = self.dropout(self.multi_attention(self.norm1(x)))
            x = x + short_cut

            # Feed-forward network + residual connection
            short_cut = x
            x = self.dropout(self.ff(self.norm2(x)))
            x = x + short_cut

            return x

        """✅ Inner class"""
        class FeedForward(nn.Module):
            """
            Feed-forward layer with GELU activation and dropout.
            """
            def __init__(self, emb_dim: int):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(emb_dim, 4 * emb_dim),
                    GELU(),
                    nn.Linear(4 * emb_dim, emb_dim)
                )

            def forward(self, x: torch.Tensor)->torch.Tensor:
                return self.layers(x)


class NSATransformer(nn.Module):
    """✅ DeepSeek NSA Transformer"""
    def __init__(self, 
            vocab_size, # 词汇表大小 
            emb_dim, # 词向量维度
            context_len, # 上下文长度
            num_heads, # 注意力头数
            num_layers, # Transformer 层数
            ffn_dim:int=None, # FFN 维度
            dropout=0.1, 
            device="cpu"):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, emb_dim).to(device)
        self.pos_emb = nn.Parameter(torch.randn(1, context_len, emb_dim)).to(device)

        self.layers = nn.ModuleList([
            self.TransformerBlock(
                emb_dim,
                context_len,
                num_heads, 
                ffn_dim or emb_dim, 
                dropout=dropout
            ) for _ in range(num_layers)
        ]).to(device)

        self.norm = nn.LayerNorm(emb_dim).to(device)
        self.fc_out = nn.Linear(emb_dim, vocab_size).to(device)

    def forward(self, x) -> torch.Tensor:
        x = self.tok_emb(x) + self.pos_emb[:, :x.shape[1], :]
        for layer in self.layers:
            x = layer(x)
        return self.fc_out(self.norm(x))

    """Inner class"""
    class TransformerBlock(nn.Module):
        """✅ NSA Transformer Block"""
        def __init__(self, embed_dim, context_len, num_heads, ffn_dim, dropout=0.1):
            super().__init__()
            self.nsa = NativeSparseAttention(embed_dim, context_len)
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)

            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, ffn_dim),
                nn.ReLU(),
                nn.Linear(ffn_dim, embed_dim)
            )
            self.dropout = nn.Dropout(dropout)

        def forward(self, x) -> torch.Tensor:
            x = x + self.dropout(self.nsa(self.norm1(x)))
            x = x + self.dropout(self.ffn(self.norm2(x)))
            return x


