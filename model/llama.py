
import torch.nn as nn
from model.abcmodel import ABCModel
from module.activation import SiLU
from module.normalization import RMSNorm
from module.attention import GroupedQueryAttention

class Llama3Model(ABCModel):
    def __init__(self, cfg):
        super().__init__(cfg):
        self._name = "Llama"
        self._version = "3"

        vocab_size, emb_dim = cfg["vocab_size"], cfg["emb_dim"]
        layers, device = cfg["device"], cfg["n_layers"]

        self.tok_emb = nn.Embedding(vocab_size, emb_dim).to(device)
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(layers)]).to(device)
        self.final_norm = RMSNorm(emb_dim).to(device)
        self.out_head = nn.Linear(emb_dim, vocab_size, bias=False).to(device)

        print(f"Initialized {self.name} with configuration:{cfg}", flush=True)

    def forward(self, x: torch.Tensor)->torch.Tensor:
        x = x.to(self.device)
        x = self.tok_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        emb_dim, hidden_dim = cfg["emb_dim"], cfg["hidden_dim"]
        self.fc1 = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.fc3 = nn.Linear(hidden_dim, emb_dim, bias=False)
        self.silu = SiLU()

    def forward(self, x: torch.Tensor)->torch.Tensor:
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = self.silu(x_fc1) * x_fc2
        return self.fc3(x)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"],
            rope_base=cfg["rope_base"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"])
        self.norm2 = RMSNorm(cfg["emb_dim"])

    def forward(self, x: torch.Tensor)->torch.Tensor:
        # Shortcut connection for attention block
        short_cut = x
        x = self.norm1(x)
        x = self.att(x) # (batch_size, num_tokens, emb_size)
        x = x + short_cut # Add the original input

        # Shortcut connection for feed-forward block
        short_cut = x
        x = self.norm1(x)
        x = self.ff(x) # (batch_size, num_tokens, emb_size)
        x = x + short_cut # Add the original input

        return x

        
