
import torch
import torch.nn as nn
from model.abcmodel import ABCModel
from module.activation import SiLU
from module.normalization import RMSNorm
from module.attention import GroupedQueryAttention

LLAMA32_CONFIG = {
    "vocab_size": 128_256,      # Vocabulary size
    "context_length": 256, #131_072,  # Context length
    "emb_dim": 2048,            # Embedding dimension
    "n_heads": 32,              # Number of attention heads
    "n_layers": 16,             # Number of layers
    "hidden_dim": 8192,         # Size of the intermediate dimension in FeedForward
    "n_kv_groups": 8,           # Key-Value groups for grouped-query attention
    "rope_base": 500_000,     # The base in RoPE's "theta"
    "dtype": torch.bfloat16,    # Lower-precision dtype to reduce memory usage
    "drop_rate": 0.1,           # Dropout rate
    "qkv_bias": False,          # Whether to use bias in QKV projections
    "lr": 4e-5,                 # Learning rate
    "decay": 0.1,               # Weight decay
    "max_grad_norm": 1.0,       # Max gradient clip
    "accumulation_steps": 4,    # Gradient accumulation update steps
    "warmup_steps": 1, #1_000,      # Warmup steps with a larger lr
    "device": torch.device(
        "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu")
    )
}

class Llama3Model(ABCModel):
    def __init__(self, cfg):
        self.init(cfg)

    def init(self, cfg):
        super().__init__(cfg)
        self._name = "Llama" if "name" not in cfg else cfg["name"]
        self._version = "3" if "version" not in cfg else cfg["version"]
        vocab_size, emb_dim = cfg["vocab_size"], cfg["emb_dim"]
        device, layers, dtype = cfg["device"], cfg["n_layers"], cfg["dtype"]

        self.tok_emb = nn.Embedding(vocab_size, emb_dim, dtype=dtype).to(device)
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(layers)]).to(device)
        self.final_norm = RMSNorm(emb_dim).to(dtype).to(device)
        self.out_head = nn.Linear(emb_dim, vocab_size, bias=False, dtype=dtype).to(device)

        # Optimizer
        self.reset_optimizer()
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

        emb_dim, hidden_dim, dtype = cfg["emb_dim"], cfg["hidden_dim"], cfg["dtype"]
        self.fc1 = nn.Linear(emb_dim, hidden_dim, bias=False, dtype=dtype)
        self.fc2 = nn.Linear(emb_dim, hidden_dim, bias=False, dtype=dtype)
        self.fc3 = nn.Linear(hidden_dim, emb_dim, bias=False, dtype=dtype)
        self.silu = SiLU()

    def forward(self, x: torch.Tensor)->torch.Tensor:
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = self.silu(x_fc1) * x_fc2
        return self.fc3(x)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        emb_dim = cfg["emb_dim"]
        self.dtype = cfg["dtype"]
        self.att = GroupedQueryAttention(
            d_in=emb_dim,
            d_out=emb_dim,
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"],
            rope_base=cfg["rope_base"],
            dtype=self.dtype,
        )
        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(emb_dim)
        self.norm2 = RMSNorm(emb_dim)

    def forward(self, x: torch.Tensor)->torch.Tensor:
        x = x.to(self.dtype)
        
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

        
