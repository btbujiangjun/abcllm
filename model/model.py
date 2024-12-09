#
#
#
#
#

import torch
import torch.nn as nn
from attention.attention import MultiHeadAttention

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
    "lr": 4e-5,
    "decay": 0.1,
    "device": torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
}

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        device=cfg["device"]
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"]).to(device)
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"]).to(device)
        self.drop_emb = nn.Dropout(cfg["drop_rate"]).to(device)
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        ).to(device)
        self.final_norm = LayerNorm(cfg["emb_dim"]).to(device)
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        ).to(device)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=cfg["lr"], weight_decay=cfg["decay"])

        print(f"initalize model with config:", cfg)

    def reset_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.parameters()
            ,lr=self.cfg["lr"]
            ,weight_decay=self.cfg["decay"]
        )
        
    def device(self):
        return next(self.parameters()).device

    def param_size(self):
        return sum(p.numel() for p in self.parameters())

    def size_byte(self):
        return sum(p.numel() * torch.tensor([], dtype=p.dtype).element_size() for p in self.parameters())

    def size_mb(self):
        return (f"{self.size_byte() / (1024 * 1024):.2f}")

    def forward(self, batch):
        batch_size, seq_len = batch.shape
        batch = batch.to(self.device())
        tok_embeds = self.tok_emb(batch)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=batch.device))

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
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
        short_cut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.dropout(x)
        x = x + short_cut #residual connection

        short_cut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + short_cut #residual connection

        return x

class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class FeedForward(nn.Module):
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
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.emb_dim = emb_dim
        self.weight = nn.Parameter(torch.ones(emb_dim)).float()

    def forward(self, x):
        mean = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(means + self.eps)
        return (x_normed * self.weight).to(dtype=x.dtype)

class ModelWrapper:
    def __init__(self):
        pass
    def __generate(self
            ,model
            ,ids
            ,max_generate_tokens
            ,context_length=None
            ,temperature=0.0
            ,top_k=None
            ,eos_id=None):
        if context_length is None:
            context_length = model.cfg["context_length"]
        ids = ids.to(model.device())

        for _ in range(max_generate_tokens):
            with torch.no_grad():
                #truncate last tokens of context_length
                logits = model(ids[:, -context_length:])
            #focus only on the last timestep
            #-1: (batch, num_tokens, vacab_size) -> (batch, vocab_size)
            logits = logits[:, -1, :] #

            if top_k is not None:
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(
                    logits < min_val
                    ,torch.tensor(float("-inf")).to(logits.device)
                    ,logits
                )

            if temperature > 0.0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                ids_next = torch.multinomial(probs, num_samples=1)
            else:
                ids_next = torch.argmax(logits, dim=-1, keepdim=True)

            if ids_next == eos_id:
                break

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
        encode_ids = tokenizer.encode(start_context, allowed_special={'<|endoftext|>'})
        encode_tensor = torch.tensor(encode_ids).unsqueeze(0)

        if eos_id is None:
            eos_id = tokenizer.eos_id()

        out_ids = self.__generate(
            model
            ,encode_tensor
            ,max_generate_tokens
            ,context_length=context_length
            ,temperature=temperature
            ,top_k=top_k
            ,eos_id=eos_id
        )

        out_ids_flat = out_ids.squeeze(0)

        return tokenizer.decode(out_ids_flat.tolist()).strip()

