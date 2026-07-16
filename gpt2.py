import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

class CasualSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)   # Key, query and value projections but in a batch
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)       # Output projection
        self.c_proj.NANOGPT_SCALE_INIT = 1.0                        # Special normalization for residual streams

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer("bias", torch.tril(torch.ones((config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)))

    def forward(self, x):
        B, T, C = x.size() # Batch, sequence length, embedding dimension

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Rearrange k,q,v so both B and bh become batch dimensions.
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Self attention (code is commented for better understanding)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # att = F.softmax(att, dim=-1)
        #y = att @ v
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # Same thing as above but optimized

        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Final projection
        y = self.c_proj(y)

        return y

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)         # Linear layer (n -> 4n)
        self.gelu = nn.GELU("tanh")                                     # GELU activation with tanh approximation. Approximation is no longer needed today, but original GPT-2 used tanh.
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)       # Linear layer (4n -> n)
        self.c_proj.NANOGPT_SCALE_INIT = 1.0                            # Special normalization for residual streams

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # We apply layer normalization before attention to have better gradient flow in the residual stream.
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),                   # Token Embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd),                   # Positional Encoding
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),      # Hidden
            ln_f = nn.LayerNorm(config.n_embd)                                      # Final Layer Normalization
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)      # Classifier

        # Weight sharing
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # PyTorch initializes LayerNorms just as GPT-2 so no need to init them
        # Also we are initializing wte.weight but Karpathy said that's okay
        if isinstance(module, nn.Linear):
            std = 0.02 # Standard number used for GPT-2, in reality we want to use sqrt(n_embd) but 0.2 is close enough
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2.0 * self.config.n_layer) ** -0.5 # Special normalization for residual streams
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # Token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # (T)
        pos_emb = self.transformer.wpe(pos) # (T, n_embd)
        tok_emb = self.transformer.wte(idx) # (B, T, n_embd)
        x = pos_emb + tok_emb

        # Forward the blocks of transformer
        for block in self.transformer.h:
            x = block(x)

        # Final projection
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # Loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss