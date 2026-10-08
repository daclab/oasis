import torch
import torch.nn.functional as F
from torch import nn


class attention(nn.Module):
    """Multi-head self-attention (from Stream-HLS MultiHeadSelfAttention); optional causal mask."""

    def __init__(self, embed_dim=32, num_heads=4, causal=False):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.causal = causal
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch, seq, embed_dim = x.size()
        q = self.query(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.head_dim**0.5
        if self.causal:
            future = torch.ones(seq, seq, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(future, float("-inf"))
        context = torch.matmul(F.softmax(scores, dim=-1), v)
        context = context.transpose(1, 2).contiguous().view(batch, seq, embed_dim)
        return self.out(context)


class transformer_block(nn.Module):
    """Pre-LayerNorm transformer block (GPT-2 layout): attention + feed-forward, residuals.

    The feed-forward uses ReLU, like ffnn, instead of GELU (no math.erf).
    """

    def __init__(self, embed_dim=32, num_heads=4, ff_dim=64, causal=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = attention(embed_dim, num_heads, causal)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff1 = nn.Linear(embed_dim, ff_dim)
        self.ff2 = nn.Linear(ff_dim, embed_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff2(F.relu(self.ff1(self.norm2(x))))
        return x


class tiny_llm(nn.Module):
    """Decoder-only language model: token + position embeddings, causal transformer blocks,
    final LayerNorm, LM head. forward(token_ids) -> next-token logits for every position."""

    def __init__(
        self, vocab_size=64, max_seq=16, embed_dim=32, num_heads=4, ff_dim=64, num_layers=2
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq, embed_dim)
        self.blocks = nn.ModuleList(
            [
                transformer_block(embed_dim, num_heads, ff_dim, causal=True)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, ids):
        seq = ids.size(1)
        x = self.tok_emb(ids) + self.pos_emb.weight[:seq]
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))
