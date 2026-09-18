from __future__ import annotations

from typing import Callable

from .genome import BlockGene, Genome


def _torch():
    import torch
    from torch import nn
    return torch, nn


def activation(name: str):
    _, nn = _torch()
    return {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }.get(name, nn.GELU)()


def build_model(genome: Genome, vocab_size: int = 8192):
    torch, nn = _torch()

    class ConvBlock(nn.Module):
        def __init__(self, in_dim: int, gene: BlockGene):
            super().__init__()
            self.proj = nn.Identity() if in_dim == gene.width else nn.Linear(in_dim, gene.width)
            self.conv = nn.Conv1d(gene.width, gene.width, kernel_size=gene.kernel, padding=gene.kernel // 2, groups=1)
            self.norm = nn.LayerNorm(gene.width)
            self.act = activation(gene.activation)
            self.drop = nn.Dropout(gene.dropout)
            self.residual = gene.residual
        def forward(self, x, mask):
            base = self.proj(x)
            y = self.conv(base.transpose(1, 2)).transpose(1, 2)
            y = self.drop(self.act(y))
            y = self.norm(y + base) if self.residual else self.norm(y)
            return y * mask.unsqueeze(-1)

    class AttentionBlock(nn.Module):
        def __init__(self, in_dim: int, gene: BlockGene):
            super().__init__()
            self.proj = nn.Identity() if in_dim == gene.width else nn.Linear(in_dim, gene.width)
            heads = max(h for h in (1, 2, 4) if h <= gene.heads and gene.width % h == 0)
            self.attn = nn.MultiheadAttention(gene.width, heads, dropout=gene.dropout, batch_first=True)
            self.norm = nn.LayerNorm(gene.width)
            self.drop = nn.Dropout(gene.dropout)
            self.residual = gene.residual
        def forward(self, x, mask):
            base = self.proj(x)
            y, _ = self.attn(base, base, base, key_padding_mask=~mask, need_weights=False)
            y = self.drop(y)
            y = self.norm(y + base) if self.residual else self.norm(y)
            return y * mask.unsqueeze(-1)

    class GRUBlock(nn.Module):
        def __init__(self, in_dim: int, gene: BlockGene):
            super().__init__()
            self.gru = nn.GRU(in_dim, gene.width, batch_first=True)
            self.proj = nn.Identity() if in_dim == gene.width else nn.Linear(in_dim, gene.width)
            self.norm = nn.LayerNorm(gene.width)
            self.drop = nn.Dropout(gene.dropout)
            self.residual = gene.residual
        def forward(self, x, mask):
            y, _ = self.gru(x)
            y = self.drop(y)
            base = self.proj(x)
            y = self.norm(y + base) if self.residual else self.norm(y)
            return y * mask.unsqueeze(-1)

    class MLPBlock(nn.Module):
        def __init__(self, in_dim: int, gene: BlockGene):
            super().__init__()
            hidden = max(gene.width, min(256, gene.width * 2))
            self.proj = nn.Identity() if in_dim == gene.width else nn.Linear(in_dim, gene.width)
            self.ff = nn.Sequential(
                nn.Linear(gene.width, hidden), activation(gene.activation), nn.Dropout(gene.dropout),
                nn.Linear(hidden, gene.width), nn.Dropout(gene.dropout),
            )
            self.norm = nn.LayerNorm(gene.width)
            self.residual = gene.residual
        def forward(self, x, mask):
            base = self.proj(x)
            y = self.ff(base)
            y = self.norm(y + base) if self.residual else self.norm(y)
            return y * mask.unsqueeze(-1)

    block_map: dict[str, Callable] = {
        "conv": ConvBlock,
        "attention": AttentionBlock,
        "gru": GRUBlock,
        "mlp": MLPBlock,
    }

    class EvolvedTextClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, genome.embed_dim, padding_idx=0)
            modules = []
            dim = genome.embed_dim
            for gene in genome.blocks:
                modules.append(block_map[gene.kind](dim, gene))
                dim = gene.width
            self.blocks = nn.ModuleList(modules)
            self.final_norm = nn.LayerNorm(dim)
            self.dropout = nn.Dropout(genome.dropout)
            self.head = nn.Linear(dim, 2)
        def forward(self, ids, mask):
            x = self.embedding(ids)
            x = x * mask.unsqueeze(-1)
            for block in self.blocks:
                x = block(x, mask)
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
            pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / denom
            return self.head(self.dropout(self.final_norm(pooled)))

    return EvolvedTextClassifier()


def parameter_count(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
