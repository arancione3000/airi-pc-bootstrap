from __future__ import annotations

from dataclasses import asdict, dataclass, field
import copy
import random
from typing import Any

BLOCK_TYPES = ("conv", "attention", "gru", "mlp")
ACTIVATIONS = ("relu", "gelu", "silu", "tanh")
WIDTHS_SAFE = (32, 48, 64, 96)
WIDTHS_EXPERIMENTAL = (24, 32, 48, 64, 96, 128)
KERNELS = (3, 5, 7)
HEADS = (1, 2, 4)


@dataclass
class BlockGene:
    kind: str
    width: int
    activation: str = "gelu"
    dropout: float = 0.1
    kernel: int = 3
    heads: int = 2
    residual: bool = True

    def normalized(self) -> "BlockGene":
        self.kind = self.kind if self.kind in BLOCK_TYPES else "mlp"
        self.width = max(8, int(self.width))
        self.activation = self.activation if self.activation in ACTIVATIONS else "gelu"
        self.dropout = min(0.5, max(0.0, float(self.dropout)))
        self.kernel = int(self.kernel) if int(self.kernel) in KERNELS else 3
        possible = [h for h in HEADS if self.width % h == 0]
        self.heads = int(self.heads) if int(self.heads) in possible else possible[0]
        self.residual = bool(self.residual)
        return self


@dataclass
class Genome:
    genome_id: str
    embed_dim: int = 48
    max_len: int = 128
    blocks: list[BlockGene] = field(default_factory=list)
    dropout: float = 0.1
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    mutation_rate: float = 0.35
    structural_rate: float = 0.30
    generation: int = 0
    parents: list[str] = field(default_factory=list)

    def normalize(self, *, mode: str = "safe") -> "Genome":
        widths = WIDTHS_EXPERIMENTAL if mode == "experimental" else WIDTHS_SAFE
        self.embed_dim = min(widths, key=lambda x: abs(x - int(self.embed_dim)))
        self.max_len = min((64, 96, 128, 192, 256), key=lambda x: abs(x - int(self.max_len)))
        self.dropout = min(0.5, max(0.0, float(self.dropout)))
        self.learning_rate = min(1e-2, max(1e-5, float(self.learning_rate)))
        self.weight_decay = min(1e-2, max(0.0, float(self.weight_decay)))
        self.batch_size = min((8, 16, 24, 32, 48, 64), key=lambda x: abs(x - int(self.batch_size)))
        self.mutation_rate = min(0.95, max(0.05, float(self.mutation_rate)))
        self.structural_rate = min(0.95, max(0.05, float(self.structural_rate)))
        max_blocks = 7 if mode == "experimental" else 4
        self.blocks = [b.normalized() for b in self.blocks[:max_blocks]]
        if not self.blocks:
            self.blocks = [BlockGene("conv", self.embed_dim, "gelu", self.dropout, 3, 1, True)]
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Genome":
        data = dict(raw)
        data["blocks"] = [BlockGene(**b) for b in data.get("blocks", [])]
        return cls(**data)

    def clone(self, new_id: str | None = None) -> "Genome":
        g = copy.deepcopy(self)
        if new_id:
            g.genome_id = new_id
        return g

    def signature(self) -> tuple:
        return (
            self.embed_dim,
            self.max_len,
            tuple((b.kind, b.width, b.activation, round(b.dropout, 3), b.kernel, b.heads, b.residual) for b in self.blocks),
        )


def _rand_block(rng: random.Random, width: int, mode: str) -> BlockGene:
    widths = WIDTHS_EXPERIMENTAL if mode == "experimental" else WIDTHS_SAFE
    out = rng.choice(widths)
    heads = rng.choice([h for h in HEADS if out % h == 0])
    return BlockGene(
        kind=rng.choice(BLOCK_TYPES),
        width=out if rng.random() < 0.75 else width,
        activation=rng.choice(ACTIVATIONS),
        dropout=round(rng.uniform(0.0, 0.25), 3),
        kernel=rng.choice(KERNELS),
        heads=heads,
        residual=rng.random() < 0.8,
    ).normalized()


def random_genome(rng: random.Random, genome_id: str, mode: str = "safe", generation: int = 0) -> Genome:
    widths = WIDTHS_EXPERIMENTAL if mode == "experimental" else WIDTHS_SAFE
    embed = rng.choice(widths)
    block_count = rng.randint(1, 5 if mode == "experimental" else 3)
    blocks: list[BlockGene] = []
    width = embed
    for _ in range(block_count):
        b = _rand_block(rng, width, mode)
        blocks.append(b)
        width = b.width
    return Genome(
        genome_id=genome_id,
        embed_dim=embed,
        max_len=rng.choice((64, 96, 128, 192)),
        blocks=blocks,
        dropout=round(rng.uniform(0.05, 0.25), 3),
        learning_rate=10 ** rng.uniform(-3.4, -2.3),
        weight_decay=10 ** rng.uniform(-6.0, -3.0),
        batch_size=rng.choice((16, 24, 32, 48)),
        mutation_rate=round(rng.uniform(0.2, 0.5), 3),
        structural_rate=round(rng.uniform(0.2, 0.45), 3),
        generation=generation,
    ).normalize(mode=mode)


def mutate(parent: Genome, rng: random.Random, new_id: str, mode: str = "safe") -> Genome:
    g = parent.clone(new_id)
    g.parents = [parent.genome_id]
    g.generation = parent.generation + 1
    widths = WIDTHS_EXPERIMENTAL if mode == "experimental" else WIDTHS_SAFE
    max_blocks = 7 if mode == "experimental" else 4

    # Self-adaptation is deliberately stronger in experimental mode.
    if mode == "experimental" and rng.random() < 0.45:
        g.mutation_rate *= rng.uniform(0.75, 1.35)
        g.structural_rate *= rng.uniform(0.70, 1.45)

    if rng.random() < g.mutation_rate:
        g.embed_dim = rng.choice(widths)
    if rng.random() < g.mutation_rate * 0.6:
        g.max_len = rng.choice((64, 96, 128, 192, 256 if mode == "experimental" else 192))
    if rng.random() < g.mutation_rate:
        g.dropout += rng.uniform(-0.06, 0.06)
    if rng.random() < g.mutation_rate:
        g.learning_rate *= 10 ** rng.uniform(-0.25, 0.25)
    if rng.random() < g.mutation_rate * 0.7:
        g.weight_decay *= 10 ** rng.uniform(-0.5, 0.5)
    if rng.random() < g.mutation_rate * 0.5:
        g.batch_size = rng.choice((8, 16, 24, 32, 48, 64))

    if rng.random() < g.structural_rate and len(g.blocks) < max_blocks:
        pos = rng.randint(0, len(g.blocks))
        prev_width = g.embed_dim if pos == 0 else g.blocks[pos - 1].width
        g.blocks.insert(pos, _rand_block(rng, prev_width, mode))
    if rng.random() < g.structural_rate * 0.55 and len(g.blocks) > 1:
        del g.blocks[rng.randrange(len(g.blocks))]
    if rng.random() < g.structural_rate * 0.35 and len(g.blocks) > 1:
        rng.shuffle(g.blocks)

    for b in g.blocks:
        if rng.random() < g.mutation_rate:
            b.kind = rng.choice(BLOCK_TYPES)
        if rng.random() < g.mutation_rate:
            b.width = rng.choice(widths)
        if rng.random() < g.mutation_rate:
            b.activation = rng.choice(ACTIVATIONS)
        if rng.random() < g.mutation_rate * 0.8:
            b.dropout += rng.uniform(-0.08, 0.08)
        if rng.random() < g.mutation_rate * 0.6:
            b.kernel = rng.choice(KERNELS)
        if rng.random() < g.mutation_rate * 0.6:
            b.heads = rng.choice(HEADS)
        if rng.random() < g.mutation_rate * 0.3:
            b.residual = not b.residual
    return g.normalize(mode=mode)


def crossover(a: Genome, b: Genome, rng: random.Random, new_id: str, mode: str = "safe") -> Genome:
    if not a.blocks:
        return mutate(b, rng, new_id, mode)
    if not b.blocks:
        return mutate(a, rng, new_id, mode)
    cut_a = rng.randint(1, len(a.blocks))
    cut_b = rng.randint(0, len(b.blocks) - 1)
    blocks = copy.deepcopy(a.blocks[:cut_a] + b.blocks[cut_b:])
    child = Genome(
        genome_id=new_id,
        embed_dim=rng.choice((a.embed_dim, b.embed_dim)),
        max_len=rng.choice((a.max_len, b.max_len)),
        blocks=blocks,
        dropout=(a.dropout + b.dropout) / 2,
        learning_rate=(a.learning_rate * b.learning_rate) ** 0.5,
        weight_decay=(max(a.weight_decay, 1e-8) * max(b.weight_decay, 1e-8)) ** 0.5,
        batch_size=rng.choice((a.batch_size, b.batch_size)),
        mutation_rate=(a.mutation_rate + b.mutation_rate) / 2,
        structural_rate=(a.structural_rate + b.structural_rate) / 2,
        generation=max(a.generation, b.generation) + 1,
        parents=[a.genome_id, b.genome_id],
    ).normalize(mode=mode)
    return child
