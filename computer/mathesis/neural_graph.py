from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from typing import Iterable


INTENTS = (
    "arithmetic",
    "equation",
    "identity",
    "synthesize_program",
    "research_claim",
    "evolve_model",
    "unknown",
)


def _features(text: str, size: int = 64) -> list[float]:
    out = [0.0] * size
    normalized = " ".join(str(text or "").lower().split())
    units = normalized.split()
    units += [normalized[i : i + 2] for i in range(max(0, len(normalized) - 1))]
    for unit in units:
        digest = hashlib.sha256(unit.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % size
        sign = 1.0 if digest[4] & 1 else -1.0
        out[idx] += sign
    norm = math.sqrt(sum(x * x for x in out)) or 1.0
    return [x / norm for x in out]


def _softmax(values: list[float]) -> list[float]:
    peak = max(values)
    exp = [math.exp(max(-40.0, min(40.0, value - peak))) for value in values]
    total = sum(exp) or 1.0
    return [value / total for value in exp]


@dataclass
class GrowingNeuralRouter:
    input_size: int = 64
    hidden_size: int = 16
    output_size: int = len(INTENTS)
    seed: int = 1337
    w1: list[list[float]] | None = None
    b1: list[float] | None = None
    w2: list[list[float]] | None = None
    b2: list[float] | None = None

    def __post_init__(self):
        if self.w1 is None:
            rng = random.Random(self.seed)
            self.w1 = [
                [rng.uniform(-0.08, 0.08) for _ in range(self.input_size)]
                for _ in range(self.hidden_size)
            ]
            self.b1 = [0.0] * self.hidden_size
            self.w2 = [
                [rng.uniform(-0.08, 0.08) for _ in range(self.hidden_size)]
                for _ in range(self.output_size)
            ]
            self.b2 = [0.0] * self.output_size

    def _forward(self, x: list[float]):
        hidden = [
            math.tanh(sum(w * value for w, value in zip(row, x)) + self.b1[i])
            for i, row in enumerate(self.w1)
        ]
        logits = [
            sum(w * value for w, value in zip(row, hidden)) + self.b2[i]
            for i, row in enumerate(self.w2)
        ]
        return hidden, _softmax(logits)

    def predict(self, text: str) -> tuple[str, float]:
        _, probs = self._forward(_features(text, self.input_size))
        idx = max(range(len(probs)), key=probs.__getitem__)
        return INTENTS[idx], probs[idx]

    def train(self, samples: Iterable[tuple[str, str]], *, epochs: int = 80, lr: float = 0.08) -> None:
        rows = [(text, INTENTS.index(label)) for text, label in samples if label in INTENTS]
        if not rows:
            return
        rng = random.Random(self.seed + self.hidden_size)
        for _ in range(max(1, int(epochs))):
            shuffled = list(rows)
            rng.shuffle(shuffled)
            for text, target in shuffled:
                x = _features(text, self.input_size)
                hidden, probs = self._forward(x)
                dlogits = list(probs)
                dlogits[target] -= 1.0

                old_w2 = [row[:] for row in self.w2]
                for o in range(self.output_size):
                    for h in range(self.hidden_size):
                        self.w2[o][h] -= lr * dlogits[o] * hidden[h]
                    self.b2[o] -= lr * dlogits[o]

                dhidden = []
                for h in range(self.hidden_size):
                    upstream = sum(old_w2[o][h] * dlogits[o] for o in range(self.output_size))
                    dhidden.append(upstream * (1.0 - hidden[h] * hidden[h]))

                for h in range(self.hidden_size):
                    for i in range(self.input_size):
                        self.w1[h][i] -= lr * dhidden[h] * x[i]
                    self.b1[h] -= lr * dhidden[h]

    def accuracy(self, samples: Iterable[tuple[str, str]]) -> float:
        rows = list(samples)
        if not rows:
            return 0.0
        correct = sum(self.predict(text)[0] == label for text, label in rows)
        return correct / len(rows)

    def grow(self, add_hidden: int = 4) -> "GrowingNeuralRouter":
        add_hidden = max(1, min(32, int(add_hidden)))
        new_hidden = self.hidden_size + add_hidden
        grown = GrowingNeuralRouter(
            input_size=self.input_size,
            hidden_size=new_hidden,
            output_size=self.output_size,
            seed=self.seed + 17,
        )
        for h in range(self.hidden_size):
            grown.w1[h] = self.w1[h][:]
            grown.b1[h] = self.b1[h]
        for o in range(self.output_size):
            for h in range(self.hidden_size):
                grown.w2[o][h] = self.w2[o][h]
            grown.b2[o] = self.b2[o]
        return grown

    def to_dict(self) -> dict:
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "seed": self.seed,
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
        }

    @classmethod
    def from_dict(cls, value: dict) -> "GrowingNeuralRouter":
        return cls(
            input_size=int(value["input_size"]),
            hidden_size=int(value["hidden_size"]),
            output_size=int(value["output_size"]),
            seed=int(value.get("seed", 1337)),
            w1=[[float(x) for x in row] for row in value["w1"]],
            b1=[float(x) for x in value["b1"]],
            w2=[[float(x) for x in row] for row in value["w2"]],
            b2=[float(x) for x in value["b2"]],
        )
