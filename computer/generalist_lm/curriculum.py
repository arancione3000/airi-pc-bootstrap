from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .training import SFTExample


@dataclass(frozen=True)
class ResearchRow:
    domain: str
    messages: list[dict[str, str]]

    def sft(self) -> SFTExample:
        return SFTExample(self.messages)


def train_rows() -> list[ResearchRow]:
    return [
        ResearchRow("language", [{"role":"user","content":"Reply with the word blue."},{"role":"assistant","content":"blue"}]),
        ResearchRow("language", [{"role":"user","content":"Rispondi con la parola ciao."},{"role":"assistant","content":"ciao"}]),
        ResearchRow("coding", [{"role":"user","content":"Python function square(x). Return code only."},{"role":"assistant","content":"def square(x):\n    return x * x"}]),
        ResearchRow("coding", [{"role":"user","content":"Python function add(a,b). Return code only."},{"role":"assistant","content":"def add(a, b):\n    return a + b"}]),
        ResearchRow("data", [{"role":"user","content":"Mean of 2,4,6? Number only."},{"role":"assistant","content":"4"}]),
        ResearchRow("data", [{"role":"user","content":"Sum of 3,5,7? Number only."},{"role":"assistant","content":"15"}]),
        ResearchRow("reasoning", [{"role":"user","content":"What is 17*19? Number only."},{"role":"assistant","content":"323"}]),
        ResearchRow("reasoning", [{"role":"user","content":"If all A are B and x is A, is x B? yes/no"},{"role":"assistant","content":"yes"}]),
        ResearchRow("tools", [{"role":"user","content":"Use calculator for 8*9."},{"role":"assistant","content":"<tool_call>{\"name\":\"calculator\",\"arguments\":{\"expression\":\"8*9\"}}</tool_call>"}]),
        ResearchRow("structured", [{"role":"user","content":"Return JSON with ok true."},{"role":"assistant","content":"{\"ok\":true}"}]),
    ]


def validation_rows() -> list[ResearchRow]:
    return [
        ResearchRow("language", [{"role":"user","content":"Reply with the word orange."},{"role":"assistant","content":"orange"}]),
        ResearchRow("coding", [{"role":"user","content":"Python function sub(a,b). Return code only."},{"role":"assistant","content":"def sub(a, b):\n    return a - b"}]),
        ResearchRow("data", [{"role":"user","content":"Mean of 10,20,30? Number only."},{"role":"assistant","content":"20"}]),
        ResearchRow("reasoning", [{"role":"user","content":"What is 12*12? Number only."},{"role":"assistant","content":"144"}]),
        ResearchRow("tools", [{"role":"user","content":"Use calculator for 6*7."},{"role":"assistant","content":"<tool_call>{\"name\":\"calculator\",\"arguments\":{\"expression\":\"6*7\"}}</tool_call>"}]),
        ResearchRow("structured", [{"role":"user","content":"Return JSON with count 2."},{"role":"assistant","content":"{\"count\":2}"}]),
    ]
