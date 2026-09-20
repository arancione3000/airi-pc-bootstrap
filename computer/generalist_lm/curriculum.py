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


DOMAINS = ("language", "coding", "data", "reasoning", "tools", "structured")


def _language_rows(words: list[str]) -> list[ResearchRow]:
    return [
        ResearchRow(
            "language",
            [
                {"role": "user", "content": f"Reply with exactly the word {word.upper()} and nothing else."},
                {"role": "assistant", "content": word.upper()},
            ],
        )
        for word in words
    ]


def _coding_rows(prefix: str, constants: Iterable[int], *, operation: str) -> list[ResearchRow]:
    rows: list[ResearchRow] = []
    for value in constants:
        name = f"{prefix}_{value}"
        if operation == "add":
            body = f"def {name}(x):\n    return x + {value}"
            prompt = f"Return only Python code defining {name}(x) that adds {value}."
        elif operation == "multiply":
            body = f"def {name}(x):\n    return x * {value}"
            prompt = f"Return only Python code defining {name}(x) that multiplies by {value}."
        else:
            raise ValueError(operation)
        rows.append(ResearchRow("coding", [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": body},
        ]))
    return rows


def _data_rows(starts: Iterable[int]) -> list[ResearchRow]:
    rows: list[ResearchRow] = []
    for start in starts:
        values = [start, start + 2, start + 4]
        mean = start + 2
        rows.append(ResearchRow("data", [
            {"role": "user", "content": f"Mean of {values[0]},{values[1]},{values[2]}? Number only."},
            {"role": "assistant", "content": str(mean)},
        ]))
    return rows


def _reasoning_rows(pairs: Iterable[tuple[int, int]]) -> list[ResearchRow]:
    return [
        ResearchRow("reasoning", [
            {"role": "user", "content": f"What is {a}*{b}? Number only."},
            {"role": "assistant", "content": str(a * b)},
        ])
        for a, b in pairs
    ]


def _tool_rows(expressions: Iterable[tuple[int, str, int]]) -> list[ResearchRow]:
    rows: list[ResearchRow] = []
    for left, op, right in expressions:
        expr = f"{left}{op}{right}"
        rows.append(ResearchRow("tools", [
            {"role": "user", "content": f"Use calculator for {expr}."},
            {"role": "assistant", "content": f'<tool_call>{{"name":"calculator","arguments":{{"expression":"{expr}"}}}}</tool_call>'},
        ]))
    return rows


def _structured_rows(values: Iterable[int]) -> list[ResearchRow]:
    return [
        ResearchRow("structured", [
            {"role": "user", "content": f"Return JSON with value {value} and ok true."},
            {"role": "assistant", "content": f'{{"value":{value},"ok":true}}'},
        ])
        for value in values
    ]


def _generated_train_rows() -> list[ResearchRow]:
    rows: list[ResearchRow] = []
    rows.extend(_language_rows(["amber", "violet", "silver", "coral", "indigo", "pearl"]))
    rows.extend(_coding_rows("add_const", range(3, 9), operation="add"))
    rows.extend(_data_rows(range(31, 43, 2)))
    rows.extend(_reasoning_rows([(21, 3), (22, 4), (24, 5), (26, 3), (28, 4), (32, 3)]))
    rows.extend(_tool_rows([(14, "+", 9), (15, "*", 4), (28, "-", 11), (9, "*", 8), (44, "+", 7), (63, "-", 19)]))
    rows.extend(_structured_rows(range(11, 17)))
    return rows


def _generated_validation_rows() -> list[ResearchRow]:
    rows: list[ResearchRow] = []
    rows.extend(_language_rows(["teal", "crimson", "ivory", "mint"]))
    rows.extend(_coding_rows("mul_const", range(3, 7), operation="multiply"))
    rows.extend(_data_rows(range(51, 59, 2)))
    rows.extend(_reasoning_rows([(33, 3), (27, 5), (34, 4), (29, 6)]))
    rows.extend(_tool_rows([(37, "+", 8), (12, "*", 6), (71, "-", 23), (16, "*", 5)]))
    rows.extend(_structured_rows(range(21, 25)))
    return rows


def train_rows() -> list[ResearchRow]:
    base = [
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
    return base + _generated_train_rows()


def validation_rows() -> list[ResearchRow]:
    base = [
        ResearchRow("language", [{"role":"user","content":"Reply with the word orange."},{"role":"assistant","content":"orange"}]),
        ResearchRow("coding", [{"role":"user","content":"Python function sub(a,b). Return code only."},{"role":"assistant","content":"def sub(a, b):\n    return a - b"}]),
        ResearchRow("data", [{"role":"user","content":"Mean of 10,20,30? Number only."},{"role":"assistant","content":"20"}]),
        ResearchRow("reasoning", [{"role":"user","content":"What is 12*12? Number only."},{"role":"assistant","content":"144"}]),
        ResearchRow("tools", [{"role":"user","content":"Use calculator for 6*7."},{"role":"assistant","content":"<tool_call>{\"name\":\"calculator\",\"arguments\":{\"expression\":\"6*7\"}}</tool_call>"}]),
        ResearchRow("structured", [{"role":"user","content":"Return JSON with count 2."},{"role":"assistant","content":"{\"count\":2}"}]),
    ]
    return base + _generated_validation_rows()


def curriculum_manifest() -> dict:
    train = train_rows()
    validation = validation_rows()
    train_prompts = {row.messages[0]["content"] for row in train}
    validation_prompts = {row.messages[0]["content"] for row in validation}
    return {
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_domains": {domain: sum(row.domain == domain for row in train) for domain in DOMAINS},
        "validation_domains": {domain: sum(row.domain == domain for row in validation) for domain in DOMAINS},
        "prompt_overlap": sorted(train_prompts & validation_prompts),
        "mechanically_labeled": True,
    }
