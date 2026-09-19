from __future__ import annotations

import ast
import json
from pathlib import Path

from .types import ArchitectureGenome


def render_model_module(genome: ArchitectureGenome) -> str:
    payload = pprint.pformat(genome.to_dict(), sort_dicts=True, width=100)
    source = (
        '"""Generated MATHESIS-Ω architecture model.\n\n'
        "This file is data-bearing executable Python with no imports/calls. "
        "It is generated transactionally and never replaces the verifier kernel.\n"
        '"""\n\n'
        f"MODEL = {payload}\n"
    )
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Call, ast.FunctionDef, ast.ClassDef)):
            raise ValueError("generated model module contains executable behavior")
    return source


def write_model_module(path: str | Path, genome: ArchitectureGenome) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    source = render_model_module(genome)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(source, encoding="utf-8")
    tmp.replace(path)
