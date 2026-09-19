from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from .types import ProofCell


@dataclass
class ProofCellGraph:
    max_cells: int = 64
    cells: dict[str, ProofCell] = field(default_factory=dict)
    root_id: str | None = None

    def _id(self, goal: str, strategy: str, parent: str | None) -> str:
        payload = f"{goal}|{strategy}|{parent or ''}|{len(self.cells)}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def add(
        self,
        goal: str,
        *,
        strategy: str = "unassigned",
        assumptions: list[str] | None = None,
        parent: str | None = None,
    ) -> ProofCell:
        if len(self.cells) >= self.max_cells:
            raise RuntimeError("proof-cell budget exhausted")
        cell_id = self._id(goal, strategy, parent)
        cell = ProofCell(
            cell_id=cell_id,
            goal=str(goal),
            assumptions=list(assumptions or []),
            strategy=strategy,
            dependencies=[parent] if parent else [],
        )
        self.cells[cell_id] = cell
        if self.root_id is None:
            self.root_id = cell_id
        return cell

    def complete(self, cell_id: str, result: Any, *, status: str = "proved") -> ProofCell:
        cell = self.cells[cell_id]
        cell.status = status
        cell.result = result
        return cell

    def fail(self, cell_id: str, result: Any) -> ProofCell:
        return self.complete(cell_id, result, status="failed")

    def pending(self) -> list[ProofCell]:
        return [cell for cell in self.cells.values() if cell.status == "pending"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_id": self.root_id,
            "max_cells": self.max_cells,
            "cells": [cell.to_dict() for cell in self.cells.values()],
        }
