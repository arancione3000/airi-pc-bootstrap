from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from coding import ROOT
from .store import now, redact

SAFE_EXECUTABLES = {
    "python", "python3", "pytest", "ruff", "mypy", "pyright",
    "npm", "pnpm", "yarn", "node", "go", "cargo", "dotnet", "git",
}
SKIP_DIRS = {".git", ".ai", ".venv", "node_modules", "__pycache__", ".pytest_cache", ".cache", "logs"}


class TrustedJudge:
    """Independent fail-closed verifier executed against an isolated copy.

    The implementation loop cannot pass merely by asserting success: Judge runs
    explicit checks in a throw-away sandbox and emits a content-addressed
    attestation. Test/build commands may mutate the sandbox but never the live
    workspace.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or ROOT).resolve()

    def _resolve(self, project_path: str) -> Path:
        raw = Path(project_path).expanduser()
        path = raw.resolve() if raw.is_absolute() else (self.root / raw).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("judge project outside workspace") from exc
        return path

    @staticmethod
    def _run(argv: list[str], cwd: Path, timeout: int) -> dict[str, Any]:
        if not argv:
            raise ValueError("empty command")
        exe = Path(argv[0]).name.lower()
        if exe not in SAFE_EXECUTABLES:
            raise PermissionError(f"judge executable not allowed: {exe}")
        env = os.environ.copy()
        env.update({
            "PYTHONDONTWRITEBYTECODE": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "GIT_CONFIG_NOSYSTEM": "1",
            "AIRI_JUDGE_SANDBOX": "1",
        })
        started = time.monotonic()
        process = subprocess.run(
            argv,
            cwd=str(cwd),
            env=env,
            text=True,
            capture_output=True,
            timeout=max(1, min(int(timeout), 300)),
            check=False,
        )
        return {
            "argv": argv,
            "returncode": process.returncode,
            "stdout": process.stdout[-12000:],
            "stderr": process.stderr[-12000:],
            "duration_ms": round((time.monotonic() - started) * 1000, 2),
        }

    @staticmethod
    def _git_changes(cwd: Path) -> list[str]:
        process = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=str(cwd), text=True, capture_output=True, timeout=30, check=False,
        )
        if process.returncode != 0:
            return []
        changes = []
        for line in process.stdout.splitlines():
            if len(line) >= 4:
                path = line[3:]
                if " -> " in path:
                    path = path.split(" -> ", 1)[1]
                changes.append(path.strip())
        return sorted(set(changes))

    @staticmethod
    def _copy_project(source: Path, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        for root, dirs, files in os.walk(source, topdown=True, followlinks=False):
            root_path = Path(root)
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not (root_path / d).is_symlink()]
            rel = root_path.relative_to(source)
            out_dir = destination / rel
            out_dir.mkdir(parents=True, exist_ok=True)
            for name in files:
                src = root_path / name
                if src.is_symlink() or name.endswith((".pyc", ".pyo")):
                    continue
                dst = out_dir / name
                try:
                    shutil.copy2(src, dst)
                except (OSError, PermissionError):
                    continue

    @staticmethod
    def _sandbox_path(sandbox: Path, value: str) -> Path:
        candidate = Path(value)
        candidate = (sandbox / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
        try:
            candidate.relative_to(sandbox.resolve())
        except ValueError as exc:
            raise ValueError("judge check path outside sandbox") from exc
        return candidate

    def evaluate(
        self,
        *,
        goal: str,
        project_path: str = ".",
        checks: list[dict[str, Any]] | None = None,
        forbidden_paths: list[str] | None = None,
        require_checks: bool = True,
    ) -> dict[str, Any]:
        source = self._resolve(project_path)
        checks = list(checks or [])
        if require_checks and not checks:
            return {"ok": False, "verdict": "FAIL", "reason": "independent checks required", "goal": goal}
        live_changes = self._git_changes(source)
        forbidden = [
            path for path in live_changes
            if any(path == f or path.startswith(str(f).rstrip("/") + "/") for f in (forbidden_paths or []))
        ]
        outcomes: list[dict[str, Any]] = []
        with tempfile.TemporaryDirectory(prefix="airi-judge-") as temp:
            sandbox = Path(temp) / "project"
            self._copy_project(source, sandbox)
            for check in checks:
                kind = check.get("type")
                try:
                    if kind == "command":
                        raw = check.get("argv") or check.get("command")
                        argv = list(raw) if isinstance(raw, list) else shlex.split(str(raw or ""))
                        result = self._run(argv, sandbox, int(check.get("timeout", 120)))
                        ok = result["returncode"] == int(check.get("expect_returncode", 0))
                        outcomes.append({"type": kind, "ok": ok, "result": redact(result)})
                    elif kind == "file_exists":
                        path = self._sandbox_path(sandbox, str(check["path"]))
                        outcomes.append({"type": kind, "ok": path.exists(), "path": str(check["path"])})
                    elif kind == "file_contains":
                        path = self._sandbox_path(sandbox, str(check["path"]))
                        needle = str(check["text"])
                        text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
                        outcomes.append({"type": kind, "ok": needle in text, "path": str(check["path"]), "needle": needle[:200]})
                    elif kind == "json_key":
                        path = self._sandbox_path(sandbox, str(check["path"]))
                        data = json.loads(path.read_text(encoding="utf-8"))
                        current: Any = data
                        for key in str(check["key"]).split("."):
                            current = current[key]
                        ok = "equals" not in check or current == check.get("equals")
                        outcomes.append({"type": kind, "ok": ok, "path": str(check["path"]), "value": redact(current)})
                    else:
                        outcomes.append({"type": str(kind), "ok": False, "error": "unsupported judge check"})
                except Exception as exc:
                    outcomes.append({"type": str(kind), "ok": False, "error": str(exc)})
        ok = bool(outcomes) and all(item.get("ok") is True for item in outcomes) and not forbidden
        report = {
            "ok": ok,
            "verdict": "PASS" if ok else "FAIL",
            "goal": goal,
            "checks": outcomes,
            "forbidden_changes": forbidden,
            "live_changes_observed": live_changes,
            "sandboxed": True,
            "live_workspace_mutated_by_judge": False,
            "timestamp": now(),
        }
        canonical = json.dumps(redact(report), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        report["attestation_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return redact(report)
