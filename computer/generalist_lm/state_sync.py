from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any

from .qualification import qualification_status, qualify_checkpoint


_SAFE_REF = re.compile(r"^[A-Za-z0-9._/-]+$")
_BASE_CHECKPOINT_FILES = ("config.json", "metadata.json", "benchmark.json")


def _validate_git_name(value: str, label: str) -> str:
    value = str(value).strip()
    if not value or value.startswith("-") or not _SAFE_REF.fullmatch(value):
        raise ValueError(f"invalid {label}")
    return value


def _run_git(repo_root: Path, args: list[str], *, stdout=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        stdout=stdout if stdout is not None else subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _ref_has_file(repo_root: Path, ref: str, source_path: str) -> bool:
    check = _run_git(repo_root, ["cat-file", "-e", f"{ref}:{source_path}"])
    return check.returncode == 0


def _extract_file(repo_root: Path, ref: str, source_path: str, target: Path) -> None:
    check = _run_git(repo_root, ["cat-file", "-e", f"{ref}:{source_path}"])
    if check.returncode != 0:
        raise FileNotFoundError(f"missing state checkpoint file: {source_path}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        shown = _run_git(repo_root, ["show", f"{ref}:{source_path}"], stdout=handle)
    if shown.returncode != 0:
        target.unlink(missing_ok=True)
        raise RuntimeError(f"failed to extract checkpoint file: {source_path}")


def sync_production_checkpoint(
    repo_root: str | Path,
    target_dir: str | Path,
    *,
    remote: str = "origin",
    branch: str = "generalist-state",
) -> dict[str, Any]:
    """Fetch and transactionally install a qualified production checkpoint.

    The source branch is treated as untrusted until the copied checkpoint passes
    the exact-digest qualification attestation. Existing local production stays
    untouched on fetch, extraction, or qualification failure.
    """
    repo_root = Path(repo_root).expanduser().resolve()
    target = Path(target_dir).expanduser().resolve()
    remote = _validate_git_name(remote, "remote")
    branch = _validate_git_name(branch, "branch")

    if not (repo_root / ".git").exists():
        raise FileNotFoundError("Airi-PC repository is not a Git working tree")

    ref = f"refs/remotes/{remote}/{branch}"
    fetched = _run_git(
        repo_root,
        ["fetch", "--quiet", remote, f"{branch}:{ref}"],
    )
    if fetched.returncode != 0:
        raise RuntimeError(
            "failed to fetch generalist state branch: "
            + fetched.stderr.decode("utf-8", errors="replace")[-2000:]
        )

    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".generalist-sync-", dir=str(parent)) as tmp:
        candidate = Path(tmp) / "candidate"
        candidate.mkdir()
        prefix = "generalist-state/production"
        for name in _BASE_CHECKPOINT_FILES:
            _extract_file(
                repo_root,
                ref,
                f"{prefix}/{name}",
                candidate / name,
            )

        config = json.loads((candidate / "config.json").read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("remote checkpoint config must be a JSON object")
        tokenizer_version = str(config.get("tokenizer_version", "byte-v1"))
        if tokenizer_version == "bpe-v1":
            _extract_file(
                repo_root,
                ref,
                f"{prefix}/tokenizer.json",
                candidate / "tokenizer.json",
            )
        elif tokenizer_version != "byte-v1":
            raise ValueError(
                f"unsupported remote checkpoint tokenizer: {tokenizer_version}"
            )

        if _ref_has_file(repo_root, ref, f"{prefix}/model.pt"):
            _extract_file(
                repo_root,
                ref,
                f"{prefix}/model.pt",
                candidate / "model.pt",
            )
        else:
            _extract_file(
                repo_root,
                ref,
                f"{prefix}/model.index.json",
                candidate / "model.index.json",
            )
            index = json.loads(
                (candidate / "model.index.json").read_text(encoding="utf-8")
            )
            shards = index.get("shards") if isinstance(index, dict) else None
            if not isinstance(shards, list) or not shards:
                raise ValueError("remote model shard index has no shards")
            seen = set()
            for row in shards:
                if not isinstance(row, dict):
                    raise ValueError("invalid remote model shard entry")
                name = str(row.get("file") or "")
                if (
                    not name
                    or name in seen
                    or Path(name).name != name
                    or not name.startswith("model-shard-")
                    or not name.endswith(".pt")
                ):
                    raise ValueError("unsafe remote model shard filename")
                seen.add(name)
                _extract_file(
                    repo_root,
                    ref,
                    f"{prefix}/{name}",
                    candidate / name,
                )

        # Never trust the remote benchmark JSON as the authority. Re-run the
        # protected qualification suite locally against the copied weights.
        minimum_score = float(os.environ.get("AIRI_GENERALIST_PRODUCTION_MIN_SCORE", "85"))
        status = qualify_checkpoint(candidate, minimum_score=minimum_score)
        if not status.get("qualified"):
            raise RuntimeError(
                "remote Generalist production checkpoint failed local requalification"
            )

        remote_digest = status.get("checkpoint_digest")
        current = qualification_status(target) if target.exists() else {"qualified": False}
        current_digest = (
            current.get("current_checkpoint_digest")
            or current.get("checkpoint_digest")
        )
        if current.get("qualified") and current_digest == remote_digest:
            return {
                "ok": True,
                "updated": False,
                "reason": "local checkpoint already matches qualified remote production",
                "checkpoint_digest": remote_digest,
            }

        staged = parent / ".generalist-production-staged"
        backup = parent / ".generalist-production-backup"
        shutil.rmtree(staged, ignore_errors=True)
        shutil.rmtree(backup, ignore_errors=True)
        shutil.copytree(candidate, staged)

        try:
            if target.exists():
                target.replace(backup)
            staged.replace(target)
            final = qualification_status(target)
            if not final.get("qualified"):
                raise RuntimeError("installed checkpoint failed post-swap qualification")
            shutil.rmtree(backup, ignore_errors=True)
        except Exception:
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            if backup.exists():
                backup.replace(target)
            shutil.rmtree(staged, ignore_errors=True)
            raise

    return {
        "ok": True,
        "updated": True,
        "reason": "installed qualified production checkpoint from generalist-state",
        "checkpoint_digest": remote_digest,
    }


def main() -> int:
    root = Path(os.environ.get("AIRI_ROOT", "/home/user/airi")).expanduser().resolve()
    target = Path(
        os.environ.get(
            "AIRI_GENERALIST_STATE",
            str(root / ".ai" / "generalist-lm" / "champion"),
        )
    ).expanduser().resolve()
    remote = os.environ.get("AIRI_GENERALIST_STATE_REMOTE", "origin")
    branch = os.environ.get("AIRI_GENERALIST_STATE_BRANCH", "generalist-state")
    try:
        result = sync_production_checkpoint(root, target, remote=remote, branch=branch)
    except Exception as exc:
        print(json.dumps({
            "ok": False,
            "updated": False,
            "error": f"{type(exc).__name__}: {exc}",
        }, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
