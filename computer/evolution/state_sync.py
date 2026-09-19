from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from . import lab

STATE_BRANCH = os.environ.get("AIRI_EVOLUTION_STATE_BRANCH", "airi-evolution-state")
SYNC_ROOT = Path(os.environ.get("AIRI_EVOLUTION_SYNC_DIR") or (lab.ROOT / ".ai" / "evolution-git")).resolve()
CHECKOUT = SYNC_ROOT / "checkout"
EXPORT_REL = Path("evolution-state") / "shadow-router"
SYNC_META = lab.LAB_STATE / "sync-meta.json"
HEARTBEAT_SECONDS = max(1800, int(os.environ.get("AIRI_EVOLUTION_HEARTBEAT_SECONDS", "7200")))


def _run(args: list[str], cwd: Path | None = None, timeout: int = 90) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-12000:],
            "stderr": proc.stderr[-12000:],
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "returncode": -1, "stdout": "", "stderr": repr(exc)}


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_sha() -> str:
    override = os.environ.get("AIRI_EVOLUTION_SOURCE_SHA", "").strip()
    if override:
        return override
    result = _run(["git", "-C", str(lab.ROOT), "rev-parse", "HEAD"], timeout=10)
    return result["stdout"].strip() if result["ok"] else ""


def dataset_digest() -> str:
    return _sha256(lab.DATA) if lab.DATA.exists() else ""


def _git_url() -> str:
    override = os.environ.get("AIRI_EVOLUTION_GIT_URL", "").strip()
    if override:
        return override
    result = _run(["git", "-C", str(lab.ROOT), "remote", "get-url", "origin"], timeout=10)
    if not result["ok"] or not result["stdout"].strip():
        raise RuntimeError("evolution state sync requires a configured git origin")
    return result["stdout"].strip()


def _clean_public(value: Any) -> Any:
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            k = str(key)
            lower = k.lower()
            if lower in {"state_dir", "run_dir", "selected_trial_path", "dataset"} or lower.endswith("_path"):
                continue
            out[k] = _clean_public(item)
        return out
    if isinstance(value, list):
        return [_clean_public(item) for item in value]
    if isinstance(value, str):
        if value.startswith("/") or (len(value) > 2 and value[1:3] == ":\\"):
            return "<local-path>"
        return value
    return value


def privacy_audit_dataset(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"ok": True, "records": 0, "errors": []}
    errors: list[str] = []
    records = 0
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            records += 1
            try:
                row = json.loads(line)
            except Exception as exc:
                errors.append(f"line {lineno}: invalid json: {exc}")
                continue
            if int(row.get("feature_schema", 0) or 0) != lab.FEATURE_SCHEMA:
                errors.append(f"line {lineno}: legacy feature schema")
            text = str(row.get("text", ""))
            if not text.startswith("goal_shape "):
                errors.append(f"line {lineno}: non-structural feature")
            lowered = text.lower()
            if "http://" in lowered or "https://" in lowered or "@" in text or "/" in text or "\\" in text:
                errors.append(f"line {lineno}: possible user data in export feature")
    return {"ok": not errors, "records": records, "errors": errors[:50]}


def _state_digest() -> str:
    h = hashlib.sha256()
    candidates = [
        lab.DATA,
        lab.META,
        lab.LAB_STATE / "data" / "canary_ids.json",
        lab.LAB_STATE / "data" / "split_manifest.json",
        lab.LAB_STATE / "champion" / "genome.json",
        lab.LAB_STATE / "champion" / "metrics.json",
        lab.LAB_STATE / "champion" / "model.pt",
    ]
    for path in candidates:
        h.update(str(path.name).encode("utf-8"))
        if path.exists():
            h.update(_sha256(path).encode("ascii"))
    return h.hexdigest()


def export_snapshot(destination: Path, *, exported_at: float | None = None) -> dict[str, Any]:
    destination = Path(destination).resolve()
    audit = privacy_audit_dataset(lab.DATA)
    if not audit["ok"]:
        raise RuntimeError(f"privacy audit failed: {audit['errors']}")

    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "data").mkdir(parents=True, exist_ok=True)

    if lab.DATA.exists():
        shutil.copy2(lab.DATA, destination / "data" / "verified.jsonl")

    for name in ("canary_ids.json", "split_manifest.json"):
        src = lab.LAB_STATE / "data" / name
        if src.exists():
            _write_json(destination / "data" / name, _clean_public(_read_json(src, {})))

    meta = _read_json(lab.META, {})
    compatible = int(meta.get("feature_schema", 0) or 0) == lab.FEATURE_SCHEMA
    if meta:
        _write_json(destination / "lab-meta.json", _clean_public(meta))

    if compatible:
        champion_src = lab.LAB_STATE / "champion"
        champion_dst = destination / "champion"
        for name in ("genome.json", "metrics.json", "provenance.json"):
            src = champion_src / name
            if src.exists():
                champion_dst.mkdir(parents=True, exist_ok=True)
                _write_json(champion_dst / name, _clean_public(_read_json(src, {})))
        model = champion_src / "model.pt"
        if model.exists() and model.stat().st_size <= 20 * 1024 * 1024:
            champion_dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(model, champion_dst / "model.pt")

    now = float(exported_at or time.time())
    status = lab.status()
    public_status = {
        "ok": True,
        "feature_schema": lab.FEATURE_SCHEMA,
        "exported_at": now,
        "source_sha": _source_sha(),
        "raw_observations_local_only": int(status.get("raw_observations", 0)),
        "training_records": int(status.get("training_records", 0)),
        "success_records": int(status.get("success_records", 0)),
        "failure_records": int(status.get("failure_records", 0)),
        "unique_route_features": int(status.get("unique_route_features", 0)),
        "pending_observations": int(status.get("pending_observations", 0)),
        "champion_compatible": bool(status.get("champion_compatible", False)),
        "champion": _clean_public(status.get("champion")),
        "last_cycle": _clean_public(status.get("last_cycle")),
    }
    _write_json(destination / "status.json", public_status)

    manifest_files: dict[str, str] = {}
    for path in sorted(destination.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            manifest_files[str(path.relative_to(destination))] = _sha256(path)
    manifest = {
        "version": 1,
        "feature_schema": lab.FEATURE_SCHEMA,
        "exported_at": now,
        "state_digest": _state_digest(),
        "files": manifest_files,
        "privacy": {
            "raw_observations_exported": False,
            "goal_text_exported": False,
            "argument_values_exported": False,
            "dataset_audit": audit,
        },
    }
    _write_json(destination / "manifest.json", manifest)
    return manifest


def _ensure_checkout() -> dict[str, Any]:
    url = _git_url()
    SYNC_ROOT.mkdir(parents=True, exist_ok=True)
    if not (CHECKOUT / ".git").exists():
        if CHECKOUT.exists():
            shutil.rmtree(CHECKOUT)
        clone = _run(
            ["git", "clone", "--filter=blob:none", "--single-branch", "--branch", STATE_BRANCH, url, str(CHECKOUT)],
            timeout=180,
        )
        if not clone["ok"]:
            return {"ok": False, "reason": "clone_failed", "git": clone}
    else:
        fetch = _run(["git", "fetch", "origin", STATE_BRANCH], cwd=CHECKOUT, timeout=120)
        if not fetch["ok"]:
            return {"ok": False, "reason": "fetch_failed", "git": fetch}
        reset = _run(["git", "checkout", "-B", STATE_BRANCH, f"origin/{STATE_BRANCH}"], cwd=CHECKOUT)
        if not reset["ok"]:
            return {"ok": False, "reason": "checkout_failed", "git": reset}
        _run(["git", "reset", "--hard", f"origin/{STATE_BRANCH}"], cwd=CHECKOUT)
    _run(["git", "config", "user.name", "Airi Evolution Lab"], cwd=CHECKOUT)
    _run(["git", "config", "user.email", "airi-evolution@users.noreply.github.com"], cwd=CHECKOUT)
    return {"ok": True, "checkout": str(CHECKOUT)}


def sync_to_git(*, force_heartbeat: bool = False) -> dict[str, Any]:
    checkout = _ensure_checkout()
    if not checkout["ok"]:
        return checkout

    remote_manifest = _read_json(CHECKOUT / EXPORT_REL / "manifest.json", {})
    digest = _state_digest()
    last_export = float(remote_manifest.get("exported_at", 0) or 0)
    if (
        not force_heartbeat
        and remote_manifest.get("state_digest") == digest
        and time.time() - last_export < HEARTBEAT_SECONDS
    ):
        return {
            "ok": True,
            "synced": False,
            "reason": "state_unchanged",
            "remote_exported_at": last_export,
            "state_digest": digest,
        }

    manifest = export_snapshot(CHECKOUT / EXPORT_REL)
    _run(["git", "add", "--", str(EXPORT_REL)], cwd=CHECKOUT)
    diff = _run(["git", "diff", "--cached", "--quiet"], cwd=CHECKOUT)
    if diff["returncode"] == 0:
        return {"ok": True, "synced": False, "reason": "nothing_to_commit", "manifest": manifest}

    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    commit = _run(["git", "commit", "-m", f"evolution-state: sync {stamp}"], cwd=CHECKOUT)
    if not commit["ok"]:
        return {"ok": False, "synced": False, "reason": "commit_failed", "git": commit}

    sha = _run(["git", "rev-parse", "HEAD"], cwd=CHECKOUT)["stdout"].strip()
    push = _run(["git", "push", "origin", f"HEAD:{STATE_BRANCH}"], cwd=CHECKOUT, timeout=120)
    if not push["ok"]:
        return {"ok": False, "synced": False, "reason": "push_failed", "local_commit": sha, "git": push}

    verify = _run(["git", "ls-remote", "origin", f"refs/heads/{STATE_BRANCH}"], cwd=CHECKOUT, timeout=60)
    remote_sha = verify["stdout"].split()[0] if verify["ok"] and verify["stdout"].split() else ""
    ok = bool(remote_sha and remote_sha == sha)
    result = {
        "ok": ok,
        "synced": ok,
        "commit": sha,
        "remote_sha": remote_sha,
        "branch": STATE_BRANCH,
        "manifest": manifest,
        "reason": "verified" if ok else "remote_verification_failed",
    }
    _write_json(SYNC_META, {**result, "updated_at": time.time()})
    return result


def restore_from_git(*, force: bool = False) -> dict[str, Any]:
    checkout = _ensure_checkout()
    if not checkout["ok"]:
        return checkout
    src = CHECKOUT / EXPORT_REL
    manifest = _read_json(src / "manifest.json", {})
    if int(manifest.get("feature_schema", 0) or 0) != lab.FEATURE_SCHEMA:
        return {"ok": False, "restored": False, "reason": "remote_feature_schema_mismatch"}

    remote_data = src / "data" / "verified.jsonl"
    audit = privacy_audit_dataset(remote_data)
    if not audit["ok"]:
        return {"ok": False, "restored": False, "reason": "remote_privacy_audit_failed", "audit": audit}

    local_records = len(lab._read_observations()) if lab.RAW.exists() else 0
    local_training = len(__import__("evolution.data", fromlist=["load_records"]).load_records(lab.DATA))
    if not force and (local_records > 0 or local_training > 0):
        return {"ok": True, "restored": False, "reason": "local_state_present"}

    lab.LAB_STATE.mkdir(parents=True, exist_ok=True)
    if remote_data.exists():
        lab.DATA.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(remote_data, lab.DATA)
    for name in ("canary_ids.json", "split_manifest.json"):
        source = src / "data" / name
        if source.exists():
            target = lab.LAB_STATE / "data" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    source_meta = src / "lab-meta.json"
    if source_meta.exists():
        shutil.copy2(source_meta, lab.META)
    source_champion = src / "champion"
    if source_champion.exists():
        target = lab.LAB_STATE / "champion"
        target.mkdir(parents=True, exist_ok=True)
        for name in ("genome.json", "metrics.json", "provenance.json", "model.pt"):
            source = source_champion / name
            if source.exists():
                shutil.copy2(source, target / name)

    return {
        "ok": True,
        "restored": True,
        "records": audit["records"],
        "feature_schema": lab.FEATURE_SCHEMA,
        "remote_exported_at": manifest.get("exported_at"),
    }
