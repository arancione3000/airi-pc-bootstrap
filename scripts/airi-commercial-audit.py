#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def audit(repo_root: str | Path, *, state_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    license_files = [
        path.name
        for path in root.iterdir()
        if path.is_file() and path.name.upper() in {"LICENSE", "LICENSE.TXT", "LICENSE.MD", "COPYING"}
    ]
    dependency_files = [
        name for name in (
            "requirements.txt",
            "requirements-dev.txt",
            "computer/requirements.txt",
            "airi-pc-companion/requirements.txt",
        )
        if (root / name).is_file()
    ]
    data_policy = root / "computer" / "generalist_lm" / "generalist_data_growth.py"
    security_policy = root / "SECURITY.md"
    privacy_docs = [
        name for name in ("PRIVACY.md", "docs/PRIVACY.md", "docs/SECURITY.md")
        if (root / name).is_file()
    ]

    training_state: dict[str, Any] = {
        "available": False,
        "manifest_present": False,
        "ambiguous_or_missing_license_rows": None,
    }
    if state_dir:
        state = Path(state_dir).expanduser().resolve()
        manifest_path = state / "autodata" / "manifest.json"
        manifest = _json(manifest_path, {})
        files = manifest.get("files") if isinstance(manifest, dict) else None
        if isinstance(files, list):
            ambiguous = 0
            for row in files:
                if not isinstance(row, dict):
                    ambiguous += 1
                    continue
                license_id = str(row.get("license") or row.get("spdx") or "").strip().upper()
                if not license_id or license_id in {"UNKNOWN", "NOASSERTION", "NONE"}:
                    ambiguous += 1
            training_state = {
                "available": True,
                "manifest_present": True,
                "rows": len(files),
                "ambiguous_or_missing_license_rows": ambiguous,
            }
        else:
            training_state = {
                "available": state.exists(),
                "manifest_present": manifest_path.is_file(),
                "ambiguous_or_missing_license_rows": None,
            }

    blockers: list[str] = []
    warnings: list[str] = []
    if not license_files:
        blockers.append("repository has no explicit LICENSE/COPYING file")
    if not data_policy.is_file():
        blockers.append("training-data provenance/allowlist implementation is missing")
    if not security_policy.is_file():
        blockers.append("SECURITY.md is missing")
    if training_state.get("manifest_present") and training_state.get("ambiguous_or_missing_license_rows"):
        blockers.append("persisted training-data manifest contains ambiguous/missing license rows")
    if not privacy_docs:
        warnings.append("no dedicated privacy document was found")
    if len(dependency_files) < 2:
        warnings.append("dependency manifests are incomplete for a redistribution audit")

    return {
        "schema": 1,
        "commercial_ready": not blockers,
        "legal_clearance": False,
        "legal_clearance_note": (
            "Automated checks cannot grant legal clearance. Human review of ownership, "
            "third-party licenses, trademarks, privacy and distribution terms is still required."
        ),
        "repository_license_files": license_files,
        "dependency_manifests": dependency_files,
        "training_data_policy_present": data_policy.is_file(),
        "security_policy_present": security_policy.is_file(),
        "privacy_documents": privacy_docs,
        "training_state": training_state,
        "blockers": blockers,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="AIRI commercial-readiness audit")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--state-dir")
    parser.add_argument("--output")
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()
    report = audit(args.repo, state_dir=args.state_dir)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 2 if args.enforce and not report["commercial_ready"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
