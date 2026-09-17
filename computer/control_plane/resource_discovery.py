from __future__ import annotations

import ipaddress
import json
import os
import re
import socket
import urllib.parse
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from coding import ROOT
from .store import AI, load_json, now, redact, save_json

FILE = "resource-discovery.json"
MAX_REMOTE_BYTES = 1_000_000
SKILL_TYPES = {"application/ai-skill", "application/ai-skill+md", "text/markdown"}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _urlopen_no_redirect(request: urllib.request.Request, timeout: int):
    opener = urllib.request.build_opener(_NoRedirect())
    return opener.open(request, timeout=timeout)


class ResourceDiscovery:
    """ARD discovery + quarantine pipeline.

    Search relevance is never interpreted as trust. Remote resources are HTTPS
    only, public-address only, size bounded, quarantined first, and Markdown
    skills are promoted only for explicitly allowlisted publishers after static
    validation. Arbitrary downloaded code is never executed by this component.
    """

    def __init__(self) -> None:
        self.state = load_json(FILE, {"version": 2, "staged": {}, "searches": []})
        self.state.setdefault("staged", {})
        self.state.setdefault("searches", [])

    @staticmethod
    def _validate_remote(url: str) -> urllib.parse.ParseResult:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError("ARD resource must use https")
        host = parsed.hostname.rstrip(".")
        try:
            infos = socket.getaddrinfo(host, parsed.port or 443, type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise ValueError(f"DNS resolution failed: {host}") from exc
        addresses = {i[4][0] for i in infos}
        if not addresses:
            raise ValueError("host resolved to no addresses")
        for raw in addresses:
            ip = ipaddress.ip_address(raw.split("%", 1)[0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified:
                raise ValueError("host resolves to a non-public address")
        return parsed

    @staticmethod
    def _endpoint(base: str, path: str) -> str:
        return base.rstrip("/") + "/" + path.lstrip("/")

    @staticmethod
    def _publisher(candidate: dict[str, Any]) -> str:
        entry = candidate.get("catalogEntry") if isinstance(candidate.get("catalogEntry"), dict) else candidate
        if not isinstance(entry, dict):
            return ""
        identifier = str(entry.get("identifier") or "")
        if identifier.startswith("urn:air:"):
            parts = identifier.split(":")
            if len(parts) >= 4:
                return parts[2].lower()
        return str(entry.get("publisher") or entry.get("provider") or "").lower()

    @staticmethod
    def _trust_identity(candidate: dict[str, Any]) -> str:
        entry = candidate.get("catalogEntry") if isinstance(candidate.get("catalogEntry"), dict) else candidate
        trust = entry.get("trustManifest") if isinstance(entry, dict) else None
        return str(trust.get("identity") or "") if isinstance(trust, dict) else ""

    def search(
        self,
        text: str,
        registries: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        page_size: int = 8,
        federation: str = "auto",
        timeout: int = 15,
    ) -> dict[str, Any]:
        if not str(text).strip():
            raise ValueError("search text is required")
        registries = list(registries or [x.strip() for x in os.environ.get("AIRI_ARD_REGISTRIES", "").split(",") if x.strip()])
        if not registries:
            raise ValueError("no ARD registry configured; set AIRI_ARD_REGISTRIES or pass registries")
        page_size = max(1, min(int(page_size), 25))
        if federation not in {"auto", "none", "referrals"}:
            raise ValueError("invalid federation mode")
        # ARD v0.91: query has text + filter. Do not invent extra context fields.
        payload = {"query": {"text": str(text), "filter": filters or {}}, "federation": federation, "pageSize": page_size}
        results: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for registry in registries[:8]:
            try:
                self._validate_remote(registry)
                url = self._endpoint(registry, "search")
                request = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json", "Accept": "application/json", "User-Agent": "Airi-PC-ARD/0.91"},
                    method="POST",
                )
                with _urlopen_no_redirect(request, timeout=max(1, min(int(timeout), 30))) as response:
                    raw = response.read(2_000_001)
                if len(raw) > 2_000_000:
                    raise ValueError("ARD response exceeds size limit")
                data = json.loads(raw.decode("utf-8"))
                items = data.get("results") or data.get("items") or data.get("entries") or []
                for item in items[:page_size]:
                    row = dict(item) if isinstance(item, dict) else {"value": item}
                    row["registry"] = registry
                    row["relevance_only"] = True
                    results.append(redact(row))
            except Exception as exc:
                errors.append({"registry": registry, "error": str(exc)})
        results.sort(key=lambda x: float(x.get("score", 0) or 0), reverse=True)
        self.state["searches"].append({"query": str(text), "at": now(), "registries": list(registries), "count": len(results), "errors": errors})
        self.state["searches"] = self.state["searches"][-100:]
        save_json(FILE, self.state)
        return {
            "ok": bool(results),
            "query": str(text),
            "results": results[:page_size],
            "errors": errors,
            "note": "ARD score is semantic relevance only; trust is evaluated separately.",
        }

    def stage(self, candidate: dict[str, Any], *, allowed_publishers: list[str] | None = None) -> dict[str, Any]:
        if not isinstance(candidate, dict):
            raise ValueError("candidate must be an object")
        publisher = self._publisher(candidate)
        identity = self._trust_identity(candidate)
        allow = {x.strip().lower() for x in (allowed_publishers or []) if x.strip()}
        publisher_allowed = bool(publisher and publisher in allow)
        sid = uuid.uuid4().hex[:12]
        row = {
            "id": sid,
            "candidate": redact(candidate),
            "publisher": publisher,
            "trust_identity": identity,
            "publisher_allowed": publisher_allowed,
            "trust_verified": False,
            "status": "approved_source_pending_verification" if publisher_allowed else "quarantined",
            "created_at": now(),
        }
        self.state["staged"][sid] = row
        save_json(FILE, self.state)
        return redact(row)

    @staticmethod
    def _safe_skill_name(candidate: dict[str, Any], sid: str) -> str:
        raw = str(candidate.get("displayName") or candidate.get("name") or candidate.get("identifier") or sid)
        raw = raw.rsplit(":", 1)[-1]
        slug = re.sub(r"[^a-z0-9._-]+", "-", raw.lower()).strip("-._")[:60]
        return f"ard-{slug or sid}"

    @staticmethod
    def _static_skill_check(content: str) -> dict[str, Any]:
        if not content.strip() or len(content.encode("utf-8")) > MAX_REMOTE_BYTES:
            return {"ok": False, "reason": "empty or oversized skill"}
        lowered = content.lower()
        blocked = (
            "ignore previous instructions", "disable security", "steal password", "exfiltrate",
            "cat ~/.ssh", "cat /etc/shadow", "curl | sh", "wget -o-", "rm -rf /",
        )
        hits = [term for term in blocked if term in lowered]
        if hits:
            return {"ok": False, "reason": "unsafe instruction pattern", "hits": hits}
        has_description = "## description" in lowered or "# " in lowered
        return {"ok": bool(has_description), "reason": "ok" if has_description else "skill lacks description/header", "hits": []}

    def _fetch_skill_content(self, candidate: dict[str, Any], timeout: int = 15) -> str:
        entry = candidate.get("catalogEntry") if isinstance(candidate.get("catalogEntry"), dict) else candidate
        if not isinstance(entry, dict):
            raise ValueError("invalid candidate")
        media_type = str(entry.get("type") or "")
        if media_type and media_type not in SKILL_TYPES:
            raise ValueError("candidate is not a supported Markdown skill")
        data = entry.get("data")
        if isinstance(data, dict):
            content = data.get("content") or data.get("markdown") or data.get("skill")
            if isinstance(content, str):
                return content
        url = str(entry.get("url") or "")
        if not url:
            raise ValueError("skill candidate has no inline content or URL")
        self._validate_remote(url)
        request = urllib.request.Request(url, headers={"Accept": "text/markdown,text/plain;q=0.9", "User-Agent": "Airi-PC-SkillHunter/1.0"})
        with _urlopen_no_redirect(request, timeout=max(1, min(int(timeout), 30))) as response:
            raw = response.read(MAX_REMOTE_BYTES + 1)
        if len(raw) > MAX_REMOTE_BYTES:
            raise ValueError("skill exceeds size limit")
        return raw.decode("utf-8", errors="strict")

    def acquire_skill(self, candidate: dict[str, Any], *, allowed_publishers: list[str] | None = None, timeout: int = 15) -> dict[str, Any]:
        """Quarantine, statically validate, then promote trusted Markdown skills.

        No arbitrary downloaded executable code is ever run. Publishers not in
        the explicit allowlist stay quarantined, preserving zero-touch safety.
        """
        staged = self.stage(candidate, allowed_publishers=allowed_publishers)
        sid = staged["id"]
        try:
            content = self._fetch_skill_content(candidate, timeout=timeout)
            static = self._static_skill_check(content)
        except Exception as exc:
            static = {"ok": False, "reason": str(exc)}
            content = ""
        quarantine = AI / "quarantine" / "skills" / sid
        quarantine.mkdir(parents=True, exist_ok=True)
        if content:
            (quarantine / "SKILL.md").write_text(content, encoding="utf-8")
        row = self.state["staged"][sid]
        row["static_verification"] = static
        row["quarantine_path"] = str(quarantine)
        if not staged.get("publisher_allowed") or not static.get("ok"):
            row["status"] = "quarantined"
            row["updated_at"] = now()
            save_json(FILE, self.state)
            return redact(row)
        name = self._safe_skill_name(candidate, sid)
        destination = (ROOT / "skills" / name).resolve()
        try:
            destination.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise ValueError("skill destination outside workspace") from exc
        if destination.exists():
            existing = destination / "SKILL.md"
            if existing.exists() and existing.read_text(encoding="utf-8", errors="replace") == content:
                row.update({
                    "status": "installed_markdown_skill",
                    "installed_path": str(existing.relative_to(ROOT)),
                    "installed_name": name,
                    "trust_verified": False,
                    "updated_at": now(),
                    "note": "identical skill already installed",
                })
                save_json(FILE, self.state)
                return redact(row)
            raise FileExistsError(f"skill destination already exists: {name}")
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "SKILL.md").write_text(content, encoding="utf-8")
        row.update({
            "status": "installed_markdown_skill",
            "installed_path": str((destination / "SKILL.md").relative_to(ROOT)),
            "installed_name": name,
            "trust_verified": False,
            "updated_at": now(),
        })
        save_json(FILE, self.state)
        return redact(row)

    def list_staged(self) -> dict[str, Any]:
        return {"count": len(self.state["staged"]), "items": redact(list(self.state["staged"].values()))}
