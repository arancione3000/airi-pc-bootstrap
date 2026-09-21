"""Resolve Akoma Ntoso <ref> hrefs to relative corpus links or Normattiva URLs."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

NORMATTIVA_URI_RES = "https://www.normattiva.it/uri-res/N2Ls"

_HREF_ENCODE: dict[str, str] = {
    " ": "%20",
    "(": "%28",
    ")": "%29",
    "[": "%5B",
    "]": "%5D",
    "#": "%23",
    '"': "%22",
    "<": "%3C",
    ">": "%3E",
    "\\": "%5C",
}


def _encode_href(href: str) -> str:
    return "".join(_HREF_ENCODE.get(c, c) for c in href)


@dataclass(frozen=True)
class RefContext:
    urn_index: dict[str, str]
    source_repo_path: str


def _camel_to_dots(value: str) -> str:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1.\2", value)
    return value.lower()


def href_to_urn(href: str) -> str | None:
    """Map an href (urn:nir or /akn/...) to a urn:nir identifier for index lookup."""
    path, _, _ = href.partition("#")
    if path.startswith("urn:nir:"):
        return path
    if not path.startswith("/akn/"):
        return None

    parts = path.strip("/").split("/")
    if len(parts) < 7 or parts[:3] != ["akn", "it", "act"]:
        return None

    act_type, authority, date, number = parts[3], parts[4], parts[5], parts[6]
    urn_type = _camel_to_dots(act_type)
    urn_authority = authority.lower().replace("_", ".")
    return f"urn:nir:{urn_authority}:{urn_type}:{date};{number}"


def _relative_link(source_repo_path: str, target_repo_path: str) -> str:
    source_dir = os.path.dirname(source_repo_path)
    start = source_dir or "."
    rel = os.path.relpath(target_repo_path, start=start).replace(os.sep, "/")
    return _encode_href(rel)


def normattiva_url(href: str) -> str:
    """Build a Normattiva uri-res link for an unresolved reference."""
    urn, _, fragment = href.partition("#")
    if not urn.startswith("urn:nir:"):
        derived = href_to_urn(href)
        if derived:
            urn = derived
    url = f"{NORMATTIVA_URI_RES}?{urn}"
    if fragment:
        url = f"{url}#{fragment}"
    return url


def resolve_ref(href: str, label: str, ctx: RefContext) -> str:
    """Convert a <ref> href to a markdown link."""
    if not href:
        return label

    urn, _, fragment = href.partition("#")
    lookup_urn = href_to_urn(href) or urn
    target = ctx.urn_index.get(lookup_urn)
    if target:
        link = _relative_link(ctx.source_repo_path, target)
        return f"[{label}]({link})"

    return f"[{label}]({normattiva_url(href)})"
