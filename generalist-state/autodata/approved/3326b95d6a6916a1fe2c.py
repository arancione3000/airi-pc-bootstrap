"""Extract metadata and body text from Normattiva Akoma Ntoso XML."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from .refs import RefContext, resolve_ref

AKN_NS = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"
ELI_NS = "http://data.europa.eu/eli/ontology#"
NS = {"akn": AKN_NS, "eli": ELI_NS}


@dataclass(frozen=True)
class AknFrontmatter:
    tipo: str | None
    numero: str | None
    data: str | None
    titolo: str | None
    urn: str | None
    codice_redazionale: str | None
    vigente: bool


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _text(el: ET.Element | None) -> str | None:
    if el is None:
        return None
    text = "".join(el.itertext()).strip()
    return text or None


def _find_one(root: ET.Element, path: str) -> ET.Element | None:
    found = root.find(path, NS)
    return found


def extract_frontmatter(root: ET.Element) -> AknFrontmatter:
    """Extract YAML frontmatter fields from an Akoma Ntoso document root."""
    tipo = _text(_find_one(root, ".//akn:preface//akn:docType"))
    numero = _text(_find_one(root, ".//akn:preface//akn:docNumber"))

    doc_date = _find_one(root, ".//akn:preface//akn:docDate")
    data = doc_date.get("date") if doc_date is not None else None

    titolo_raw = _text(_find_one(root, ".//akn:preface//akn:docTitle"))
    titolo = " ".join(titolo_raw.split()) if titolo_raw else None

    urn_el = _find_one(
        root, ".//akn:meta/akn:identification//akn:FRBRalias[@name='urn:nir']"
    )
    urn = urn_el.get("value") if urn_el is not None else None

    codice_el = _find_one(root, ".//akn:meta/akn:proprietary//eli:id_local")
    codice_redazionale = _text(codice_el)

    repeal_events = root.findall(
        ".//akn:meta/akn:lifecycle//akn:eventRef[@type='repeal']", NS
    )
    vigente = len(repeal_events) == 0

    return AknFrontmatter(
        tipo=tipo,
        numero=numero,
        data=data,
        titolo=titolo,
        urn=urn,
        codice_redazionale=codice_redazionale,
        vigente=vigente,
    )


def format_frontmatter(fm: AknFrontmatter) -> str:
    """Serialize frontmatter fields to YAML between --- markers."""
    fields: list[tuple[str, str | bool]] = []
    if fm.tipo is not None:
        fields.append(("tipo", fm.tipo))
    if fm.numero is not None:
        fields.append(("numero", fm.numero))
    if fm.data is not None:
        fields.append(("data", fm.data))
    if fm.titolo is not None:
        fields.append(("titolo", fm.titolo))
    if fm.urn is not None:
        fields.append(("urn", fm.urn))
    if fm.codice_redazionale is not None:
        fields.append(("codice_redazionale", fm.codice_redazionale))
    fields.append(("vigente", fm.vigente))

    lines = ["---"]
    for key, value in fields:
        if isinstance(value, bool):
            lines.append(f"{key}: {'true' if value else 'false'}")
        elif key == "titolo":
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'{key}: "{escaped}"')
        else:
            lines.append(f"{key}: {value}")
    lines.append("---")
    return "\n".join(lines) + "\n\n"


def _render_inline(el: ET.Element, ctx: RefContext) -> str:
    tag = _local(el.tag)
    if tag == "ref":
        href = el.get("href") or ""
        label = _text(el) or href
        if href:
            return resolve_ref(href, label, ctx)
        return label

    parts: list[str] = []
    if el.text:
        parts.append(el.text)
    for child in el:
        parts.append(_render_inline(child, ctx))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts).strip()


def _render_block(
    el: ET.Element, lines: list[str], ctx: RefContext, heading_level: int = 2
) -> None:
    tag = _local(el.tag)

    if tag == "article":
        num_el = el.find(f"{{{AKN_NS}}}num")
        heading_el = el.find(f"{{{AKN_NS}}}heading")
        num = _text(num_el)
        heading = _text(heading_el)
        title = " — ".join(part for part in (num, heading) if part)
        if title:
            lines.append(f"{'#' * heading_level} {title}")
            lines.append("")
        for child in el:
            if _local(child.tag) not in {"num", "heading"}:
                _render_block(child, lines, ctx, heading_level + 1)
        return

    if tag in {"paragraph", "content", "p", "blockList", "item", "point"}:
        text = _render_inline(el, ctx) if tag == "p" else None
        if text:
            lines.append(text)
            lines.append("")
            return
        for child in el:
            _render_block(child, lines, ctx, heading_level)
        return

    if tag in {"section", "chapter", "part", "division", "title", "subtitle"}:
        heading_el = el.find(f"{{{AKN_NS}}}heading")
        heading = _text(heading_el)
        if heading:
            lines.append(f"{'#' * min(heading_level, 6)} {heading}")
            lines.append("")
        for child in el:
            if heading_el is not None and child is heading_el:
                continue
            _render_block(child, lines, ctx, heading_level + 1)
        return

    text = _render_inline(el, ctx)
    if text and tag not in {
        "meta",
        "preface",
        "body",
        "preamble",
        "conclusions",
        "act",
    }:
        lines.append(text)
        lines.append("")
        return

    for child in el:
        _render_block(child, lines, ctx, heading_level)


def body_to_markdown(root: ET.Element, ctx: RefContext) -> str:
    """Convert preamble, body, and conclusions to markdown."""
    lines: list[str] = []
    for section in ("preamble", "body", "conclusions"):
        section_el = _find_one(root, f".//akn:{section}")
        if section_el is not None:
            _render_block(section_el, lines, ctx)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def parse_akn_xml(content: str) -> ET.Element:
    if not content or not content.strip():
        raise ValueError("Empty content")
    return ET.fromstring(content)


def akn_xml_to_markdown(
    content: str,
    urn_index: dict[str, str],
    source_repo_path: str,
) -> tuple[AknFrontmatter, str]:
    root = parse_akn_xml(content)
    fm = extract_frontmatter(root)
    ctx = RefContext(urn_index=urn_index, source_repo_path=source_repo_path)
    body = body_to_markdown(root, ctx)
    return fm, format_frontmatter(fm) + body
