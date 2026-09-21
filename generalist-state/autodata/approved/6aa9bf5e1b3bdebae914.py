import re
import xml.etree.ElementTree as ET
from pathlib import Path

from .akn import akn_xml_to_markdown, extract_frontmatter, parse_akn_xml
from .config import logger
from .filename import collection_subdir_name, fit_md_basename, safe_filename

_ILLEGAL_XML10 = re.compile(rb"[\x00-\x08\x0B\x0C\x0E-\x1F]")


def natural_sort_key(p: Path, base: Path) -> tuple:
    """Natural sort key on relative path: numeric parts sorted as int."""
    rel = p.relative_to(base).as_posix()
    return tuple(
        int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", rel)
    )


def _unique_md_path(md_dir: Path, base_name: str) -> Path:
    out_path = md_dir / f"{base_name}.md"
    if not out_path.exists():
        return out_path
    counter = 2
    while True:
        candidate = md_dir / f"{base_name}_{counter}.md"
        if not candidate.exists():
            return candidate
        counter += 1


def convert_akn_dir_to_md(
    src_dir: str,
    md_dir: str,
    urn_index: dict[str, str],
    collection_name: str,
) -> int:
    src = Path(src_dir)
    collection_subdir = collection_subdir_name(collection_name)
    xml_files = sorted(
        (p for p in src.rglob("*.xml") if p.is_file()),
        key=lambda p: natural_sort_key(p, src),
    )
    md_path = Path(md_dir)
    count = 0
    skipped = 0
    for xml_file in xml_files:
        content = xml_file.read_text(encoding="utf-8", errors="replace")
        try:
            root = parse_akn_xml(content)
        except ET.ParseError:
            raw = xml_file.read_bytes()
            content = _ILLEGAL_XML10.sub(b"", raw).decode("utf-8", errors="replace")
            logger.warning(
                "[converter] Sanitizing illegal XML 1.0 bytes in %s", xml_file.name
            )
            try:
                root = parse_akn_xml(content)
            except Exception as e:
                logger.warning(
                    "[converter] Failed to fix through sanitization, skipping %s: %s",
                    xml_file.name,
                    e,
                )
                skipped += 1
                continue
        except Exception as e:
            logger.warning("[converter] Skipping %s: %s", xml_file.name, e)
            skipped += 1
            continue

        try:
            fm_preview = extract_frontmatter(root)
        except Exception as e:
            logger.warning("[converter] Skipping %s: %s", xml_file.name, e)
            skipped += 1
            continue

        title = fm_preview.titolo or fm_preview.codice_redazionale or xml_file.stem
        base_name = fit_md_basename(safe_filename(title))
        out_path = _unique_md_path(md_path, base_name)
        source_repo_path = f"{collection_subdir}/{out_path.name}"

        try:
            fm, markdown = akn_xml_to_markdown(content, urn_index, source_repo_path)
        except Exception as e:
            logger.warning("[converter] Skipping %s: %s", xml_file.name, e)
            skipped += 1
            continue

        out_path.write_text(markdown, encoding="utf-8")
        if fm.urn:
            urn_index[fm.urn] = source_repo_path
        count += 1

    if skipped:
        logger.warning(f"[converter] {skipped} file(s) skipped due to errors")
    return count
