import hashlib
import os

_MAX_MD_BASENAME_BYTES = 200


def safe_repo_name(name: str) -> str:
    """Convert a collection name into a safe filesystem slug."""
    name = (name or "").strip().lower()
    name = name.replace("\0", "")
    keep = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "."):
            keep.append(ch)
        elif ch in (" ", "_", "/", "\\", "(", ")", ","):
            keep.append("-")
    name = "".join(keep)
    while "--" in name:
        name = name.replace("--", "-")
    name = name.strip("-.")
    return (name or "collezione")[:100]


def safe_filename(name: str) -> str:
    """Sanitize a string for use as a filesystem filename."""
    name = (name or "").strip()
    name = name.replace("\0", "")
    name = name.replace(os.sep, "-")
    if os.altsep:
        name = name.replace(os.altsep, "-")
    name = " ".join(name.split())
    keep = []
    for ch in name:
        if ch.isalnum() or ch in (" ", ".", "_", "-"):
            keep.append(ch)
    name = "".join(keep).strip(" .-_")
    return name or "document"


def fit_md_basename(stem: str, max_bytes: int = _MAX_MD_BASENAME_BYTES) -> str:
    """Shorten stem so the final *.md name fits OS limits; keep uniqueness via a hash suffix."""
    raw = stem.encode("utf-8")
    if len(raw) <= max_bytes:
        return stem
    digest = hashlib.sha256(raw).hexdigest()[:12]
    tag = f"_{digest}"
    budget = max_bytes - len(tag.encode("ascii"))
    if budget < 1:
        return digest[:max_bytes]
    cut = raw[:budget]
    while cut:
        try:
            return cut.decode("utf-8") + tag
        except UnicodeDecodeError:
            cut = cut[:-1]
    return digest


def collection_subdir_name(collection_name: str) -> str:
    """
    Folder name inside the aggregated repo: spaces preserved, no forced hyphens.
    Hyphens in the API name become spaces; only path-illegal characters are removed.
    """
    name = (collection_name or "").strip().replace("\0", "")
    name = name.replace("-", " ")
    name = " ".join(name.split())
    invalid = '\\/:*?"<>|\r\n'
    buf = []
    for ch in name:
        buf.append(" " if ch in invalid else ch)
    name = " ".join("".join(buf).split())
    if not name:
        name = "collezione"
    return fit_md_basename(name, max_bytes=_MAX_MD_BASENAME_BYTES)
