from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


FOUNDATION_SCHEMA_VERSION = 1
FOUNDATION_MANIFEST_FILENAME = "airi-foundation-manifest.json"
FOUNDATION_DOMAINS = (
    "language",
    "coding",
    "data",
    "reasoning",
    "tools",
    "structured",
    "long_context",
    "robustness",
)
_ATTESTATION_FILENAMES = {
    ".airi-qualification.json",
    ".airi-foundation-qualification.json",
}


@dataclass(frozen=True)
class FoundationManifest:
    model_id: str
    source_revision: str
    license: str
    architecture: str
    context_length: int
    parameter_count: int = 0
    dtype: str = "unknown"
    quantization: str = "none"
    intended_domains: tuple[str, ...] = FOUNDATION_DOMAINS
    backend: str = "transformers"
    local_files_only: bool = True
    trust_remote_code: bool = False
    schema_version: int = FOUNDATION_SCHEMA_VERSION

    def validate(self) -> "FoundationManifest":
        if type(self.schema_version) is not int:
            raise ValueError("foundation schema_version must be an integer")
        if self.schema_version != FOUNDATION_SCHEMA_VERSION:
            raise ValueError("unsupported foundation manifest schema")
        if self.backend != "transformers":
            raise ValueError("foundation backend must be transformers")
        if self.local_files_only is not True:
            raise ValueError("foundation models must remain local-files-only")
        if self.trust_remote_code is not False:
            raise ValueError("foundation models must keep trust_remote_code disabled")

        for name in ("model_id", "source_revision", "license", "architecture", "dtype", "quantization"):
            raw = getattr(self, name)
            if not isinstance(raw, str):
                raise ValueError(f"foundation manifest field {name} must be a string")
            value = raw.strip()
            if not value:
                raise ValueError(f"foundation manifest field {name} must be non-empty")
            if len(value) > 512:
                raise ValueError(f"foundation manifest field {name} is too long")

        if type(self.context_length) is not int:
            raise ValueError("foundation context_length must be an integer")
        context = self.context_length
        if context < 1024 or context > 1_048_576:
            raise ValueError("foundation context_length must be between 1024 and 1048576")
        if type(self.parameter_count) is not int:
            raise ValueError("foundation parameter_count must be an integer")
        params = self.parameter_count
        if params < 0 or params > 100_000_000_000_000:
            raise ValueError("foundation parameter_count is out of bounds")

        if not isinstance(self.intended_domains, (tuple, list)):
            raise ValueError("foundation intended_domains must be a list or tuple")
        if not all(isinstance(x, str) and x.strip() for x in self.intended_domains):
            raise ValueError("foundation intended_domains entries must be non-empty strings")
        domains = tuple(x.strip() for x in self.intended_domains)
        if len(domains) != len(set(domains)):
            raise ValueError("foundation intended_domains must be unique")
        unknown = sorted(set(domains) - set(FOUNDATION_DOMAINS))
        if unknown:
            raise ValueError(f"unsupported foundation domains: {unknown}")
        missing = sorted(set(FOUNDATION_DOMAINS) - set(domains))
        if missing:
            raise ValueError(f"foundation manifest is missing protected domains: {missing}")
        return self

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["intended_domains"] = list(self.intended_domains)
        return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _model_inventory(root: Path) -> dict[str, Any]:
    supplied = root.expanduser()
    if supplied.is_symlink():
        raise ValueError("foundation model directory must not be a symlink")
    root = supplied.resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError("foundation model directory does not exist")
    if not (root / "config.json").is_file():
        raise FileNotFoundError("foundation model is missing config.json")

    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"foundation model tree must not contain symlinks: {path.name}")
        if path.is_file():
            files.append(path)

    tokenizer_files = [
        path for path in files
        if path.name in {
            "tokenizer.json",
            "tokenizer_config.json",
            "tokenizer.model",
            "spiece.model",
            "sentencepiece.bpe.model",
            "vocab.json",
            "vocab.txt",
            "merges.txt",
        }
    ]
    if not tokenizer_files:
        raise FileNotFoundError("foundation model has no recognized tokenizer artifacts")

    weight_files = [
        path for path in files
        if path.suffix == ".safetensors"
        or (path.name.startswith("pytorch_model") and path.suffix == ".bin")
    ]
    if not weight_files:
        raise FileNotFoundError("foundation model has no recognized causal-LM weight files")

    try:
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"foundation config.json is invalid: {type(exc).__name__}") from exc
    if not isinstance(config, dict):
        raise ValueError("foundation config.json must be a JSON object")

    declared_contexts = []
    for key in ("max_position_embeddings", "n_positions", "n_ctx", "seq_length"):
        value = config.get(key)
        if type(value) is int and value > 0:
            declared_contexts.append(value)

    relative = [path.relative_to(root).as_posix() for path in files]
    return {
        "files": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "weight_files": [path.relative_to(root).as_posix() for path in weight_files],
        "weight_bytes": sum(path.stat().st_size for path in weight_files),
        "tokenizer_files": [path.relative_to(root).as_posix() for path in tokenizer_files],
        "attestations": sorted(name for name in relative if Path(name).name in _ATTESTATION_FILENAMES),
        "config_context_limit": min(declared_contexts) if declared_contexts else None,
    }


def load_foundation_manifest(model_dir: str | Path) -> FoundationManifest:
    supplied = Path(model_dir).expanduser()
    if supplied.is_symlink():
        raise ValueError("foundation model directory must not be a symlink")
    root = supplied.resolve()
    inventory = _model_inventory(root)
    raw = json.loads((root / FOUNDATION_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("foundation manifest must be a JSON object")
    raw = dict(raw)
    raw["intended_domains"] = tuple(raw.get("intended_domains") or ())
    try:
        manifest = FoundationManifest(**raw).validate()
    except TypeError as exc:
        raise ValueError(f"invalid foundation manifest schema: {exc}") from exc
    config_limit = inventory.get("config_context_limit")
    if config_limit is not None and manifest.context_length > int(config_limit):
        raise ValueError(
            "foundation manifest context_length exceeds the local config.json limit"
        )
    return manifest


def foundation_manifest_digest(model_dir: str | Path) -> str:
    manifest = load_foundation_manifest(model_dir)
    return hashlib.sha256(_canonical_json(manifest.to_dict())).hexdigest()


def write_foundation_manifest(
    model_dir: str | Path,
    manifest: FoundationManifest,
) -> dict[str, Any]:
    supplied = Path(model_dir).expanduser()
    if supplied.is_symlink():
        raise ValueError("foundation model directory must not be a symlink")
    root = supplied.resolve()
    inventory = _model_inventory(root)
    checked = manifest.validate()
    config_limit = inventory.get("config_context_limit")
    if config_limit is not None and checked.context_length > int(config_limit):
        raise ValueError(
            "foundation manifest context_length exceeds the local config.json limit"
        )
    value = checked.to_dict()
    path = root / FOUNDATION_MANIFEST_FILENAME
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest = foundation_manifest_digest(root)
    inventory = _model_inventory(root)
    return {
        "ok": True,
        "manifest_path": str(path),
        "manifest_digest": digest,
        "manifest": value,
        "inventory": inventory,
        "policy": "local files only; remote code disabled; qualification remains mandatory",
    }


def foundation_identity(model_dir: str | Path) -> dict[str, Any]:
    root = Path(model_dir).expanduser().resolve()
    manifest = load_foundation_manifest(root)
    return {
        "ok": True,
        "manifest": manifest.to_dict(),
        "manifest_digest": foundation_manifest_digest(root),
        "inventory": _model_inventory(root),
    }
