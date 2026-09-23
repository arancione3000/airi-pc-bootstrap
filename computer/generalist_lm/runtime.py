from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
import uuid

from .bpe_tokenizer import BPETokenizer
from .model import CausalTransformerLM, GeneralistLMConfig, parameter_count
from .tokenizer import ASSISTANT, EOS, ByteTokenizer
from .tool_protocol import ToolCall, parse_tool_call, tool_prompt


MODEL_INDEX_FILENAME = "model.index.json"
MODEL_SHARD_FORMAT = "airi-generalist-sharded-state-v1"
MODEL_SHARD_TARGET_BYTES = 64 * 1024 * 1024
MODEL_SHARD_HARD_LIMIT_BYTES = 90 * 1024 * 1024


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_checkpoint_member(root: Path, name: str) -> Path:
    candidate = (root / str(name)).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"checkpoint member escapes root: {name}") from exc
    return candidate


def checkpoint_model_files(state_dir: str | Path) -> list[Path]:
    """Return the active model payload files for legacy or sharded checkpoints."""
    root = Path(state_dir)
    legacy = root / "model.pt"
    if legacy.is_file():
        return [legacy]

    index_path = root / MODEL_INDEX_FILENAME
    if not index_path.is_file():
        return []
    manifest = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("model shard index must be a JSON object")
    if int(manifest.get("schema", 0) or 0) != 1:
        raise ValueError("unsupported model shard index schema")
    if str(manifest.get("format") or "") != MODEL_SHARD_FORMAT:
        raise ValueError("unsupported model shard format")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("model shard index has no shards")

    files: list[Path] = []
    seen: set[str] = set()
    for row in shards:
        if not isinstance(row, dict):
            raise ValueError("invalid model shard entry")
        name = str(row.get("file") or "")
        if not name or name in seen:
            raise ValueError("invalid or duplicate model shard filename")
        seen.add(name)
        path = _safe_checkpoint_member(root, name)
        if not path.is_file():
            raise FileNotFoundError(f"missing model shard: {name}")
        declared_bytes = int(row.get("bytes", 0) or 0)
        if declared_bytes and path.stat().st_size != declared_bytes:
            raise ValueError(f"model shard size mismatch: {name}")
        declared_sha = str(row.get("sha256") or "")
        if declared_sha and _sha256_file(path) != declared_sha:
            raise ValueError(f"model shard digest mismatch: {name}")
        files.append(path)
    return files


def checkpoint_has_model(state_dir: str | Path) -> bool:
    try:
        return bool(checkpoint_model_files(state_dir))
    except Exception:
        return False


def checkpoint_model_sha256(state_dir: str | Path) -> str:
    """Stable checkpoint-model digest.

    Preserve the historical single-file digest for legacy checkpoints. Sharded
    checkpoints use a canonical digest over shard names + shard digests.
    """
    root = Path(state_dir)
    files = checkpoint_model_files(root)
    if not files:
        raise FileNotFoundError("checkpoint has no model payload")
    if len(files) == 1 and files[0].name == "model.pt":
        return _sha256_file(files[0])

    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _load_model_state(root: Path, torch):
    legacy = root / "model.pt"
    if legacy.is_file():
        return torch.load(legacy, map_location="cpu", weights_only=True)

    index_path = root / MODEL_INDEX_FILENAME
    files = checkpoint_model_files(root)
    if not index_path.is_file() or not files:
        raise FileNotFoundError(
            "generalist checkpoint requires model.pt or a valid model.index.json"
        )
    manifest = json.loads(index_path.read_text(encoding="utf-8"))
    shard_rows = manifest.get("shards") or []
    by_name = {str(row.get("file")): row for row in shard_rows if isinstance(row, dict)}

    merged = {}
    for path in files:
        shard = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(shard, dict):
            raise ValueError(f"model shard is not a state dict: {path.name}")
        declared = by_name.get(path.name, {})
        declared_keys = declared.get("keys")
        if isinstance(declared_keys, list) and list(shard.keys()) != [str(x) for x in declared_keys]:
            raise ValueError(f"model shard key manifest mismatch: {path.name}")
        overlap = set(merged).intersection(shard)
        if overlap:
            raise ValueError(f"duplicate model tensor keys across shards: {sorted(overlap)[:8]}")
        merged.update(shard)
    return merged


def _save_model_state(root: Path, state_dict, torch) -> dict[str, Any]:
    """Persist a state dict without ever creating a Git-hostile giant file."""
    total_tensor_bytes = 0
    for name, tensor in state_dict.items():
        if not hasattr(tensor, "numel") or not hasattr(tensor, "element_size"):
            raise TypeError(f"unsupported non-tensor state entry: {name}")
        total_tensor_bytes += int(tensor.numel()) * int(tensor.element_size())

    legacy = root / "model.pt"
    index_path = root / MODEL_INDEX_FILENAME
    if total_tensor_bytes <= MODEL_SHARD_TARGET_BYTES:
        tmp_model = root / "model.pt.tmp"
        torch.save(state_dict, tmp_model)
        if tmp_model.stat().st_size > MODEL_SHARD_HARD_LIMIT_BYTES:
            tmp_model.unlink(missing_ok=True)
        else:
            tmp_model.replace(legacy)
            index_path.unlink(missing_ok=True)
            for stale in root.glob("model-shard-*.pt"):
                stale.unlink(missing_ok=True)
            return {
                "kind": "single",
                "format": "torch-state-dict",
                "files": [{
                    "file": legacy.name,
                    "bytes": int(legacy.stat().st_size),
                    "sha256": _sha256_file(legacy),
                }],
                "tensor_bytes": int(total_tensor_bytes),
            }

    nonce = uuid.uuid4().hex[:12]
    shard_specs: list[tuple[str, dict[str, Any]]] = []
    current: dict[str, Any] = {}
    current_bytes = 0

    def flush() -> None:
        nonlocal current, current_bytes
        if not current:
            return
        shard_specs.append((f"model-shard-{nonce}-{len(shard_specs)+1:05d}.pt", current))
        current = {}
        current_bytes = 0

    for name, tensor in state_dict.items():
        tensor_bytes = int(tensor.numel()) * int(tensor.element_size())
        if tensor_bytes > MODEL_SHARD_HARD_LIMIT_BYTES:
            raise ValueError(
                f"single tensor {name} is too large for safe checkpoint sharding: "
                f"{tensor_bytes} bytes"
            )
        if current and current_bytes + tensor_bytes > MODEL_SHARD_TARGET_BYTES:
            flush()
        current[name] = tensor
        current_bytes += tensor_bytes
    flush()

    rows = []
    created: list[Path] = []
    try:
        for filename, shard in shard_specs:
            path = root / filename
            torch.save(shard, path)
            created.append(path)
            size = int(path.stat().st_size)
            if size > MODEL_SHARD_HARD_LIMIT_BYTES:
                raise ValueError(
                    f"serialized model shard exceeds hard limit: {filename}={size}"
                )
            rows.append({
                "file": filename,
                "bytes": size,
                "sha256": _sha256_file(path),
                "keys": list(shard.keys()),
            })

        manifest = {
            "schema": 1,
            "format": MODEL_SHARD_FORMAT,
            "tensor_bytes": int(total_tensor_bytes),
            "total_serialized_bytes": int(sum(row["bytes"] for row in rows)),
            "shards": rows,
        }
        tmp_index = root / f"{MODEL_INDEX_FILENAME}.tmp"
        tmp_index.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_index.replace(index_path)

        # The new index is now authoritative. Remove stale payloads only after
        # the atomic index switch so an interrupted save keeps a loadable model.
        legacy.unlink(missing_ok=True)
        active_names = {row["file"] for row in rows}
        for stale in root.glob("model-shard-*.pt"):
            if stale.name not in active_names:
                stale.unlink(missing_ok=True)
        return {
            "kind": "sharded",
            "format": MODEL_SHARD_FORMAT,
            "files": [
                {"file": row["file"], "bytes": row["bytes"], "sha256": row["sha256"]}
                for row in rows
            ],
            "tensor_bytes": int(total_tensor_bytes),
        }
    except Exception:
        # New shard names are unique and cannot be referenced by the previous
        # active index, so they are safe to remove on failure.
        for path in created:
            path.unlink(missing_ok=True)
        raise


class GeneralistRuntime:
    def __init__(self, model, config: GeneralistLMConfig, tokenizer=None, *, device: str = "cpu"):
        import torch
        self.torch = torch
        self.config = config.validate()
        if tokenizer is None:
            if self.config.tokenizer_version != "byte-v1":
                raise ValueError("non-byte checkpoints require an explicit tokenizer artifact")
            tokenizer = ByteTokenizer()
        self.tokenizer = tokenizer
        if self.tokenizer.version != self.config.tokenizer_version:
            raise ValueError("tokenizer version does not match model config")
        if self.tokenizer.vocab_size != self.config.vocab_size:
            raise ValueError("tokenizer vocabulary does not match model config")
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()

    @classmethod
    def fresh(cls, config: GeneralistLMConfig | None = None, *, tokenizer=None, device: str = "cpu") -> "GeneralistRuntime":
        config = (config or GeneralistLMConfig()).validate()
        if tokenizer is None and config.tokenizer_version != "byte-v1":
            raise ValueError("fresh non-byte models require an explicit tokenizer")
        return cls(CausalTransformerLM(config), config, tokenizer=tokenizer, device=device)

    @classmethod
    def from_checkpoint(cls, state_dir: str | Path, *, device: str = "cpu") -> "GeneralistRuntime":
        import torch
        root = Path(state_dir)
        cfg_path = root / "config.json"
        if not cfg_path.exists() or not checkpoint_has_model(root):
            raise FileNotFoundError(
                "generalist checkpoint requires config.json and a valid model payload"
            )
        config = GeneralistLMConfig.from_dict(json.loads(cfg_path.read_text(encoding="utf-8")))
        if config.tokenizer_version == "byte-v1":
            tokenizer = ByteTokenizer()
        elif config.tokenizer_version == "bpe-v1":
            tokenizer_path = root / "tokenizer.json"
            if not tokenizer_path.exists():
                raise FileNotFoundError("bpe-v1 checkpoint requires tokenizer.json")
            tokenizer = BPETokenizer.load(tokenizer_path)
        else:  # validate() should already make this unreachable.
            raise ValueError(f"unsupported tokenizer version: {config.tokenizer_version}")

        metadata_path = root / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                raise ValueError("checkpoint metadata must be a JSON object")
            declared_version = str(metadata.get("tokenizer_version", config.tokenizer_version))
            if declared_version != config.tokenizer_version:
                raise ValueError("checkpoint tokenizer version does not match metadata")
            declared_digest = metadata.get("tokenizer_digest")
            if config.tokenizer_version == "bpe-v1":
                if str(declared_digest or "") != tokenizer.digest:
                    raise ValueError("checkpoint tokenizer digest does not match metadata")
            elif declared_digest not in {None, ""}:
                raise ValueError("byte-v1 checkpoint metadata must not declare a tokenizer digest")

        model = CausalTransformerLM(config)
        model.load_state_dict(_load_model_state(root, torch), strict=True)
        return cls(model, config, tokenizer=tokenizer, device=device)

    def save_checkpoint(self, state_dir: str | Path, *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        root = Path(state_dir)
        root.mkdir(parents=True, exist_ok=True)
        self.model.to("cpu")
        try:
            model_storage = _save_model_state(
                root,
                self.model.state_dict(),
                self.torch,
            )
        finally:
            self.model.to(self.device)
        (root / "config.json").write_text(json.dumps(self.config.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        tokenizer_path = root / "tokenizer.json"
        if self.tokenizer.version == "bpe-v1":
            if not isinstance(self.tokenizer, BPETokenizer):
                raise TypeError("bpe-v1 runtime requires BPETokenizer")
            self.tokenizer.save(tokenizer_path)
            tokenizer_digest = self.tokenizer.digest
        else:
            tokenizer_path.unlink(missing_ok=True)
            tokenizer_digest = None
        info = {
            **dict(metadata or {}),
            "format": "airi-generalist-lm-v1",
            "tokenizer_version": self.tokenizer.version,
            "tokenizer_digest": tokenizer_digest,
            "parameters": parameter_count(self.model),
            "model_storage": model_storage,
            "model_sha256": checkpoint_model_sha256(root),
        }
        (root / "metadata.json").write_text(json.dumps(info, indent=2, sort_keys=True), encoding="utf-8")
        return info

    def _generate_ids(
        self,
        prompt_ids: list[int],
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
    ) -> list[int]:
        context = prompt_ids[-self.config.context_length:]
        tensor = self.torch.tensor([context], dtype=self.torch.long, device=self.device)
        out = self.model.generate(
            tensor,
            max_new_tokens=max_new_tokens,
            eos_token_id=EOS,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )[0].tolist()
        return out[len(context):]

    def complete(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
    ) -> str:
        ids = self.tokenizer.encode(prompt, bos=True)
        generated = self._generate_ids(
            ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        return self.tokenizer.decode(generated).split("\x00", 1)[0]

    def generate(self, prompt: str, *, max_new_tokens: int = 128) -> str:
        """Benchmark/backend-compatible deterministic text generation."""
        return self.complete(prompt, max_new_tokens=max_new_tokens, temperature=0.0)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int = 192,
        temperature: float = 0.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
    ) -> str:
        ids = self.tokenizer.serialize_messages(messages, add_generation_prompt=True)
        generated = self._generate_ids(
            ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        return self.tokenizer.decode(generated)

    def complete_code(self, instruction: str, *, language: str = "python", max_new_tokens: int = 256) -> str:
        return self.chat([
            {"role": "system", "content": "You are a precise coding assistant. Return code unless explanation is explicitly requested."},
            {"role": "user", "content": f"Language: {language}\nTask: {instruction}"},
        ], max_new_tokens=max_new_tokens)

    def analyze_data(self, rows: list[dict[str, Any]], question: str, *, max_new_tokens: int = 256) -> str:
        payload = json.dumps(rows[:200], ensure_ascii=False, separators=(",", ":"))
        return self.chat([
            {"role": "system", "content": "Analyze structured data carefully. Show computed values and state uncertainty when evidence is insufficient."},
            {"role": "user", "content": f"DATA={payload}\nQUESTION={question}"},
        ], max_new_tokens=max_new_tokens)

    def request_tool(
        self,
        messages: list[dict[str, str]],
        tools: dict[str, dict[str, Any]],
        *,
        max_new_tokens: int = 192,
    ) -> ToolCall:
        augmented = [
            {"role": "system", "content": tool_prompt(tools)},
            *messages,
        ]
        text = self.chat(augmented, max_new_tokens=max_new_tokens)
        return parse_tool_call(text, allowed_tools=set(tools))
