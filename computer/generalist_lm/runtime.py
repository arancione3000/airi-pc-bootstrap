from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .bpe_tokenizer import BPETokenizer
from .model import CausalTransformerLM, GeneralistLMConfig, parameter_count
from .tokenizer import ASSISTANT, EOS, ByteTokenizer
from .tool_protocol import ToolCall, parse_tool_call, tool_prompt



MODEL_MANIFEST_FORMAT = "airi-generalist-sharded-state-v1"
MODEL_SHARD_RAW_BYTES = 32 * 1024 * 1024
MODEL_SHARD_PREFIX = "model-shard-"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_model_manifest(state_dir: str | Path) -> dict[str, Any] | None:
    root = Path(state_dir)
    model_path = root if root.name == "model.pt" else root / "model.pt"
    if not model_path.is_file():
        return None
    try:
        raw = json.loads(model_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict) or raw.get("format") != MODEL_MANIFEST_FORMAT:
        return None
    shards = raw.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("sharded checkpoint manifest requires non-empty shards")
    return raw


def checkpoint_model_shard_names(state_dir: str | Path) -> list[str]:
    manifest = checkpoint_model_manifest(state_dir)
    if manifest is None:
        return []
    names: list[str] = []
    for row in manifest["shards"]:
        if not isinstance(row, dict):
            raise ValueError("invalid sharded checkpoint entry")
        name = str(row.get("name") or "")
        if (
            not name.startswith(MODEL_SHARD_PREFIX)
            or not name.endswith(".pt")
            or "/" in name
            or "\\" in name
            or name in names
        ):
            raise ValueError("invalid sharded checkpoint file name")
        names.append(name)
    return names


def checkpoint_model_files(
    state_dir: str | Path,
    *,
    verify: bool = True,
) -> list[Path]:
    root = Path(state_dir)
    model_path = root / "model.pt"
    if not model_path.is_file():
        raise FileNotFoundError("generalist checkpoint requires model.pt")
    manifest = checkpoint_model_manifest(root)
    files = [model_path]
    if manifest is None:
        return files
    by_name = {
        str(row.get("name") or ""): row
        for row in manifest.get("shards") or []
        if isinstance(row, dict)
    }
    for name in checkpoint_model_shard_names(root):
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"missing checkpoint model shard: {name}")
        if verify:
            row = by_name[name]
            expected_size = int(row.get("size", 0) or 0)
            if expected_size and path.stat().st_size != expected_size:
                raise ValueError(f"checkpoint shard size mismatch: {name}")
            expected_sha = str(row.get("sha256") or "")
            if expected_sha and _sha256_file(path) != expected_sha:
                raise ValueError(f"checkpoint shard digest mismatch: {name}")
        files.append(path)
    return files


def checkpoint_model_size(state_dir: str | Path) -> int:
    return sum(path.stat().st_size for path in checkpoint_model_files(state_dir))


def checkpoint_model_digest(state_dir: str | Path) -> str:
    digest = hashlib.sha256()
    for path in checkpoint_model_files(state_dir):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def load_checkpoint_state_dict(state_dir: str | Path) -> dict[str, Any]:
    import torch

    root = Path(state_dir)
    model_path = root / "model.pt"
    manifest = checkpoint_model_manifest(root)
    if manifest is None:
        return torch.load(model_path, map_location="cpu", weights_only=True)

    state: dict[str, Any] = {}
    for shard_path in checkpoint_model_files(root)[1:]:
        shard = torch.load(shard_path, map_location="cpu", weights_only=True)
        if not isinstance(shard, dict):
            raise ValueError(f"invalid checkpoint shard payload: {shard_path.name}")
        overlap = set(state).intersection(shard)
        if overlap:
            raise ValueError(
                "checkpoint shards contain duplicate tensors: "
                + ", ".join(sorted(overlap)[:8])
            )
        state.update(shard)
    expected = int(manifest.get("tensor_count", 0) or 0)
    if expected and len(state) != expected:
        raise ValueError(
            f"checkpoint tensor count mismatch: expected={expected} actual={len(state)}"
        )
    return state


def _tensor_storage_bytes(tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _partition_state_dict(
    state: dict[str, Any],
    *,
    max_raw_bytes: int,
) -> list[dict[str, Any]]:
    shards: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    current_bytes = 0
    for name, tensor in state.items():
        size = _tensor_storage_bytes(tensor)
        if current and current_bytes + size > max_raw_bytes:
            shards.append(current)
            current = {}
            current_bytes = 0
        current[name] = tensor
        current_bytes += size
    if current:
        shards.append(current)
    return shards


def save_checkpoint_state_dict(
    torch_module,
    state_dir: str | Path,
    state: dict[str, Any],
    *,
    shard_raw_bytes: int = MODEL_SHARD_RAW_BYTES,
) -> dict[str, Any]:
    root = Path(state_dir)
    root.mkdir(parents=True, exist_ok=True)
    model_path = root / "model.pt"
    total_raw_bytes = sum(_tensor_storage_bytes(tensor) for tensor in state.values())

    if total_raw_bytes <= int(shard_raw_bytes):
        tmp_model = root / "model.pt.tmp"
        torch_module.save(state, tmp_model)
        tmp_model.replace(model_path)
        for stale in root.glob(f"{MODEL_SHARD_PREFIX}*.pt"):
            stale.unlink(missing_ok=True)
        return {
            "storage": "monolithic",
            "files": ["model.pt"],
            "raw_tensor_bytes": int(total_raw_bytes),
        }

    groups = _partition_state_dict(
        state,
        max_raw_bytes=max(1, int(shard_raw_bytes)),
    )
    total = len(groups)
    manifest_rows: list[dict[str, Any]] = []
    live_names: set[str] = set()

    for index, shard in enumerate(groups, 1):
        tmp_shard = root / f".model-shard-{index:05d}.tmp"
        torch_module.save(shard, tmp_shard)
        sha = _sha256_file(tmp_shard)
        name = (
            f"{MODEL_SHARD_PREFIX}{index:05d}-of-{total:05d}-"
            f"{sha[:12]}.pt"
        )
        target = root / name
        tmp_shard.replace(target)
        live_names.add(name)
        manifest_rows.append({
            "name": name,
            "sha256": sha,
            "size": int(target.stat().st_size),
            "tensors": list(shard.keys()),
        })

    manifest = {
        "format": MODEL_MANIFEST_FORMAT,
        "version": 1,
        "tensor_count": len(state),
        "raw_tensor_bytes": int(total_raw_bytes),
        "shard_max_raw_bytes": int(shard_raw_bytes),
        "shards": manifest_rows,
    }
    tmp_manifest = root / "model.pt.tmp"
    tmp_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp_manifest.replace(model_path)

    for stale in root.glob(f"{MODEL_SHARD_PREFIX}*.pt"):
        if stale.name not in live_names:
            stale.unlink(missing_ok=True)

    # Verify every shard after the atomic manifest swap. The manifest hashes
    # commit the logical model identity to all shard bytes.
    checkpoint_model_files(root, verify=True)
    return {
        "storage": "sharded",
        "files": ["model.pt", *sorted(live_names)],
        "raw_tensor_bytes": int(total_raw_bytes),
        "shards": int(total),
    }


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
        model_path = root / "model.pt"
        if not cfg_path.exists() or not model_path.exists():
            raise FileNotFoundError("generalist checkpoint requires config.json and model.pt")
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
        model.load_state_dict(load_checkpoint_state_dict(root))
        return cls(model, config, tokenizer=tokenizer, device=device)

    def save_checkpoint(self, state_dir: str | Path, *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        root = Path(state_dir)
        root.mkdir(parents=True, exist_ok=True)
        self.model.to("cpu")
        try:
            storage = save_checkpoint_state_dict(
                self.torch,
                root,
                self.model.state_dict(),
            )
            (root / "config.json").write_text(
                json.dumps(self.config.to_dict(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
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
                "model_storage": storage,
            }
            (root / "metadata.json").write_text(
                json.dumps(info, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            return info
        finally:
            self.model.to(self.device)

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
