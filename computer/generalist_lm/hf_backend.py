from __future__ import annotations

import importlib.util
from pathlib import Path
import re
from typing import Any


_DEVICE_MAP_STRATEGIES = {"auto", "balanced", "balanced_low_0", "sequential"}
_DTYPE_NAMES = {"auto", "float16", "bfloat16", "float32"}
_MEMORY_RE = re.compile(r"^\d+(?:\.\d+)?\s*(?:B|KB|MB|GB|TB|KiB|MiB|GiB|TiB)$", re.IGNORECASE)


def _normalized_device_map(value: str | dict[str, Any] | None) -> str | dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        token = value.strip().lower()
        if not token:
            return None
        if token not in _DEVICE_MAP_STRATEGIES:
            raise ValueError(f"unsupported Transformers device_map strategy: {value!r}")
        return token
    if not isinstance(value, dict) or not value:
        raise ValueError("device_map must be a supported strategy string or non-empty mapping")
    checked: dict[str, Any] = {}
    for key, target in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("device_map keys must be non-empty strings")
        if not isinstance(target, (str, int)) or isinstance(target, bool):
            raise ValueError("device_map targets must be device strings or integer CUDA indices")
        checked[key] = target
    return checked


def _normalized_max_memory(value: dict[Any, Any] | None) -> dict[Any, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or not value:
        raise ValueError("max_memory must be a non-empty mapping")
    checked: dict[Any, Any] = {}
    for raw_key, raw_limit in value.items():
        if isinstance(raw_key, str) and raw_key.isdigit():
            key: Any = int(raw_key)
        elif isinstance(raw_key, (str, int)) and not isinstance(raw_key, bool):
            key = raw_key
        else:
            raise ValueError("max_memory device keys must be strings or integers")
        if isinstance(raw_limit, int) and not isinstance(raw_limit, bool):
            if raw_limit <= 0:
                raise ValueError("max_memory integer limits must be positive")
            limit: Any = raw_limit
        elif isinstance(raw_limit, str) and _MEMORY_RE.fullmatch(raw_limit.strip()):
            limit = raw_limit.strip()
        else:
            raise ValueError("max_memory limits must be positive bytes or explicit memory strings")
        checked[key] = limit
    return checked


def _resolved_torch_dtype(name: str | None, torch_module: Any) -> Any:
    if name is None:
        return None
    token = str(name).strip().lower()
    if not token:
        return None
    if token not in _DTYPE_NAMES:
        raise ValueError(f"unsupported Transformers dtype: {name!r}")
    if token == "auto":
        return "auto"
    return {
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "float32": torch_module.float32,
    }[token]


class LocalTransformersBackend:
    """Local-only Hugging Face causal-LM adapter with optional sharded loading."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cpu",
        device_map: str | dict[str, Any] | None = None,
        torch_dtype: str | None = None,
        max_memory: dict[Any, Any] | None = None,
        offload_folder: str | Path | None = None,
        local_files_only: bool = True,
    ):
        root = Path(model_path).expanduser().resolve()
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError("local transformers model directory does not exist")
        if not local_files_only:
            raise ValueError("autonomous transformers backend must remain local-files-only")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("torch and transformers must be installed") from exc

        checked_device_map = _normalized_device_map(device_map)
        checked_max_memory = _normalized_max_memory(max_memory)
        checked_dtype = _resolved_torch_dtype(torch_dtype, torch)
        if checked_device_map is not None and importlib.util.find_spec("accelerate") is None:
            raise RuntimeError("Transformers device_map loading requires accelerate")

        load_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "trust_remote_code": False,
        }
        if checked_device_map is not None:
            load_kwargs["device_map"] = checked_device_map
            load_kwargs["low_cpu_mem_usage"] = True
        if checked_dtype is not None:
            load_kwargs["torch_dtype"] = checked_dtype
        if checked_max_memory is not None:
            if checked_device_map is None:
                raise ValueError("max_memory requires device_map")
            load_kwargs["max_memory"] = checked_max_memory
        if offload_folder is not None:
            if checked_device_map is None:
                raise ValueError("offload_folder requires device_map")
            folder = Path(offload_folder).expanduser().resolve()
            folder.mkdir(parents=True, exist_ok=True)
            load_kwargs["offload_folder"] = str(folder)

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(root),
            local_files_only=True,
            trust_remote_code=False,
        )
        self.model = AutoModelForCausalLM.from_pretrained(str(root), **load_kwargs)
        if checked_device_map is None:
            self.model.to(device)
        self.model.eval()

        self.model_type = str(
            getattr(getattr(self.model, "config", None), "model_type", "") or ""
        ).strip().lower()
        if self.model_type == "gpt_oss":
            from .harmony_adapter import harmony_runtime_status
            harmony = harmony_runtime_status()
            if not harmony.get("available"):
                raise RuntimeError(str(harmony.get("reason") or "Harmony runtime unavailable"))

        self.device = str(device)
        self.device_map = checked_device_map
        self.input_device = self._resolve_input_device(torch, fallback=device)
        self.load_policy = {
            "device": self.device,
            "device_map": checked_device_map,
            "torch_dtype": str(torch_dtype).strip().lower() if torch_dtype else None,
            "max_memory": checked_max_memory,
            "offload_folder": str(Path(offload_folder).expanduser().resolve()) if offload_folder else None,
        }

    def _resolve_input_device(self, torch_module: Any, *, fallback: str) -> Any:
        try:
            embeddings = self.model.get_input_embeddings()
            weight = getattr(embeddings, "weight", None)
            device = getattr(weight, "device", None)
            if device is not None and str(device) != "meta":
                return device
        except Exception:
            pass
        device = getattr(self.model, "device", None)
        if device is not None and str(device) != "meta":
            return device
        mapping = getattr(self.model, "hf_device_map", None)
        if isinstance(mapping, dict):
            for target in mapping.values():
                if isinstance(target, int):
                    return torch_module.device(f"cuda:{target}")
                if isinstance(target, str) and target not in {"disk", "meta"}:
                    return torch_module.device(target)
        return torch_module.device(fallback)

    def _generate_encoded(
        self,
        encoded: dict[str, Any],
        *,
        max_new_tokens: int,
        eos_token_id: int | list[int] | None = None,
    ) -> list[int]:
        import torch

        placed = {key: value.to(self.input_device) for key, value in encoded.items()}
        kwargs: dict[str, Any] = {
            **placed,
            "max_new_tokens": int(max_new_tokens),
            "do_sample": False,
        }
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is not None:
            kwargs["pad_token_id"] = pad_token_id
        if eos_token_id is not None:
            kwargs["eos_token_id"] = eos_token_id

        with torch.no_grad():
            out = self.model.generate(**kwargs)
        generated = out[0, placed["input_ids"].shape[1]:]
        return [int(token) for token in generated.detach().cpu().tolist()]

    def generate(self, prompt: str, *, max_new_tokens: int = 192) -> str:
        encoded = self.tokenizer(str(prompt), return_tensors="pt")
        generated = self._generate_encoded(
            dict(encoded),
            max_new_tokens=max_new_tokens,
        )
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def _chat_harmony(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int,
    ) -> str:
        from .harmony_adapter import harmony_stop_tokens, parse_harmony_assistant_tokens

        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise RuntimeError("gpt-oss requires a Transformers Harmony chat template")
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except (ValueError, TypeError) as exc:
            raise RuntimeError("gpt-oss Harmony chat template could not render messages") from exc
        if not isinstance(prompt, str) or not prompt:
            raise RuntimeError("gpt-oss Harmony chat template returned an empty prompt")

        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        generated = self._generate_encoded(
            dict(encoded),
            max_new_tokens=max_new_tokens,
            eos_token_id=harmony_stop_tokens(),
        )
        parsed = parse_harmony_assistant_tokens(generated)
        if not parsed.get("ok"):
            raise RuntimeError(str(parsed.get("reason") or "Harmony completion is not a final answer"))
        return str(parsed.get("final_text") or "")

    def chat(self, messages: list[dict[str, str]], *, max_new_tokens: int = 256) -> str:
        if self.model_type == "gpt_oss":
            return self._chat_harmony(messages, max_new_tokens=max_new_tokens)

        prompt = None
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except (ValueError, TypeError):
                prompt = None
        if not prompt:
            prompt = "\n".join(
                f"{m.get('role','user')}: {m.get('content','')}"
                for m in messages
            ) + "\nassistant:"
        return self.generate(prompt, max_new_tokens=max_new_tokens)
