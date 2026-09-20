from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .model import CausalTransformerLM, GeneralistLMConfig, parameter_count
from .tokenizer import ASSISTANT, EOS, ByteTokenizer
from .tool_protocol import ToolCall, parse_tool_call, tool_prompt


class GeneralistRuntime:
    def __init__(self, model, config: GeneralistLMConfig, tokenizer: ByteTokenizer | None = None, *, device: str = "cpu"):
        import torch
        self.torch = torch
        self.config = config.validate()
        self.tokenizer = tokenizer or ByteTokenizer()
        if self.tokenizer.vocab_size != self.config.vocab_size:
            raise ValueError("tokenizer vocabulary does not match model config")
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()

    @classmethod
    def fresh(cls, config: GeneralistLMConfig | None = None, *, device: str = "cpu") -> "GeneralistRuntime":
        config = (config or GeneralistLMConfig()).validate()
        return cls(CausalTransformerLM(config), config, device=device)

    @classmethod
    def from_checkpoint(cls, state_dir: str | Path, *, device: str = "cpu") -> "GeneralistRuntime":
        import torch
        root = Path(state_dir)
        cfg_path = root / "config.json"
        model_path = root / "model.pt"
        if not cfg_path.exists() or not model_path.exists():
            raise FileNotFoundError("generalist checkpoint requires config.json and model.pt")
        config = GeneralistLMConfig.from_dict(json.loads(cfg_path.read_text(encoding="utf-8")))
        model = CausalTransformerLM(config)
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        return cls(model, config, device=device)

    def save_checkpoint(self, state_dir: str | Path, *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        root = Path(state_dir)
        root.mkdir(parents=True, exist_ok=True)
        tmp_model = root / "model.pt.tmp"
        self.torch.save(self.model.to("cpu").state_dict(), tmp_model)
        tmp_model.replace(root / "model.pt")
        (root / "config.json").write_text(json.dumps(self.config.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        info = {
            "format": "airi-generalist-lm-v1",
            "tokenizer_version": self.tokenizer.version,
            "parameters": parameter_count(self.model),
            **dict(metadata or {}),
        }
        (root / "metadata.json").write_text(json.dumps(info, indent=2, sort_keys=True), encoding="utf-8")
        self.model.to(self.device)
        return info

    def _generate_ids(self, prompt_ids: list[int], *, max_new_tokens: int, temperature: float = 0.0, top_k: int | None = None) -> list[int]:
        context = prompt_ids[-self.config.context_length:]
        tensor = self.torch.tensor([context], dtype=self.torch.long, device=self.device)
        out = self.model.generate(
            tensor,
            max_new_tokens=max_new_tokens,
            eos_token_id=EOS,
            temperature=temperature,
            top_k=top_k,
        )[0].tolist()
        return out[len(context):]

    def complete(self, prompt: str, *, max_new_tokens: int = 128, temperature: float = 0.0, top_k: int | None = None) -> str:
        ids = self.tokenizer.encode(prompt, bos=True)
        generated = self._generate_ids(ids, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
        return self.tokenizer.decode(generated).split("\x00", 1)[0]

    def generate(self, prompt: str, *, max_new_tokens: int = 128) -> str:
        """Benchmark/backend-compatible deterministic text generation."""
        return self.complete(prompt, max_new_tokens=max_new_tokens, temperature=0.0)

    def chat(self, messages: list[dict[str, str]], *, max_new_tokens: int = 192, temperature: float = 0.0) -> str:
        ids = self.tokenizer.serialize_messages(messages, add_generation_prompt=True)
        generated = self._generate_ids(ids, max_new_tokens=max_new_tokens, temperature=temperature)
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
