from __future__ import annotations

from pathlib import Path
from typing import Any


class LocalTransformersBackend:
    """Optional adapter for already-downloaded Hugging Face causal LMs.

    Autonomous execution is local-files-only by default. This allows a capable
    open-weight model to participate in the same benchmark/promotion system
    without letting the evolution worker download code or arbitrary artifacts.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cpu",
        local_files_only: bool = True,
    ):
        root = Path(model_path).expanduser().resolve()
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError("local transformers model directory does not exist")
        if not local_files_only:
            raise ValueError("autonomous transformers backend must remain local-files-only")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers is not installed") from exc

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(root),
            local_files_only=True,
            trust_remote_code=False,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            str(root),
            local_files_only=True,
            trust_remote_code=False,
        )
        self.model.to(device)
        self.model.eval()
        self.device = device

    def generate(self, prompt: str, *, max_new_tokens: int = 192) -> str:
        import torch
        encoded = self.tokenizer(str(prompt), return_tensors="pt")
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.tokenizer.eos_token_id
            out = self.model.generate(
                **encoded,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                pad_token_id=pad_token_id,
            )
        generated = out[0, encoded["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def chat(self, messages: list[dict[str, str]], *, max_new_tokens: int = 256) -> str:
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
