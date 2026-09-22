from __future__ import annotations

from .model import GeneralistLMConfig, _torch


class CausalGatedRecurrentLM:
    """Factory wrapper for AIRI's non-Transformer gated recurrent causal LM.

    The family replaces self-attention with a stack of recurrent token mixers.
    It keeps the same causal-LM forward/generation contract as the Transformer
    so training, evaluation and promotion use identical held-out benchmarks.

    This implementation is intentionally a trusted primitive, not generated
    Python. Architecture Search may select/configure it through Architecture IR
    but may not rewrite the primitive itself.
    """

    def __new__(cls, config: GeneralistLMConfig):
        torch, nn, F = _torch()
        config = config.validate()
        if config.architecture_family != "gated_recurrent_v1":
            raise ValueError("CausalGatedRecurrentLM requires gated_recurrent_v1")

        class RMSNorm(nn.Module):
            def __init__(self, width: int, eps: float = 1e-6):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(width))
                self.eps = eps

            def forward(self, x):
                scale = torch.rsqrt(
                    x.pow(2).mean(dim=-1, keepdim=True) + self.eps
                )
                return x * scale * self.weight

        def norm():
            if config.norm_type == "rmsnorm":
                return RMSNorm(config.d_model)
            return nn.LayerNorm(config.d_model)

        class SwiGLU(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = nn.Linear(
                    config.d_model,
                    config.d_ff,
                    bias=config.bias,
                )
                self.value = nn.Linear(
                    config.d_model,
                    config.d_ff,
                    bias=config.bias,
                )
                self.down = nn.Linear(
                    config.d_ff,
                    config.d_model,
                    bias=config.bias,
                )

            def forward(self, x):
                return self.down(F.silu(self.gate(x)) * self.value(x))

        class GeluFF(nn.Module):
            def __init__(self):
                super().__init__()
                self.up = nn.Linear(
                    config.d_model,
                    config.d_ff,
                    bias=config.bias,
                )
                self.down = nn.Linear(
                    config.d_ff,
                    config.d_model,
                    bias=config.bias,
                )

            def forward(self, x):
                return self.down(F.gelu(self.up(x)))

        class RecurrentBlock(nn.Module):
            def __init__(self, layer_index: int):
                super().__init__()
                self.layer_index = int(layer_index)
                self.mixer_norm = norm()
                self.ff_norm = norm()
                self.rnn = nn.GRU(
                    input_size=config.d_model,
                    hidden_size=config.d_model,
                    num_layers=1,
                    bias=config.bias,
                    batch_first=True,
                    dropout=0.0,
                    bidirectional=False,
                )
                self.ff = (
                    SwiGLU()
                    if config.ff_variant == "swiglu"
                    else GeluFF()
                )
                self.drop = nn.Dropout(config.dropout)

            def forward(
                self,
                x,
                *,
                past_state=None,
                use_cache: bool = False,
            ):
                if config.norm_placement == "pre":
                    recurrent_out, state = self.rnn(
                        self.mixer_norm(x),
                        past_state,
                    )
                    x = x + self.drop(recurrent_out)
                    x = x + self.drop(self.ff(self.ff_norm(x)))
                else:
                    recurrent_out, state = self.rnn(x, past_state)
                    x = self.mixer_norm(x + self.drop(recurrent_out))
                    x = self.ff_norm(x + self.drop(self.ff(x)))
                return x, state if use_cache else None

        class GatedRecurrentLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = config
                self.token_embedding = nn.Embedding(
                    config.vocab_size,
                    config.d_model,
                )
                self.blocks = nn.ModuleList([
                    RecurrentBlock(layer_index)
                    for layer_index in range(config.n_layers)
                ])
                self.final_norm = (
                    norm()
                    if config.norm_placement == "pre"
                    else nn.Identity()
                )
                self.lm_head = nn.Linear(
                    config.d_model,
                    config.vocab_size,
                    bias=False,
                )
                if config.tie_embeddings:
                    self.lm_head.weight = self.token_embedding.weight
                self.apply(self._init_weights)

            @staticmethod
            def _init_weights(module):
                if isinstance(module, nn.Linear):
                    nn.init.normal_(
                        module.weight,
                        mean=0.0,
                        std=0.02,
                    )
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(
                        module.weight,
                        mean=0.0,
                        std=0.02,
                    )
                elif isinstance(module, nn.GRU):
                    for name, parameter in module.named_parameters():
                        if "weight" in name:
                            nn.init.normal_(
                                parameter,
                                mean=0.0,
                                std=0.02,
                            )
                        elif "bias" in name:
                            nn.init.zeros_(parameter)

            def forward(
                self,
                input_ids,
                labels=None,
                *,
                past_key_values=None,
                use_cache: bool = False,
            ):
                if input_ids.ndim != 2:
                    raise ValueError(
                        "input_ids must have shape [batch, time]"
                    )
                bsz, seqlen = input_ids.shape
                if seqlen <= 0:
                    raise ValueError("input_ids cannot be empty")
                if (
                    past_key_values is None
                    and seqlen > config.context_length
                ):
                    raise ValueError(
                        "sequence exceeds model context_length"
                    )

                if past_key_values is None:
                    states = [None] * len(self.blocks)
                else:
                    if len(past_key_values) != len(self.blocks):
                        raise ValueError(
                            "recurrent cache length must match layers"
                        )
                    states = list(past_key_values)
                    for state in states:
                        if state is None:
                            continue
                        if state.ndim != 3:
                            raise ValueError(
                                "recurrent state must be [1,batch,width]"
                            )
                        if tuple(state.shape) != (
                            1,
                            int(bsz),
                            int(config.d_model),
                        ):
                            raise ValueError(
                                "invalid recurrent cache shape"
                            )

                x = self.token_embedding(input_ids)
                presents = []
                for block, state in zip(self.blocks, states):
                    x, present = block(
                        x,
                        past_state=state,
                        use_cache=use_cache,
                    )
                    if use_cache:
                        presents.append(present)

                logits = self.lm_head(self.final_norm(x))
                loss = None
                if labels is not None:
                    if labels.shape != input_ids.shape:
                        raise ValueError(
                            "labels must match input_ids shape"
                        )
                    loss = F.cross_entropy(
                        logits[:, :-1, :]
                        .contiguous()
                        .view(-1, config.vocab_size),
                        labels[:, 1:]
                        .contiguous()
                        .view(-1),
                        ignore_index=-100,
                    )
                return {
                    "logits": logits,
                    "loss": loss,
                    "past_key_values": (
                        tuple(presents)
                        if use_cache
                        else None
                    ),
                }

            @torch.no_grad()
            def generate(
                self,
                input_ids,
                *,
                max_new_tokens: int = 64,
                eos_token_id: int | None = 2,
                temperature: float = 0.0,
                top_k: int | None = None,
                top_p: float | None = None,
                repetition_penalty: float = 1.0,
                use_cache: bool = True,
            ):
                self.eval()
                out = input_ids[:, -config.context_length:]
                prompt_length = int(out.shape[1])

                if use_cache:
                    result = self(out, use_cache=True)
                    past_key_values = result["past_key_values"]
                    logits = result["logits"][:, -1, :]
                    cached_tokens = int(out.shape[1])
                else:
                    past_key_values = None
                    logits = None
                    cached_tokens = 0

                for step in range(max(0, int(max_new_tokens))):
                    if not use_cache or logits is None:
                        window = out[:, -config.context_length:]
                        result = self(
                            window,
                            use_cache=bool(use_cache),
                        )
                        logits = result["logits"][:, -1, :]
                        past_key_values = (
                            result["past_key_values"]
                            if use_cache
                            else None
                        )
                        cached_tokens = int(window.shape[1])

                    if temperature is None or float(temperature) <= 0.0:
                        next_token = logits.argmax(
                            dim=-1,
                            keepdim=True,
                        )
                    else:
                        sampled_logits = logits.clone()
                        penalty = max(
                            1.0,
                            float(repetition_penalty or 1.0),
                        )
                        if (
                            penalty > 1.0
                            and int(out.shape[1]) > prompt_length
                        ):
                            for batch_index in range(
                                int(out.shape[0])
                            ):
                                repeated = torch.unique(
                                    out[
                                        batch_index,
                                        prompt_length:,
                                    ]
                                )
                                values = sampled_logits[
                                    batch_index,
                                    repeated,
                                ]
                                sampled_logits[
                                    batch_index,
                                    repeated,
                                ] = torch.where(
                                    values < 0,
                                    values * penalty,
                                    values / penalty,
                                )
                        sampled_logits = sampled_logits / max(
                            1e-5,
                            float(temperature),
                        )
                        if (
                            top_k is not None
                            and 0 < int(top_k) < sampled_logits.shape[-1]
                        ):
                            values, _ = torch.topk(
                                sampled_logits,
                                int(top_k),
                            )
                            cutoff = values[:, -1].unsqueeze(-1)
                            sampled_logits = sampled_logits.masked_fill(
                                sampled_logits < cutoff,
                                float("-inf"),
                            )
                        if (
                            top_p is not None
                            and 0.0 < float(top_p) < 1.0
                        ):
                            sorted_logits, sorted_indices = torch.sort(
                                sampled_logits,
                                descending=True,
                                dim=-1,
                            )
                            sorted_probs = torch.softmax(
                                sorted_logits,
                                dim=-1,
                            )
                            cumulative = torch.cumsum(
                                sorted_probs,
                                dim=-1,
                            )
                            remove = cumulative > float(top_p)
                            remove[..., 1:] = remove[
                                ..., :-1
                            ].clone()
                            remove[..., 0] = False
                            sorted_logits = sorted_logits.masked_fill(
                                remove,
                                float("-inf"),
                            )
                            filtered = torch.full_like(
                                sampled_logits,
                                float("-inf"),
                            )
                            filtered.scatter_(
                                dim=-1,
                                index=sorted_indices,
                                src=sorted_logits,
                            )
                            sampled_logits = filtered
                        probs = torch.softmax(
                            sampled_logits,
                            dim=-1,
                        )
                        next_token = torch.multinomial(
                            probs,
                            num_samples=1,
                        )

                    out = torch.cat([out, next_token], dim=1)
                    if (
                        eos_token_id is not None
                        and bool(
                            torch.all(
                                next_token
                                == int(eos_token_id)
                            )
                        )
                    ):
                        break

                    if not use_cache:
                        logits = None
                        continue

                    # Keep recurrent memory bounded to the declared context.
                    # Once the cached token count reaches the context limit,
                    # recompute from the last context window and discard older
                    # hidden state instead of silently granting infinite memory.
                    if cached_tokens >= config.context_length:
                        window = out[:, -config.context_length:]
                        result = self(window, use_cache=True)
                        past_key_values = result["past_key_values"]
                        logits = result["logits"][:, -1, :]
                        cached_tokens = int(window.shape[1])
                    else:
                        result = self(
                            next_token,
                            past_key_values=past_key_values,
                            use_cache=True,
                        )
                        past_key_values = result["past_key_values"]
                        logits = result["logits"][:, -1, :]
                        cached_tokens += 1
                return out

        return GatedRecurrentLM()
