from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any


LATTICE_FAMILY = "airi-native-lattice"
LATTICE_ARCHITECTURE_VERSION = "airi-lattice-v0"


def _torch():
    import torch
    from torch import nn
    from torch.nn import functional as F
    return torch, nn, F


@dataclass
class AiriLatticeConfig:
    """Experimental AIRI-native recurrent architecture.

    The design is intentionally optimized for the AIRI constraints:
      - linear-time recurrent sequence processing instead of quadratic attention;
      - constant-size multi-timescale state;
      - surprise-gated writes into slower/deeper memory bands;
      - learned communication between memory bands;
      - sparse top-k experts whose router sees both token and memory state;
      - shared adaptive reasoning passes, with more passes only on novel inputs.

    It is a research family. It must beat equal-budget Native Transformer
    baselines before it can become canonical.
    """

    vocab_size: int = 264
    context_length: int = 256
    d_model: int = 96
    n_cells: int = 2
    memory_bands: int = 4
    d_expert: int = 192
    n_experts: int = 8
    active_experts: int = 2
    max_reasoning_steps: int = 4
    surprise_threshold: float = 0.20
    predictive_error_memory: bool = False
    local_recurrence: bool = False
    surprise_power: float = 1.5
    deep_write_power: float = 1.6
    lattice_mix: float = 0.20
    memory_decay_min: float = 0.25
    memory_decay_max: float = 0.995
    dropout: float = 0.0
    rms_eps: float = 1e-6
    init_std: float = 0.02
    tokenizer_version: str = "byte-v1"
    tie_embeddings: bool = True
    architecture_version: str = LATTICE_ARCHITECTURE_VERSION

    def validate(self) -> "AiriLatticeConfig":
        self.vocab_size = int(self.vocab_size)
        self.context_length = int(self.context_length)
        self.d_model = int(self.d_model)
        self.n_cells = int(self.n_cells)
        self.memory_bands = int(self.memory_bands)
        self.d_expert = int(self.d_expert)
        self.n_experts = int(self.n_experts)
        self.active_experts = int(self.active_experts)
        self.max_reasoning_steps = int(self.max_reasoning_steps)
        self.surprise_threshold = float(self.surprise_threshold)
        self.predictive_error_memory = bool(self.predictive_error_memory)
        self.local_recurrence = bool(self.local_recurrence)
        self.surprise_power = float(self.surprise_power)
        self.deep_write_power = float(self.deep_write_power)
        self.lattice_mix = float(self.lattice_mix)
        self.memory_decay_min = float(self.memory_decay_min)
        self.memory_decay_max = float(self.memory_decay_max)
        self.dropout = float(self.dropout)
        self.rms_eps = float(self.rms_eps)
        self.init_std = float(self.init_std)

        if self.architecture_version != LATTICE_ARCHITECTURE_VERSION:
            raise ValueError("unsupported AIRI Lattice architecture_version")
        if self.tokenizer_version not in {"byte-v1", "bpe-v1"}:
            raise ValueError("unsupported AIRI Lattice tokenizer_version")
        if not (16 <= self.vocab_size <= 262144):
            raise ValueError("vocab_size out of AIRI Lattice bounds")
        if not (16 <= self.context_length <= 1_048_576):
            raise ValueError("context_length out of AIRI Lattice bounds")
        if not (16 <= self.d_model <= 8192):
            raise ValueError("d_model out of AIRI Lattice bounds")
        if not (1 <= self.n_cells <= 32):
            raise ValueError("n_cells out of AIRI Lattice bounds")
        if not (2 <= self.memory_bands <= 32):
            raise ValueError("memory_bands out of AIRI Lattice bounds")
        if not (self.d_model <= self.d_expert <= 65536):
            raise ValueError("d_expert out of AIRI Lattice bounds")
        if not (2 <= self.n_experts <= 256):
            raise ValueError("n_experts out of AIRI Lattice bounds")
        if not (1 <= self.active_experts <= min(16, self.n_experts)):
            raise ValueError("active_experts out of AIRI Lattice bounds")
        if not (1 <= self.max_reasoning_steps <= 32):
            raise ValueError("max_reasoning_steps out of AIRI Lattice bounds")
        if not (0.0 <= self.surprise_threshold < 0.95):
            raise ValueError("surprise_threshold out of AIRI Lattice bounds")
        if not (0.25 <= self.surprise_power <= 8.0):
            raise ValueError("surprise_power out of AIRI Lattice bounds")
        if not (0.25 <= self.deep_write_power <= 8.0):
            raise ValueError("deep_write_power out of AIRI Lattice bounds")
        if not (0.0 <= self.lattice_mix <= 1.0):
            raise ValueError("lattice_mix out of AIRI Lattice bounds")
        if not (0.001 <= self.memory_decay_min < self.memory_decay_max < 0.99999):
            raise ValueError("invalid AIRI Lattice memory decay range")
        if not (0.0 <= self.dropout <= 0.5):
            raise ValueError("dropout out of AIRI Lattice bounds")
        if not (1e-8 <= self.rms_eps <= 1e-3):
            raise ValueError("rms_eps out of AIRI Lattice bounds")
        if not (1e-5 <= self.init_std <= 0.2):
            raise ValueError("init_std out of AIRI Lattice bounds")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AiriLatticeConfig":
        if not isinstance(raw, dict):
            raise ValueError("AIRI Lattice config must be an object")
        return cls(**dict(raw)).validate()


class AiriLatticeLM:
    """Factory for the experimental AIRI Lattice causal language model.

    Unlike a Transformer, the model never attends over the whole prefix.
    Each token updates a constant-size hierarchy of recurrent memory states.
    Slow memory bands only receive strong writes for surprising information.
    Learned band-to-band routing forms the "lattice". Sparse experts and a
    shared recurrent reasoner spend additional compute only where needed.
    """

    def __new__(cls, config: AiriLatticeConfig):
        torch, nn, F = _torch()
        config = config.validate()

        class RMSNorm(nn.Module):
            def __init__(self, width: int):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(width))

            def forward(self, x):
                scale = torch.rsqrt(
                    x.float().pow(2).mean(dim=-1, keepdim=True)
                    + config.rms_eps
                )
                return (x * scale.to(dtype=x.dtype)) * self.weight

        class SparseExperts(nn.Module):
            """Vectorized top-k experts.

            Expert tensors are gathered only for selected routes. This keeps
            sparse semantics while avoiding a Python loop over every expert,
            which was the dominant CPU overhead in the first real swarm.
            """
            def __init__(self):
                super().__init__()
                self.router = nn.Linear(config.d_model * 2, config.n_experts, bias=False)
                self.up_weight = nn.Parameter(torch.empty(
                    config.n_experts,
                    config.d_model,
                    config.d_expert,
                ))
                self.gate_weight = nn.Parameter(torch.empty(
                    config.n_experts,
                    config.d_model,
                    config.d_expert,
                ))
                self.down_weight = nn.Parameter(torch.empty(
                    config.n_experts,
                    config.d_expert,
                    config.d_model,
                ))
                nn.init.normal_(self.up_weight, mean=0.0, std=config.init_std)
                nn.init.normal_(self.gate_weight, mean=0.0, std=config.init_std)
                nn.init.normal_(self.down_weight, mean=0.0, std=config.init_std)

            def forward(self, x, memory):
                router_input = torch.cat([x, memory], dim=-1)
                logits = self.router(router_input)
                values, indices = torch.topk(
                    logits,
                    k=config.active_experts,
                    dim=-1,
                )
                weights = torch.softmax(values, dim=-1)

                # [batch, top_k, d_model, d_expert]
                selected_up = self.up_weight[indices]
                selected_gate = self.gate_weight[indices]
                up = torch.einsum("bd,bkdh->bkh", x, selected_up)
                gate = torch.einsum("bd,bkdh->bkh", x, selected_gate)
                hidden = F.silu(gate) * up

                # [batch, top_k, d_expert, d_model]
                selected_down = self.down_weight[indices]
                expert_output = torch.einsum(
                    "bkh,bkhd->bkd",
                    hidden,
                    selected_down,
                )
                output = (expert_output * weights.unsqueeze(-1)).sum(dim=1)
                usage = torch.bincount(
                    indices.reshape(-1),
                    minlength=config.n_experts,
                ).to(device=x.device, dtype=x.dtype)
                return output, usage

            def expert_parameter_count(self):
                return (
                    self.up_weight.numel()
                    + self.gate_weight.numel()
                    + self.down_weight.numel()
                )

        class LatticeCell(nn.Module):
            def __init__(self, cell_index: int):
                super().__init__()
                self.cell_index = int(cell_index)
                self.input_norm = RMSNorm(config.d_model)
                self.memory_norm = RMSNorm(config.d_model)
                self.local_gate = (
                    nn.Linear(config.d_model * 2, config.d_model, bias=False)
                    if config.local_recurrence else None
                )
                self.local_candidate = (
                    nn.Linear(config.d_model * 2, config.d_model, bias=False)
                    if config.local_recurrence else None
                )
                self.predict_proj = (
                    nn.Linear(config.d_model, config.d_model, bias=False)
                    if config.predictive_error_memory else None
                )
                self.write_proj = nn.Linear(
                    config.d_model * 2,
                    config.d_model,
                    bias=False,
                )
                self.write_gate = nn.Linear(
                    config.d_model,
                    config.memory_bands,
                    bias=True,
                )
                self.query_proj = nn.Linear(
                    config.d_model,
                    config.d_model,
                    bias=False,
                )
                self.memory_out = nn.Linear(
                    config.d_model,
                    config.d_model,
                    bias=False,
                )
                self.band_routes = nn.Parameter(
                    torch.eye(config.memory_bands)
                    + 0.01 * torch.randn(config.memory_bands, config.memory_bands)
                )
                self.experts = SparseExperts()
                self.output_norm = RMSNorm(config.d_model)
                self.dropout = nn.Dropout(config.dropout)

                # Geometric decay schedule: early bands react quickly, deep
                # bands integrate over much longer horizons. They remain
                # trainable, so MATHESIS can later initialize/evolve them.
                lo = math.log(config.memory_decay_min)
                hi = math.log(config.memory_decay_max)
                target = torch.exp(
                    torch.linspace(lo, hi, config.memory_bands)
                ).clamp(1e-5, 1 - 1e-5)
                logits = torch.log(target / (1.0 - target))
                self.decay_logits = nn.Parameter(logits)

            def initial_state(self, batch_size: int, *, device, dtype):
                return torch.zeros(
                    config.memory_bands,
                    batch_size,
                    config.d_model,
                    device=device,
                    dtype=dtype,
                )

            def forward(self, x, state):
                # x: [batch, d_model], state: [bands, batch, d_model]
                if state.ndim != 3:
                    raise ValueError("invalid AIRI Lattice state rank")
                if state.shape[0] != config.memory_bands:
                    raise ValueError("invalid AIRI Lattice memory band count")
                if state.shape[1] != x.shape[0] or state.shape[2] != config.d_model:
                    raise ValueError("invalid AIRI Lattice state shape")

                fast = self.memory_norm(state[0])
                base = x
                if self.local_gate is not None and self.local_candidate is not None:
                    local_input = torch.cat([self.input_norm(x), fast], dim=-1)
                    local_gate = torch.sigmoid(self.local_gate(local_input))
                    local_candidate = torch.tanh(self.local_candidate(local_input))
                    base = x + local_gate * local_candidate
                x_norm = self.input_norm(base)

                route = torch.softmax(self.band_routes, dim=-1)
                routed = torch.einsum("ij,jbd->ibd", route, state)
                previous_summary = state.mean(dim=0)

                if self.predict_proj is not None:
                    prediction = self.predict_proj(previous_summary)
                    reference = prediction
                    write_source = x_norm - prediction
                else:
                    reference = fast
                    write_source = x_norm

                # Surprise can either measure mismatch with fast memory or,
                # in predictive-error mode, a learned latent prediction error.
                cosine = F.cosine_similarity(
                    x_norm.float(),
                    reference.float(),
                    dim=-1,
                    eps=1e-6,
                ).clamp(-1.0, 1.0)
                surprise = ((1.0 - cosine) * 0.5).to(dtype=x.dtype)

                candidate = torch.tanh(
                    self.write_proj(torch.cat([write_source, previous_summary], dim=-1))
                )
                raw_gate = torch.sigmoid(self.write_gate(x_norm))

                band_depth = torch.linspace(
                    0.0,
                    1.0,
                    config.memory_bands,
                    device=x.device,
                    dtype=x.dtype,
                )
                surprise_base = surprise.clamp_min(1e-4).unsqueeze(-1)
                depth_exponent = (
                    1.0 + band_depth * config.deep_write_power
                ).unsqueeze(0)
                depth_gate = surprise_base.pow(depth_exponent)
                write_strength = (raw_gate * depth_gate).transpose(0, 1).unsqueeze(-1)

                decay = torch.sigmoid(self.decay_logits).to(
                    device=x.device,
                    dtype=x.dtype,
                ).view(config.memory_bands, 1, 1)
                proposal = (
                    write_strength * candidate.unsqueeze(0)
                    + (1.0 - write_strength)
                    * config.lattice_mix
                    * routed
                )
                new_state = decay * state + (1.0 - decay) * proposal

                query = self.query_proj(x_norm)
                scores = torch.einsum(
                    "bd,ibd->bi",
                    query,
                    self.memory_norm(new_state),
                ) / math.sqrt(config.d_model)
                band_weights = torch.softmax(scores, dim=-1)
                memory = torch.einsum("bi,ibd->bd", band_weights, new_state)

                expert_delta, usage = self.experts(x_norm, memory)
                out = base + self.dropout(self.memory_out(memory) + expert_delta)
                return self.output_norm(out), new_state, surprise, usage

        class Reasoner(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = RMSNorm(config.d_model)
                self.gate = nn.Linear(config.d_model * 2, config.d_model, bias=False)
                self.up = nn.Linear(config.d_model, config.d_expert, bias=False)
                self.down = nn.Linear(config.d_expert, config.d_model, bias=False)

            def forward(self, x, memory):
                h = self.norm(x)
                gate = torch.sigmoid(self.gate(torch.cat([h, memory], dim=-1)))
                delta = self.down(F.silu(self.up(h)))
                return x + gate * delta

        class Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = config
                self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
                self.cells = nn.ModuleList(
                    [LatticeCell(i) for i in range(config.n_cells)]
                )
                self.reasoner = Reasoner()
                self.final_norm = RMSNorm(config.d_model)
                self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
                if config.tie_embeddings:
                    self.lm_head.weight = self.token_embedding.weight
                self.apply(self._init_weights)

            @staticmethod
            def _init_weights(module):
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=config.init_std)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0.0, std=config.init_std)

            def initial_state(self, batch_size: int, *, device=None, dtype=None):
                if device is None:
                    device = self.token_embedding.weight.device
                if dtype is None:
                    dtype = self.token_embedding.weight.dtype
                return tuple(
                    cell.initial_state(batch_size, device=device, dtype=dtype)
                    for cell in self.cells
                )

            def _reasoning_budget(self, surprise):
                if config.max_reasoning_steps <= 1:
                    return torch.ones_like(surprise, dtype=torch.long)
                normalized = (
                    (surprise - config.surprise_threshold)
                    / max(1e-6, 1.0 - config.surprise_threshold)
                ).clamp(0.0, 1.0)
                shaped = normalized.pow(config.surprise_power)
                extra = torch.floor(
                    shaped * (config.max_reasoning_steps - 1) + 1e-6
                ).to(dtype=torch.long)
                return 1 + extra

            def forward(
                self,
                input_ids,
                labels=None,
                *,
                lattice_state=None,
                return_state: bool = False,
            ):
                if input_ids.ndim != 2:
                    raise ValueError("input_ids must have shape [batch, time]")
                batch_size, sequence_length = input_ids.shape
                if sequence_length <= 0:
                    raise ValueError("AIRI Lattice requires at least one token")
                if sequence_length > config.context_length:
                    raise ValueError("sequence exceeds AIRI Lattice context_length")
                if labels is not None and labels.shape != input_ids.shape:
                    raise ValueError("labels must match input_ids shape")

                x_tokens = self.token_embedding(input_ids)
                if lattice_state is None:
                    states = list(self.initial_state(
                        batch_size,
                        device=x_tokens.device,
                        dtype=x_tokens.dtype,
                    ))
                else:
                    if len(lattice_state) != len(self.cells):
                        raise ValueError("AIRI Lattice state cell count mismatch")
                    states = [item for item in lattice_state]

                outputs = []
                surprise_sum = x_tokens.new_zeros(())
                reasoning_sum = x_tokens.new_zeros(())
                token_count = 0
                expert_usage = x_tokens.new_zeros(config.n_experts)

                for time_index in range(sequence_length):
                    h = x_tokens[:, time_index, :]
                    cell_surprises = []
                    for cell_index, cell in enumerate(self.cells):
                        h, new_state, surprise, usage = cell(
                            h,
                            states[cell_index],
                        )
                        states[cell_index] = new_state
                        cell_surprises.append(surprise)
                        expert_usage = expert_usage + usage

                    surprise = torch.stack(cell_surprises, dim=0).mean(dim=0)
                    memory_summary = torch.stack(
                        [state.mean(dim=0) for state in states],
                        dim=0,
                    ).mean(dim=0)
                    budgets = self._reasoning_budget(surprise)

                    # Shared recurrent reasoning weights: hard tokens spend more
                    # compute without multiplying the parameter count.
                    reasoning_steps = torch.zeros(
                        batch_size,
                        device=h.device,
                        dtype=torch.long,
                    )
                    for step in range(config.max_reasoning_steps):
                        active = budgets > step
                        if not bool(active.any()):
                            break
                        rows = active.nonzero(as_tuple=True)[0]
                        refined = self.reasoner(
                            h.index_select(0, rows),
                            memory_summary.index_select(0, rows),
                        )
                        delta = refined - h.index_select(0, rows)
                        h = h.index_add(0, rows, delta)
                        reasoning_steps = reasoning_steps + active.to(torch.long)

                    outputs.append(h)
                    surprise_sum = surprise_sum + surprise.mean()
                    reasoning_sum = reasoning_sum + reasoning_steps.float().mean()
                    token_count += 1

                hidden = torch.stack(outputs, dim=1)
                logits = self.lm_head(self.final_norm(hidden))
                loss = None
                if labels is not None:
                    loss = F.cross_entropy(
                        logits[:, :-1, :].contiguous().view(-1, config.vocab_size),
                        labels[:, 1:].contiguous().view(-1),
                        ignore_index=-100,
                    )

                denom = max(1, token_count)
                usage_total = expert_usage.sum().clamp_min(1.0)
                stats = {
                    "mean_surprise": float(
                        (surprise_sum / denom).detach().float().cpu()
                    ),
                    "mean_reasoning_steps": float(
                        (reasoning_sum / denom).detach().float().cpu()
                    ),
                    "expert_usage": (
                        expert_usage / usage_total
                    ).detach().float().cpu().tolist(),
                    "state_elements": int(
                        config.n_cells
                        * config.memory_bands
                        * batch_size
                        * config.d_model
                    ),
                }
                return {
                    "logits": logits,
                    "loss": loss,
                    "lattice_state": tuple(states) if return_state else None,
                    "stats": stats,
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
            ):
                self.eval()
                if input_ids.ndim != 2:
                    raise ValueError("input_ids must have shape [batch, time]")
                prompt = input_ids[:, -config.context_length:]
                result = self(prompt, return_state=True)
                state = result["lattice_state"]
                out = input_ids
                logits = result["logits"][:, -1, :]

                for _ in range(max(0, int(max_new_tokens))):
                    if temperature is None or float(temperature) <= 0:
                        token = logits.argmax(dim=-1, keepdim=True)
                    else:
                        scaled = logits / max(1e-5, float(temperature))
                        if top_k is not None and 0 < int(top_k) < scaled.shape[-1]:
                            values, _ = torch.topk(scaled, int(top_k))
                            cutoff = values[:, -1].unsqueeze(-1)
                            scaled = scaled.masked_fill(
                                scaled < cutoff,
                                float("-inf"),
                            )
                        token = torch.multinomial(torch.softmax(scaled, dim=-1), 1)

                    out = torch.cat([out, token], dim=1)
                    if eos_token_id is not None and bool(
                        torch.all(token == int(eos_token_id))
                    ):
                        break
                    result = self(
                        token,
                        lattice_state=state,
                        return_state=True,
                    )
                    state = result["lattice_state"]
                    logits = result["logits"][:, -1, :]
                return out

        return Decoder()


def lattice_parameter_count(config: AiriLatticeConfig) -> int:
    model = AiriLatticeLM(config)
    return sum(parameter.numel() for parameter in model.parameters())


def lattice_active_parameter_estimate(config: AiriLatticeConfig) -> int:
    """Estimate parameters touched per token.

    Embeddings/output and shared cell/reasoner parameters are counted fully.
    Only active_experts/n_experts of expert parameters are counted because
    top-k routing executes only those expert MLPs for each token.
    """
    config = config.validate()
    torch, nn, _F = _torch()
    model = AiriLatticeLM(config)
    total = sum(p.numel() for p in model.parameters())

    expert_total = sum(
        int(cell.experts.expert_parameter_count())
        for cell in model.cells
    )
    active_fraction = config.active_experts / config.n_experts
    return int(round(total - expert_total + expert_total * active_fraction))


def lattice_state_bytes(
    config: AiriLatticeConfig,
    *,
    batch_size: int = 1,
    bytes_per_element: int = 4,
) -> int:
    config = config.validate()
    return (
        config.n_cells
        * config.memory_bands
        * int(batch_size)
        * config.d_model
        * int(bytes_per_element)
    )
