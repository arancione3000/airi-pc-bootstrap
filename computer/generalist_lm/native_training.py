from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
from typing import Any, Sequence

from .native_data import (
    NativeCorpusDocument,
    NativeTokenBlock,
    load_native_corpus,
    load_native_tokenizer,
    pack_native_blocks,
    split_native_corpus,
)
from .native_foundation import (
    NATIVE_CONFIG_FILENAME,
    NATIVE_FAMILY,
    NATIVE_MANIFEST_FILENAME,
    NATIVE_MODEL_FILENAME,
    NATIVE_TOKENIZER_FILENAME,
    NativeFoundationConfig,
    NativeFoundationProvenance,
    _sha256_file,
    _sha256_json,
    load_native_checkpoint,
    native_checkpoint_status,
)
from .tokenizer import PAD


NATIVE_TRAINING_VERSION = "native-training-v1"
NATIVE_TRAINING_MANIFEST_FILENAME = "native-training.json"
NATIVE_TRAINER_STATE_FILENAME = "trainer-state.pt"


@dataclass
class NativeTrainConfig:
    max_steps: int = 1000
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    validation_fraction: float = 0.05
    max_eval_blocks: int = 128
    seed: int = 17
    device: str = "auto"
    precision: str = "auto"
    save_optimizer_state: bool = True
    domain_weights: dict[str, float] | None = None

    def validate(self) -> "NativeTrainConfig":
        self.max_steps = int(self.max_steps)
        self.micro_batch_size = int(self.micro_batch_size)
        self.gradient_accumulation_steps = int(self.gradient_accumulation_steps)
        self.warmup_steps = int(self.warmup_steps)
        self.max_eval_blocks = int(self.max_eval_blocks)
        self.seed = int(self.seed)
        self.learning_rate = float(self.learning_rate)
        self.min_learning_rate = float(self.min_learning_rate)
        self.weight_decay = float(self.weight_decay)
        self.grad_clip = float(self.grad_clip)
        self.validation_fraction = float(self.validation_fraction)
        self.device = str(self.device).strip().lower()
        self.precision = str(self.precision).strip().lower()

        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.micro_batch_size < 1:
            raise ValueError("micro_batch_size must be positive")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if not (0.0 < self.learning_rate <= 1.0):
            raise ValueError("learning_rate out of bounds")
        if not (0.0 <= self.min_learning_rate <= self.learning_rate):
            raise ValueError("min_learning_rate out of bounds")
        if not (0 <= self.warmup_steps < self.max_steps):
            raise ValueError("warmup_steps must be smaller than max_steps")
        if not (0.0 <= self.weight_decay <= 1.0):
            raise ValueError("weight_decay out of bounds")
        if not (0.0 < self.grad_clip <= 100.0):
            raise ValueError("grad_clip out of bounds")
        if not (0.0 < self.validation_fraction < 0.5):
            raise ValueError("validation_fraction must be between 0 and 0.5")
        if self.max_eval_blocks < 1:
            raise ValueError("max_eval_blocks must be positive")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu or cuda")
        if self.precision not in {"auto", "fp32", "fp16", "bf16"}:
            raise ValueError("precision must be auto, fp32, fp16 or bf16")

        if self.domain_weights is not None:
            clean: dict[str, float] = {}
            for raw_key, raw_value in self.domain_weights.items():
                key = str(raw_key).strip().lower()
                value = float(raw_value)
                if not key or not (0.01 <= value <= 100.0):
                    raise ValueError("invalid native domain weight")
                clean[key] = value
            self.domain_weights = clean
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    created_process_group: bool

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @property
    def main_process(self) -> bool:
        return self.rank == 0


def _resolve_device(torch, requested: str, local_rank: int):
    key = str(requested).strip().lower()
    if key == "auto":
        key = "cuda" if torch.cuda.is_available() else "cpu"
    if key == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for native training but is unavailable")
        torch.cuda.set_device(int(local_rank))
        return torch.device("cuda", int(local_rank))
    return torch.device("cpu")


def _distributed_context(torch, requested_device: str) -> tuple[DistributedContext, Any]:
    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    device = _resolve_device(torch, requested_device, local_rank)
    created = False
    if world_size > 1:
        if not torch.distributed.is_available():
            raise RuntimeError("torch.distributed is unavailable")
        if not torch.distributed.is_initialized():
            backend = "nccl" if device.type == "cuda" else "gloo"
            torch.distributed.init_process_group(backend=backend, init_method="env://")
            created = True
        actual_world = int(torch.distributed.get_world_size())
        actual_rank = int(torch.distributed.get_rank())
        if actual_world != world_size or actual_rank != rank:
            raise RuntimeError("distributed environment does not match process group")
    return DistributedContext(rank, local_rank, world_size, created), device


def _resolved_precision(torch, requested: str, device) -> tuple[str, Any | None]:
    key = str(requested).strip().lower()
    if key == "auto":
        if device.type == "cuda":
            key = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        else:
            key = "fp32"
    if key == "fp16" and device.type != "cuda":
        raise ValueError("fp16 native training currently requires CUDA")
    if key == "bf16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise ValueError("bf16 requested but unsupported by this CUDA device")
        if device.type not in {"cuda", "cpu"}:
            raise ValueError("bf16 native training requires CUDA or CPU")
        return key, torch.bfloat16
    if key == "fp16":
        return key, torch.float16
    return "fp32", None


def _autocast(torch, device, dtype):
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def _make_grad_scaler(torch, enabled: bool):
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


def _batch(torch, blocks: Sequence[NativeTokenBlock], indices: Sequence[int], *, device):
    ids = torch.tensor(
        [blocks[index].ids for index in indices],
        dtype=torch.long,
        device=device,
    )
    labels = ids.clone()
    labels[labels == PAD] = -100
    return ids, labels


def _learning_rate(config: NativeTrainConfig, step: int) -> float:
    if config.warmup_steps > 0 and step < config.warmup_steps:
        return config.learning_rate * float(step + 1) / float(config.warmup_steps)
    span = max(1, config.max_steps - config.warmup_steps)
    progress = min(1.0, max(0.0, float(step - config.warmup_steps) / float(span)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_learning_rate + (
        config.learning_rate - config.min_learning_rate
    ) * cosine


def _weighted_block_weights(
    blocks: Sequence[NativeTokenBlock],
    domain_weights: dict[str, float] | None,
) -> list[float]:
    domain_weights = domain_weights or {}
    weights: list[float] = []
    for block in blocks:
        weights.append(
            max(
                1e-9,
                float(block.weight) * float(domain_weights.get(block.domain, 1.0)),
            )
        )
    return weights


def _evaluate(
    torch,
    model,
    blocks: Sequence[NativeTokenBlock],
    *,
    device,
    dtype,
    context: DistributedContext,
    batch_size: int,
    max_blocks: int,
) -> float:
    if not blocks:
        raise ValueError("native validation produced no token blocks")
    indices = list(range(min(len(blocks), max(1, int(max_blocks)))))
    local_indices = indices[context.rank::context.world_size]
    weighted = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(local_indices), max(1, int(batch_size))):
            rows = local_indices[start:start + max(1, int(batch_size))]
            if not rows:
                continue
            ids, labels = _batch(torch, blocks, rows, device=device)
            with _autocast(torch, device, dtype):
                loss = model(ids, labels=labels)["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite native validation loss")
            weighted += float(loss.detach().float().cpu()) * len(rows)
            count += len(rows)

    if context.distributed:
        payload = torch.tensor([weighted, float(count)], dtype=torch.float64, device=device)
        torch.distributed.all_reduce(payload, op=torch.distributed.ReduceOp.SUM)
        weighted = float(payload[0].cpu())
        count = int(payload[1].cpu())
    if count <= 0:
        raise RuntimeError("native validation had no distributed samples")
    return weighted / count


def _load_resume_state(
    torch,
    source: Path,
    *,
    corpus_digest: str,
    optimizer,
    rng: random.Random,
) -> tuple[int, bool]:
    training_path = source / NATIVE_TRAINING_MANIFEST_FILENAME
    trainer_path = source / NATIVE_TRAINER_STATE_FILENAME
    if not training_path.is_file() or not trainer_path.is_file():
        return 0, False

    training = json.loads(training_path.read_text(encoding="utf-8"))
    if training.get("version") != NATIVE_TRAINING_VERSION:
        return 0, False
    if training.get("corpus_digest") != corpus_digest:
        return 0, False
    expected = str(training.get("trainer_state_digest", ""))
    if len(expected) != 64 or _sha256_file(trainer_path) != expected:
        raise RuntimeError("native trainer state digest mismatch")

    state = torch.load(trainer_path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or int(state.get("global_step", -1)) < 0:
        raise RuntimeError("invalid native trainer state")
    optimizer.load_state_dict(state["optimizer"])
    if "python_rng_state" in state:
        rng.setstate(state["python_rng_state"])
    if "torch_rng_state" in state:
        torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available() and state.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
    return int(state["global_step"]), True


def _save_descendant(
    torch,
    *,
    source: Path,
    output: Path,
    model,
    config: NativeFoundationConfig,
    source_status: dict[str, Any],
    optimizer,
    rng: random.Random,
    global_step: int,
    training_payload: dict[str, Any],
    save_optimizer_state: bool,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("native training output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)

    config_target = output / NATIVE_CONFIG_FILENAME
    config_target.write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tokenizer_digest = None
    if config.tokenizer_version == "bpe-v1":
        source_tokenizer = source / NATIVE_TOKENIZER_FILENAME
        if not source_tokenizer.is_file():
            raise FileNotFoundError(NATIVE_TOKENIZER_FILENAME)
        target_tokenizer = output / NATIVE_TOKENIZER_FILENAME
        shutil.copyfile(source_tokenizer, target_tokenizer)
        tokenizer_digest = _sha256_file(target_tokenizer)
        if tokenizer_digest != source_status.get("tokenizer_digest"):
            raise RuntimeError("native tokenizer changed while training")

    bare_model = getattr(model, "module", model)
    model_target = output / NATIVE_MODEL_FILENAME
    torch.save(bare_model.state_dict(), model_target)

    provenance = NativeFoundationProvenance(
        family=NATIVE_FAMILY,
        architecture_version=config.architecture_version,
        checkpoint_version=1,
        weights_origin="airi-native-descendant",
        root_seed=int(source_status["root_seed"]),
        tokenizer_version=config.tokenizer_version,
        tokenizer_digest=tokenizer_digest,
        config_digest=_sha256_json(config.to_dict()),
        model_digest=_sha256_file(model_target),
        parent_checkpoint_digest=str(source_status["checkpoint_digest"]),
        external_pretrained=False,
    ).validate()
    (output / NATIVE_MANIFEST_FILENAME).write_text(
        json.dumps(provenance.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    trainer_digest = None
    if save_optimizer_state:
        trainer_state = {
            "version": NATIVE_TRAINING_VERSION,
            "global_step": int(global_step),
            "optimizer": optimizer.state_dict(),
            "python_rng_state": rng.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        }
        trainer_path = output / NATIVE_TRAINER_STATE_FILENAME
        torch.save(trainer_state, trainer_path)
        trainer_digest = _sha256_file(trainer_path)

    payload = {
        **training_payload,
        "version": NATIVE_TRAINING_VERSION,
        "global_step": int(global_step),
        "trainer_state_digest": trainer_digest,
        "parent_checkpoint_digest": str(source_status["checkpoint_digest"]),
        "weights_origin": "airi-native-descendant",
        "external_pretrained": False,
    }
    (output / NATIVE_TRAINING_MANIFEST_FILENAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    status = native_training_status(output)
    if not status.get("ok"):
        raise RuntimeError(str(status.get("reason") or "invalid native training checkpoint"))
    return status


def native_training_status(state_dir: str | Path) -> dict[str, Any]:
    root = Path(state_dir).expanduser().resolve()
    checkpoint = native_checkpoint_status(root)
    if not checkpoint.get("ok"):
        return {
            "ok": False,
            "checkpoint": checkpoint,
            "reason": checkpoint.get("reason", "invalid native checkpoint"),
        }
    training_path = root / NATIVE_TRAINING_MANIFEST_FILENAME
    if not training_path.is_file():
        return {
            "ok": True,
            "checkpoint": checkpoint,
            "training": None,
            "reason": "valid native checkpoint without trainer metadata",
        }
    try:
        training = json.loads(training_path.read_text(encoding="utf-8"))
        if training.get("version") != NATIVE_TRAINING_VERSION:
            raise ValueError("unsupported native training version")
        if training.get("external_pretrained") is not False:
            raise ValueError("native training manifest permits external pretrained weights")
        if training.get("weights_origin") != "airi-native-descendant":
            raise ValueError("invalid native training weights origin")
        provenance = NativeFoundationProvenance.from_dict(
            json.loads((root / NATIVE_MANIFEST_FILENAME).read_text(encoding="utf-8"))
        )
        if training.get("parent_checkpoint_digest") != provenance.parent_checkpoint_digest:
            raise ValueError("native training parent checkpoint digest mismatch")
        trainer_digest = training.get("trainer_state_digest")
        if trainer_digest is not None:
            trainer = root / NATIVE_TRAINER_STATE_FILENAME
            if not trainer.is_file():
                raise FileNotFoundError(NATIVE_TRAINER_STATE_FILENAME)
            if _sha256_file(trainer) != trainer_digest:
                raise ValueError("native trainer state digest mismatch")
        return {
            "ok": True,
            "checkpoint": checkpoint,
            "training": training,
            "reason": "verified AIRI Native training checkpoint",
        }
    except Exception as exc:
        return {
            "ok": False,
            "checkpoint": checkpoint,
            "reason": f"invalid_native_training:{type(exc).__name__}:{exc}",
        }


def train_native_foundation(
    state_dir: str | Path,
    corpus_manifest: str | Path,
    *,
    allowed_roots: Sequence[str | Path],
    output_dir: str | Path,
    config: NativeTrainConfig | None = None,
    max_total_bytes: int = 2_000_000_000,
) -> dict[str, Any]:
    """Continue AIRI Native pretraining without any external pretrained model."""
    import torch

    train_config = (config or NativeTrainConfig()).validate()
    source = Path(state_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    report = load_native_corpus(
        corpus_manifest,
        allowed_roots=allowed_roots,
        max_total_bytes=int(max_total_bytes),
    )

    context, device = _distributed_context(torch, train_config.device)
    precision_name, autocast_dtype = _resolved_precision(
        torch,
        train_config.precision,
        device,
    )

    try:
        model, model_config, source_status = load_native_checkpoint(
            source,
            device=str(device),
        )
        tokenizer = load_native_tokenizer(source, model_config.tokenizer_version)
        train_documents, validation_documents = split_native_corpus(
            report.documents,
            validation_fraction=train_config.validation_fraction,
            seed=train_config.seed,
        )
        train_blocks = pack_native_blocks(
            train_documents,
            tokenizer,
            context_length=model_config.context_length,
        )
        validation_blocks = pack_native_blocks(
            validation_documents,
            tokenizer,
            context_length=model_config.context_length,
        )
        if not train_blocks or not validation_blocks:
            raise ValueError("native corpus produced empty train or validation blocks")

        if context.distributed:
            from torch.nn.parallel import DistributedDataParallel
            model = DistributedDataParallel(
                model,
                device_ids=[context.local_rank] if device.type == "cuda" else None,
                output_device=context.local_rank if device.type == "cuda" else None,
            )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=train_config.weight_decay,
        )
        rng = random.Random(train_config.seed + context.rank)
        torch.manual_seed(train_config.seed + context.rank)

        start_step, optimizer_resumed = _load_resume_state(
            torch,
            source,
            corpus_digest=report.corpus_digest,
            optimizer=optimizer,
            rng=rng,
        )
        if start_step >= train_config.max_steps:
            raise ValueError(
                "max_steps must be greater than the resumed native global_step"
            )

        block_weights = _weighted_block_weights(
            train_blocks,
            train_config.domain_weights,
        )
        initial_validation_loss = _evaluate(
            torch,
            model,
            validation_blocks,
            device=device,
            dtype=autocast_dtype,
            context=context,
            batch_size=train_config.micro_batch_size,
            max_blocks=train_config.max_eval_blocks,
        )

        scaler = _make_grad_scaler(
            torch,
            enabled=(precision_name == "fp16" and device.type == "cuda"),
        )
        train_losses: list[float] = []
        last_lr = train_config.learning_rate

        for step in range(start_step, train_config.max_steps):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0.0
            last_lr = _learning_rate(train_config, step)
            for group in optimizer.param_groups:
                group["lr"] = last_lr

            for micro_step in range(train_config.gradient_accumulation_steps):
                rows = rng.choices(
                    range(len(train_blocks)),
                    weights=block_weights,
                    k=train_config.micro_batch_size,
                )
                ids, labels = _batch(torch, train_blocks, rows, device=device)
                sync_context = nullcontext()
                if (
                    context.distributed
                    and micro_step < train_config.gradient_accumulation_steps - 1
                    and hasattr(model, "no_sync")
                ):
                    sync_context = model.no_sync()
                with sync_context:
                    with _autocast(torch, device, autocast_dtype):
                        loss = model(ids, labels=labels)["loss"]
                        scaled_loss = loss / train_config.gradient_accumulation_steps
                    if not torch.isfinite(loss):
                        raise RuntimeError("non-finite AIRI Native training loss")
                    if scaler is not None:
                        scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()
                accumulated += float(loss.detach().float().cpu())

            if scaler is not None:
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                train_config.grad_clip,
            )
            if not torch.isfinite(grad_norm):
                raise RuntimeError("non-finite AIRI Native gradient norm")
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            train_losses.append(
                accumulated / train_config.gradient_accumulation_steps
            )

        final_validation_loss = _evaluate(
            torch,
            model,
            validation_blocks,
            device=device,
            dtype=autocast_dtype,
            context=context,
            batch_size=train_config.micro_batch_size,
            max_blocks=train_config.max_eval_blocks,
        )

        if context.distributed:
            torch.distributed.barrier()

        result: dict[str, Any] = {
            "ok": bool(final_validation_loss < initial_validation_loss),
            "training_version": NATIVE_TRAINING_VERSION,
            "source_checkpoint_digest": source_status["checkpoint_digest"],
            "corpus_digest": report.corpus_digest,
            "documents": len(report.documents),
            "train_documents": len(train_documents),
            "validation_documents": len(validation_documents),
            "train_blocks": len(train_blocks),
            "validation_blocks": len(validation_blocks),
            "start_step": start_step,
            "global_step": train_config.max_steps,
            "steps_run": train_config.max_steps - start_step,
            "optimizer_resumed": optimizer_resumed,
            "initial_validation_loss": initial_validation_loss,
            "final_validation_loss": final_validation_loss,
            "validation_loss_improvement": (
                initial_validation_loss - final_validation_loss
            ),
            "best_train_loss": min(train_losses),
            "last_train_loss": train_losses[-1],
            "last_learning_rate": last_lr,
            "precision": precision_name,
            "device_type": device.type,
            "world_size": context.world_size,
            "micro_batch_size": train_config.micro_batch_size,
            "gradient_accumulation_steps": train_config.gradient_accumulation_steps,
            "effective_batch_size": (
                train_config.micro_batch_size
                * train_config.gradient_accumulation_steps
                * context.world_size
            ),
            "config": train_config.to_dict(),
            "external_pretrained": False,
        }

        if context.main_process:
            status = _save_descendant(
                torch,
                source=source,
                output=output,
                model=model,
                config=model_config,
                source_status=source_status,
                optimizer=optimizer,
                rng=rng,
                global_step=train_config.max_steps,
                training_payload=result,
                save_optimizer_state=train_config.save_optimizer_state,
            )
            result["checkpoint"] = status["checkpoint"]
            result["output"] = str(output)
        if context.distributed:
            torch.distributed.barrier()
        return result
    finally:
        if context.created_process_group and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
