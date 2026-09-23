from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import re
from collections import Counter
import shutil
import time
from typing import Any

from .airi_pc_lab import (
    build_airi_pc_lab_rows,
    load_verified_lab_experiences,
    snapshot_airi_pc_lab,
    run_airi_pc_lab_probe,
)
from .bootstrap_data import (
    build_bootstrap_bundle,
    load_bootstrap_replay,
    write_bootstrap_replay,
)
from .curriculum import train_rows, validation_rows
from .curriculum_memory import CurriculumMemory, canary_rows
from .evolution import GeneralistGenome, progressive_scale_candidate
from .model import CausalTransformerLM, parameter_count
from .phase5_diagnostics import (
    degeneration_gate,
    evaluate_phase5_language,
    evaluate_sft_validation,
    protected_bootstrap_texts,
    sft_validation_gate,
)
from .pretraining import (
    corpus_loss,
    load_packed_block_cache,
    pack_causal_blocks,
    save_packed_block_cache,
)
from .research_cycle import (
    _continual_candidate_genome,
    _grouped_validation,
    _load_champion,
    _research_eligible,
    _research_score,
    _save_champion,
    _transfer_compatible_weights,
)
from .runtime import GeneralistRuntime
from .lineage_migration import refresh_live_lineage_manifest
from .tokenizer import PAD
from .training import (
    SFTExample,
    causal_training_objective,
    train_sft,
    train_sft_residual_recovery,
)


PHASE5_BOOTSTRAP_VERSION = "phase5-language-bootstrap-v3"
PHASE5_SFT_GUARD_VERSION = "phase5-sft-guard-v2"
PHASE5_ANTICOLLAPSE_VERSION = "phase5-anticollapse-v2"
PHASE5_LANGUAGE_REHABILITATION_VERSION = "phase5-language-rehabilitation-v1"
PHASE5_LANGUAGE_REHABILITATION_STAGES = (
    "R1_bilingual_foundations",
    "R2_simple_responses",
    "R3_short_dialogue",
)
PHASE5_CAPACITY_GROWTH_VERSION = "phase5-capacity-growth-v2"
PHASE5_BASE_UNIQUE_CORPUS_TOKENS = 5_000_000
PHASE5_100M_UNIQUE_CORPUS_TOKENS = 20_000_000
PHASE5_250M_UNIQUE_CORPUS_TOKENS = 40_000_000
PHASE5_500M_UNIQUE_CORPUS_TOKENS = 60_000_000
PHASE5_1B_UNIQUE_CORPUS_TOKENS = 100_000_000

# One-time human-assisted capacity gift for the current AIRI lineage only.
# Once this lineage reaches the requested tier the marker is persisted and this
# path becomes permanently dormant; future growth returns to AIRI's own search.
ASSISTED_CAPACITY_LINEAGE_ID = "airi-5d3d25177d2e83f7"
ASSISTED_CAPACITY_TARGET_PARAMETERS = 50_000_000
ASSISTED_CAPACITY_COMPLETION_FLOOR = 49_000_000


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp.replace(path)


def _load_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.is_file():
        return dict(default or {})
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"expected JSON object: {path}")
    return raw


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



OPTIMIZER_MANIFEST_FORMAT = "airi-phase5-sharded-optimizer-v1"
OPTIMIZER_SHARD_RAW_BYTES = 32 * 1024 * 1024
OPTIMIZER_SHARD_PREFIX = "optimizer-shard-"
OPTIMIZER_META_PREFIX = "optimizer-meta-"


def _optimizer_tensor_bytes(value: Any) -> int:
    if hasattr(value, "numel") and hasattr(value, "element_size"):
        return int(value.numel()) * int(value.element_size())
    if isinstance(value, dict):
        return sum(_optimizer_tensor_bytes(row) for row in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_optimizer_tensor_bytes(row) for row in value)
    return 0


def _optimizer_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict) or raw.get("format") != OPTIMIZER_MANIFEST_FORMAT:
        return None
    shards = raw.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("sharded optimizer manifest requires non-empty shards")
    return raw


def _optimizer_shard_paths(path: Path, *, verify: bool = True) -> list[Path]:
    manifest = _optimizer_manifest(path)
    if manifest is None:
        return []
    result: list[Path] = []
    seen: set[str] = set()
    for row in manifest.get("shards") or []:
        if not isinstance(row, dict):
            raise ValueError("invalid optimizer shard entry")
        name = str(row.get("name") or "")
        if (
            not name.startswith(OPTIMIZER_SHARD_PREFIX)
            or not name.endswith(".pt")
            or "/" in name
            or "\\" in name
            or name in seen
        ):
            raise ValueError("invalid optimizer shard file name")
        seen.add(name)
        shard = path.parent / name
        if not shard.is_file():
            raise FileNotFoundError(f"missing optimizer shard: {name}")
        if verify:
            expected_size = int(row.get("size", 0) or 0)
            if expected_size and shard.stat().st_size != expected_size:
                raise ValueError(f"optimizer shard size mismatch: {name}")
            expected_sha = str(row.get("sha256") or "")
            if expected_sha and _sha256_file(shard) != expected_sha:
                raise ValueError(f"optimizer shard digest mismatch: {name}")
        result.append(shard)
    return result


def _remove_optimizer_checkpoint(path: Path) -> None:
    path.unlink(missing_ok=True)
    for shard in path.parent.glob(f"{OPTIMIZER_SHARD_PREFIX}*.pt"):
        shard.unlink(missing_ok=True)
    for meta in path.parent.glob(f"{OPTIMIZER_META_PREFIX}*.pt"):
        meta.unlink(missing_ok=True)


def _save_optimizer_checkpoint(
    optimizer,
    path: Path,
    *,
    shard_raw_bytes: int = OPTIMIZER_SHARD_RAW_BYTES,
) -> dict[str, Any]:
    """Persist AdamW resume state without exceeding GitHub's per-file limit."""
    import torch

    state_dict = optimizer.state_dict()
    states = dict(state_dict.get("state") or {})
    param_groups = list(state_dict.get("param_groups") or [])
    total_raw_bytes = sum(_optimizer_tensor_bytes(value) for value in states.values())
    shard_raw_bytes = max(1, int(shard_raw_bytes))

    if total_raw_bytes <= shard_raw_bytes:
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(state_dict, tmp)
        tmp.replace(path)
        for stale in path.parent.glob(f"{OPTIMIZER_SHARD_PREFIX}*.pt"):
            stale.unlink(missing_ok=True)
        for stale in path.parent.glob(f"{OPTIMIZER_META_PREFIX}*.pt"):
            stale.unlink(missing_ok=True)
        return {
            "storage": "monolithic",
            "files": [path.name],
            "raw_tensor_bytes": int(total_raw_bytes),
        }

    chunks: list[dict[Any, Any]] = []
    current: dict[Any, Any] = {}
    current_bytes = 0
    for param_id, value in states.items():
        size = _optimizer_tensor_bytes(value)
        if current and current_bytes + size > shard_raw_bytes:
            chunks.append(current)
            current = {}
            current_bytes = 0
        current[param_id] = value
        current_bytes += size
    if current:
        chunks.append(current)

    rows: list[dict[str, Any]] = []
    live_names: set[str] = set()
    total = len(chunks)
    for index, chunk in enumerate(chunks, 1):
        tmp = path.parent / f".optimizer-shard-{index:05d}.tmp"
        torch.save({"state": chunk}, tmp)
        sha = _sha256_file(tmp)
        name = (
            f"{OPTIMIZER_SHARD_PREFIX}{index:05d}-of-{total:05d}-"
            f"{sha[:12]}.pt"
        )
        target = path.parent / name
        tmp.replace(target)
        live_names.add(name)
        rows.append({
            "name": name,
            "sha256": sha,
            "size": int(target.stat().st_size),
            "state_entries": len(chunk),
        })

    tmp_meta = path.parent / ".optimizer-meta.tmp"
    torch.save({"param_groups": param_groups}, tmp_meta)
    meta_sha = _sha256_file(tmp_meta)
    meta_name = f"{OPTIMIZER_META_PREFIX}{meta_sha[:12]}.pt"
    meta_target = path.parent / meta_name
    tmp_meta.replace(meta_target)

    manifest = {
        "format": OPTIMIZER_MANIFEST_FORMAT,
        "version": 1,
        "raw_tensor_bytes": int(total_raw_bytes),
        "param_groups_file": {
            "name": meta_name,
            "sha256": meta_sha,
            "size": int(meta_target.stat().st_size),
        },
        "state_entries": len(states),
        "shards": rows,
    }
    tmp_manifest = path.with_suffix(path.suffix + ".tmp")
    tmp_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp_manifest.replace(path)

    for stale in path.parent.glob(f"{OPTIMIZER_SHARD_PREFIX}*.pt"):
        if stale.name not in live_names:
            stale.unlink(missing_ok=True)
    for stale in path.parent.glob(f"{OPTIMIZER_META_PREFIX}*.pt"):
        if stale.name != meta_name:
            stale.unlink(missing_ok=True)
    _optimizer_shard_paths(path, verify=True)
    return {
        "storage": "sharded",
        "files": [path.name, meta_name, *sorted(live_names)],
        "raw_tensor_bytes": int(total_raw_bytes),
        "shards": int(total),
    }


def _load_optimizer_checkpoint(optimizer, path: Path) -> dict[str, Any]:
    import torch

    manifest = _optimizer_manifest(path)
    if manifest is None:
        optimizer.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True)
        )
        return {"storage": "monolithic", "files": [path.name]}

    meta_row = manifest.get("param_groups_file")
    if not isinstance(meta_row, dict):
        raise ValueError("sharded optimizer manifest requires param_groups_file")
    meta_name = str(meta_row.get("name") or "")
    if (
        not meta_name.startswith(OPTIMIZER_META_PREFIX)
        or not meta_name.endswith(".pt")
        or "/" in meta_name
        or "\\" in meta_name
    ):
        raise ValueError("invalid optimizer metadata file name")
    meta_path = path.parent / meta_name
    if not meta_path.is_file():
        raise FileNotFoundError(f"missing optimizer metadata file: {meta_name}")
    expected_meta_size = int(meta_row.get("size", 0) or 0)
    if expected_meta_size and meta_path.stat().st_size != expected_meta_size:
        raise ValueError("optimizer metadata size mismatch")
    expected_meta_sha = str(meta_row.get("sha256") or "")
    if expected_meta_sha and _sha256_file(meta_path) != expected_meta_sha:
        raise ValueError("optimizer metadata digest mismatch")
    meta_payload = torch.load(meta_path, map_location="cpu", weights_only=True)
    if (
        not isinstance(meta_payload, dict)
        or not isinstance(meta_payload.get("param_groups"), list)
    ):
        raise ValueError("invalid optimizer metadata payload")
    param_groups = meta_payload["param_groups"]

    state: dict[Any, Any] = {}
    shards = _optimizer_shard_paths(path, verify=True)
    for shard in shards:
        payload = torch.load(shard, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("state"), dict):
            raise ValueError(f"invalid optimizer shard payload: {shard.name}")
        overlap = set(state).intersection(payload["state"])
        if overlap:
            raise ValueError("optimizer shards contain duplicate state entries")
        state.update(payload["state"])

    expected = int(manifest.get("state_entries", 0) or 0)
    if expected and len(state) != expected:
        raise ValueError(
            f"optimizer state count mismatch: expected={expected} actual={len(state)}"
        )
    optimizer.load_state_dict({
        "state": state,
        "param_groups": param_groups,
    })
    return {
        "storage": "sharded",
        "files": [path.name, meta_name, *[shard.name for shard in shards]],
        "shards": len(shards),
    }


def _bootstrap_corpus_target(training_target_tokens: int) -> int:
    """Grow reviewed unique text together with cumulative optimization.

    AIRI keeps one checkpoint lineage, but larger rungs should not merely loop
    over the same small corpus forever. The streaming FineWeb/FineWeb2 sources
    already support bounded deterministic growth, so increase unique language
    gradually while keeping the corpus much smaller than the optimization
    budget for resumable CPU training.
    """
    target = max(100_000, int(training_target_tokens))
    if target >= 1_000_000_000:
        ceiling = PHASE5_1B_UNIQUE_CORPUS_TOKENS
    elif target >= 500_000_000:
        ceiling = PHASE5_500M_UNIQUE_CORPUS_TOKENS
    elif target >= 250_000_000:
        ceiling = PHASE5_250M_UNIQUE_CORPUS_TOKENS
    elif target >= 100_000_000:
        ceiling = PHASE5_100M_UNIQUE_CORPUS_TOKENS
    else:
        ceiling = PHASE5_BASE_UNIQUE_CORPUS_TOKENS
    return min(target, ceiling)

def _bootstrap_capacity_target(training_target_tokens: int) -> int | None:
    """Capacity ladder paired with cumulative language optimization.

    The 100M rung is the requested conversational/knowledge bootstrap budget.
    Larger explicit targets remain supported so the language lane does not
    become a permanent capacity ceiling after that rung.
    """
    target = max(0, int(training_target_tokens))
    if target >= 1_000_000_000:
        return 32_000_000
    if target >= 500_000_000:
        return 20_000_000
    if target >= 250_000_000:
        return 12_000_000
    if target >= 100_000_000:
        return 7_000_000
    if target >= 50_000_000:
        return 3_000_000
    if target >= 20_000_000:
        return 1_250_000
    return None


def _assisted_capacity_target(
    progress: dict[str, Any],
    *,
    current_parameters: int,
) -> int | None:
    """Return the one-time 50M assist only for Thomas's current AIRI lineage."""
    if str(progress.get("lineage_id") or "") != ASSISTED_CAPACITY_LINEAGE_ID:
        return None
    if bool(progress.get("assisted_capacity_growth_completed")):
        return None
    if int(current_parameters) >= ASSISTED_CAPACITY_COMPLETION_FLOOR:
        return None
    return ASSISTED_CAPACITY_TARGET_PARAMETERS


def _grow_bootstrap_runtime(
    base_genome: GeneralistGenome,
    runtime: GeneralistRuntime,
    *,
    target_parameters: int,
) -> tuple[GeneralistGenome, GeneralistRuntime, dict[str, Any]]:
    """Grow a Phase-5 candidate before long training and retain compatible weights."""
    source_parameters = parameter_count(runtime.model)
    target = max(source_parameters + 1, int(target_parameters))
    grown_genome = progressive_scale_candidate(
        base_genome,
        target_parameters=target,
        vocab_size=runtime.tokenizer.vocab_size,
        max_width=512,
        max_layers=12,
        prefer_function_preserving=True,
    )
    config = grown_genome.model_config(runtime.tokenizer.vocab_size)
    model = CausalTransformerLM(config)
    transfer = _transfer_compatible_weights(
        runtime.model,
        model,
        source_tokenizer=runtime.tokenizer,
        target_tokenizer=runtime.tokenizer,
    )
    if not bool(transfer.get("function_preserving_growth")):
        raise RuntimeError(
            "Phase-5 capacity growth must preserve the learned function; "
            "refusing destructive width/topology reset"
        )
    grown = GeneralistRuntime(
        model,
        config,
        tokenizer=runtime.tokenizer,
        device="cpu",
    )
    return grown_genome, grown, {
        "version": PHASE5_CAPACITY_GROWTH_VERSION,
        "requested_parameters": int(target_parameters),
        "source_parameters": int(source_parameters),
        "parameters": int(parameter_count(model)),
        "genome": grown_genome.to_dict(),
        "weight_transfer": transfer,
    }


def _dead_capacity_revival_due(progress: dict[str, Any]) -> bool:
    """Revive the zero-gradient FFN capacity created by the legacy 50M growth."""
    rehabilitation = dict(progress.get("language_rehabilitation") or {})
    revival = dict(progress.get("dead_capacity_revival") or {})
    return bool(
        str(progress.get("lineage_id") or "") == ASSISTED_CAPACITY_LINEAGE_ID
        and bool(progress.get("assisted_capacity_growth_completed"))
        and int(rehabilitation.get("consecutive_rejections", 0) or 0) >= 4
        and not bool(revival.get("completed"))
    )


def _dead_capacity_source_ff_width(
    progress: dict[str, Any],
    *,
    current_d_ff: int,
) -> int:
    """Resolve the FF width immediately before the assisted large-model growth."""
    history = list(progress.get("capacity_growth_history") or [])
    source_parameters = None
    for entry in reversed(history):
        genome = entry.get("genome")
        if not isinstance(genome, dict):
            continue
        if int(genome.get("d_ff", 0) or 0) != int(current_d_ff):
            continue
        source_parameters = int(entry.get("source_parameters", 0) or 0)
        if source_parameters > 0:
            break

    if source_parameters:
        for entry in reversed(history):
            if int(entry.get("parameters", 0) or 0) != source_parameters:
                continue
            genome = entry.get("genome")
            if isinstance(genome, dict):
                width = int(genome.get("d_ff", 0) or 0)
                if 0 < width < int(current_d_ff):
                    return width

    candidates = []
    for entry in history:
        genome = entry.get("genome")
        if not isinstance(genome, dict):
            continue
        width = int(genome.get("d_ff", 0) or 0)
        if 0 < width < int(current_d_ff):
            candidates.append(width)
    if not candidates:
        raise RuntimeError("cannot resolve pre-growth FF width for dead-capacity revival")
    return max(candidates)


def _revive_dead_ffn_model_capacity(
    runtime: GeneralistRuntime,
    *,
    source_d_ff: int,
    seed: int = 50_041_536,
) -> dict[str, Any]:
    """Revive legacy zeroed FFN coordinates without changing model logits."""
    import torch

    if str(runtime.config.ff_variant) != "swiglu":
        raise RuntimeError("dead-capacity revival currently requires SwiGLU")
    current_d_ff = int(runtime.config.d_ff)
    source_d_ff = int(source_d_ff)
    if source_d_ff <= 0 or source_d_ff >= current_d_ff:
        raise RuntimeError("dead-capacity revival requires a valid smaller source FF width")

    probe_ids = torch.tensor(
        [[1, 40, 41, 42, 43, 44, 45, 46]],
        dtype=torch.long,
        device=runtime.device,
    )
    runtime.model.eval()
    with torch.no_grad():
        before_logits = runtime.model(probe_ids)["logits"].detach().clone()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    revived_rows = 0
    revived_columns = 0
    layers = 0
    before_max_new_up = 0.0
    before_max_new_down = 0.0

    with torch.no_grad():
        for block in runtime.model.blocks:
            ff = getattr(block, "ff", None)
            up = getattr(ff, "up", None)
            down = getattr(ff, "down", None)
            if up is None or down is None:
                raise RuntimeError("unexpected FFN layout during dead-capacity revival")
            if tuple(up.weight.shape) != (2 * current_d_ff, int(runtime.config.d_model)):
                raise RuntimeError("unexpected SwiGLU up-projection shape")
            if tuple(down.weight.shape) != (int(runtime.config.d_model), current_d_ff):
                raise RuntimeError("unexpected SwiGLU down-projection shape")

            new_gate = up.weight[source_d_ff:current_d_ff]
            new_value = up.weight[
                current_d_ff + source_d_ff : 2 * current_d_ff
            ]
            new_down = down.weight[:, source_d_ff:current_d_ff]

            layer_up_max = max(
                float(new_gate.abs().max().item()) if new_gate.numel() else 0.0,
                float(new_value.abs().max().item()) if new_value.numel() else 0.0,
            )
            layer_down_max = (
                float(new_down.abs().max().item()) if new_down.numel() else 0.0
            )
            before_max_new_up = max(before_max_new_up, layer_up_max)
            before_max_new_down = max(before_max_new_down, layer_down_max)

            if layer_up_max > 1.0e-12 or layer_down_max > 1.0e-12:
                raise RuntimeError(
                    "expanded FFN capacity is not an untouched zero-gradient branch; "
                    "refusing destructive revival"
                )

            gate_noise = torch.randn(
                tuple(new_gate.shape),
                generator=generator,
                dtype=new_gate.dtype,
                device="cpu",
            ).to(device=new_gate.device)
            value_noise = torch.randn(
                tuple(new_value.shape),
                generator=generator,
                dtype=new_value.dtype,
                device="cpu",
            ).to(device=new_value.device)
            new_gate.copy_(gate_noise * 0.02)
            new_value.copy_(value_noise * 0.02)
            revived_rows += int(new_gate.shape[0] + new_value.shape[0])
            revived_columns += int(new_down.shape[1])
            layers += 1

    runtime.model.eval()
    with torch.no_grad():
        after_logits = runtime.model(probe_ids)["logits"]
    max_logit_delta = float((before_logits - after_logits).abs().max().item())
    if max_logit_delta > 1.0e-7:
        raise RuntimeError(
            f"dead-capacity revival changed live logits: {max_logit_delta}"
        )

    runtime.model.train()
    runtime.model.zero_grad(set_to_none=True)
    loss = runtime.model(probe_ids, labels=probe_ids)["loss"]
    loss.backward()
    gradient_sum = 0.0
    for block in runtime.model.blocks:
        down = block.ff.down.weight
        if down.grad is not None:
            gradient_sum += float(
                down.grad[:, source_d_ff:current_d_ff].abs().sum().item()
            )
    runtime.model.zero_grad(set_to_none=True)
    runtime.model.eval()
    if not math.isfinite(gradient_sum) or gradient_sum <= 0.0:
        raise RuntimeError("revived FFN output columns still receive zero gradient")

    return {
        "source_d_ff": source_d_ff,
        "target_d_ff": current_d_ff,
        "layers": layers,
        "revived_up_rows": revived_rows,
        "revived_down_columns": revived_columns,
        "before_max_new_up": before_max_new_up,
        "before_max_new_down": before_max_new_down,
        "max_logit_delta": max_logit_delta,
        "new_down_gradient_sum": gradient_sum,
    }


def _revive_dead_ffn_capacity(
    root: Path,
    progress: dict[str, Any],
    runtime: GeneralistRuntime,
    *,
    candidate_dir: Path,
    optimizer_path: Path,
    base_model_sha: str,
    seed: int = 50_041_536,
) -> dict[str, Any]:
    """Persist a function-preserving revival of the legacy zero-gradient 50M FFN."""
    current_d_ff = int(runtime.config.d_ff)
    source_d_ff = _dead_capacity_source_ff_width(
        progress,
        current_d_ff=current_d_ff,
    )
    revival = _revive_dead_ffn_model_capacity(
        runtime,
        source_d_ff=source_d_ff,
        seed=seed,
    )

    _remove_optimizer_checkpoint(optimizer_path)
    bootstrap_root = root / "bootstrap-data"
    shutil.rmtree(bootstrap_root / "best", ignore_errors=True)
    shutil.rmtree(bootstrap_root / ".segment-best", ignore_errors=True)
    shutil.rmtree(bootstrap_root / "pre-sft", ignore_errors=True)
    shutil.rmtree(bootstrap_root / "pre-anticollapse", ignore_errors=True)
    runtime.save_checkpoint(
        candidate_dir,
        metadata={
            "role": "phase5_dead_capacity_revived_candidate",
            "production_qualified": False,
            "base_champion_model_sha256": base_model_sha,
            "tokens_processed": int(progress.get("tokens_processed", 0) or 0),
            "valid_tokens_processed": int(
                progress.get("valid_tokens_processed", 0) or 0
            ),
            "source_d_ff": source_d_ff,
            "target_d_ff": current_d_ff,
        },
    )

    report = {
        "completed": True,
        "version": "phase5-dead-capacity-revival-v1",
        **revival,
        "tokens_preserved": int(progress.get("tokens_processed", 0) or 0),
        "valid_tokens_preserved": int(
            progress.get("valid_tokens_processed", 0) or 0
        ),
        "completed_at_unix": int(time.time()),
    }
    progress["dead_capacity_revival"] = report
    progress["best_validation_loss"] = None
    progress["bad_eval_count"] = 0
    progress["language_rehabilitation"] = {
        "version": PHASE5_LANGUAGE_REHABILITATION_VERSION,
        "next_stage_index": 0,
        "completed_cycles": 0,
        "consecutive_rejections": 0,
        "last_accepted": False,
        "last_stage": "dead_capacity_revival",
        "updated_at_unix": int(time.time()),
    }
    progress["updated_at_unix"] = int(time.time())
    _atomic_json(root / "bootstrap-data" / "progress.json", progress)
    _atomic_json(root / "bootstrap-data" / "dead-capacity-revival.json", report)
    refresh_live_lineage_manifest(
        root,
        reason="phase5_dead_capacity_revival",
    )
    return report


def _historical_language_recovery_due(progress: dict[str, Any]) -> bool:
    """Use historical replacement only after structural revival also fails."""
    rehabilitation = dict(progress.get("language_rehabilitation") or {})
    recovery = dict(progress.get("historical_language_recovery") or {})
    revival = dict(progress.get("dead_capacity_revival") or {})
    return bool(
        str(progress.get("lineage_id") or "") == ASSISTED_CAPACITY_LINEAGE_ID
        and bool(revival.get("completed"))
        and int(rehabilitation.get("consecutive_rejections", 0) or 0) >= 6
        and not bool(recovery.get("completed"))
    )


def _historical_language_source_genome(
    progress: dict[str, Any],
    *,
    source_parameters: int,
) -> GeneralistGenome:
    """Recover the exact genome that produced a historical checkpoint."""
    matches: list[dict[str, Any]] = []
    for entry in progress.get("capacity_growth_history") or []:
        if int(entry.get("parameters", 0) or 0) != int(source_parameters):
            continue
        genome = entry.get("genome")
        if isinstance(genome, dict):
            matches.append(genome)
    if not matches:
        raise RuntimeError(
            "historical language recovery cannot identify the source genome "
            f"for {int(source_parameters)} parameters"
        )
    return GeneralistGenome(**matches[-1]).validate()


def _recover_historical_language_best(
    root: Path,
    progress: dict[str, Any],
    *,
    source_dir: Path,
    base_model_sha: str,
) -> tuple[GeneralistGenome, GeneralistRuntime, dict[str, Any]]:
    """Rebuild the live large model from a verified healthy historical best.

    Repeated rehabilitation failures mean the active weights are no longer a
    useful optimization starting point.  This path deliberately rolls effective
    checkpoint token accounting back to the historical best, then regrows the
    *same lineage* with an exactly function-preserving, trainable expansion.
    Historical compute is retained as provenance but is not misreported as
    effective live-checkpoint training.
    """
    import torch

    bootstrap_root = root / "bootstrap-data"
    candidate_dir = bootstrap_root / "candidate"
    optimizer_path = bootstrap_root / "optimizer.pt"
    recovery_report_path = bootstrap_root / "language-recovery.json"

    if not source_dir.is_dir():
        raise RuntimeError(f"historical recovery source is missing: {source_dir}")
    source_metadata = _load_json(source_dir / "metadata.json")
    source_tokens = int(source_metadata.get("tokens_processed", 0) or 0)
    if source_tokens <= 0:
        raise RuntimeError("historical recovery source has no token provenance")

    source_runtime = GeneralistRuntime.from_checkpoint(source_dir, device="cpu")
    source_parameters = int(parameter_count(source_runtime.model))
    source_genome = _historical_language_source_genome(
        progress,
        source_parameters=source_parameters,
    )
    expected_source_config = source_genome.model_config(
        source_runtime.tokenizer.vocab_size
    ).to_dict()
    if source_runtime.config.to_dict() != expected_source_config:
        raise RuntimeError("historical recovery source config does not match its genome")

    source_language = evaluate_phase5_language(source_runtime)
    current_live_language = _load_json(bootstrap_root / "before.json", {})
    source_multi = float(source_language.get("multiword_output_rate", 0.0) or 0.0)
    source_nll = float(source_language.get("language_nll", float("inf")))
    current_multi = float(current_live_language.get("multiword_output_rate", 0.0) or 0.0)
    current_nll = float(current_live_language.get("language_nll", float("inf")))
    source_run = int(source_language.get("longest_repeated_token_run", 0) or 0)
    current_run = int(current_live_language.get("longest_repeated_token_run", 0) or 0)
    source_is_clear = not bool(source_language.get("pathological_repetition"))
    source_is_materially_better = bool(
        source_nll <= current_nll - 0.35
        and source_multi >= current_multi + 0.10
        and source_run < current_run
        and float(source_language.get("non_empty_rate", 0.0) or 0.0) >= 0.70
    )
    if not (source_is_clear or source_is_materially_better):
        raise RuntimeError(
            "historical recovery source is not materially healthier than the live checkpoint"
        )
    if source_multi < 0.30:
        raise RuntimeError("historical recovery source lacks usable multi-word language")

    target_parameters = max(
        int(progress.get("capacity_target_parameters", 0) or 0),
        ASSISTED_CAPACITY_TARGET_PARAMETERS,
    )
    direct_capacity_restore = bool(source_parameters >= target_parameters)
    if direct_capacity_restore:
        grown_genome = source_genome
        grown = source_runtime
        growth = {
            "version": "phase5-direct-capacity-recovery-v1",
            "requested_parameters": int(target_parameters),
            "source_parameters": int(source_parameters),
            "parameters": int(source_parameters),
            "direct_capacity_restore": True,
            "weight_transfer": {"function_preserving_growth": True},
        }
    else:
        grown_genome, grown, growth = _grow_bootstrap_runtime(
            source_genome,
            source_runtime,
            target_parameters=target_parameters,
        )
    if not bool((growth.get("weight_transfer") or {}).get("function_preserving_growth")):
        raise RuntimeError("historical recovery regrowth is not function preserving")

    # Verify the actual function, not merely the structural compatibility flag.
    source_language = evaluate_phase5_language(source_runtime)
    grown_language = evaluate_phase5_language(grown)
    source_traces = source_language.get("traces") or []
    grown_traces = grown_language.get("traces") or []
    if len(source_traces) != len(grown_traces):
        raise RuntimeError("historical recovery probe count changed after regrowth")
    for before_trace, after_trace in zip(source_traces, grown_traces):
        if before_trace.get("generated_token_ids") != after_trace.get("generated_token_ids"):
            raise RuntimeError("historical recovery changed greedy language outputs")
    if abs(
        float(source_language.get("language_nll", float("inf")))
        - float(grown_language.get("language_nll", float("inf")))
    ) > 1.0e-5:
        raise RuntimeError("historical recovery changed protected language NLL")
    if (
        bool(grown_language.get("pathological_repetition"))
        and not source_is_materially_better
    ):
        raise RuntimeError("historical recovery regrowth reintroduced repetition collapse")

    revival_report = None
    if direct_capacity_restore:
        source_d_ff = _dead_capacity_source_ff_width(
            progress,
            current_d_ff=int(grown.config.d_ff),
        )
        revival_report = _revive_dead_ffn_model_capacity(
            grown,
            source_d_ff=source_d_ff,
            seed=50_041_536,
        )
        post_revival_language = evaluate_phase5_language(grown)
        if (
            [row.get("generated_token_ids") for row in source_language.get("traces") or []]
            != [row.get("generated_token_ids") for row in post_revival_language.get("traces") or []]
        ):
            raise RuntimeError("direct 50M recovery revival changed protected language outputs")
        grown_language = post_revival_language

    # A second direct-logit check catches transfer errors outside the protected
    # generation traces while remaining independent of decoding.
    probe_ids = torch.tensor(
        [[1, 40, 41, 42, 43, 44, 45, 46]],
        dtype=torch.long,
    )
    if direct_capacity_restore:
        max_logit_delta = float((revival_report or {}).get("max_logit_delta", 0.0))
    else:
        source_runtime.model.eval()
        grown.model.eval()
        with torch.no_grad():
            source_logits = source_runtime.model(probe_ids)["logits"]
            grown_logits = grown.model(probe_ids)["logits"]
        max_logit_delta = float((source_logits - grown_logits).abs().max().item())
        if max_logit_delta > 1.0e-5:
            raise RuntimeError(
                f"historical recovery logit mismatch after regrowth: {max_logit_delta}"
            )

    lineage_before = str(_load_json(root / "lineage.json").get("lineage_id") or "")
    if lineage_before and lineage_before != ASSISTED_CAPACITY_LINEAGE_ID:
        raise RuntimeError("historical recovery attempted on an unexpected lineage")

    valid_before = int(progress.get("valid_tokens_processed", 0) or 0)
    causal_before = int(progress.get("tokens_processed", 0) or 0)
    shutil.rmtree(candidate_dir, ignore_errors=True)
    shutil.rmtree(bootstrap_root / "best", ignore_errors=True)
    shutil.rmtree(bootstrap_root / ".segment-best", ignore_errors=True)
    _remove_optimizer_checkpoint(optimizer_path)
    grown.save_checkpoint(
        candidate_dir,
        metadata={
            "role": "phase5_historical_language_recovery_candidate",
            "production_qualified": False,
            "base_champion_model_sha256": base_model_sha,
            "tokens_processed": source_tokens,
            "valid_tokens_processed": source_tokens,
            "historical_source_parameters": source_parameters,
            "historical_source_tokens": source_tokens,
        },
    )

    progress["tokens_processed"] = source_tokens
    progress["valid_tokens_processed"] = source_tokens
    progress["accepted_rehabilitation_tokens"] = 0
    progress["best_validation_loss"] = None
    progress["bad_eval_count"] = 0
    progress["capacity_target_parameters"] = int(parameter_count(grown.model))
    progress["capacity_genome"] = grown_genome.to_dict()
    progress["assisted_capacity_growth_completed"] = True
    progress["segment_guard_consecutive_rejections"] = 0
    progress["segment_guard_lr_scale"] = min(
        float(progress.get("segment_guard_lr_scale", 1.0) or 1.0),
        1.0 / 64.0,
    )
    progress["segment_guard_recovery_hold"] = True
    progress["segment_guard_last_accepted"] = False
    progress["segment_guard_stable_fast_lane"] = False
    progress["causal_recovery_mode"] = True
    if revival_report is not None:
        progress["dead_capacity_revival"] = {
            "completed": True,
            "version": "phase5-dead-capacity-revival-v1",
            **revival_report,
            "tokens_preserved": int(source_tokens),
            "valid_tokens_preserved": int(source_tokens),
            "completed_at_unix": int(time.time()),
        }
    progress["language_rehabilitation"] = {
        "version": PHASE5_LANGUAGE_REHABILITATION_VERSION,
        "next_stage_index": 0,
        "completed_cycles": 0,
        "consecutive_rejections": 0,
        "last_accepted": False,
        "last_stage": "historical_best_recovery",
        "updated_at_unix": int(time.time()),
    }
    progress["historical_language_recovery"] = {
        "completed": True,
        "version": "phase5-historical-language-recovery-v2",
        "source_parameters": source_parameters,
        "direct_capacity_restore": bool(direct_capacity_restore),
        "causal_recovery_mode": True,
        "dead_capacity_revival": revival_report,
        "source_tokens": source_tokens,
        "target_parameters": int(parameter_count(grown.model)),
        "valid_tokens_before_recovery": valid_before,
        "causal_tokens_before_recovery": causal_before,
        "discarded_effective_tokens": max(0, valid_before - source_tokens),
        "function_preserving_growth": True,
        "max_logit_delta": max_logit_delta,
        "source_language": source_language,
        "grown_language": grown_language,
        "completed_at_unix": int(time.time()),
    }
    progress["updated_at_unix"] = int(time.time())
    _atomic_json(bootstrap_root / "progress.json", progress)
    _atomic_json(
        recovery_report_path,
        {
            "schema": 1,
            **progress["historical_language_recovery"],
            "lineage_id_before": lineage_before,
        },
    )

    lineage_manifest = refresh_live_lineage_manifest(
        root,
        reason="phase5_historical_language_recovery",
    )
    lineage_after = str(lineage_manifest.get("lineage_id") or "")
    if lineage_before and lineage_after != lineage_before:
        raise RuntimeError("historical language recovery changed the live lineage id")
    report = _load_json(recovery_report_path)
    report["lineage_id_after"] = lineage_after
    report["lineage_preserved"] = bool(
        not lineage_before or lineage_after == lineage_before
    )
    _atomic_json(recovery_report_path, report)
    return grown_genome, grown, report


def _phase5_memory_safe_batch_plan(
    *,
    parameters: int,
    context_length: int,
    requested_batch_size: int,
) -> tuple[int, int]:
    """Return (micro_batch, accumulation_steps) for CPU Phase-5 training.

    Keep the caller's effective batch unchanged while bounding activation
    memory as AIRI grows. The current ~50M assisted lineage uses micro-batch 2;
    larger future descendants fall back to micro-batch 1.
    """
    requested = max(1, int(requested_batch_size))
    pressure = int(parameters) * max(1.0, float(context_length) / 128.0)
    if pressure <= 12_000_000:
        micro = requested
    elif pressure <= 32_000_000:
        micro = min(requested, 8)
    elif pressure <= 64_000_000:
        micro = min(requested, 2)
    else:
        micro = 1
    accumulation = int(math.ceil(requested / max(1, micro)))
    return int(micro), max(1, accumulation)


def _phase5_parameter_segment_cap(
    *,
    parameters: int,
    context_length: int,
) -> int | None:
    """Bound transaction size as live AIRI grows on CPU runners.

    Smaller transactions do not change optimizer semantics or effective batch;
    they only persist/quality-gate progress more frequently so a 50M+ model
    cannot spend hours inside one uncommitted Phase-5 segment.
    """
    pressure = int(parameters) * max(1.0, float(context_length) / 128.0)
    if pressure < 20_000_000:
        return None
    if pressure < 40_000_000:
        return 250_000
    if pressure < 64_000_000:
        return 62_500
    return 31_250


def _phase5_recovery_segment_budget(
    requested_tokens: int,
    *,
    consecutive_rejections: int,
) -> int:
    """Make language-guard retries progressively more conservative.

    Rejected work remains uncounted.  Only the next attempted segment becomes
    smaller, preserving the strict quality gate while avoiding repeated
    million-token overshoots when the live lineage is already in recovery.
    """
    requested = max(1, int(requested_tokens))
    rejected = max(0, int(consecutive_rejections))
    if rejected <= 0:
        return requested
    divisor = 2 if rejected == 1 else 4 if rejected == 2 else 8
    return max(62_500, min(requested, int(math.ceil(requested / divisor))))


def _phase5_recovery_plan(
    requested_tokens: int,
    *,
    parameters: int,
    context_length: int,
    persisted_lr_scale: float,
    consecutive_rejections: int,
    recovery_hold: bool = False,
    last_segment_accepted: bool = False,
) -> dict[str, Any]:
    """Return a bounded rescue plan that cannot livelock at the old LR floor.

    After repeated language-gate rejections we keep the gate strict, but change
    the training dynamics instead of retrying the same transaction forever:
    use a smaller transaction, progressively lower learning rates down to
    1/64x, reset stale AdamW momentum, and temporarily focus on short-sentence
    completion until one segment is accepted.
    """
    rejected = max(0, int(consecutive_rejections))
    requested = max(1, int(requested_tokens))
    parameter_cap = _phase5_parameter_segment_cap(
        parameters=int(parameters),
        context_length=int(context_length),
    )
    stable_fast_lane = bool(
        int(consecutive_rejections) == 0
        and bool(last_segment_accepted)
        and not bool(recovery_hold)
        and int(parameters) < 64_000_000
        and int(context_length) <= 128
    )
    if stable_fast_lane and parameter_cap is not None:
        # A 50M checkpoint that just passed the protected language gate can
        # safely amortize evaluation/checkpoint overhead over a larger unit.
        # Any rejection immediately clears last_segment_accepted and returns
        # the next transaction to the conservative recovery budget.
        parameter_cap = max(int(parameter_cap), 250_000)
    segment_budget = _phase5_recovery_segment_budget(
        requested,
        consecutive_rejections=rejected,
    )
    if parameter_cap is not None:
        segment_budget = min(int(segment_budget), int(parameter_cap))

    persisted_scale = max(
        1.0 / 64.0,
        min(1.0, float(persisted_lr_scale or 1.0)),
    )
    # Once a 50M lineage proves that conservative rescue can make progress,
    # keep that regime sticky until the durable language anchor is recovered.
    # A single accepted segment must not immediately double the transaction
    # and raise LR again; that was the source of the post-rescue relapse.
    sticky_recovery = bool(recovery_hold) or (
        rejected >= 1 and persisted_scale <= 0.0625
    )
    stall_recovery = rejected >= 2 or sticky_recovery
    if stall_recovery:
        # 50M CPU retries previously sat at ~62.5k forever.  Halving the
        # transaction makes each quality decision faster without counting
        # rejected tokens.
        segment_budget = min(int(segment_budget), 31_250)

    if rejected <= 0:
        lr_scale = persisted_scale
    else:
        scheduled_scale = max(1.0 / 64.0, 0.5 ** (rejected + 1))
        lr_scale = min(persisted_scale, scheduled_scale)

    return {
        "requested_budget_tokens": int(requested),
        "effective_budget_tokens": int(segment_budget),
        "parameter_cap_tokens": (
            int(parameter_cap) if parameter_cap is not None else None
        ),
        "learning_rate_scale": float(lr_scale),
        "stall_recovery": bool(stall_recovery),
        "stable_fast_lane": bool(stable_fast_lane),
        # Reset stale AdamW momentum only when entering rescue because of a
        # rejection.  After an accepted rescue segment, keep its valid
        # optimizer state while the sticky hold continues.
        "reset_optimizer": bool(stall_recovery and rejected > 0),
        "forced_stage": (
            "B_short_sentence_completion" if stall_recovery else None
        ),
    }


def _batch(blocks: list[list[int]], indices: list[int], *, device):
    import torch
    ids = torch.tensor([blocks[i] for i in indices], dtype=torch.long, device=device)
    labels = ids.clone()
    labels[labels == PAD] = -100
    return ids, labels


def _effective_bootstrap_target(
    requested_target: int,
    persisted_progress: dict[str, Any] | None,
) -> int:
    """Never shrink an already-started cumulative bootstrap rung.

    Push-triggered workflow invocations historically defaulted to 1M tokens.
    After a 5M rung completed, those maintenance runs could rebuild the
    displayed manifest at 1M even though the persisted candidate/progress still
    represented 5M.  The cumulative target is part of state and therefore must
    be monotonic.
    """
    requested = max(100_000, int(requested_target))
    persisted = int((persisted_progress or {}).get("target_tokens", 0) or 0)
    return max(requested, persisted)


def _learning_rate(
    *,
    base_lr: float,
    processed_tokens: int,
    target_tokens: int,
    warmup_tokens: int,
) -> float:
    processed = max(0, int(processed_tokens))
    target = max(1, int(target_tokens))
    warmup = max(1, min(int(warmup_tokens), target // 3))
    if processed < warmup:
        return float(base_lr) * max(0.05, processed / warmup)
    progress = min(1.0, (processed - warmup) / max(1, target - warmup))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(base_lr) * (0.10 + 0.90 * cosine)


def _phase5_success(before: dict[str, Any], after: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if float(after.get("language_nll", float("inf"))) >= float(before.get("language_nll", float("inf"))) - 0.02:
        reasons.append("language NLL did not improve by at least 0.02 byte-NLL")
    if bool(after.get("pathological_repetition")):
        reasons.append("pathological repetition remains")
    if float(after.get("repetition_rate", 1.0)) > min(
        0.60,
        float(before.get("repetition_rate", 0.0)) + 0.02,
    ):
        reasons.append("repetition rate did not improve enough")
    if float(after.get("non_empty_rate", 0.0)) < 0.70:
        reasons.append("non-empty rate is below 70%")
    if float(after.get("word_output_rate", 0.0)) < 0.40:
        reasons.append("fewer than 40% of held-out prompts produce word-like output")
    if float(after.get("multiword_output_rate", 0.0)) < 0.40:
        reasons.append("fewer than 40% of held-out prompts produce multi-word output")
    return (not reasons), reasons


def _write_latest_research(
    root: Path,
    runtime: GeneralistRuntime,
    *,
    report: dict[str, Any],
    cycle: int,
    base_model_sha256: str,
) -> dict[str, Any]:
    latest = root / "latest-research"
    tmp = root / ".phase5-latest-research"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    runtime.save_checkpoint(
        tmp,
        metadata={
            "role": "phase5_language_bootstrap",
            "research_only": True,
            "production_qualified": False,
            "cycle": int(cycle),
            "base_champion_model_sha256": base_model_sha256,
        },
    )
    _atomic_json(tmp / "research-metrics.json", report)
    summary = {
        "candidate_id": f"phase5-bootstrap-{cycle}",
        "kind": "language_bootstrap",
        "cycle": int(cycle),
        "stage": 5,
        "parameters": int(report.get("parameters", 0) or 0),
        "score": float(report.get("score", 0.0) or 0.0),
        "mean_nll_per_byte": float(report.get("nll_per_byte", 0.0) or 0.0),
        "mean_generation_accuracy": float(report.get("generation_exact_accuracy", 0.0) or 0.0),
        "mean_generation_similarity": float(report.get("generation_similarity", 0.0) or 0.0),
        "mean_generation_nonempty_rate": float(report.get("generation_nonempty_rate", 0.0) or 0.0),
        "worst_domain_regression": float(report.get("worst_domain_regression", 0.0) or 0.0),
        "all_seed_eligible": bool(report.get("research_gate_passed")),
        "any_seed_eligible": bool(report.get("research_gate_passed")),
        "tokenizer_vocab_size": int(runtime.tokenizer.vocab_size),
        "research_only": True,
        "external_pretrained": False,
    }
    _atomic_json(tmp / "research-summary.json", summary)
    shutil.rmtree(latest, ignore_errors=True)
    tmp.replace(latest)
    return summary


def _frequent_word_documents(documents):
    counts: Counter[str] = Counter()
    parsed: list[tuple[Any, list[str]]] = []
    for document in documents:
        words = re.findall(r"[^\\W\\d_]+", document.text.casefold(), flags=re.UNICODE)
        parsed.append((document, words))
        counts.update(words)
    frequent = {word for word, _count in counts.most_common(768)}
    selected = [
        document
        for document, words in parsed
        if 1 <= len(words) <= 12
        and words
        and sum(word in frequent for word in words) / len(words) >= 0.70
        and len(document.text) <= 100
    ]
    return selected or [row for row in documents if len(row.text) <= 100] or list(documents)


def _causal_curriculum_stage(processed: int, target: int) -> str:
    ratio = max(0.0, min(1.0, processed / max(1, target)))
    if ratio < 0.15:
        return "A_frequent_word_contexts"
    if ratio < 0.40:
        return "B_short_sentence_completion"
    return "C_causal_next_sentence"


def _anti_collapse_weights(
    stage: str,
    diagnostics: dict[str, Any],
) -> tuple[float, float]:
    pathological = bool(diagnostics.get("pathological_repetition"))
    repetition = float(diagnostics.get("repetition_rate", 0.0) or 0.0)
    if not pathological and repetition < 0.60:
        return 0.0, 1.0
    if stage == "A_frequent_word_contexts":
        return 0.0, 1.20
    if stage == "B_short_sentence_completion":
        return 0.050, 1.60
    return 0.100, 2.00


def _anti_collapse_rescue_gate(
    before: dict[str, Any],
    after: dict[str, Any],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    before_rep = float(before.get("repetition_rate", 1.0) or 1.0)
    after_rep = float(after.get("repetition_rate", 1.0) or 1.0)
    before_nll = float(before.get("language_nll", float("inf")))
    after_nll = float(after.get("language_nll", float("inf")))

    if after_nll > before_nll + 0.08:
        reasons.append("language NLL regressed by more than 0.08")
    if float(after.get("non_empty_rate", 0.0) or 0.0) < max(
        0.50,
        float(before.get("non_empty_rate", 0.0) or 0.0) - 0.15,
    ):
        reasons.append("non-empty generation rate regressed")
    if float(after.get("token_entropy", 0.0) or 0.0) < max(
        1.50,
        float(before.get("token_entropy", 0.0) or 0.0) - 0.50,
    ):
        reasons.append("token entropy collapsed")
    if float(after.get("generation_similarity", 0.0) or 0.0) < (
        float(before.get("generation_similarity", 0.0) or 0.0) - 0.05
    ):
        reasons.append("generation similarity regressed")
    if (
        bool(after.get("pathological_repetition"))
        and before_rep - after_rep < 0.025
    ):
        reasons.append("pathological repetition did not improve by at least 0.025")
    if int(after.get("longest_repeated_token_run", 0) or 0) > (
        int(before.get("longest_repeated_token_run", 0) or 0) + 3
    ):
        reasons.append("longest repeated-token run worsened")
    return (not reasons), reasons



def _language_quality(report: dict[str, Any]) -> float:
    """Small deterministic scalar used only to compare the same held-out suite."""
    nll = float(report.get("language_nll", 1_000_000.0) or 1_000_000.0)
    repetition = float(report.get("repetition_rate", 1.0) or 1.0)
    similarity = float(report.get("generation_similarity", 0.0) or 0.0)
    multiword = float(report.get("multiword_output_rate", 0.0) or 0.0)
    nonempty = float(report.get("non_empty_rate", 0.0) or 0.0)
    entropy = min(5.0, max(0.0, float(report.get("token_entropy", 0.0) or 0.0)))
    pathological = 1.0 if bool(report.get("pathological_repetition")) else 0.0
    return (
        -nll
        - 1.50 * repetition
        + 1.50 * similarity
        + 0.75 * multiword
        + 0.25 * nonempty
        + 0.10 * entropy
        - 0.50 * pathological
    )


def _language_guard_violations(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> list[str]:
    """Return material held-out language regressions versus a durable anchor."""
    reasons: list[str] = []
    ref_nll = float(reference.get("language_nll", float("inf")))
    cand_nll = float(candidate.get("language_nll", float("inf")))
    ref_rep = float(reference.get("repetition_rate", 1.0) or 1.0)
    cand_rep = float(candidate.get("repetition_rate", 1.0) or 1.0)
    ref_similarity = float(reference.get("generation_similarity", 0.0) or 0.0)
    cand_similarity = float(candidate.get("generation_similarity", 0.0) or 0.0)
    ref_multi = float(reference.get("multiword_output_rate", 0.0) or 0.0)
    cand_multi = float(candidate.get("multiword_output_rate", 0.0) or 0.0)
    ref_nonempty = float(reference.get("non_empty_rate", 0.0) or 0.0)
    cand_nonempty = float(candidate.get("non_empty_rate", 0.0) or 0.0)
    ref_entropy = float(reference.get("token_entropy", 0.0) or 0.0)
    cand_entropy = float(candidate.get("token_entropy", 0.0) or 0.0)

    if cand_nll > ref_nll + 0.15:
        reasons.append("language NLL exceeds durable anchor by more than 0.15")
    if cand_rep > ref_rep + 0.08:
        reasons.append("repetition rate exceeds durable anchor by more than 0.08")
    if (
        not bool(reference.get("pathological_repetition"))
        and bool(candidate.get("pathological_repetition"))
    ):
        reasons.append("pathological repetition reappeared")
    if cand_similarity < ref_similarity - 0.05:
        reasons.append("generation similarity fell below durable anchor")
    if cand_multi < max(0.0, ref_multi - 0.15):
        reasons.append("multiword output rate fell below durable anchor")
    if cand_nonempty < max(0.50, ref_nonempty - 0.10):
        reasons.append("non-empty generation rate fell below durable anchor")
    if cand_entropy < max(1.25, ref_entropy - 0.60):
        reasons.append("token entropy fell below durable anchor")
    return reasons


def _segment_language_gate(
    before: dict[str, Any],
    after: dict[str, Any],
    anchor: dict[str, Any],
    *,
    attempted_tokens: int | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Fail closed when a training segment makes the live AIRI language worse.

    If the lineage is already below a historical best anchor, enter recovery
    mode: a segment may be accepted only when it moves held-out quality toward
    that anchor without introducing a new material regression.
    """
    before_anchor = _language_guard_violations(anchor, before)
    after_anchor = _language_guard_violations(anchor, after)
    local_reasons: list[str] = []

    before_nll = float(before.get("language_nll", float("inf")))
    after_nll = float(after.get("language_nll", float("inf")))
    before_rep = float(before.get("repetition_rate", 1.0) or 1.0)
    after_rep = float(after.get("repetition_rate", 1.0) or 1.0)
    before_similarity = float(before.get("generation_similarity", 0.0) or 0.0)
    after_similarity = float(after.get("generation_similarity", 0.0) or 0.0)

    if after_nll > before_nll + 0.08:
        local_reasons.append("segment language NLL regressed by more than 0.08")
    if after_rep > before_rep + 0.06:
        local_reasons.append("segment repetition rate regressed by more than 0.06")
    if (
        not bool(before.get("pathological_repetition"))
        and bool(after.get("pathological_repetition"))
    ):
        local_reasons.append("segment introduced pathological repetition")
    if after_similarity < before_similarity - 0.04:
        local_reasons.append("segment generation similarity regressed")
    if float(after.get("multiword_output_rate", 0.0) or 0.0) < max(
        0.0,
        float(before.get("multiword_output_rate", 0.0) or 0.0) - 0.15,
    ):
        local_reasons.append("segment multiword output rate regressed")

    before_quality = _language_quality(before)
    after_quality = _language_quality(after)
    recovery_mode = bool(before_anchor)

    if recovery_mode:
        # Tiny transactional rescue segments move the held-out composite in
        # smaller increments than the older 250k-1M token segments.  Keep the
        # gate monotonic, but scale the minimum measurable improvement to the
        # transaction size so genuine 31k-token recovery is not discarded.
        recovery_delta = 0.005
        if attempted_tokens is not None and int(attempted_tokens) <= 40_000:
            recovery_delta = 0.002
        improved = (
            after_quality >= before_quality + recovery_delta
            or len(after_anchor) < len(before_anchor)
        )
        if not improved:
            local_reasons.append(
                "lineage is below its durable language anchor and this segment "
                "did not measurably recover quality"
            )
        accepted = not local_reasons
    else:
        accepted = not local_reasons and not after_anchor

    return bool(accepted), {
        "accepted": bool(accepted),
        "recovery_mode": recovery_mode,
        "before_quality": before_quality,
        "after_quality": after_quality,
        "anchor_quality": _language_quality(anchor),
        "recovery_minimum_quality_delta": (
            0.002
            if recovery_mode
            and attempted_tokens is not None
            and int(attempted_tokens) <= 40_000
            else 0.005
        ),
        "attempted_tokens": (
            int(attempted_tokens) if attempted_tokens is not None else None
        ),
        "before_anchor_violations": before_anchor,
        "after_anchor_violations": after_anchor,
        "local_reasons": local_reasons,
        "validation_before": before,
        "validation_after": after,
        "anchor": anchor,
    }


def _run_anti_collapse_rescue(
    runtime: GeneralistRuntime,
    stage_blocks: dict[str, list[list[int]]],
    *,
    bootstrap_root: Path,
    base_model_sha: str,
    seed: int,
    steps: int = 128,
    batch_size: int = 16,
) -> tuple[GeneralistRuntime, dict[str, Any]]:
    import torch

    before = evaluate_phase5_language(runtime)
    if (
        not bool(before.get("pathological_repetition"))
        and float(before.get("repetition_rate", 0.0) or 0.0) < 0.60
    ):
        return runtime, {
            "accepted": False,
            "rolled_back": False,
            "reason": "anti-collapse rescue not needed",
            "validation_before": before,
            "validation_after": before,
            "steps": 0,
            "tokens_processed": 0,
        }

    pre_dir = bootstrap_root / "pre-anticollapse"
    trial_dir = bootstrap_root / ".anticollapse-selected"
    shutil.rmtree(pre_dir, ignore_errors=True)
    shutil.rmtree(trial_dir, ignore_errors=True)
    runtime.save_checkpoint(
        pre_dir,
        metadata={
            "role": "phase5_pre_anticollapse_checkpoint",
            "production_qualified": False,
            "base_champion_model_sha256": base_model_sha,
        },
    )
    trial = GeneralistRuntime.from_checkpoint(pre_dir, device="cpu")
    optimizer = torch.optim.AdamW(
        trial.model.parameters(),
        lr=5e-5,
        weight_decay=0.01,
    )
    rng = random.Random(int(seed))
    objective_tail: dict[str, Any] = {}
    losses: list[float] = []
    trained_tokens = 0
    trial.model.train()

    for step in range(max(1, int(steps))):
        stage = (
            "B_short_sentence_completion"
            if step % 3 != 2
            else "C_causal_next_sentence"
        )
        blocks = stage_blocks[stage]
        indices = [
            rng.randrange(len(blocks))
            for _ in range(max(1, int(batch_size)))
        ]
        ids, labels = _batch(blocks, indices, device=trial.device)
        optimizer.zero_grad(set_to_none=True)
        result = trial.model(ids)
        loss, objective_tail = causal_training_objective(
            result["logits"],
            labels,
            ids,
            eos_loss_weight=2.0,
            repetition_unlikelihood_weight=0.08,
            repetition_window=16,
        )
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite anti-collapse rescue loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trial.model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        trained_tokens += int((labels[:, 1:] != -100).sum().item())

    after = evaluate_phase5_language(trial)
    accepted, reasons = _anti_collapse_rescue_gate(before, after)
    report = {
        "accepted": bool(accepted),
        "rolled_back": not bool(accepted),
        "reason": (
            "held-out anti-collapse rescue improved degeneration safely"
            if accepted
            else "anti-collapse rescue rejected by held-out safety gate"
        ),
        "gate_reasons": reasons,
        "validation_before": before,
        "validation_after": after,
        "steps": max(1, int(steps)),
        "tokens_processed": int(trained_tokens),
        "mean_objective_loss": sum(losses) / max(1, len(losses)),
        "last_objective": objective_tail,
        "external_pretrained_weights": False,
        "decoding_modified": False,
    }

    if not accepted:
        restored = GeneralistRuntime.from_checkpoint(pre_dir, device="cpu")
        shutil.rmtree(trial_dir, ignore_errors=True)
        return restored, report

    trial.save_checkpoint(
        trial_dir,
        metadata={
            "role": "phase5_anticollapse_rescue_selected",
            "production_qualified": False,
            "base_champion_model_sha256": base_model_sha,
            "steps": max(1, int(steps)),
            "tokens_processed": int(trained_tokens),
        },
    )
    selected = GeneralistRuntime.from_checkpoint(trial_dir, device="cpu")
    shutil.rmtree(trial_dir, ignore_errors=True)
    return selected, report


def _sft_row_fingerprint(row) -> str:
    raw = json.dumps(row.messages, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _normal_content(text: str) -> str:
    return " ".join(str(text).strip().casefold().split())


def _filter_protected_replay(rows, *, heldout_sft=()):
    heldout_hashes = {
        _sft_row_fingerprint(row)
        for row in heldout_sft
    }
    protected_texts = protected_bootstrap_texts()
    unique = {}
    filtered = 0
    for row in rows:
        key = _sft_row_fingerprint(row)
        contents = {
            _normal_content(message.get("content", ""))
            for message in row.messages
        }
        if key in heldout_hashes or bool(contents & protected_texts):
            filtered += 1
            continue
        unique.setdefault(key, row)
    return list(unique.values()), int(filtered)


def _elementary_rehabilitation_rows() -> dict[str, dict[str, list[SFTExample]]]:
    """Return a small bilingual curriculum that is disjoint from Phase-5 probes.

    These rows teach the same elementary skills without copying protected
    prompts or targets.  They are intentionally reviewed, bounded and local;
    the larger persisted replay remains the anti-forgetting source.
    """

    def pair(user: str, assistant: str) -> SFTExample:
        return SFTExample([
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ])

    def dialogue(*messages: tuple[str, str]) -> SFTExample:
        return SFTExample([
            {"role": role, "content": content}
            for role, content in messages
        ])

    return {
        "R1_bilingual_foundations": {
            "it": [
                pair("Saluta una persona al mattino.", "Buongiorno! Come posso aiutarti?"),
                pair("Qual è il tuo nome?", "Sono AIRI, la tua assistente."),
                pair("Rispondi in italiano: come stai?", "Sto bene, grazie."),
                pair("Nomina due colori.", "Rosso e blu."),
                pair("Che animale abbaia?", "Il cane abbaia."),
                pair("Conta da uno a tre.", "Uno, due, tre."),
            ],
            "en": [
                pair("Greet someone in the morning.", "Good morning! How can I help?"),
                pair("What is your name?", "I am AIRI, your assistant."),
                pair("Answer in English: how are you?", "I am well, thank you."),
                pair("Name two colors.", "Red and blue."),
                pair("Which animal barks?", "A dog barks."),
                pair("Count from one to three.", "One, two, three."),
            ],
        },
        "R2_simple_responses": {
            "it": [
                pair("Scrivi una frase breve sul sole.", "Il sole illumina la strada."),
                pair("Descrivi un cane in poche parole.", "Il cane è allegro e corre nel parco."),
                pair("Completa: L'acqua è", "trasparente e fresca."),
                pair("Rispondi con una frase gentile.", "Certo, sarò felice di aiutarti."),
                pair("Dove si legge un libro?", "Si può leggere un libro in un luogo tranquillo."),
                pair("Spiega che cosa fa una lampada.", "Una lampada produce luce quando è accesa."),
            ],
            "en": [
                pair("Write a short sentence about the sun.", "The sun lights the road."),
                pair("Describe a dog in a few words.", "The dog is happy and runs in the park."),
                pair("Complete: Water is", "clear and fresh."),
                pair("Reply with one kind sentence.", "Of course, I will be happy to help."),
                pair("Where can someone read a book?", "Someone can read a book in a quiet place."),
                pair("Explain what a lamp does.", "A lamp produces light when it is on."),
            ],
        },
        "R3_short_dialogue": {
            "it": [
                dialogue(
                    ("user", "Buongiorno, puoi aiutarmi?"),
                    ("assistant", "Sì, dimmi pure di cosa hai bisogno."),
                    ("user", "Vorrei una risposta breve."),
                    ("assistant", "Va bene, userò una frase semplice."),
                ),
                dialogue(
                    ("user", "Parli italiano?"),
                    ("assistant", "Sì, posso rispondere in italiano."),
                    ("user", "Dimmi qualcosa sul mare."),
                    ("assistant", "Il mare è ampio e le onde arrivano sulla riva."),
                ),
                dialogue(
                    ("user", "Ho una domanda facile."),
                    ("assistant", "Sono pronta ad ascoltarla."),
                    ("user", "Di che colore sono le foglie in estate?"),
                    ("assistant", "Di solito le foglie sono verdi."),
                ),
                dialogue(
                    ("user", "Possiamo fare due domande?"),
                    ("assistant", "Sì, facciamole una alla volta."),
                    ("user", "Quanto fa due più due?"),
                    ("assistant", "Due più due fa quattro."),
                ),
            ],
            "en": [
                dialogue(
                    ("user", "Good morning, can you help me?"),
                    ("assistant", "Yes, tell me what you need."),
                    ("user", "I would like a short answer."),
                    ("assistant", "All right, I will use a simple sentence."),
                ),
                dialogue(
                    ("user", "Do you speak English?"),
                    ("assistant", "Yes, I can answer in English."),
                    ("user", "Tell me something about the sea."),
                    ("assistant", "The sea is wide and waves reach the shore."),
                ),
                dialogue(
                    ("user", "I have an easy question."),
                    ("assistant", "I am ready to hear it."),
                    ("user", "What color are leaves in summer?"),
                    ("assistant", "Leaves are usually green."),
                ),
                dialogue(
                    ("user", "May I ask two questions?"),
                    ("assistant", "Yes, let us take them one at a time."),
                    ("user", "What is two plus two?"),
                    ("assistant", "Two plus two is four."),
                ),
            ],
        },
    }


def _rehabilitation_replay_rows(
    stage: str,
    curriculum: dict[str, dict[str, list[SFTExample]]],
    protected_replay,
    *,
    heldout_sft=(),
    max_replay_rows: int = 192,
    replay_offset: int = 0,
) -> tuple[list[SFTExample], dict[str, int]]:
    if stage not in PHASE5_LANGUAGE_REHABILITATION_STAGES:
        raise ValueError(f"unknown language rehabilitation stage: {stage}")
    stage_rows = curriculum.get(stage) or {}
    italian = list(stage_rows.get("it") or [])
    english = list(stage_rows.get("en") or [])
    if not italian or len(italian) != len(english):
        raise RuntimeError("language rehabilitation curriculum must be bilingual and balanced")

    safe_replay, filtered = _filter_protected_replay(
        list(protected_replay),
        heldout_sft=heldout_sft,
    )
    safe_replay = sorted(safe_replay, key=_sft_row_fingerprint)
    replay_limit = max(0, int(max_replay_rows))
    offset = max(0, int(replay_offset))
    if safe_replay and replay_limit:
        offset %= len(safe_replay)
        rotated = safe_replay[offset:] + safe_replay[:offset]
        selected_replay = rotated[:replay_limit]
    else:
        selected_replay = []
    elementary: list[SFTExample] = []
    for it_row, en_row in zip(italian, english):
        elementary.extend((it_row, en_row))
    combined, second_filtered = _filter_protected_replay(
        elementary + selected_replay,
        heldout_sft=heldout_sft,
    )
    if len(combined) < len(elementary):
        raise RuntimeError("elementary rehabilitation rows overlap a protected holdout")
    return combined, {
        "elementary_it_rows": len(italian),
        "elementary_en_rows": len(english),
        "protected_replay_rows": len(selected_replay),
        "protected_replay_offset": int(offset),
        "protected_rows_filtered": int(filtered + second_filtered),
        "total_rows": len(combined),
    }


def _rehabilitation_needed(
    current: dict[str, Any],
    anchor: dict[str, Any],
) -> bool:
    return bool(
        current.get("pathological_repetition")
        or _language_guard_violations(anchor, current)
    )


def _rehabilitation_strategy_rejection_count(
    progress: dict[str, Any],
    rehabilitation_state: dict[str, Any],
) -> int:
    """Return rejections for the *current* rehabilitation optimizer strategy.

    Full-model SFT failures must not make a newly introduced residual/KL
    strategy start at an artificially tiny learning rate. Once dead-capacity
    revival is active, residual failures are tracked independently.
    """
    revival = dict(progress.get("dead_capacity_revival") or {})
    if bool(revival.get("completed")):
        return max(
            0,
            int(
                rehabilitation_state.get(
                    "residual_consecutive_rejections",
                    0,
                )
                or 0
            ),
        )
    return max(
        0,
        int(rehabilitation_state.get("consecutive_rejections", 0) or 0),
    )


def _rehabilitation_cycle_due(
    current: dict[str, Any],
    anchor: dict[str, Any],
    next_stage_index: int,
) -> bool:
    """Finish an active progressive cycle even if an early gate clears collapse."""
    stage_count = len(PHASE5_LANGUAGE_REHABILITATION_STAGES)
    stage_index = max(0, int(next_stage_index))
    cycle_active = 0 < stage_index < stage_count
    return bool(
        cycle_active
        or (
            _rehabilitation_needed(current, anchor)
            and (stage_index < stage_count or current.get("pathological_repetition"))
        )
    )


def _language_rehabilitation_gate(
    stage: str,
    before: dict[str, Any],
    after: dict[str, Any],
    anchor: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """Apply increasingly strict held-out gates without training on the probes."""
    if stage not in PHASE5_LANGUAGE_REHABILITATION_STAGES:
        raise ValueError(f"unknown language rehabilitation stage: {stage}")
    reasons: list[str] = []
    before_nll = float(before.get("language_nll", float("inf")))
    after_nll = float(after.get("language_nll", float("inf")))
    before_rep = float(before.get("repetition_rate", 1.0) or 1.0)
    after_rep = float(after.get("repetition_rate", 1.0) or 1.0)
    before_similarity = float(before.get("generation_similarity", 0.0) or 0.0)
    after_similarity = float(after.get("generation_similarity", 0.0) or 0.0)
    before_multi = float(before.get("multiword_output_rate", 0.0) or 0.0)
    after_multi = float(after.get("multiword_output_rate", 0.0) or 0.0)

    if after_nll > before_nll + 0.10:
        reasons.append("protected language NLL regressed by more than 0.10")
    if float(after.get("non_empty_rate", 0.0) or 0.0) < max(
        0.60,
        float(before.get("non_empty_rate", 0.0) or 0.0) - 0.10,
    ):
        reasons.append("protected non-empty output rate regressed")
    before_anchor_violations = _language_guard_violations(anchor, before)
    recovery_mode = bool(before_anchor_violations)
    if recovery_mode:
        # A collapsed checkpoint can have *higher* entropy because it is
        # uncertain while emitting garbage/repeated bytes. During recovery,
        # using that entropy as the floor rejects genuine improvements. The
        # durable non-pathological anchor is the correct stability reference.
        entropy_floor = max(
            1.25,
            float(anchor.get("token_entropy", 0.0) or 0.0) - 0.60,
        )
    else:
        entropy_floor = max(
            1.25,
            float(before.get("token_entropy", 0.0) or 0.0) - 0.75,
        )
    if float(after.get("token_entropy", 0.0) or 0.0) < entropy_floor:
        reasons.append("protected token entropy collapsed")

    if stage == "R1_bilingual_foundations":
        gate = "repetition_recovery"
        run_gain = int(before.get("longest_repeated_token_run", 0) or 0) - int(
            after.get("longest_repeated_token_run", 0) or 0
        )
        if (
            bool(after.get("pathological_repetition"))
            and before_rep - after_rep < 0.04
            and run_gain < 2
        ):
            reasons.append("pathological repetition did not measurably recover")
        if (
            not bool(after.get("pathological_repetition"))
            and after_rep > min(0.66, before_rep + 0.02)
        ):
            reasons.append("repetition rate did not clear the foundation gate")
    elif stage == "R2_simple_responses":
        gate = "elementary_language"
        if bool(after.get("pathological_repetition")):
            reasons.append("pathological repetition remains after elementary recovery")
        if (
            after_nll > before_nll - 0.03
            and _language_quality(after) < _language_quality(before) + 0.02
        ):
            reasons.append("elementary language quality did not improve")
        if after_rep > before_rep + 0.02:
            reasons.append("repetition regressed during elementary recovery")
        if after_multi < max(0.35, before_multi - 0.02):
            reasons.append("multi-word output did not clear the elementary gate")
    else:
        gate = "short_dialogue"
        if bool(after.get("pathological_repetition")):
            reasons.append("pathological repetition remains after dialogue recovery")
        if after_rep > 0.60:
            reasons.append("dialogue repetition exceeds the fail-closed limit")
        if after_similarity < before_similarity - 0.01:
            reasons.append("dialogue similarity regressed")
        if after_multi < max(0.40, before_multi - 0.02):
            reasons.append("multi-word dialogue rate regressed")
        if _language_quality(after) < _language_quality(before) - 0.01:
            reasons.append("overall protected language quality regressed")

    accepted = not reasons
    return accepted, {
        "accepted": bool(accepted),
        "stage": stage,
        "gate": gate,
        "reasons": reasons,
        "before_quality": _language_quality(before),
        "after_quality": _language_quality(after),
        "anchor_quality": _language_quality(anchor),
        "recovery_mode": bool(recovery_mode),
        "protected_entropy_floor": float(entropy_floor),
        "anchor_violations_before": before_anchor_violations,
        "anchor_violations_after": _language_guard_violations(anchor, after),
    }


def _residual_language_rehabilitation_attempts(
    *,
    stage: str,
    consecutive_rejections: int,
) -> tuple[tuple[int, float, float, float, float, bool], ...]:
    """Conservative trust-region plan for the revived 50M residual branch.

    The legacy network stays frozen during rehabilitation. Attempts first train
    only the zero-initialized residual output columns, then optionally allow the
    revived SwiGLU input rows to adapt while KL keeps behavior near the
    pre-attempt checkpoint. Repeated rejections lower LR and strengthen KL
    instead of escalating anti-repetition/EOS pressure.
    """
    if stage not in PHASE5_LANGUAGE_REHABILITATION_STAGES:
        raise ValueError(f"unknown language rehabilitation stage: {stage}")
    rejected = max(0, int(consecutive_rejections))
    lr_scale = 0.5 ** min(3, rejected)
    # Live 50M evidence showed that a ~1e-5 output-residual step with strong
    # behavioral KL removed pathological repetition while passing replay. Start
    # in that measured trust region; retries get smaller and more constrained.
    kl_base = min(2.5, 1.75 + 0.25 * rejected)

    plans = {
        "R1_bilingual_foundations": (
            (64, 1.0e-5 * lr_scale, 0.03, 1.15, kl_base, False),
            (48, 5.0e-6 * lr_scale, 0.03, 1.15, min(2.75, kl_base + 0.25), False),
            (48, 2.5e-6 * lr_scale, 0.04, 1.20, min(3.0, kl_base + 0.50), True),
        ),
        "R2_simple_responses": (
            (72, 7.5e-6 * lr_scale, 0.025, 1.10, kl_base, False),
            (56, 3.75e-6 * lr_scale, 0.035, 1.15, min(2.75, kl_base + 0.25), True),
        ),
        "R3_short_dialogue": (
            (80, 5.0e-6 * lr_scale, 0.02, 1.10, kl_base, False),
            (56, 2.5e-6 * lr_scale, 0.03, 1.15, min(2.75, kl_base + 0.25), True),
        ),
    }
    return plans[stage]


def _rehabilitation_replay_limit(stage: str, consecutive_rejections: int) -> int:
    """Increase protected replay diversity after a failed rehabilitation attempt."""
    if stage not in PHASE5_LANGUAGE_REHABILITATION_STAGES:
        raise ValueError(f"unknown language rehabilitation stage: {stage}")
    base = {
        "R1_bilingual_foundations": 48,
        "R2_simple_responses": 64,
        "R3_short_dialogue": 96,
    }[stage]
    multiplier = 1 + min(3, max(0, int(consecutive_rejections)))
    return min(192, int(base * multiplier))


def _language_rehabilitation_attempts(
    *,
    parameters: int,
    stage: str,
    consecutive_rejections: int,
) -> tuple[tuple[int, float, float, float], ...]:
    """Return an adaptive, fail-closed SFT retry schedule.

    A rejected stage is always rolled back. Repeating the same deterministic
    seed/checkpoint with the same hyperparameters would simply reproduce the
    same failure, so later retries become shorter, lower-LR, more strongly
    anti-repetition updates while protected replay diversity increases.
    """
    if stage not in PHASE5_LANGUAGE_REHABILITATION_STAGES:
        raise ValueError(f"unknown language rehabilitation stage: {stage}")
    rejected = max(0, int(consecutive_rejections))
    large = int(parameters) >= 20_000_000

    if large and rejected <= 0:
        plans = {
            "R1_bilingual_foundations": ((256, 2.0e-5, 0.12, 2.00), (128, 1.0e-5, 0.10, 2.00)),
            "R2_simple_responses": ((320, 2.0e-5, 0.08, 1.75), (160, 1.0e-5, 0.06, 1.75)),
            "R3_short_dialogue": ((384, 1.5e-5, 0.06, 1.50), (192, 7.5e-6, 0.04, 1.50)),
        }
    elif large and rejected == 1:
        plans = {
            "R1_bilingual_foundations": ((96, 5.0e-6, 0.18, 1.50), (64, 2.5e-6, 0.16, 1.50)),
            "R2_simple_responses": ((128, 5.0e-6, 0.12, 1.50), (80, 2.5e-6, 0.10, 1.50)),
            "R3_short_dialogue": ((160, 3.75e-6, 0.10, 1.35), (96, 1.875e-6, 0.08, 1.35)),
        }
    elif large:
        # After two failed 50M rehabilitation attempts, the remaining failure
        # mode is usually a long greedy repeated-token run rather than basic
        # language loss.  Break that loop without weakening any held-out gate:
        # keep LR tiny, increase target-safe unlikelihood, and strongly teach
        # EOS on the short reviewed responses.  Later rejections become even
        # gentler in LR while retaining the repetition-breaking objective.
        extra = min(3, max(0, rejected - 2))
        lr_scale = 0.5 ** extra
        anti_scale = 1.0 + 0.20 * extra
        eos_scale = 1.0 + 0.15 * extra
        plans = {
            "R1_bilingual_foundations": (
                (96, 2.5e-6 * lr_scale, 0.75 * anti_scale, 4.00 * eos_scale),
                (64, 1.25e-6 * lr_scale, 1.00 * anti_scale, 5.00 * eos_scale),
                (48, 0.625e-6 * lr_scale, 1.25 * anti_scale, 6.00 * eos_scale),
            ),
            "R2_simple_responses": (
                (96, 2.5e-6 * lr_scale, 0.40 * anti_scale, 2.50 * eos_scale),
                (64, 1.25e-6 * lr_scale, 0.55 * anti_scale, 3.00 * eos_scale),
                (48, 0.625e-6 * lr_scale, 0.70 * anti_scale, 3.50 * eos_scale),
            ),
            "R3_short_dialogue": (
                (112, 1.875e-6 * lr_scale, 0.25 * anti_scale, 2.00 * eos_scale),
                (72, 0.9375e-6 * lr_scale, 0.35 * anti_scale, 2.50 * eos_scale),
                (48, 0.46875e-6 * lr_scale, 0.45 * anti_scale, 3.00 * eos_scale),
            ),
        }
    elif rejected <= 0:
        plans = {
            "R1_bilingual_foundations": ((128, 3.0e-5, 0.12, 2.00), (64, 1.5e-5, 0.10, 2.00)),
            "R2_simple_responses": ((160, 3.0e-5, 0.08, 1.75), (80, 1.5e-5, 0.06, 1.75)),
            "R3_short_dialogue": ((192, 2.0e-5, 0.06, 1.50), (96, 1.0e-5, 0.04, 1.50)),
        }
    elif rejected == 1:
        plans = {
            "R1_bilingual_foundations": ((64, 7.5e-6, 0.18, 1.50), (32, 3.75e-6, 0.16, 1.50)),
            "R2_simple_responses": ((80, 7.5e-6, 0.12, 1.50), (40, 3.75e-6, 0.10, 1.50)),
            "R3_short_dialogue": ((96, 5.0e-6, 0.10, 1.35), (48, 2.5e-6, 0.08, 1.35)),
        }
    else:
        extra = min(3, max(0, rejected - 2))
        lr_scale = 0.5 ** extra
        anti_scale = 1.0 + 0.20 * extra
        eos_scale = 1.0 + 0.15 * extra
        plans = {
            "R1_bilingual_foundations": (
                (48, 3.75e-6 * lr_scale, 0.60 * anti_scale, 3.00 * eos_scale),
                (24, 1.875e-6 * lr_scale, 0.85 * anti_scale, 4.00 * eos_scale),
            ),
            "R2_simple_responses": (
                (56, 3.75e-6 * lr_scale, 0.35 * anti_scale, 2.25 * eos_scale),
                (28, 1.875e-6 * lr_scale, 0.50 * anti_scale, 2.75 * eos_scale),
            ),
            "R3_short_dialogue": (
                (64, 2.5e-6 * lr_scale, 0.25 * anti_scale, 1.75 * eos_scale),
                (32, 1.25e-6 * lr_scale, 0.35 * anti_scale, 2.25 * eos_scale),
            ),
        }
    return plans[stage]


def _run_language_rehabilitation_stage(
    runtime: GeneralistRuntime,
    stage: str,
    rows: list[SFTExample],
    heldout_sft,
    anchor: dict[str, Any],
    *,
    bootstrap_root: Path,
    base_model_sha: str,
    seed: int,
    consecutive_rejections: int = 0,
) -> tuple[GeneralistRuntime, dict[str, Any]]:
    """Train one transactional rehabilitation stage and roll back on any gate."""
    if not rows:
        raise RuntimeError("language rehabilitation stage has no training rows")
    heldout = list(heldout_sft)
    if not heldout:
        raise RuntimeError("language rehabilitation requires held-out SFT rows")

    pre_dir = bootstrap_root / ".language-rehabilitation-pre"
    selected_dir = bootstrap_root / ".language-rehabilitation-selected"
    shutil.rmtree(pre_dir, ignore_errors=True)
    shutil.rmtree(selected_dir, ignore_errors=True)
    runtime.save_checkpoint(
        pre_dir,
        metadata={
            "role": "phase5_language_rehabilitation_rollback",
            "production_qualified": False,
            "stage": stage,
            "base_champion_model_sha256": base_model_sha,
        },
    )
    rollback_manifest_sha = _sha256_file(pre_dir / "model.pt")
    before = evaluate_phase5_language(runtime)
    sft_before = evaluate_sft_validation(
        runtime,
        heldout,
        max_examples=16,
        max_new_tokens=32,
    )
    parameters = parameter_count(runtime.model)
    persisted_progress = _load_json(bootstrap_root / "progress.json")
    revival = dict(persisted_progress.get("dead_capacity_revival") or {})
    residual_source_d_ff = int(revival.get("source_d_ff", 0) or 0)
    residual_mode = bool(
        revival.get("completed")
        and residual_source_d_ff > 0
        and residual_source_d_ff < int(runtime.config.d_ff)
    )

    residual_supervised_rows = list(rows)
    residual_anchor_rows: list[SFTExample] = []
    if residual_mode:
        stage_curriculum = _elementary_rehabilitation_rows().get(stage) or {}
        elementary_fingerprints = {
            _sft_row_fingerprint(row)
            for language_rows in stage_curriculum.values()
            for row in language_rows
        }
        residual_supervised_rows = [
            row for row in rows
            if _sft_row_fingerprint(row) in elementary_fingerprints
        ]
        residual_anchor_rows = [
            row for row in rows
            if _sft_row_fingerprint(row) not in elementary_fingerprints
        ]
        if not residual_supervised_rows:
            raise RuntimeError("residual rehabilitation has no elementary supervision rows")
        if not residual_anchor_rows:
            raise RuntimeError("residual rehabilitation has no protected replay anchors")

    if residual_mode:
        stage_attempts = [
            {
                "steps": int(steps),
                "learning_rate": float(learning_rate),
                "repetition_unlikelihood_weight": float(anti_weight),
                "eos_loss_weight": float(eos_weight),
                "teacher_kl_weight": float(kl_weight),
                "train_upstream": bool(train_upstream),
                "mode": "residual_kl_recovery",
            }
            for (
                steps,
                learning_rate,
                anti_weight,
                eos_weight,
                kl_weight,
                train_upstream,
            ) in _residual_language_rehabilitation_attempts(
                stage=stage,
                consecutive_rejections=int(consecutive_rejections),
            )
        ]
    else:
        stage_attempts = [
            {
                "steps": int(steps),
                "learning_rate": float(learning_rate),
                "repetition_unlikelihood_weight": float(anti_weight),
                "eos_loss_weight": float(eos_weight),
                "teacher_kl_weight": 0.0,
                "train_upstream": True,
                "mode": "full_model_sft",
            }
            for steps, learning_rate, anti_weight, eos_weight in _language_rehabilitation_attempts(
                parameters=int(parameters),
                stage=stage,
                consecutive_rejections=int(consecutive_rejections),
            )
        ]

    attempts: list[dict[str, Any]] = []
    selected_index: int | None = None

    for index, plan in enumerate(stage_attempts):
        trial = GeneralistRuntime.from_checkpoint(pre_dir, device="cpu")
        if residual_mode:
            reference = GeneralistRuntime.from_checkpoint(pre_dir, device="cpu")
            training = train_sft_residual_recovery(
                trial.model,
                reference.model,
                trial.tokenizer,
                residual_supervised_rows,
                anchor_examples=residual_anchor_rows,
                source_d_ff=residual_source_d_ff,
                steps=int(plan["steps"]),
                batch_size=2,
                learning_rate=float(plan["learning_rate"]),
                seed=int(seed + index * 101),
                device="cpu",
                repetition_unlikelihood_weight=float(
                    plan["repetition_unlikelihood_weight"]
                ),
                eos_loss_weight=float(plan["eos_loss_weight"]),
                repetition_window=16,
                kl_weight=float(plan["teacher_kl_weight"]),
                train_upstream=bool(plan["train_upstream"]),
            )
            del reference
        else:
            training = train_sft(
                trial.model,
                trial.tokenizer,
                rows,
                steps=int(plan["steps"]),
                batch_size=2,
                learning_rate=float(plan["learning_rate"]),
                weight_decay=0.01,
                seed=int(seed + index * 101),
                device="cpu",
                gradient_accumulation_steps=1,
                precision="fp32",
                repetition_unlikelihood_weight=float(
                    plan["repetition_unlikelihood_weight"]
                ),
                eos_loss_weight=float(plan["eos_loss_weight"]),
                repetition_window=16,
            )
        after = evaluate_phase5_language(trial)
        sft_after = evaluate_sft_validation(
            trial,
            heldout,
            max_examples=16,
            max_new_tokens=32,
        )
        stage_ok, stage_gate = _language_rehabilitation_gate(
            stage,
            before,
            after,
            anchor,
        )
        replay_ok, replay_reasons = sft_validation_gate(
            sft_before,
            sft_after,
            max_repetition_regression=0.03,
            max_nll_regression=0.04,
            min_entropy_fraction=0.80,
            max_unique_ratio_drop=0.05,
        )
        attempt = {
            "index": int(index),
            "training_mode": str(plan["mode"]),
            "residual_source_d_ff": (
                int(residual_source_d_ff) if residual_mode else None
            ),
            "residual_supervision_rows": (
                len(residual_supervised_rows) if residual_mode else None
            ),
            "residual_anchor_rows": (
                len(residual_anchor_rows) if residual_mode else None
            ),
            "training": training,
            "validation_before": before,
            "validation_after": after,
            "stage_gate": stage_gate,
            "protected_replay_gate_passed": bool(replay_ok),
            "protected_replay_gate_reasons": list(replay_reasons),
            "protected_replay_validation_before": sft_before,
            "protected_replay_validation_after": sft_after,
            "accepted": bool(stage_ok and replay_ok),
        }
        attempts.append(attempt)
        if stage_ok and replay_ok:
            trial.save_checkpoint(
                selected_dir,
                metadata={
                    "role": "phase5_language_rehabilitation_selected",
                    "production_qualified": False,
                    "stage": stage,
                    "attempt_index": int(index),
                    "base_champion_model_sha256": base_model_sha,
                },
            )
            selected_index = index
            break

    if selected_index is None:
        restored = GeneralistRuntime.from_checkpoint(pre_dir, device="cpu")
        rollback_verified = _sha256_file(pre_dir / "model.pt") == rollback_manifest_sha
        shutil.rmtree(pre_dir, ignore_errors=True)
        shutil.rmtree(selected_dir, ignore_errors=True)
        return restored, {
            "schema": 1,
            "version": PHASE5_LANGUAGE_REHABILITATION_VERSION,
            "stage": stage,
            "accepted": False,
            "rolled_back": True,
            "rollback_verified": bool(rollback_verified),
            "rollback_model_manifest_sha256": rollback_manifest_sha,
            "accepted_supervised_tokens": 0,
            "consecutive_rejections_before": int(consecutive_rejections),
            "training_mode": (
                "residual_kl_recovery" if residual_mode else "full_model_sft"
            ),
            "residual_source_d_ff": (
                int(residual_source_d_ff) if residual_mode else None
            ),
            "attempt_plan": [dict(plan) for plan in stage_attempts],
            "attempts": attempts,
            "reason": "all rehabilitation attempts failed progressive or protected replay gates",
        }

    selected = GeneralistRuntime.from_checkpoint(selected_dir, device="cpu")
    chosen = attempts[selected_index]
    accepted_tokens = int(
        (chosen.get("training") or {}).get("supervised_tokens", 0) or 0
    )
    shutil.rmtree(pre_dir, ignore_errors=True)
    shutil.rmtree(selected_dir, ignore_errors=True)
    return selected, {
        "schema": 1,
        "version": PHASE5_LANGUAGE_REHABILITATION_VERSION,
        "stage": stage,
        "accepted": True,
        "rolled_back": False,
        "rollback_verified": True,
        "rollback_model_manifest_sha256": rollback_manifest_sha,
        "accepted_supervised_tokens": accepted_tokens,
        "consecutive_rejections_before": int(consecutive_rejections),
        "training_mode": (
            "residual_kl_recovery" if residual_mode else "full_model_sft"
        ),
        "residual_source_d_ff": (
            int(residual_source_d_ff) if residual_mode else None
        ),
        "attempt_plan": [dict(plan) for plan in stage_attempts],
        "selected_attempt": int(selected_index),
        "attempts": attempts,
        "reason": "progressive language and protected replay gates passed",
    }


def _mixed_replay_rows(
    root: Path,
    bootstrap_sft,
    repo: Path,
    *,
    heldout_sft=(),
) -> tuple[list, dict[str, int]]:
    one_turn = [row for row in bootstrap_sft if len(row.messages) <= 2]
    multi_turn = [row for row in bootstrap_sft if len(row.messages) > 2]
    base = [row.sft() for row in train_rows()]
    lab = build_airi_pc_lab_rows(snapshot_airi_pc_lab(repo), max_rows=18)
    verified = load_verified_lab_experiences(root, max_rows=32)

    stage_rows = {
        "D_simple_prompt_response": one_turn[:900] + base[:200],
        "E_simple_multiturn_chat": multi_turn[:600] + one_turn[:300],
        "F_airi_pc_lab_tool_use": [row.sft() for row in (lab + verified)] + base,
    }
    rows = []
    counts: dict[str, int] = {}
    for stage, stage_items in stage_rows.items():
        counts[stage] = len(stage_items)
        rows.extend(stage_items)
    try:
        memory = CurriculumMemory(root, max_rows=4000)
        memory_rows = [row.sft() for row in memory.rows()[:2400]]
        rows.extend(memory_rows)
        counts["persistent_replay"] = len(memory_rows)
    except Exception:
        counts["persistent_replay"] = 0

    # Deterministic de-duplication plus structural exclusion of every
    # held-out SFT row and the canonical Phase-5 evaluation strings.
    filtered_rows, filtered = _filter_protected_replay(
        rows,
        heldout_sft=heldout_sft,
    )
    counts["held_out_filtered"] = int(filtered)
    return filtered_rows, counts


def _sft_attempt_score(
    before: dict[str, Any],
    after: dict[str, Any],
) -> float:
    return float(
        (float(before.get("language_nll", 0.0)) - float(after.get("language_nll", 0.0)))
        + 0.75 * (
            float(before.get("repetition_rate", 0.0))
            - float(after.get("repetition_rate", 0.0))
        )
        + 0.25 * (
            float(after.get("generation_similarity", 0.0))
            - float(before.get("generation_similarity", 0.0))
        )
    )


def _run_guarded_sft(
    runtime: GeneralistRuntime,
    replay,
    heldout_sft,
    *,
    bootstrap_root: Path,
    base_model_sha: str,
    target_tokens: int,
    seed: int,
) -> tuple[GeneralistRuntime, dict[str, Any]]:
    pre_sft_dir = bootstrap_root / "pre-sft"
    selected_dir = bootstrap_root / ".sft-selected"
    shutil.rmtree(pre_sft_dir, ignore_errors=True)
    shutil.rmtree(selected_dir, ignore_errors=True)
    runtime.save_checkpoint(
        pre_sft_dir,
        metadata={
            "role": "phase5_pre_sft_checkpoint",
            "production_qualified": False,
            "base_champion_model_sha256": base_model_sha,
            "target_tokens": int(target_tokens),
        },
    )

    if not replay:
        return runtime, {
            "accepted": False,
            "rolled_back": True,
            "reason": "no replay rows",
            "attempts": [],
        }
    heldout = list(heldout_sft)
    if not heldout:
        return runtime, {
            "accepted": False,
            "rolled_back": True,
            "reason": "no held-out SFT validation rows",
            "attempts": [],
        }

    before = evaluate_sft_validation(
        runtime,
        heldout,
        max_examples=24,
        max_new_tokens=32,
    )
    collapsed = bool(before.get("pathological_repetition")) or (
        float(before.get("repetition_rate", 0.0) or 0.0) >= 0.60
    )
    parameters = parameter_count(runtime.model)
    sft_anti_collapse_weight = 0.08 if collapsed else 0.0
    sft_eos_loss_weight = 1.75 if collapsed else 1.0
    if int(target_tokens) >= 100_000_000 or parameters >= 3_000_000:
        attempts = (
            {"steps": 1600, "learning_rate": 4e-5},
            {"steps": 800, "learning_rate": 2.5e-5},
            {"steps": 320, "learning_rate": 1.25e-5},
        )
    elif int(target_tokens) >= 20_000_000 or parameters >= 750_000:
        attempts = (
            {"steps": 800, "learning_rate": 5e-5},
            {"steps": 400, "learning_rate": 3e-5},
            {"steps": 160, "learning_rate": 1.5e-5},
        )
    elif parameters >= 250_000:
        attempts = (
            {"steps": 320, "learning_rate": 5e-5},
            {"steps": 160, "learning_rate": 3e-5},
            {"steps": 80, "learning_rate": 1.5e-5},
        )
    else:
        attempts = (
            {"steps": 120, "learning_rate": 5e-5},
            {"steps": 60, "learning_rate": 3e-5},
            {"steps": 30, "learning_rate": 1.5e-5},
        )
    reports: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_index: int | None = None

    for index, config in enumerate(attempts):
        trial = GeneralistRuntime.from_checkpoint(pre_sft_dir, device="cpu")
        train_report = train_sft(
            trial.model,
            trial.tokenizer,
            replay,
            steps=int(config["steps"]),
            batch_size=8,
            learning_rate=float(config["learning_rate"]),
            weight_decay=0.01,
            seed=int(seed + index * 101),
            device="cpu",
            gradient_accumulation_steps=1,
            precision="fp32",
            repetition_unlikelihood_weight=sft_anti_collapse_weight,
            eos_loss_weight=sft_eos_loss_weight,
        )
        after = evaluate_sft_validation(
            trial,
            heldout,
            max_examples=24,
            max_new_tokens=32,
        )
        gate_ok, gate_reasons = sft_validation_gate(before, after)
        score = _sft_attempt_score(before, after)
        report = {
            "index": index,
            "steps": int(config["steps"]),
            "learning_rate": float(config["learning_rate"]),
            "training": train_report,
            "validation": after,
            "gate_passed": bool(gate_ok),
            "gate_reasons": list(gate_reasons),
            "selection_score": float(score),
            "anti_collapse_objective": {
                "enabled": bool(sft_anti_collapse_weight > 0.0),
                "repetition_unlikelihood_weight": float(sft_anti_collapse_weight),
                "eos_loss_weight": float(sft_eos_loss_weight),
                "decoding_modified": False,
            },
        }
        reports.append(report)
        if gate_ok and score > best_score:
            best_score = float(score)
            best_index = index
            shutil.rmtree(selected_dir, ignore_errors=True)
            trial.save_checkpoint(
                selected_dir,
                metadata={
                    "role": "phase5_guarded_sft_selected",
                    "production_qualified": False,
                    "base_champion_model_sha256": base_model_sha,
                    "target_tokens": int(target_tokens),
                    "attempt_index": int(index),
                },
            )

    if best_index is None:
        restored = GeneralistRuntime.from_checkpoint(pre_sft_dir, device="cpu")
        shutil.rmtree(selected_dir, ignore_errors=True)
        return restored, {
            "accepted": False,
            "rolled_back": True,
            "reason": "all SFT attempts failed held-out degeneration gate",
            "validation_before": before,
            "attempts": reports,
        }

    selected = GeneralistRuntime.from_checkpoint(selected_dir, device="cpu")
    shutil.rmtree(selected_dir, ignore_errors=True)
    return selected, {
        "accepted": True,
        "rolled_back": False,
        "reason": "best held-out-safe SFT attempt selected",
        "validation_before": before,
        "selected_attempt": int(best_index),
        "selected_score": float(best_score),
        "attempts": reports,
    }


def run_segment(
    state_dir: str | Path,
    repo_root: str | Path,
    cache_dir: str | Path,
    *,
    target_tokens: int,
    segment_tokens: int = 250_000,
    batch_size: int = 16,
    base_learning_rate: float = 3e-4,
    eval_every_steps: int = 32,
    historical_recovery_source: str | Path | None = None,
) -> dict[str, Any]:
    import torch

    root = Path(state_dir).expanduser().resolve()
    repo = Path(repo_root).expanduser().resolve()
    cache = Path(cache_dir).expanduser().resolve()
    bootstrap_root = root / "bootstrap-data"
    candidate_dir = bootstrap_root / "candidate"
    progress_path = bootstrap_root / "progress.json"
    optimizer_path = bootstrap_root / "optimizer.pt"
    manifest_path = bootstrap_root / "manifest.json"
    before_path = bootstrap_root / "before.json"
    after_path = bootstrap_root / "after.json"
    language_guard_path = bootstrap_root / "language-guard.json"
    segment_guard_path = bootstrap_root / "segment-guard-last.json"
    rehabilitation_report_path = bootstrap_root / "language-rehabilitation.json"
    segment_best_dir = bootstrap_root / ".segment-best"
    bootstrap_root.mkdir(parents=True, exist_ok=True)

    status = _load_json(root / "status.json")
    cycle = int(status.get("cycle", 0) or 0)
    loaded = _load_champion(root, device="cpu")
    if loaded is None:
        raise RuntimeError("Phase 5 requires an existing champion checkpoint")
    champion_genome, champion_runtime = loaded
    champion_model_path = root / "champion" / "model.pt"
    base_model_sha = _sha256_file(champion_model_path)

    persisted_progress = _load_json(progress_path, {}) if progress_path.is_file() else {}
    target_tokens = _effective_bootstrap_target(target_tokens, persisted_progress)

    previous_manifest = _load_json(manifest_path) if manifest_path.is_file() else None
    corpus_target_tokens = _bootstrap_corpus_target(target_tokens)
    bundle = build_bootstrap_bundle(
        champion_runtime.tokenizer,
        cache_dir=cache,
        target_tokens=int(corpus_target_tokens),
        previous_manifest=previous_manifest,
    )
    if int(bundle.manifest.get("actual_selected_tokens", 0) or 0) < int(corpus_target_tokens * 0.95):
        raise RuntimeError(
            "bootstrap corpus coverage is below 95% of reviewed unique-corpus target: "
            f"selected={bundle.manifest.get('actual_selected_tokens')} "
            f"target={corpus_target_tokens}"
        )
    _atomic_json(manifest_path, bundle.manifest)
    replay_manifest = write_bootstrap_replay(
        bundle,
        champion_runtime.tokenizer,
        output_dir=bootstrap_root,
        max_tokens=min(1_000_000, max(250_000, int(target_tokens // 4))),
        max_sft_conversations=512,
    )
    protected_replay = load_bootstrap_replay(bootstrap_root)

    if not before_path.is_file():
        _atomic_json(before_path, evaluate_phase5_language(champion_runtime))
    before = _load_json(before_path)

    progress = _load_json(progress_path, {
        "schema": 1,
        "version": PHASE5_BOOTSTRAP_VERSION,
        "base_champion_model_sha256": base_model_sha,
        "target_tokens": int(target_tokens),
        "tokens_processed": 0,
        "steps": 0,
        "best_validation_loss": None,
        "bad_eval_count": 0,
        "completed_rungs": [],
        "sft_completed_rungs": [],
    })
    progress["schema"] = 1
    progress["version"] = PHASE5_BOOTSTRAP_VERSION
    progress["accepted_rehabilitation_tokens"] = int(
        progress.get("accepted_rehabilitation_tokens", 0) or 0
    )
    progress["valid_tokens_processed"] = int(
        progress.get("tokens_processed", 0) or 0
    ) + int(progress["accepted_rehabilitation_tokens"])
    rebase_to_champion = False
    if str(progress.get("base_champion_model_sha256")) != base_model_sha:
        # A completed previous rung may have gone through the converged swarm
        # and promoted a jointly-trained descendant. Resume the next language
        # rung from that stronger champion, never from stale pre-fusion weights.
        # A mid-rung champion change still fails closed.
        previous_target = int(progress.get("target_tokens", 0) or 0)
        previous_processed = int(progress.get("tokens_processed", 0) or 0)
        previous_sft = {
            int(value)
            for value in (progress.get("sft_completed_rungs") or [])
            if isinstance(value, (int, float))
        }
        if (
            previous_processed < previous_target
            or previous_target not in previous_sft
        ):
            raise RuntimeError("champion changed during an incomplete Phase 5 bootstrap")
        progress["base_champion_model_sha256"] = base_model_sha
        rebase_to_champion = True

    progress["target_tokens"] = max(int(progress.get("target_tokens", 0) or 0), int(target_tokens))
    target_tokens = int(progress["target_tokens"])
    progress["unique_corpus_target_tokens"] = int(corpus_target_tokens)
    progress["optimization_passes_target"] = float(
        target_tokens / max(1, corpus_target_tokens)
    )

    candidate_genome = champion_genome
    capacity_genome_raw = progress.get("capacity_genome")
    if not rebase_to_champion and isinstance(capacity_genome_raw, dict):
        try:
            candidate_genome = GeneralistGenome(**capacity_genome_raw).validate()
        except Exception:
            candidate_genome = champion_genome

    recovery_source = (
        Path(historical_recovery_source).expanduser().resolve()
        if historical_recovery_source
        else None
    )
    if _historical_language_recovery_due(progress):
        if recovery_source is None:
            raise RuntimeError(
                "historical language recovery is due but no recovery source was provided"
            )
        candidate_genome, runtime, recovery_report = _recover_historical_language_best(
            root,
            progress,
            source_dir=recovery_source,
            base_model_sha=base_model_sha,
        )
        return {
            "ok": True,
            "lineage_id": recovery_report.get("lineage_id_after"),
            "version": PHASE5_BOOTSTRAP_VERSION,
            "target_tokens": int(target_tokens),
            "tokens_processed": int(progress.get("tokens_processed", 0) or 0),
            "valid_tokens_processed": int(progress.get("valid_tokens_processed", 0) or 0),
            "accepted_rehabilitation_tokens": 0,
            "segment_tokens_processed": 0,
            "parameters": int(parameter_count(runtime.model)),
            "historical_recovery_only": True,
            "historical_recovery_report": recovery_report,
            "rung_complete": False,
            "early_stopped": False,
        }

    if rebase_to_champion:
        shutil.rmtree(candidate_dir, ignore_errors=True)
        _remove_optimizer_checkpoint(optimizer_path)
        shutil.rmtree(bootstrap_root / "best", ignore_errors=True)
        shutil.rmtree(bootstrap_root / "pre-sft", ignore_errors=True)
        shutil.rmtree(bootstrap_root / "pre-anticollapse", ignore_errors=True)
        progress["capacity_genome"] = champion_genome.to_dict()
        progress["capacity_rebased_from_champion"] = {
            "model_sha256": base_model_sha,
            "parameters": int(parameter_count(champion_runtime.model)),
            "training_target_tokens": int(target_tokens),
        }

    if candidate_dir.is_dir():
        runtime = GeneralistRuntime.from_checkpoint(candidate_dir, device="cpu")
    else:
        runtime = GeneralistRuntime(
            copy.deepcopy(champion_runtime.model),
            champion_runtime.config,
            tokenizer=champion_runtime.tokenizer,
            device="cpu",
        )
        runtime.save_checkpoint(
            candidate_dir,
            metadata={
                "role": "phase5_language_bootstrap_candidate",
                "production_qualified": False,
                "base_champion_model_sha256": base_model_sha,
                "rebased_from_converged_champion": bool(rebase_to_champion),
            },
        )

    if _dead_capacity_revival_due(progress):
        revival_report = _revive_dead_ffn_capacity(
            root,
            progress,
            runtime,
            candidate_dir=candidate_dir,
            optimizer_path=optimizer_path,
            base_model_sha=base_model_sha,
        )
        return {
            "ok": True,
            "lineage_id": str(progress.get("lineage_id") or ""),
            "version": PHASE5_BOOTSTRAP_VERSION,
            "target_tokens": int(target_tokens),
            "tokens_processed": int(progress.get("tokens_processed", 0) or 0),
            "valid_tokens_processed": int(
                progress.get("valid_tokens_processed", 0) or 0
            ),
            "accepted_rehabilitation_tokens": int(
                progress.get("accepted_rehabilitation_tokens", 0) or 0
            ),
            "segment_tokens_processed": 0,
            "parameters": int(parameter_count(runtime.model)),
            "dead_capacity_revival_only": True,
            "dead_capacity_revival_report": revival_report,
            "rung_complete": False,
            "early_stopped": False,
        }

    desired_capacity = _bootstrap_capacity_target(target_tokens)
    current_capacity = int(parameter_count(runtime.model))
    assisted_capacity = _assisted_capacity_target(
        progress,
        current_parameters=current_capacity,
    )
    if assisted_capacity is not None:
        desired_capacity = max(int(desired_capacity or 0), int(assisted_capacity))

    if desired_capacity is not None and current_capacity < int(desired_capacity):
        candidate_genome, runtime, growth = _grow_bootstrap_runtime(
            candidate_genome,
            runtime,
            target_parameters=int(desired_capacity),
        )
        progress["capacity_target_parameters"] = int(desired_capacity)
        progress["capacity_genome"] = candidate_genome.to_dict()
        history = list(progress.get("capacity_growth_history") or [])
        history.append({
            **growth,
            "training_target_tokens": int(target_tokens),
            "tokens_processed_before_growth": int(progress.get("tokens_processed", 0) or 0),
        })
        progress["capacity_growth_history"] = history
        grown_parameters = int(parameter_count(runtime.model))
        if (
            assisted_capacity is not None
            and grown_parameters >= ASSISTED_CAPACITY_COMPLETION_FLOOR
        ):
            progress["assisted_capacity_growth_completed"] = True
            progress["assisted_capacity_growth"] = {
                "lineage_id": ASSISTED_CAPACITY_LINEAGE_ID,
                "requested_parameters": int(ASSISTED_CAPACITY_TARGET_PARAMETERS),
                "actual_parameters": grown_parameters,
                "tokens_processed_before_growth": int(
                    progress.get("tokens_processed", 0) or 0
                ),
                "generation": int(candidate_genome.generation),
                "genome_id": str(candidate_genome.genome_id),
                "function_preserving_growth": bool(
                    (growth.get("weight_transfer") or {}).get(
                        "function_preserving_growth"
                    )
                ),
            }
        progress["best_validation_loss"] = None
        progress["bad_eval_count"] = 0
        _remove_optimizer_checkpoint(optimizer_path)
        shutil.rmtree(bootstrap_root / "best", ignore_errors=True)
        shutil.rmtree(bootstrap_root / "pre-sft", ignore_errors=True)
        shutil.rmtree(bootstrap_root / "pre-anticollapse", ignore_errors=True)
        runtime.save_checkpoint(
            candidate_dir,
            metadata={
                "role": "phase5_language_bootstrap_candidate",
                "production_qualified": False,
                "base_champion_model_sha256": base_model_sha,
                "capacity_growth": growth,
            },
        )

        if (
            assisted_capacity is not None
            and bool(progress.get("assisted_capacity_growth_completed"))
        ):
            # The 50M assist is a structural gift, not another hidden training
            # phase. Persist it transactionally first, then let the next normal
            # bootstrap run continue learning from the enlarged live lineage.
            progress["rung_complete"] = False
            progress["early_stopped"] = False
            progress["updated_at_unix"] = int(time.time())
            _atomic_json(progress_path, progress)
            lineage_manifest = refresh_live_lineage_manifest(
                root,
                reason="assisted_50m_capacity_growth",
            )
            return {
                "ok": True,
                "version": PHASE5_BOOTSTRAP_VERSION,
                "lineage_id": lineage_manifest.get("lineage_id"),
                "active_lineage_checkpoint": lineage_manifest.get(
                    "active_checkpoint"
                ),
                "target_tokens": int(target_tokens),
                "tokens_processed": int(progress.get("tokens_processed", 0) or 0),
                "valid_tokens_processed": int(
                    progress.get("valid_tokens_processed", 0) or 0
                ),
                "segment_tokens_processed": 0,
                "steps": int(progress.get("steps", 0) or 0),
                "parameters": int(parameter_count(runtime.model)),
                "capacity_growth_only": True,
                "assisted_capacity_growth": progress.get(
                    "assisted_capacity_growth"
                ),
                "rung_complete": False,
                "early_stopped": False,
            }

    # Packing a 20M-token corpus is expensive and previously repeated for
    # every resumable segment.  Cache the exact fixed-width token blocks in the
    # job-local data cache.  The cache identity includes the reviewed manifest,
    # tokenizer and context length, so stale data fails closed and is rebuilt.
    manifest_identity = str(bundle.manifest.get("manifest_content_sha256") or "")
    packed_cache_root = cache / "phase5-packed-blocks-v1"
    packed_cache_root.mkdir(parents=True, exist_ok=True)
    packed_cache_base = {
        "version": "phase5-stage-block-cache-v1",
        "manifest_content_sha256": manifest_identity,
        "tokenizer_version": str(runtime.tokenizer.version),
        "tokenizer_vocab_size": int(runtime.tokenizer.vocab_size),
        "context_length": int(runtime.config.context_length),
    }

    def _packed_identity(split: str) -> dict[str, Any]:
        return {**packed_cache_base, "split": str(split)}

    def _packed_path(split: str) -> Path:
        safe = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(split))
        digest = hashlib.sha256(
            json.dumps(_packed_identity(split), sort_keys=True).encode("utf-8")
        ).hexdigest()[:20]
        return packed_cache_root / f"{safe}-{digest}.bin"

    stage_blocks: dict[str, Any] = {}
    packed_cache_hits: dict[str, bool] = {}
    stage_names = (
        "A_frequent_word_contexts",
        "B_short_sentence_completion",
        "C_causal_next_sentence",
    )
    for stage in stage_names:
        cached = load_packed_block_cache(
            _packed_path(stage),
            expected_identity=_packed_identity(stage),
        )
        if cached is not None:
            stage_blocks[stage] = cached
            packed_cache_hits[stage] = True
        else:
            packed_cache_hits[stage] = False

    missing_stages = [stage for stage in stage_names if stage not in stage_blocks]
    if "A_frequent_word_contexts" in missing_stages:
        documents = _frequent_word_documents(bundle.train_documents)
        packed = pack_causal_blocks(
            documents,
            runtime.tokenizer,
            context_length=runtime.config.context_length,
        )
        save_packed_block_cache(
            _packed_path("A_frequent_word_contexts"),
            packed,
            block_size=runtime.config.context_length,
            identity=_packed_identity("A_frequent_word_contexts"),
        )
        stage_blocks["A_frequent_word_contexts"] = packed

    if "B_short_sentence_completion" in missing_stages:
        documents = [
            row for row in bundle.train_documents
            if len(row.text) <= 220
        ] or list(bundle.train_documents)
        packed = pack_causal_blocks(
            documents,
            runtime.tokenizer,
            context_length=runtime.config.context_length,
        )
        save_packed_block_cache(
            _packed_path("B_short_sentence_completion"),
            packed,
            block_size=runtime.config.context_length,
            identity=_packed_identity("B_short_sentence_completion"),
        )
        stage_blocks["B_short_sentence_completion"] = packed

    if "C_causal_next_sentence" in missing_stages:
        packed = pack_causal_blocks(
            bundle.train_documents,
            runtime.tokenizer,
            context_length=runtime.config.context_length,
        )
        save_packed_block_cache(
            _packed_path("C_causal_next_sentence"),
            packed,
            block_size=runtime.config.context_length,
            identity=_packed_identity("C_causal_next_sentence"),
        )
        stage_blocks["C_causal_next_sentence"] = packed

    validation_identity = _packed_identity("validation")
    validation_blocks = load_packed_block_cache(
        _packed_path("validation"),
        expected_identity=validation_identity,
    )
    packed_cache_hits["validation"] = validation_blocks is not None
    if validation_blocks is None:
        validation_blocks = pack_causal_blocks(
            bundle.validation_documents,
            runtime.tokenizer,
            context_length=runtime.config.context_length,
        )
        save_packed_block_cache(
            _packed_path("validation"),
            validation_blocks,
            block_size=runtime.config.context_length,
            identity=validation_identity,
        )

    if any(not rows for rows in stage_blocks.values()) or not validation_blocks:
        raise RuntimeError("bootstrap corpus did not produce curriculum train/validation blocks")

    progress["packed_block_cache"] = {
        "version": "phase5-stage-block-cache-v1",
        "manifest_content_sha256": manifest_identity,
        "hits": packed_cache_hits,
        "block_counts": {
            stage: len(rows) for stage, rows in stage_blocks.items()
        },
        "validation_blocks": len(validation_blocks),
    }

    progress["bootstrap_replay"] = replay_manifest
    progress["curriculum_schedule"] = [
        "A_frequent_word_contexts",
        "B_short_sentence_completion",
        "C_causal_next_sentence",
        "D_simple_prompt_response",
        "E_simple_multiturn_chat",
        "F_airi_pc_lab_tool_use",
    ]

    segment_language_before = evaluate_phase5_language(runtime)
    guard_payload = _load_json(language_guard_path) if language_guard_path.is_file() else {}
    guard_anchor = (
        dict(guard_payload.get("report") or {})
        if isinstance(guard_payload, dict)
        else {}
    )
    if not guard_anchor:
        historical = _load_json(after_path) if after_path.is_file() else {}
        if (
            isinstance(historical, dict)
            and historical
            and _language_quality(historical) > _language_quality(segment_language_before)
        ):
            guard_anchor = historical
            guard_source = "best_completed_rung"
        else:
            guard_anchor = segment_language_before
            guard_source = "current_live_lineage"
        _atomic_json(
            language_guard_path,
            {
                "schema": 1,
                "version": "phase5-segment-language-guard-v1",
                "source": guard_source,
                "report": guard_anchor,
                "created_at_unix": int(time.time()),
            },
        )

    rehabilitation_state = dict(progress.get("language_rehabilitation") or {})
    if (
        str(rehabilitation_state.get("version") or "")
        != PHASE5_LANGUAGE_REHABILITATION_VERSION
    ):
        rehabilitation_state = {
            "version": PHASE5_LANGUAGE_REHABILITATION_VERSION,
            "next_stage_index": 0,
            "completed_cycles": 0,
        }
    next_stage_index = max(
        0,
        int(rehabilitation_state.get("next_stage_index", 0) or 0),
    )
    rehabilitation_due = bool(
        not bool(progress.get("causal_recovery_mode", False))
        and _rehabilitation_cycle_due(
            segment_language_before,
            guard_anchor,
            next_stage_index,
        )
    )
    if rehabilitation_due:
        if next_stage_index >= len(PHASE5_LANGUAGE_REHABILITATION_STAGES):
            next_stage_index = 0
            rehabilitation_state["completed_cycles"] = int(
                rehabilitation_state.get("completed_cycles", 0) or 0
            ) + 1
        stage = PHASE5_LANGUAGE_REHABILITATION_STAGES[next_stage_index]
        curriculum = _elementary_rehabilitation_rows()
        total_rejection_count = int(
            rehabilitation_state.get("consecutive_rejections", 0) or 0
        )
        rejection_count = _rehabilitation_strategy_rejection_count(
            progress,
            rehabilitation_state,
        )
        residual_strategy_active = bool(
            (progress.get("dead_capacity_revival") or {}).get("completed")
        )
        replay_rejection_count = (
            max(3, int(rejection_count))
            if residual_strategy_active
            else int(rejection_count)
        )
        rehabilitation_rows, rehabilitation_counts = _rehabilitation_replay_rows(
            stage,
            curriculum,
            protected_replay.sft_train,
            heldout_sft=bundle.sft_validation,
            max_replay_rows=_rehabilitation_replay_limit(
                stage,
                replay_rejection_count,
            ),
            # Rotate with global history for diversity, while hyperparameters
            # use only the current strategy's rejection history.
            replay_offset=total_rejection_count * 37,
        )
        lineage_before = str(
            _load_json(root / "lineage.json").get("lineage_id") or ""
        )
        valid_before = int(progress.get("valid_tokens_processed", 0) or 0)
        runtime, rehabilitation_report = _run_language_rehabilitation_stage(
            runtime,
            stage,
            rehabilitation_rows,
            bundle.sft_validation,
            guard_anchor,
            bootstrap_root=bootstrap_root,
            base_model_sha=base_model_sha,
            seed=91 + next_stage_index * 1009 + total_rejection_count * 100_003,
            consecutive_rejections=rejection_count,
        )
        rehabilitation_report.update({
            "curriculum": rehabilitation_counts,
            "global_consecutive_rejections_before": int(total_rejection_count),
            "strategy_consecutive_rejections_before": int(rejection_count),
            "protected_replay_manifest_sha256": str(
                replay_manifest.get("corpus_sha256") or ""
            ),
            "lineage_id_before": lineage_before,
            "parameters_before": int(current_capacity),
            "parameters_after": int(parameter_count(runtime.model)),
            "valid_tokens_before": valid_before,
            "causal_tokens_before": int(progress.get("tokens_processed", 0) or 0),
        })
        accepted_rehabilitation = bool(rehabilitation_report.get("accepted"))
        if accepted_rehabilitation:
            accepted_tokens = int(
                rehabilitation_report.get("accepted_supervised_tokens", 0) or 0
            )
            if accepted_tokens <= 0:
                raise RuntimeError(
                    "accepted language rehabilitation reported no supervised tokens"
                )
            progress["accepted_rehabilitation_tokens"] = int(
                progress.get("accepted_rehabilitation_tokens", 0) or 0
            ) + accepted_tokens
            progress["valid_tokens_processed"] = int(
                progress.get("tokens_processed", 0) or 0
            ) + int(progress["accepted_rehabilitation_tokens"])
            rehabilitation_state["next_stage_index"] = next_stage_index + 1
            rehabilitation_state["last_accepted_stage"] = stage
            rehabilitation_state["last_accepted_tokens"] = accepted_tokens
            rehabilitation_state["consecutive_rejections"] = 0
            if (
                str(rehabilitation_report.get("training_mode") or "")
                == "residual_kl_recovery"
            ):
                rehabilitation_state["residual_consecutive_rejections"] = 0
            runtime.save_checkpoint(
                candidate_dir,
                metadata={
                    "role": "phase5_language_bootstrap_candidate",
                    "production_qualified": False,
                    "base_champion_model_sha256": base_model_sha,
                    "tokens_processed": int(progress.get("tokens_processed", 0) or 0),
                    "valid_tokens_processed": int(progress["valid_tokens_processed"]),
                    "language_rehabilitation_stage": stage,
                    "language_rehabilitation_version": (
                        PHASE5_LANGUAGE_REHABILITATION_VERSION
                    ),
                },
            )
            _remove_optimizer_checkpoint(optimizer_path)
            progress["optimizer_reset_after_language_rehabilitation"] = True
        else:
            rehabilitation_state["next_stage_index"] = next_stage_index
            rehabilitation_state["consecutive_rejections"] = int(
                rehabilitation_state.get("consecutive_rejections", 0) or 0
            ) + 1
            if (
                str(rehabilitation_report.get("training_mode") or "")
                == "residual_kl_recovery"
            ):
                rehabilitation_state["residual_consecutive_rejections"] = int(
                    rehabilitation_state.get(
                        "residual_consecutive_rejections",
                        0,
                    )
                    or 0
                ) + 1
            progress["valid_tokens_processed"] = valid_before
        rehabilitation_state["version"] = PHASE5_LANGUAGE_REHABILITATION_VERSION
        rehabilitation_state["last_stage"] = stage
        rehabilitation_state["last_accepted"] = accepted_rehabilitation
        rehabilitation_state["updated_at_unix"] = int(time.time())
        progress["language_rehabilitation"] = rehabilitation_state
        progress["rung_complete"] = False
        progress["early_stopped"] = False
        progress["updated_at_unix"] = int(time.time())
        _atomic_json(progress_path, progress)
        lineage_manifest = refresh_live_lineage_manifest(
            root,
            reason=(
                "phase5_language_rehabilitation_checkpoint"
                if accepted_rehabilitation
                else "phase5_language_rehabilitation_rollback"
            ),
        )
        lineage_after = str(lineage_manifest.get("lineage_id") or "")
        if lineage_before and lineage_after != lineage_before:
            raise RuntimeError("language rehabilitation changed the live lineage id")
        rehabilitation_report.update({
            "lineage_id_after": lineage_after,
            "lineage_preserved": bool(
                not lineage_before or lineage_after == lineage_before
            ),
            "valid_tokens_after": int(progress["valid_tokens_processed"]),
            "valid_tokens_increased": bool(
                int(progress["valid_tokens_processed"]) > valid_before
            ),
            "causal_tokens_after": int(progress.get("tokens_processed", 0) or 0),
        })
        _atomic_json(rehabilitation_report_path, rehabilitation_report)
        return {
            "ok": True,
            "lineage_id": lineage_after,
            "active_lineage_checkpoint": lineage_manifest.get("active_checkpoint"),
            "version": PHASE5_BOOTSTRAP_VERSION,
            "target_tokens": target_tokens,
            "tokens_processed": int(progress.get("tokens_processed", 0) or 0),
            "valid_tokens_processed": int(
                progress.get("valid_tokens_processed", 0) or 0
            ),
            "accepted_rehabilitation_tokens": int(
                progress["accepted_rehabilitation_tokens"]
            ),
            "segment_tokens_processed": 0,
            "rehabilitation_stage": stage,
            "rehabilitation_accepted": accepted_rehabilitation,
            "rehabilitation_rejected": not accepted_rehabilitation,
            "rehabilitation_report_path": str(rehabilitation_report_path),
            "steps": int(progress.get("steps", 0) or 0),
            "rung_complete": False,
            "early_stopped": False,
        }

    optimizer = torch.optim.AdamW(
        runtime.model.parameters(),
        lr=float(base_learning_rate),
        weight_decay=0.01,
    )
    optimizer_storage = None
    if optimizer_path.is_file():
        optimizer_storage = _load_optimizer_checkpoint(optimizer, optimizer_path)

    progress_before_segment = copy.deepcopy(progress)
    consecutive_rejections = int(
        progress.get("segment_guard_consecutive_rejections", 0) or 0
    )
    recovery_plan = _phase5_recovery_plan(
        segment_tokens,
        parameters=int(parameter_count(runtime.model)),
        context_length=int(runtime.config.context_length),
        persisted_lr_scale=float(
            progress.get("segment_guard_lr_scale", 1.0) or 1.0
        ),
        consecutive_rejections=consecutive_rejections,
        recovery_hold=bool(progress.get("segment_guard_recovery_hold", False)),
        last_segment_accepted=bool(progress.get("segment_guard_last_accepted", False)),
    )
    segment_lr_scale = float(recovery_plan["learning_rate_scale"])
    retry_seed_offset = int(consecutive_rejections) * 1_000_003
    shutil.rmtree(segment_best_dir, ignore_errors=True)

    processed_before_segment = int(progress.get("tokens_processed", 0) or 0)
    requested_segment_budget = int(recovery_plan["requested_budget_tokens"])
    segment_budget = int(recovery_plan["effective_budget_tokens"])
    parameter_cap = recovery_plan["parameter_cap_tokens"]
    progress["segment_guard_requested_budget_tokens"] = int(requested_segment_budget)
    progress["parameter_segment_cap_tokens"] = parameter_cap
    progress["segment_guard_effective_budget_tokens"] = int(segment_budget)
    progress["segment_guard_recovery_budget_active"] = bool(
        segment_budget < requested_segment_budget
    )
    progress["segment_guard_stall_recovery_active"] = bool(
        recovery_plan["stall_recovery"]
    )
    progress["segment_guard_stable_fast_lane"] = bool(
        recovery_plan.get("stable_fast_lane", False)
    )
    progress["segment_guard_rescue_stage"] = recovery_plan["forced_stage"]

    if bool(recovery_plan["reset_optimizer"]):
        # The model is already the exact pre-segment rollback checkpoint.  A
        # fresh optimizer removes stale AdamW momentum that can repeatedly push
        # the same 50M lineage back across the language guard.
        optimizer = torch.optim.AdamW(
            runtime.model.parameters(),
            lr=float(base_learning_rate) * segment_lr_scale,
            weight_decay=0.01,
        )
        optimizer_storage = None
        progress["segment_guard_optimizer_reset_for_stall"] = True
    else:
        progress["segment_guard_optimizer_reset_for_stall"] = False
    eval_every_steps = max(8, int(eval_every_steps))
    best_loss = progress.get("best_validation_loss")
    best_loss = float(best_loss) if best_loss is not None else float("inf")
    bad_eval_count = int(progress.get("bad_eval_count", 0) or 0)
    early_stopped = False
    last_train_loss = None
    last_validation_loss = None

    effective_batch_size = max(1, int(batch_size))
    micro_batch_size, gradient_accumulation_steps = _phase5_memory_safe_batch_plan(
        parameters=int(parameter_count(runtime.model)),
        context_length=int(runtime.config.context_length),
        requested_batch_size=effective_batch_size,
    )
    progress["training_resource_plan"] = {
        "parameters": int(parameter_count(runtime.model)),
        "context_length": int(runtime.config.context_length),
        "effective_batch_size": int(effective_batch_size),
        "micro_batch_size": int(micro_batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
    }

    runtime.model.train()
    while (
        int(progress["tokens_processed"]) < target_tokens
        and int(progress["tokens_processed"]) - processed_before_segment < segment_budget
    ):
        step = int(progress["steps"])
        forced_stage = recovery_plan["forced_stage"]
        stage = (
            str(forced_stage)
            if forced_stage
            else _causal_curriculum_stage(
                int(progress["tokens_processed"]),
                target_tokens,
            )
        )
        progress["curriculum_stage"] = stage
        train_blocks = stage_blocks[stage]
        rng = random.Random(5_000_000 + step + retry_seed_offset)
        indices = [
            rng.randrange(len(train_blocks))
            for _ in range(effective_batch_size)
        ]
        optimizer.zero_grad(set_to_none=True)

        lr = _learning_rate(
            base_lr=float(base_learning_rate) * segment_lr_scale,
            processed_tokens=int(progress["tokens_processed"]),
            target_tokens=target_tokens,
            warmup_tokens=min(100_000, max(20_000, target_tokens // 10)),
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        anti_weight, eos_weight = _anti_collapse_weights(stage, before)
        supervised = 0
        weighted_loss = 0.0
        objective_stats = {}
        micro_batches = [
            indices[offset:offset + micro_batch_size]
            for offset in range(0, len(indices), micro_batch_size)
        ]
        prepared_micro_batches = []
        for micro_indices in micro_batches:
            ids, labels = _batch(train_blocks, micro_indices, device=runtime.device)
            micro_supervised = int((labels[:, 1:] != -100).sum().item())
            prepared_micro_batches.append((ids, labels, micro_supervised))
            supervised += micro_supervised
        if supervised <= 0:
            raise RuntimeError("Phase 5 batch contains no supervised tokens")

        for ids, labels, micro_supervised in prepared_micro_batches:
            result = runtime.model(ids)
            loss, micro_objective = causal_training_objective(
                result["logits"],
                labels,
                ids,
                eos_loss_weight=eos_weight,
                repetition_unlikelihood_weight=anti_weight,
                repetition_window=16,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite Phase 5 bootstrap loss")
            weighted_loss += float(loss.detach().cpu()) * micro_supervised
            # Weight gradients by supervised-token share so accumulation
            # reproduces the full effective batch even with padded examples.
            (loss * (micro_supervised / supervised)).backward()
            objective_stats = micro_objective

        torch.nn.utils.clip_grad_norm_(runtime.model.parameters(), 1.0)
        optimizer.step()

        progress["tokens_processed"] = int(progress["tokens_processed"]) + supervised
        progress["steps"] = step + 1
        progress["learning_rate"] = float(lr)
        last_train_loss = weighted_loss / max(1, supervised)
        progress["last_train_loss"] = float(last_train_loss)
        progress["last_objective"] = objective_stats
        progress["anti_collapse_training_enabled"] = bool(anti_weight > 0.0)

        if int(progress["steps"]) % eval_every_steps == 0:
            last_validation_loss = corpus_loss(
                runtime.model,
                validation_blocks,
                device="cpu",
                batch_size=max(1, min(8, int(micro_batch_size))),
                max_blocks=min(96, len(validation_blocks)),
                seed=991,
            )
            progress["last_validation_loss"] = float(last_validation_loss)
            if last_validation_loss + 1e-4 < best_loss:
                best_loss = float(last_validation_loss)
                bad_eval_count = 0
                runtime.save_checkpoint(
                    segment_best_dir,
                    metadata={
                        "role": "phase5_language_bootstrap_segment_best",
                        "validation_loss": best_loss,
                        "tokens_processed": int(progress["tokens_processed"]),
                    },
                )
            else:
                bad_eval_count += 1
            progress["best_validation_loss"] = best_loss
            progress["bad_eval_count"] = bad_eval_count
            # A freshly grown network needs a substantial fraction of the
            # rescue rung before plateau logic is allowed to stop it.
            minimum_before_early_stop = max(
                600_000,
                int(target_tokens * 0.75),
            )
            if (
                bad_eval_count >= 4
                and int(progress["tokens_processed"]) >= minimum_before_early_stop
            ):
                early_stopped = True
                break
            runtime.model.train()

    segment_language_after = evaluate_phase5_language(runtime)
    attempted_tokens = int(progress["tokens_processed"]) - processed_before_segment
    attempted_steps = int(progress["steps"]) - int(
        progress_before_segment.get("steps", 0) or 0
    )
    segment_accepted, segment_guard = _segment_language_gate(
        segment_language_before,
        segment_language_after,
        guard_anchor,
        attempted_tokens=attempted_tokens,
    )
    segment_guard.update({
        "attempted_tokens": int(attempted_tokens),
        "attempted_steps": int(attempted_steps),
        "learning_rate_scale": float(segment_lr_scale),
        "consecutive_rejections_before": int(consecutive_rejections),
    })
    _atomic_json(segment_guard_path, segment_guard)

    if not segment_accepted:
        # candidate_dir and optimizer.pt still contain the exact pre-segment
        # state because they are written only after this gate passes.
        shutil.rmtree(segment_best_dir, ignore_errors=True)
        progress = progress_before_segment
        history = list(progress.get("segment_guard_rejections") or [])
        history.append({
            "tokens_processed": int(progress.get("tokens_processed", 0) or 0),
            "attempted_tokens": int(attempted_tokens),
            "attempted_steps": int(attempted_steps),
            "before_quality": float(segment_guard["before_quality"]),
            "after_quality": float(segment_guard["after_quality"]),
            "recovery_mode": bool(segment_guard["recovery_mode"]),
            "reasons": list(segment_guard.get("local_reasons") or [])
                + list(segment_guard.get("after_anchor_violations") or []),
            "learning_rate_scale": float(segment_lr_scale),
        })
        progress["segment_guard_rejections"] = history[-32:]
        next_rejection_count = int(consecutive_rejections + 1)
        progress["segment_guard_consecutive_rejections"] = next_rejection_count
        next_plan = _phase5_recovery_plan(
            requested_segment_budget,
            parameters=int(parameter_count(runtime.model)),
            context_length=int(runtime.config.context_length),
            persisted_lr_scale=float(segment_lr_scale) * 0.5,
            consecutive_rejections=next_rejection_count,
            recovery_hold=bool(
                progress_before_segment.get("segment_guard_recovery_hold", False)
            ),
            last_segment_accepted=False,
        )
        progress["segment_guard_lr_scale"] = float(
            next_plan["learning_rate_scale"]
        )
        # Preserve attempt telemetry across rollback.  Previously restoring
        # progress_before_segment also restored stale budget fields, making the
        # app look frozen even while smaller rescue transactions were running.
        progress["segment_guard_requested_budget_tokens"] = int(
            requested_segment_budget
        )
        progress["parameter_segment_cap_tokens"] = parameter_cap
        progress["segment_guard_effective_budget_tokens"] = int(segment_budget)
        progress["segment_guard_recovery_budget_active"] = bool(
            segment_budget < requested_segment_budget
        )
        progress["segment_guard_last_attempted_tokens"] = int(attempted_tokens)
        progress["segment_guard_last_attempted_steps"] = int(attempted_steps)
        progress["segment_guard_stall_recovery_active"] = bool(
            next_plan["stall_recovery"]
        )
        progress["segment_guard_rescue_stage"] = next_plan["forced_stage"]
        progress["segment_guard_optimizer_reset_for_stall"] = bool(
            next_plan["reset_optimizer"]
        )
        progress["segment_guard_last_accepted"] = False
        progress["segment_guard_recovery_mode"] = bool(
            segment_guard.get("recovery_mode")
        )
        progress["early_stopped"] = False
        progress["rung_complete"] = False
        progress["updated_at_unix"] = int(time.time())
        _atomic_json(progress_path, progress)
        lineage_manifest = refresh_live_lineage_manifest(
            root,
            reason="phase5_segment_language_guard_rollback",
        )
        return {
            "ok": True,
            "lineage_id": lineage_manifest.get("lineage_id"),
            "active_lineage_checkpoint": lineage_manifest.get("active_checkpoint"),
            "version": PHASE5_BOOTSTRAP_VERSION,
            "target_tokens": target_tokens,
            "tokens_processed": int(progress.get("tokens_processed", 0) or 0),
            "valid_tokens_processed": int(
                progress.get("valid_tokens_processed", 0) or 0
            ),
            "segment_tokens_processed": 0,
            "steps": int(progress.get("steps", 0) or 0),
            "rung_complete": False,
            "early_stopped": False,
            "segment_rejected": True,
            "segment_guard_path": str(segment_guard_path),
            "next_learning_rate_scale": float(progress["segment_guard_lr_scale"]),
        }

    if segment_best_dir.is_dir():
        shutil.rmtree(bootstrap_root / "best", ignore_errors=True)
        segment_best_dir.rename(bootstrap_root / "best")

    progress["segment_guard_consecutive_rejections"] = 0
    progress["valid_tokens_processed"] = int(progress["tokens_processed"]) + int(
        progress.get("accepted_rehabilitation_tokens", 0) or 0
    )
    recovery_mode = bool(segment_guard.get("recovery_mode"))
    remaining_anchor_violations = list(
        segment_guard.get("after_anchor_violations") or []
    )
    recovery_hold = bool(recovery_mode and remaining_anchor_violations)
    previous_recovery_streak = int(
        progress_before_segment.get("segment_guard_recovery_accept_streak", 0) or 0
    )
    progress["segment_guard_recovery_hold"] = recovery_hold
    if bool(progress.get("causal_recovery_mode", False)) and not remaining_anchor_violations:
        progress["causal_recovery_mode"] = False
    progress["segment_guard_recovery_accept_streak"] = (
        previous_recovery_streak + 1 if recovery_hold else 0
    )
    if recovery_hold:
        # Hold the exact conservative regime that just produced measurable
        # recovery.  Do not raise LR after one good 31k-token segment while the
        # lineage is still below the durable anchor.
        progress["segment_guard_lr_scale"] = min(
            0.0625,
            max(1.0 / 64.0, float(segment_lr_scale)),
        )
    else:
        progress["segment_guard_lr_scale"] = min(
            1.0,
            max(1.0 / 64.0, float(segment_lr_scale) * 1.20),
        )
    progress["segment_guard_last_accepted"] = True
    progress["segment_guard_stall_recovery_active"] = recovery_hold
    progress["segment_guard_rescue_stage"] = (
        "B_short_sentence_completion" if recovery_hold else None
    )
    progress["segment_guard_optimizer_reset_for_stall"] = False
    progress["segment_guard_last_attempted_tokens"] = int(attempted_tokens)
    progress["segment_guard_last_attempted_steps"] = int(attempted_steps)
    progress["segment_guard_recovery_mode"] = recovery_mode
    accepted_history = list(progress.get("segment_guard_acceptances") or [])
    accepted_history.append({
        "tokens_processed": int(progress["tokens_processed"]),
        "segment_tokens": int(attempted_tokens),
        "before_quality": float(segment_guard["before_quality"]),
        "after_quality": float(segment_guard["after_quality"]),
        "recovery_mode": bool(segment_guard["recovery_mode"]),
        "learning_rate_scale": float(segment_lr_scale),
    })
    progress["segment_guard_acceptances"] = accepted_history[-32:]

    if _language_quality(segment_language_after) > _language_quality(guard_anchor) + 0.005:
        runtime.save_checkpoint(
            bootstrap_root / "language-guard-best",
            metadata={
                "role": "phase5_language_guard_best",
                "tokens_processed": int(progress["tokens_processed"]),
                "language_quality": float(_language_quality(segment_language_after)),
            },
        )
        _atomic_json(
            language_guard_path,
            {
                "schema": 1,
                "version": "phase5-segment-language-guard-v1",
                "source": "accepted_segment_improvement",
                "report": segment_language_after,
                "tokens_processed": int(progress["tokens_processed"]),
                "updated_at_unix": int(time.time()),
            },
        )

    runtime.save_checkpoint(
        candidate_dir,
        metadata={
            "role": "phase5_language_bootstrap_candidate",
            "production_qualified": False,
            "base_champion_model_sha256": base_model_sha,
            "tokens_processed": int(progress["tokens_processed"]),
            "target_tokens": target_tokens,
            "segment_language_guard": True,
        },
    )
    optimizer_storage = _save_optimizer_checkpoint(optimizer, optimizer_path)
    progress["optimizer_storage"] = optimizer_storage

    rung_complete = int(progress["tokens_processed"]) >= target_tokens or early_stopped
    progress["early_stopped"] = bool(early_stopped)
    progress["rung_complete"] = bool(rung_complete)
    progress["updated_at_unix"] = int(time.time())

    rescue_accepted = False
    rescue_due = bool(
        rung_complete
        and (
            str(progress.get("anti_collapse_rescue_version") or "")
            != PHASE5_ANTICOLLAPSE_VERSION
            or str(progress.get("anti_collapse_base_model_sha256") or "")
            != base_model_sha
        )
    )
    if rescue_due:
        runtime_parameters = parameter_count(runtime.model)
        rescue_steps = (
            1024 if runtime_parameters >= 3_000_000
            else 512 if runtime_parameters >= 750_000
            else 256 if runtime_parameters >= 250_000
            else 128
        )
        runtime, rescue_report = _run_anti_collapse_rescue(
            runtime,
            stage_blocks,
            bootstrap_root=bootstrap_root,
            base_model_sha=base_model_sha,
            seed=71 + int(cycle),
            steps=rescue_steps,
        )
        rescue_accepted = bool(rescue_report.get("accepted"))
        progress["anti_collapse_rescue"] = rescue_report
        progress["anti_collapse_rescue_version"] = PHASE5_ANTICOLLAPSE_VERSION
        progress["anti_collapse_base_model_sha256"] = base_model_sha
        progress["anti_collapse_rescue_tokens"] = int(
            progress.get("anti_collapse_rescue_tokens", 0) or 0
        ) + int(rescue_report.get("tokens_processed", 0) or 0)
        if rescue_accepted:
            _remove_optimizer_checkpoint(optimizer_path)
            progress["optimizer_reset_after_anticollapse"] = True
        runtime.save_checkpoint(
            candidate_dir,
            metadata={
                "role": "phase5_language_bootstrap_candidate",
                "production_qualified": False,
                "base_champion_model_sha256": base_model_sha,
                "tokens_processed": int(progress["tokens_processed"]),
                "target_tokens": target_tokens,
                "anti_collapse_rescue": bool(rescue_accepted),
            },
        )

    completed_sft_before = set(
        int(x) for x in (progress.get("sft_completed_rungs") or [])
    )
    sft_guard_migration = bool(
        rung_complete
        and target_tokens in completed_sft_before
        and str(progress.get("sft_guard_version") or "") != PHASE5_SFT_GUARD_VERSION
    )
    needs_sft = bool(
        rung_complete
        and (
            target_tokens not in completed_sft_before
            or sft_guard_migration
            or rescue_accepted
        )
    )

    if needs_sft:
        progress["curriculum_stage"] = "D_to_F_supervised_replay"
        if sft_guard_migration:
            causal_best_dir = bootstrap_root / "best"
            if not (causal_best_dir / "model.pt").is_file():
                raise RuntimeError(
                    "cannot migrate legacy SFT state without causal-best checkpoint"
                )
            runtime = GeneralistRuntime.from_checkpoint(
                causal_best_dir,
                device="cpu",
            )
            progress["sft_guard_migrated_from_causal_best"] = True
        replay, replay_counts = _mixed_replay_rows(
            root,
            bundle.sft_train,
            repo,
            heldout_sft=bundle.sft_validation,
        )
        progress["curriculum_replay_rows"] = replay_counts
        runtime, sft_report = _run_guarded_sft(
            runtime,
            replay,
            bundle.sft_validation,
            bootstrap_root=bootstrap_root,
            base_model_sha=base_model_sha,
            target_tokens=target_tokens,
            seed=51 + len(progress.get("completed_rungs") or []),
        )
        progress["sft_report"] = sft_report
        completed_sft = list(progress.get("sft_completed_rungs") or [])
        completed_sft.append(target_tokens)
        progress["sft_completed_rungs"] = sorted(set(int(x) for x in completed_sft))
        if bool(sft_report.get("accepted")):
            accepted_sft = list(progress.get("sft_accepted_rungs") or [])
            accepted_sft.append(target_tokens)
            progress["sft_accepted_rungs"] = sorted(set(int(x) for x in accepted_sft))
        runtime.save_checkpoint(
            candidate_dir,
            metadata={
                "role": "phase5_language_bootstrap_candidate",
                "production_qualified": False,
                "base_champion_model_sha256": base_model_sha,
                "tokens_processed": int(progress["tokens_processed"]),
                "target_tokens": target_tokens,
                "sft_replay": bool(sft_report.get("accepted")),
                "sft_guarded": True,
                "sft_rolled_back": bool(sft_report.get("rolled_back")),
            },
        )
        # Only accepted SFT changes invalidate the causal Adam moments. If every
        # SFT attempt is rejected, the restored pre-SFT checkpoint still matches
        # the persisted causal optimizer exactly.
        progress["sft_guard_version"] = PHASE5_SFT_GUARD_VERSION
        progress["sft_guard_migration_applied"] = bool(sft_guard_migration)
        if bool(sft_report.get("accepted")) or sft_guard_migration:
            _remove_optimizer_checkpoint(optimizer_path)
            progress["optimizer_reset_after_sft"] = True
        else:
            progress["optimizer_reset_after_sft"] = False

    if rung_complete:
        after = evaluate_phase5_language(runtime)
        _atomic_json(bootstrap_root / "after.json", after)

        champion_report = _grouped_validation(
            champion_runtime.model,
            champion_runtime.tokenizer,
            validation_rows(),
            device="cpu",
        )
        champion_report["parameters"] = sum(
            int(p.numel()) for p in champion_runtime.model.parameters() if p.requires_grad
        )
        champion_report["score"] = _research_score(
            champion_report,
            champion_report["parameters"],
        )
        candidate_report = _grouped_validation(
            runtime.model,
            runtime.tokenizer,
            validation_rows(),
            device="cpu",
        )
        candidate_report["parameters"] = sum(
            int(p.numel()) for p in runtime.model.parameters() if p.requires_grad
        )
        candidate_report["score"] = _research_score(
            candidate_report,
            candidate_report["parameters"],
        )
        rotating = canary_rows(max(1, cycle))
        champion_report["canary_cycle"] = max(1, cycle)
        champion_report["canary"] = _grouped_validation(
            champion_runtime.model,
            champion_runtime.tokenizer,
            rotating,
            device="cpu",
        )
        candidate_report["canary_cycle"] = max(1, cycle)
        candidate_report["canary"] = _grouped_validation(
            runtime.model,
            runtime.tokenizer,
            rotating,
            device="cpu",
        )
        eligible, eligible_reason = _research_eligible(
            champion_report,
            candidate_report,
            minimum_loss_gain=0.01,
            max_domain_regression=0.08,
        )
        degeneration_ok, degeneration_reason = degeneration_gate(before, after)
        phase5_ok, phase5_reasons = _phase5_success(before, after)

        old_domains = champion_report.get("domain_nll_per_byte") or {}
        new_domains = candidate_report.get("domain_nll_per_byte") or {}
        regressions = [
            float(new_domains[name]) - float(old)
            for name, old in old_domains.items()
            if name in new_domains
        ]
        worst_domain_regression = max(regressions, default=0.0)
        candidate_report["worst_domain_regression"] = float(worst_domain_regression)
        candidate_report["research_gate_passed"] = bool(eligible)
        candidate_report["phase5_degeneration_gate_passed"] = bool(degeneration_ok)
        candidate_report["phase5_minimum_success"] = bool(phase5_ok)
        candidate_report["phase5_diagnostics"] = after

        latest_summary = _write_latest_research(
            root,
            runtime,
            report=candidate_report,
            cycle=cycle,
            base_model_sha256=base_model_sha,
        )

        promoted = bool(eligible and degeneration_ok and phase5_ok)
        promotion_reason = (
            "existing research gates + Phase 5 degeneration/language gates passed"
            if promoted
            else "; ".join(
                [
                    f"research={eligible_reason}" if not eligible else "",
                    f"degeneration={degeneration_reason}" if not degeneration_ok else "",
                    ("phase5=" + ", ".join(phase5_reasons)) if not phase5_ok else "",
                ]
            ).strip("; ")
        )
        if promoted:
            genome = _continual_candidate_genome(candidate_genome, cycle + 1)
            _save_champion(root, genome, runtime, candidate_report)
            status["champion"] = genome.to_dict()
            status["champion_report"] = candidate_report
            status["promoted"] = True
            status["promotion_reason"] = "phase5_bootstrap:" + promotion_reason

        status["latest_research"] = latest_summary
        status["phase5_bootstrap"] = {
            "version": PHASE5_BOOTSTRAP_VERSION,
            "target_tokens": target_tokens,
            "unique_corpus_target_tokens": int(corpus_target_tokens),
            "optimization_passes_target": float(
                target_tokens / max(1, corpus_target_tokens)
            ),
            "capacity_target_parameters": progress.get("capacity_target_parameters"),
            "capacity_growth_history": progress.get("capacity_growth_history") or [],
            "tokens_processed": int(progress["tokens_processed"]),
            "steps": int(progress["steps"]),
            "before": before,
            "after": after,
            "research_gate_passed": bool(eligible),
            "research_gate_reason": eligible_reason,
            "degeneration_gate_passed": bool(degeneration_ok),
            "degeneration_gate_reason": degeneration_reason,
            "minimum_success": bool(phase5_ok),
            "minimum_success_reasons": phase5_reasons,
            "promoted": promoted,
            "promotion_reason": promotion_reason,
            "manifest_sha256": bundle.manifest.get("manifest_content_sha256"),
            "anti_collapse": {
                "version": PHASE5_ANTICOLLAPSE_VERSION,
                "rescue": progress.get("anti_collapse_rescue"),
                "rescue_tokens": int(progress.get("anti_collapse_rescue_tokens", 0) or 0),
                "last_objective": progress.get("last_objective"),
                "decoding_modified": False,
            },
        }
        _atomic_json(root / "status.json", status)

        snapshot = snapshot_airi_pc_lab(repo)
        lab_probe = run_airi_pc_lab_probe(runtime, snapshot)
        phase5_lab = {
            "version": "airi-pc-lab-phase5-v1",
            "mode": snapshot.get("mode"),
            "candidate": lab_probe,
            "denied_capabilities": snapshot.get("denied_capabilities") or [],
            "capabilities": snapshot.get("capabilities") or [],
        }
        _atomic_json(bootstrap_root / "airi-pc-lab-after.json", phase5_lab)
        existing_lab = _load_json(root / "airi-pc-lab-report.json")
        existing_lab["phase5"] = phase5_lab
        _atomic_json(root / "airi-pc-lab-report.json", existing_lab)

        completed = list(progress.get("completed_rungs") or [])
        completed.append(target_tokens)
        progress["completed_rungs"] = sorted(set(int(x) for x in completed))
        report = {
            "schema": 1,
            "version": PHASE5_BOOTSTRAP_VERSION,
            "target_tokens": target_tokens,
            "unique_corpus_target_tokens": int(corpus_target_tokens),
            "optimization_passes_target": float(
                target_tokens / max(1, corpus_target_tokens)
            ),
            "capacity_target_parameters": progress.get("capacity_target_parameters"),
            "capacity_growth_history": progress.get("capacity_growth_history") or [],
            "tokens_processed": int(progress["tokens_processed"]),
            "steps": int(progress["steps"]),
            "model_parameters": int(candidate_report["parameters"]),
            "tokenizer_version": str(runtime.tokenizer.version),
            "tokenizer_vocab_size": int(runtime.tokenizer.vocab_size),
            "before": before,
            "after": after,
            "validation": {
                "champion": champion_report,
                "candidate": candidate_report,
            },
            "research_gate_passed": bool(eligible),
            "research_gate_reason": eligible_reason,
            "degeneration_gate_passed": bool(degeneration_ok),
            "degeneration_gate_reason": degeneration_reason,
            "minimum_success": bool(phase5_ok),
            "minimum_success_reasons": phase5_reasons,
            "promoted": promoted,
            "promotion_reason": promotion_reason,
            "airi_pc_lab_after": lab_probe,
            "manifest_sha256": bundle.manifest.get("manifest_content_sha256"),
            "anti_collapse": {
                "version": PHASE5_ANTICOLLAPSE_VERSION,
                "rescue": progress.get("anti_collapse_rescue"),
                "rescue_tokens": int(progress.get("anti_collapse_rescue_tokens", 0) or 0),
                "last_objective": progress.get("last_objective"),
                "decoding_modified": False,
            },
        }
        _atomic_json(bootstrap_root / "report.json", report)

    _atomic_json(progress_path, progress)
    lineage_manifest = refresh_live_lineage_manifest(
        root,
        reason=(
            "phase5_rung_complete"
            if rung_complete
            else "phase5_segment_checkpoint"
        ),
    )
    return {
        "ok": True,
        "lineage_id": lineage_manifest.get("lineage_id"),
        "active_lineage_checkpoint": lineage_manifest.get("active_checkpoint"),
        "version": PHASE5_BOOTSTRAP_VERSION,
        "target_tokens": target_tokens,
        "tokens_processed": int(progress["tokens_processed"]),
        "valid_tokens_processed": int(progress["valid_tokens_processed"]),
        "accepted_rehabilitation_tokens": int(
            progress.get("accepted_rehabilitation_tokens", 0) or 0
        ),
        "segment_tokens_processed": int(progress["tokens_processed"]) - processed_before_segment,
        "steps": int(progress["steps"]),
        "rung_complete": bool(rung_complete),
        "early_stopped": bool(early_stopped),
        "last_train_loss": last_train_loss,
        "last_validation_loss": last_validation_loss,
        "best_validation_loss": progress.get("best_validation_loss"),
        "report_path": str(bootstrap_root / "report.json") if rung_complete else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AIRI Generalist Phase 5 language bootstrap")
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--target-tokens", type=int, default=1_000_000)
    parser.add_argument("--segment-tokens", type=int, default=250_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--historical-recovery-source")
    args = parser.parse_args(argv)
    result = run_segment(
        args.state_dir,
        args.repo_root,
        args.cache_dir,
        target_tokens=args.target_tokens,
        segment_tokens=args.segment_tokens,
        batch_size=args.batch_size,
        base_learning_rate=args.learning_rate,
        historical_recovery_source=args.historical_recovery_source,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
