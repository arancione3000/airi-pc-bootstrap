from __future__ import annotations

import copy
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

from .data import encode_text, load_records
from .engine import decision_from_probability, load_champion_for_prediction


def edge_acceptance(
    *,
    float_bytes: int,
    int8_bytes: int,
    float_latency_ms: float,
    int8_latency_ms: float,
    mean_probability_delta: float,
    label_agreement: float,
    max_mean_delta: float = 0.05,
    min_label_agreement: float = 0.95,
    min_size_gain: float = 0.15,
    min_latency_gain: float = 0.10,
) -> dict[str, Any]:
    size_gain = 1.0 - (int8_bytes / max(1, float_bytes))
    latency_gain = 1.0 - (int8_latency_ms / max(0.0001, float_latency_ms))
    quality_ok = mean_probability_delta <= max_mean_delta and label_agreement >= min_label_agreement
    efficiency_ok = size_gain >= min_size_gain or latency_gain >= min_latency_gain
    return {
        "accepted": bool(quality_ok and efficiency_ok),
        "quality_ok": bool(quality_ok),
        "efficiency_ok": bool(efficiency_ok),
        "size_gain": size_gain,
        "latency_gain": latency_gain,
        "limits": {
            "max_mean_delta": max_mean_delta,
            "min_label_agreement": min_label_agreement,
            "min_size_gain": min_size_gain,
            "min_latency_gain": min_latency_gain,
        },
    }


def _dynamic_quantize(model):
    import torch
    quantize_dynamic = torch.ao.quantization.quantize_dynamic
    return quantize_dynamic(copy.deepcopy(model).cpu().eval(), {torch.nn.Linear, torch.nn.GRU}, dtype=torch.qint8)


def _probability(model, genome, text: str, vocab_size: int = 8192) -> float:
    import torch
    ids, mask = encode_text(text, genome.max_len, vocab_size)
    with torch.inference_mode():
        logits = model(
            torch.tensor([ids], dtype=torch.long),
            torch.tensor([mask], dtype=torch.bool),
        )
        return float(torch.softmax(logits, dim=-1)[0, 1])


def _latency(model, genome, text: str, repeats: int = 20, vocab_size: int = 8192) -> float:
    import torch
    ids, mask = encode_text(text, genome.max_len, vocab_size)
    ids_t = torch.tensor([ids], dtype=torch.long)
    mask_t = torch.tensor([mask], dtype=torch.bool)
    model.eval()
    with torch.inference_mode():
        for _ in range(3):
            model(ids_t, mask_t)
        times = []
        for _ in range(max(5, int(repeats))):
            start = time.perf_counter()
            model(ids_t, mask_t)
            times.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(times)


def _state_bytes(model) -> int:
    import torch
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        torch.save(model.state_dict(), path)
        return path.stat().st_size
    finally:
        path.unlink(missing_ok=True)


def quantize_champion(
    state_dir: Path,
    *,
    max_samples: int = 64,
    max_mean_delta: float = 0.05,
    min_label_agreement: float = 0.95,
) -> dict[str, Any]:
    import torch
    state_dir = Path(state_dir)
    genome, float_model = load_champion_for_prediction(state_dir)
    float_model = float_model.cpu().eval()
    records = load_records(state_dir / "data" / "verified.jsonl")
    if not records:
        return {"ok": False, "error": "no_verified_records_for_edge_validation"}
    samples = [row["text"] for row in records[-max(4, min(int(max_samples), len(records))):]]
    try:
        int8_model = _dynamic_quantize(float_model)
    except Exception as exc:
        return {"ok": False, "error": "dynamic_quantization_failed", "detail": repr(exc)}

    float_probs = [_probability(float_model, genome, text) for text in samples]
    int8_probs = [_probability(int8_model, genome, text) for text in samples]
    deltas = [abs(a - b) for a, b in zip(float_probs, int8_probs)]
    agreements = [
        int((a >= 0.5) == (b >= 0.5))
        for a, b in zip(float_probs, int8_probs)
    ]

    sample_text = samples[0]
    float_bytes = _state_bytes(float_model)
    int8_bytes = _state_bytes(int8_model)
    float_latency = _latency(float_model, genome, sample_text)
    int8_latency = _latency(int8_model, genome, sample_text)
    gate = edge_acceptance(
        float_bytes=float_bytes,
        int8_bytes=int8_bytes,
        float_latency_ms=float_latency,
        int8_latency_ms=int8_latency,
        mean_probability_delta=sum(deltas) / max(1, len(deltas)),
        label_agreement=sum(agreements) / max(1, len(agreements)),
        max_mean_delta=max_mean_delta,
        min_label_agreement=min_label_agreement,
    )
    metrics = {
        "samples": len(samples),
        "float_bytes": float_bytes,
        "int8_bytes": int8_bytes,
        "float_latency_ms": float_latency,
        "int8_latency_ms": int8_latency,
        "mean_probability_delta": sum(deltas) / max(1, len(deltas)),
        "max_probability_delta": max(deltas) if deltas else 0.0,
        "label_agreement": sum(agreements) / max(1, len(agreements)),
        **gate,
    }
    if gate["accepted"]:
        edge_dir = state_dir / "edge"
        edge_dir.mkdir(parents=True, exist_ok=True)
        torch.save(int8_model.state_dict(), edge_dir / "model-int8.pt")
        (edge_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "format": "airi-pc-dynamic-int8-v1",
                    "created_at": time.time(),
                    "genome_id": genome.genome_id,
                    "metrics": metrics,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return {"ok": True, "saved": gate["accepted"], "metrics": metrics}


def load_edge_model(state_dir: Path):
    import torch
    state_dir = Path(state_dir)
    edge_path = state_dir / "edge" / "model-int8.pt"
    meta_path = state_dir / "edge" / "metadata.json"
    if not edge_path.exists() or not meta_path.exists():
        raise FileNotFoundError("no accepted INT8 edge model; run edge-quantize first")
    genome, float_model = load_champion_for_prediction(state_dir)
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError("invalid INT8 edge metadata") from exc
    if metadata.get("genome_id") != genome.genome_id:
        raise RuntimeError("INT8 edge model is stale for the current champion")
    qmodel = _dynamic_quantize(float_model)
    qmodel.load_state_dict(torch.load(edge_path, map_location="cpu", weights_only=True))
    qmodel.eval()
    return genome, qmodel


def predict_edge_text(state_dir: Path, text: str) -> dict[str, Any]:
    state_dir = Path(state_dir)
    genome, model = load_edge_model(state_dir)
    p_real = _probability(model, genome, text)
    provenance = {}
    try:
        provenance = json.loads((state_dir / "champion" / "provenance.json").read_text(encoding="utf-8"))
    except Exception:
        pass
    threshold = float(provenance.get("abstain_threshold", 0.65))
    label, confidence = decision_from_probability(p_real, threshold)
    return {
        "label": label,
        "binary_preference": "likely_real" if p_real >= 0.5 else "likely_fake",
        "real_probability": p_real,
        "fake_probability": 1.0 - p_real,
        "confidence": confidence,
        "abstain_threshold": threshold,
        "genome_id": genome.genome_id,
        "edge_format": "dynamic_int8",
    }
