"""Offline sustained-update experiment; no live promotion or state writes."""

import argparse
import json
import os
from pathlib import Path
import tempfile

import torch
from generalist_lm.learning_efficiency_benchmark import (
    read_split_documents,
    run_sequence,
    trial_matrix,
    validate_output_path,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--state-dir", required=True)
parser.add_argument("--train", required=True)
parser.add_argument("--validation", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--trial", default="embeddings_normalized")
parser.add_argument("--sizes", default="8128,8128,16256,16256")
parser.add_argument("--seed", type=int, default=5000000)
args = parser.parse_args()
torch.set_num_threads(4)
output = Path(args.output).resolve()
validate_output_path(args.state_dir, output)
if output.exists():
    raise ValueError(
        "use a new output file; interrupted sequences restart from pinned source"
    )
train = read_split_documents(args.train, "train")
validation = read_split_documents(args.validation, "validation")
trial = next(t for t in trial_matrix() if t.name == args.trial)
events = []


def record(event):
    events.append(event)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=output.parent, delete=False
    ) as handle:
        handle.write(
            json.dumps(
                {
                    "scope": "offline candidate sequence, no live writes",
                    "partial": True,
                    "segments": events,
                },
                sort_keys=True,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
        temporary = handle.name
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "segment": event["sequence_index"],
                "size": event["trial"]["segment_tokens"],
                "accepted": event["accepted"],
                "sequence_accepted_tokens": event["sequence_accepted_tokens"],
                "repetition": event["after"]["repetition_rate"],
                "nll": event["after"]["language_nll"],
                "headroom": event["headroom_after"],
            }
        ),
        flush=True,
    )


result = run_sequence(
    args.state_dir,
    trial,
    [int(s) for s in args.sizes.split(",")],
    seed=args.seed,
    causal_documents=train,
    validation_documents=validation,
    measure_gradients=True,
    on_segment=record,
)
result["partial"] = False
with tempfile.NamedTemporaryFile(mode="w", dir=output.parent, delete=False) as handle:
    handle.write(json.dumps(result, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
    temporary = handle.name
os.replace(temporary, output)
