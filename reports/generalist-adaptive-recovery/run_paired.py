"""Reproduce diagnostic trials on a pinned read-only state; never promote it."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import torch
from generalist_lm.learning_efficiency_benchmark import run_trial, trial_matrix, validate_output_path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--state-dir', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
root, out = Path(args.state_dir).resolve(), Path(args.output).resolve()
validate_output_path(root, out)
if out.exists():
    raise ValueError('Refuse to overwrite diagnostic evidence')
base = next(t for t in trial_matrix(16256) if t.name == 'wide128_causal20_ar')
torch.set_num_threads(2)
for seed in (7100000, 7100001, 7100002):
    for multiplier in (1.0, 0.25, 0.0625):
        trial = replace(base, name=f'wide128_lr_multiplier_{multiplier}',
                        lr_multiplier=multiplier)
        result = run_trial(root, trial, seed=seed)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('a') as handle:
            handle.write(json.dumps(result) + '\n')
        print(seed, multiplier, result['accepted'], flush=True)
