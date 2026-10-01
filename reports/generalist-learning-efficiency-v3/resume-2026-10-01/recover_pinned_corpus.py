import json, hashlib, argparse, urllib.request
from pathlib import Path
from dataclasses import asdict
from generalist_lm.bootstrap_data import (
    SOURCES,
    _parse_tatoeba,
    _parse_oasst,
    _oasst_conversations,
    _split_documents,
)
from generalist_lm.pretraining import CorpusDocument

parser = argparse.ArgumentParser(
    description="Recover exact pinned English/OASST sources with original holdout split"
)
parser.add_argument("--state-dir", required=True)
parser.add_argument("--output-dir", required=True)
args = parser.parse_args()
state = Path(args.state_dir) / "bootstrap-data"
output = Path(args.output_dir)
output.mkdir(parents=True, exist_ok=True)
from generalist_lm.learning_efficiency_benchmark import validate_output_path

validate_output_path(args.state_dir, output)
manifest = json.loads((state / "manifest.json").read_text())
pins = {s["id"]: s for s in manifest["sources"]}
docs = []
s = next(s for s in SOURCES if s["id"] == "tatoeba-en-cc0")
raw = Path(__file__).with_name("tatoeba-en-cc0.bz2").read_bytes()
if hashlib.sha256(raw).hexdigest() != pins[s["id"]]["sha256"]:
    raise ValueError("English source pin mismatch")
for sid, text, _ in _parse_tatoeba(s, raw):
    docs.append(
        CorpusDocument(
            source=f"{s['id']}:{sid}:en",
            text=text,
            sha256=hashlib.sha256(text.encode()).hexdigest(),
            bytes=len(text.encode()),
            domain="language",
        )
    )
if len(docs) != pins[s["id"]]["imported_documents"]:
    raise ValueError("English source count mismatch")
oasst_path = output / "oasst1-human.gz"
if not oasst_path.exists():
    raw = urllib.request.urlopen(pins["oasst1-human"]["url"], timeout=60).read(60000000)
    if hashlib.sha256(raw).hexdigest() != pins["oasst1-human"]["sha256"]:
        raise ValueError("OASST source pin mismatch")
    oasst_path.write_bytes(raw)
raw = oasst_path.read_bytes()
if hashlib.sha256(raw).hexdigest() != pins["oasst1-human"]["sha256"]:
    raise ValueError("OASST source pin mismatch")
conv = _oasst_conversations(_parse_oasst(raw))
if len(conv) != pins["oasst1-human"]["imported_documents"]:
    raise ValueError("OASST source count mismatch")
for sid, messages, lang in conv:
    text = messages[-1]["content"]
    docs.append(
        CorpusDocument(
            source=f"oasst1-human:{sid}:{lang}",
            text=text,
            sha256=hashlib.sha256(text.encode()).hexdigest(),
            bytes=len(text.encode()),
            domain="dialogue",
        )
    )
train, val = _split_documents(docs)
if not {d.sha256 for d in train}.isdisjoint(d.sha256 for d in val):
    raise ValueError("Holdout contamination")
for name, split, rows in [
    ("pinned-train", "train", train),
    ("pinned-validation", "validation", val),
]:
    (output / (name + ".jsonl")).write_text(
        "".join(
            json.dumps({**asdict(d), "split": split}, ensure_ascii=False) + "\n"
            for d in rows
        )
    )
print(
    json.dumps(
        {
            "training_documents": len(train),
            "validation_documents": len(val),
            "scope": "exact pinned English Tatoeba and human OASST1; Italian rolling source hash changed so excluded; FineWeb caches unavailable",
            "source_shas": {
                k: pins[k]["sha256"] for k in ["tatoeba-en-cc0", "oasst1-human"]
            },
        }
    )
)
