"""Trains the local model jev-cc ships: TF-IDF per token block and logistic regression.

    python3 training/train_linear.py --jev-cc target/release/jev-cc --data ~/jev-cc-dataset

Tokens come from `jev-cc features`, so training sees exactly what jev-cc computes at commit
time. Writes `models/shared.bin` (the format `src/model.rs` reads), after checking that
jev-cc's probabilities match scikit-learn's. Needs numpy and scikit-learn.
"""

import argparse
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

BLOCKS = {"msg": 1.0, "paths": 1.0, "patch": 1.0, "meta": 1.0}
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "models" / "shared.bin"


def load(path):
    return [json.loads(line) for line in Path(path).read_text().split("\n") if line]


def jev_cc_lines(jev_cc, args, path):
    with open(path, "rb") as f:
        out = subprocess.run([jev_cc, *args], stdin=f, capture_output=True, check=True).stdout
    return [json.loads(line) for line in out.decode().split("\n") if line]


def identity(tokens):
    return tokens


class Features:
    """One TF-IDF vectoriser per block, each L2-normalised and scaled by the block's weight."""

    def __init__(self, vocabularies=None):
        self.vecs = {b: TfidfVectorizer(analyzer=identity, sublinear_tf=True, min_df=2,
                                        vocabulary=None if vocabularies is None else vocabularies[b])
                     for b in BLOCKS}

    def fit_transform(self, tokens):
        return sp.hstack([self.vecs[b].fit_transform([t[b] for t in tokens]) * w
                          for b, w in BLOCKS.items()]).tocsr()

    def transform(self, tokens):
        return sp.hstack([self.vecs[b].transform([t[b] for t in tokens]) * w
                          for b, w in BLOCKS.items()]).tocsr()

    def terms(self):
        return [list(self.vecs[b].get_feature_names_out()) for b in BLOCKS]


def fit(tokens, y, C, vocabularies=None):
    features = Features(vocabularies)
    model = LogisticRegression(C=C, max_iter=3000).fit(features.fit_transform(tokens), y)
    return features, model


def write_model(path, features, model):
    def s(text):
        data = text.encode()
        return struct.pack("<H", len(data)) + data

    out = bytearray(b"JCCM" + struct.pack("<I", 1))
    out += struct.pack("<I", len(model.classes_)) + b"".join(s(c) for c in model.classes_)
    out += struct.pack("<I", len(BLOCKS))
    for (block, weight), terms in zip(BLOCKS.items(), features.terms()):
        out += s(block) + struct.pack("<fI", weight, len(terms))
        out += b"".join(s(t) for t in terms)
        out += features.vecs[block].idf_.astype("<f4").tobytes()
    out += model.coef_.T.astype("<f4").tobytes()       # feature-major
    out += model.intercept_.astype("<f4").tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(out))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--jev-cc", required=True)
    parser.add_argument("--data", required=True, type=Path, help="directory with train.jsonl and heldout.jsonl")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--features", type=int, default=20_000, help="features kept after pruning")
    parser.add_argument("--C", type=float, default=3.0)
    args = parser.parse_args()
    data = args.data.expanduser()

    train_rows, heldout_rows = load(data / "train.jsonl"), load(data / "heldout.jsonl")
    train = jev_cc_lines(args.jev_cc, ["features"], data / "train.jsonl")
    heldout = jev_cc_lines(args.jev_cc, ["features"], data / "heldout.jsonl")
    y = [r["expected"]["type"] for r in train_rows]
    y_heldout = np.array([r["expected"]["type"] for r in heldout_rows])

    full_features, full = fit(train, y, args.C)
    accuracy = lambda f, m: (m.predict(f.transform(heldout)) == y_heldout).mean()
    print(f"full model: {sum(map(len, full_features.terms()))} features, "
          f"held-out accuracy {accuracy(full_features, full):.1%}")

    # Keep the features with the largest weight for any class, then refit on just those, so
    # each block's normalisation covers exactly the terms jev-cc will know.
    importance = np.abs(full.coef_).max(0)
    keep = set(np.argsort(-importance)[:args.features])
    vocabularies, offset = {}, 0
    for block, terms in zip(BLOCKS, full_features.terms()):
        vocabularies[block] = [t for i, t in enumerate(terms) if offset + i in keep]
        offset += len(terms)
    features, model = fit(train, y, args.C, vocabularies)
    print(f"pruned model: {sum(map(len, features.terms()))} features "
          f"({', '.join(f'{b} {len(v)}' for b, v in vocabularies.items())}), "
          f"held-out accuracy {accuracy(features, model):.1%}")

    write_model(args.out, features, model)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")

    # jev-cc must give the same probabilities as scikit-learn.
    sample = heldout_rows[:300]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl") as f:
        f.write("".join(json.dumps(r) + "\n" for r in sample))
        f.flush()
        ours = jev_cc_lines(args.jev_cc, ["features", "--model", str(args.out)], f.name)
    expected = model.predict_proba(features.transform(heldout[:300]))
    got = np.array([[p[c] for c in model.classes_] for p in ours])
    diff = np.abs(expected - got).max()
    print(f"parity: largest probability difference over {len(sample)} commits is {diff:.2e}")
    if diff > 1e-4:
        sys.exit("jev-cc and scikit-learn disagree")


if __name__ == "__main__":
    main()
