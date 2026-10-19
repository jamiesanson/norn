# Training the built-in model

This directory builds a dataset from public conventional-commit history and trains the
linear model built into norn.

## Build the dataset

```sh
cargo build --release
python3 training/build_dataset.py --norn target/release/norn --out ~/norn-dataset
```

This writes `train.jsonl` and `heldout.jsonl`, one commit per line, from the repositories in
`repos.txt`. Each line is the request norn would send for that commit, plus the type its
author chose. Held-out repositories are never trained on.

The rows have the `type` label only, because few commits are labelled breaking. Pass
`--breaking` to label both questions.

To export a single repository yourself, run `norn export` in it.

## Train

```sh
pip install numpy scikit-learn
python3 training/train_linear.py --norn target/release/norn --data ~/norn-dataset
cargo build --release
```

This rewrites `models/shared.bin`, which the next build embeds, and prints its accuracy on
the held-out repositories. It takes under a minute.

A fine-tuned 421M-parameter [Laya](https://github.com/NandhaKishorM/laya) model was tried on
the same data and scored the same on the held-out set (66%), so norn uses the linear model.

## Evaluate

In a repository whose history uses conventional prefixes, learn from all but the newest
commits, then score norn on those:

```sh
norn learn --skip 60
norn eval --limit 60
```
