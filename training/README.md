# Training a local model

This directory builds a dataset from public conventional-commit history and trains models on
it: the linear model built into jev-cc (`classifier = "local"`), and experimentally a
fine-tuned [Laya](https://github.com/NandhaKishorM/laya) checkpoint.

## Build the dataset

```sh
cargo build --release
python3 training/build_dataset.py --jev-cc target/release/jev-cc --out ~/jev-cc-dataset
```

This writes `train.jsonl` and `heldout.jsonl`, one commit per line, from the repositories in
`repos.txt`. Each line is the request jev-cc would send for that commit, plus the type its
author chose. Held-out repositories are never trained on.

The rows have the `type` label only, because few commits are labelled breaking. Pass
`--breaking` to label both questions.

To export a single repository yourself, run `jev-cc export` in it.

## Train the built-in model

```sh
pip install numpy scikit-learn
python3 training/train_linear.py --jev-cc target/release/jev-cc --data ~/jev-cc-dataset
cargo build --release
```

This rewrites `models/shared.bin`, which the next build embeds. It takes under a minute.

## Train a Laya model

On a Mac (slow; start with a sample):

```sh
pip install laya
shuf -n 3000 ~/jev-cc-dataset/train.jsonl > sample.jsonl
laya-train --data sample.jsonl --base english --max-len 1024 --shuffle-options \
           --eval-data ~/jev-cc-dataset/heldout.jsonl --out ./laya-commits
```

On Kaggle's free GPUs, `kaggle.py` uploads the dataset and runs `kaggle_train.py` as a private
notebook. You need a Kaggle account with a verified phone number (for GPU and internet
access), and an API token from your Kaggle account settings, saved to `~/.kaggle/access_token`
with `chmod 600`. Then:

```sh
pip install kaggle
export KAGGLE_USERNAME=your-username
python3 training/kaggle.py upload ~/jev-cc-dataset
python3 training/kaggle.py train --sample 2000 --epochs 1   # a short run, to time it
python3 training/kaggle.py status
python3 training/kaggle.py download --log-only ./kaggle-run   # the log and before/after report
python3 training/kaggle.py download ./kaggle-run   # the checkpoint is in kaggle-run/laya-commits
```

Use the short run's time to check a full run (`train` with no options) fits within Kaggle's
session limit and your weekly GPU quota, and lower `--epochs` or set `--sample` if it doesn't.
`download` fetches the latest run's output.

Laya's [fine-tuning guide](https://github.com/NandhaKishorM/laya/blob/main/docs/finetune.md)
covers the training options.

## Evaluate

Serve the checkpoint in place of Laya's English one, then compare with Jev on the held-out
repositories:

```sh
LAYA_EXTRA_MODELS='{"english": "./kaggle-run/laya-commits"}' LAYA_HOST=127.0.0.1 laya-serve
JEV_CC_BASE_URL=http://127.0.0.1:8000 jev-cc eval --limit 60
```
