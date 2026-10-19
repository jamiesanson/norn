"""Runs the fine-tune on Kaggle's free GPUs, using the Kaggle CLI (`pip install kaggle`).

    python3 training/kaggle.py upload ~/jev-cc-dataset   # once, and after rebuilding the data
    python3 training/kaggle.py train --sample 2000 --epochs 1   # a short run to time things
    python3 training/kaggle.py train                     # the full run
    python3 training/kaggle.py status
    python3 training/kaggle.py download ./kaggle-run

The dataset and the notebook are private to your Kaggle account.
"""

import argparse
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DATASET = "jev-cc-commits"
KERNEL = "jev-cc-laya-train"
TRAIN_SCRIPT = Path(__file__).with_name("kaggle_train.py")


def username(args):
    if args.user:
        return args.user
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    config = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
    try:
        return json.loads(config.read_text())["username"]
    except (OSError, KeyError, ValueError):
        sys.exit("couldn't find your Kaggle username; pass --user or set KAGGLE_USERNAME")


def kaggle(*args, check=True):
    try:
        return subprocess.run(["kaggle", *args], check=check)
    except FileNotFoundError:
        sys.exit("the Kaggle CLI isn't installed; run `pip install kaggle`")


def upload(args):
    """Uploads the dataset gzipped, from a staging directory so nothing else in `data` goes."""
    data = Path(args.data).expanduser()
    slug = f"{username(args)}/{DATASET}"
    with tempfile.TemporaryDirectory() as staging:
        for name in ("train.jsonl", "heldout.jsonl"):
            if not (data / name).is_file():
                sys.exit(f"{data / name} not found; build the dataset first")
            with open(data / name, "rb") as f, gzip.open(Path(staging, name + ".gz"), "wb") as out:
                shutil.copyfileobj(f, out)
        Path(staging, "dataset-metadata.json").write_text(json.dumps(
            {"title": DATASET, "id": slug, "licenses": [{"name": "other"}]}, indent=2))
        exists = subprocess.run(["kaggle", "datasets", "status", slug],
                                capture_output=True).returncode == 0
        if exists:
            kaggle("datasets", "version", "-p", staging, "-m", "Rebuilt dataset")
        else:
            kaggle("datasets", "create", "-p", staging)


def train(args):
    user = username(args)
    settings = {"laya_version": args.laya_version, "base": args.base, "epochs": args.epochs,
                "max_len": args.max_len, "sample": args.sample, "seed": args.seed}
    script = re.sub(r"^SETTINGS = \{.*?^\}", "SETTINGS = " + json.dumps(settings, indent=4),
                    TRAIN_SCRIPT.read_text(), count=1, flags=re.S | re.M)
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "train.py").write_text(script)
        Path(tmp, "kernel-metadata.json").write_text(json.dumps({
            "id": f"{user}/{KERNEL}",
            "title": KERNEL,
            "code_file": "train.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [f"{user}/{DATASET}"],
        }, indent=2))
        kaggle("kernels", "push", "-p", tmp)
    print(f"running at https://www.kaggle.com/code/{user}/{KERNEL}")


def status(args):
    kaggle("kernels", "status", f"{username(args)}/{KERNEL}")


def download(args):
    pattern = ["--file-pattern", r"train\.log$|train_report\.json$"] if args.log_only else []
    kaggle("kernels", "output", f"{username(args)}/{KERNEL}", "-p", args.out, *pattern)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--user", help="Kaggle username (default: $KAGGLE_USERNAME, or from "
                        "a legacy ~/.kaggle/kaggle.json)")
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("upload", help="upload or update the dataset")
    p.add_argument("data", help="directory holding train.jsonl and heldout.jsonl")
    p.set_defaults(run=upload)

    p = commands.add_parser("train", help="start a fine-tuning run")
    p.add_argument("--base", default="english", help="Laya checkpoint to start from")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--max-len", type=int, default=1024, help="tokens of each commit the model reads")
    p.add_argument("--sample", type=int, default=0, help="train on this many rows (default: all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--laya-version", default="0.4.11")
    p.set_defaults(run=train)

    p = commands.add_parser("status", help="show the run's status")
    p.set_defaults(run=status)

    p = commands.add_parser("download", help="download the run's checkpoint and log")
    p.add_argument("out", help="directory to download into")
    p.add_argument("--log-only", action="store_true",
                   help="only the training log and report, not the checkpoint")
    p.set_defaults(run=download)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
