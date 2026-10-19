"""Fine-tunes Laya on the jev-cc commit dataset. Runs on Kaggle; `training/kaggle.py train`
pushes it with the settings below filled in.

Writes the checkpoint to /kaggle/working/laya-commits and the run's log to
/kaggle/working/train.log, both downloaded by `training/kaggle.py download`.
"""

import glob
import gzip
import os
import random
import shutil
import subprocess
import sys

# Replaced by `training/kaggle.py train`.
SETTINGS = {
    "laya_version": "0.4.11",
    "base": "english",
    "epochs": 4,
    "max_len": 1024,
    "sample": 0,
    "seed": 0,
}

WORK = "/kaggle/working"
OUT = os.path.join(WORK, "laya-commits")
LOG = os.path.join(WORK, "train.log")


def find(name):
    """`name` from the attached dataset, decompressed if it was uploaded gzipped."""
    for candidate in (name, name + ".gz"):
        matches = glob.glob(f"/kaggle/input/**/{candidate}", recursive=True)
        if matches and candidate == name:
            return matches[0]
        if matches:
            out = f"/tmp/{name}"
            with gzip.open(matches[0], "rb") as f, open(out, "wb") as dest:
                shutil.copyfileobj(f, dest)
            return out
    sys.exit(f"{name} not found under /kaggle/input; is the dataset attached?")


def sample(path, n, seed):
    """`n` rows of `path`, chosen with `seed`, or `path` itself when n is 0."""
    if not n:
        return path
    with open(path) as f:
        rows = f.readlines()
    random.Random(seed).shuffle(rows)
    out = f"/tmp/train-{n}.jsonl"
    with open(out, "w") as f:
        f.writelines(rows[:n])
    return out


def run(args, log):
    """Runs a command, echoing its output to the notebook and to `log`."""
    print("$", " ".join(args), flush=True)
    log.write("$ " + " ".join(args) + "\n")
    with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as p:
        for line in p.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
    if p.returncode:
        sys.exit(f"command failed with exit code {p.returncode}")


def main():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           f"laya=={SETTINGS['laya_version']}"])
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])

    train = sample(find("train.jsonl"), SETTINGS["sample"], SETTINGS["seed"])
    heldout = find("heldout.jsonl")
    common = ["laya-train", "--data", train, "--base", SETTINGS["base"],
              "--max-len", str(SETTINGS["max_len"]), "--shuffle-options",
              "--seed", str(SETTINGS["seed"])]

    with open(LOG, "w") as log:
        log.write(f"settings: {SETTINGS}\n")
        run(common + ["--dry-run"], log)
        run(common + ["--epochs", str(SETTINGS["epochs"]), "--eval-data", heldout, "--out", OUT], log)

    # The final checkpoint is complete, so the per-epoch copy would only double the download.
    shutil.rmtree(os.path.join(OUT, "checkpoint_latest"), ignore_errors=True)
    print(f"done: {OUT}")


if __name__ == "__main__":
    main()
