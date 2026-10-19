"""Builds a training set for a commit-type model from public conventional-commit history.

Clones each repository in `repos.txt` (shallow, no checkout), runs `norn export` over it,
filters the rows and writes `train.jsonl` and `heldout.jsonl` in the format `laya-train`
reads. Repositories under `[heldout]` are kept out of training entirely, so a fine-tune can
be compared with Jev on repositories it has never seen.

    python3 training/build_dataset.py --norn target/release/norn --out ~/norn-dataset

Standard library only. Clones are deleted once exported, so peak disk use stays small.
Each repository's rows are kept in `repos/` under the output directory, and a rerun reuses
them; delete that directory after changing `--depth`, `--per-repo` or `--breaking`.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Commits a bot made. Bots don't run commit hooks, so they aren't what the model will see.
BOT_AUTHOR = re.compile(r"\[bot\]|dependabot|renovate|github-actions|release-please", re.I)
# Dependency bumps and releases have templated subjects and one obvious diff. A few teach the
# model what they look like; hundreds would drown out everything else.
DEPS_SUBJECT = re.compile(r"^\w+\(deps(-dev)?\)|: (bump|update) (dependency|dependencies|\S+ from )", re.I)
RELEASE_SUBJECT = re.compile(r"^\w+(\(release\))?!?: (release|v?\d+\.\d+\.\d+)|^\w+\(release\)", re.I)
# Squash merges append the pull request number, which nobody types at commit time.
PR_SUFFIX = re.compile(r"\s*\(#\d+\)\s*$")


def read_repos(path):
    """`{"train": [...], "heldout": [...]}` from a file of `owner/name` lines under section headers."""
    sections, current = {"train": [], "heldout": []}, "train"
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line in ("[train]", "[heldout]"):
            current = line[1:-1]
        elif line:
            sections[current].append(line)
    return sections


def git(repo_dir, *args):
    return subprocess.run(["git", "-C", str(repo_dir), *args], capture_output=True, text=True,
                          check=True).stdout


def read_jsonl(path):
    # Not `splitlines()`: it also splits on characters JSON leaves unescaped, such as U+2028.
    return [json.loads(line) for line in path.read_text().split("\n") if line]


def export_repo(repo, args, work, empty_config):
    """Rows for one repository, and a line describing what was kept."""
    cached = args.out / "repos" / (repo.replace("/", "__") + ".jsonl")
    if cached.is_file():
        rows = read_jsonl(cached)
        return rows, f"{repo}: {len(rows)} (cached)"
    rows, note = filter_rows(repo, args, work, empty_config)
    if rows:
        cached.parent.mkdir(exist_ok=True)
        cached.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    return rows, note


def filter_rows(repo, args, work, empty_config):
    clone = work / repo.replace("/", "__")
    try:
        subprocess.run(["git", "clone", "--quiet", "--depth", str(args.depth), "--single-branch",
                        "--no-tags", "--no-checkout", f"https://github.com/{repo}", str(clone)],
                       capture_output=True, text=True, check=True)
        bots = set()
        for line in git(clone, "log", "--format=%H%x00%an%x00%ae").splitlines():
            sha, name, email = line.split("\0")
            if BOT_AUTHOR.search(name) or BOT_AUTHOR.search(email):
                bots.add(sha)

        out = clone.parent / (clone.name + ".jsonl")
        env = dict(os.environ, XDG_CONFIG_HOME=str(empty_config))
        subprocess.run([args.norn, "export", "--message", "--out", str(out)], cwd=clone, env=env,
                       capture_output=True, text=True, check=True)
        exported = read_jsonl(out)
        out.unlink()
    except subprocess.CalledProcessError as e:
        return [], f"{repo}: failed: {(e.stderr or '').strip()[:200]}"
    finally:
        shutil.rmtree(clone, ignore_errors=True)

    rows, dropped = [], Counter()
    deps_cap = release_cap = max(1, args.per_repo // 20)
    for row in exported:
        if len(rows) == args.per_repo:
            break
        subject = row["subject"]
        if row["sha"] in bots:
            dropped["bot"] += 1
            continue
        if DEPS_SUBJECT.search(subject):
            if deps_cap == 0:
                dropped["deps"] += 1
                continue
            deps_cap -= 1
        elif RELEASE_SUBJECT.search(subject):
            if release_cap == 0:
                dropped["release"] += 1
                continue
            release_cap -= 1

        state = row["state"]
        message = PR_SUFFIX.sub("", state.get("message", ""))
        # Half the rows go without a message, like commits written in the editor. The sha
        # decides, so a rebuild makes the same choice.
        if message and int(hashlib.sha1(row["sha"].encode()).hexdigest(), 16) % 2 == 0:
            state["message"] = message
        else:
            state.pop("message", None)
        if not args.breaking:
            del row["expected"]["breaking"]
        rows.append({"repo": repo, **row})

    note = ", ".join(f"{n} {why}" for why, n in dropped.items())
    return rows, f"{repo}: {len(rows)} of {len(exported)} exported" + (f" (dropped {note})" if note else "")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--norn", required=True, help="path to the norn binary")
    parser.add_argument("--out", required=True, type=Path, help="directory to write the dataset to")
    parser.add_argument("--repos", type=Path, default=Path(__file__).with_name("repos.txt"))
    parser.add_argument("--depth", type=int, default=800, help="commits to clone per repository")
    parser.add_argument("--per-repo", type=int, default=300, help="most rows kept per repository")
    parser.add_argument("--breaking", action="store_true",
                        help="also label the breaking question (few commits are labelled breaking)")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    args.norn = str(Path(args.norn).resolve())

    repos = read_repos(args.repos)
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="norn-dataset-") as tmp:
        work, empty_config = Path(tmp), Path(tmp) / "config"
        empty_config.mkdir()
        for split, names in repos.items():
            path = args.out / f"{split}.jsonl"
            types, total = Counter(), 0
            with open(path, "w") as f, ThreadPoolExecutor(args.jobs) as pool:
                for rows, note in pool.map(lambda r: export_repo(r, args, work, empty_config), names):
                    print(note, file=sys.stderr, flush=True)
                    for row in rows:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        types[row["expected"]["type"]] += 1
                    total += len(rows)
            print(f"{path}: {total} rows  {dict(types.most_common())}", file=sys.stderr)


if __name__ == "__main__":
    main()
