# norn

In Norse mythology, the [norns](https://en.wikipedia.org/wiki/Norns) decide the fate of
everything. This one just decides what kind of change your commit is.

A fast git hook that classifies your staged diff and prefixes your commit message with
[Conventional Commits](https://www.conventionalcommits.org/) syntax. It runs on your machine,
or can use [TypeSafe's Jev](https://docs.typesafe.ai/api) model.

```console
$ git commit -m "handle empty config files"
$ git log -1 --format=%s
fix(config): handle empty config files
```

## How it works

- **You** write the description.
- **A classifier** picks the `type` (`feat`, `fix`, `docs`, …). By default it's a small model
  built into norn, trained on public conventional-commit history. With Jev instead, the
  type is more accurate, and Jev also decides whether the change is breaking (`!`).
- **Local rules** pick the `scope` from the paths you changed, and skip the classifier when
  the answer is obvious (for example, a docs-only diff).

If the classifier isn't confident enough, or Jev can't be reached, your message is left
unchanged.

## Install

macOS and Linux:

```sh
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/jamiesanson/norn/releases/latest/download/norn-installer.sh | sh
```

Windows:

```powershell
powershell -ExecutionPolicy Bypass -c "irm https://github.com/jamiesanson/norn/releases/latest/download/norn-installer.ps1 | iex"
```

Prebuilt archives for each platform are also on the
[releases page](https://github.com/jamiesanson/norn/releases). To build from
source, run `cargo install --path .`.

Then, in each repository:

```sh
norn install
```

To use Jev, set `classifier = "jev"` (see [Configuration](#configuration)) and run
`norn jev login` once to save your TypeSafe API key.

With `git commit -m`, the prefix goes in front of your message. With the editor, the first
line is pre-filled (e.g. `fix(config): `) for you to finish.

`install` won't overwrite existing `prepare-commit-msg` or `commit-msg` hooks. Add
`norn <hook> "$@"` to them yourself.

Try it without committing:

```console
$ git add -p
$ norn classify "handle empty config files"
fix(config): handle empty config files
  source: Model, confidence: 97%, took 37ms
```

## Configuration

Settings are read from these places, each overriding the one before:

1. `~/.config/norn/config.toml`, for your own defaults
2. `.norn.toml` at the repository root, for settings shared with your team
3. environment variables

```toml
# .norn.toml
min_confidence = 0.75
exclude = ["secrets/**", "**/*.pem"]
types = ["feat", "fix", "docs", "refactor", "test", "chore"]

[scopes]
"packages/web" = "web"
"packages/web/admin" = "admin"
"packages/api" = "api"
```

| Setting | Environment variable | Default | |
|---|---|---|---|
| `timeout_ms` | `NORN_TIMEOUT_MS` | `1000` | Request timeout |
| `deadline_ms` | `NORN_DEADLINE_MS` | `2000` | Time limit for the whole hook |
| `min_confidence` | `NORN_MIN_CONFIDENCE` | `0.6` | Minimum confidence to apply a type |
| `breaking_threshold` | `NORN_BREAKING_THRESHOLD` | `0.85` | Probability needed to add `!` |
| `disable` | `NORN_DISABLE` | `false` | |
| `classifier` | `NORN_CLASSIFIER` | `local` | `local` for the built-in model, or `jev`. Global config only |
| `exclude` | | `[]` | Paths never sent to Jev |
| `types` | | all built-in types | Types norn may choose from |
| `scopes` | | | Path prefixes mapped to scopes |
| `base_url` | `NORN_BASE_URL` | `https://api.typesafe.ai` | Global config only |

`exclude` patterns are
[git glob pathspecs](https://git-scm.com/docs/gitglossary#Documentation/gitglossary.txt-aiddefpathspecapathspec),
relative to the repository root. `*` doesn't match across directories, so use `**/*.pem` to
match at any depth. A project's patterns are added to your global ones.

`types` lists built-in types by name. For your own types, use a table of names to
descriptions. Jev uses the descriptions to tell types apart, and an empty description keeps
the built-in one:

```toml
[types]
feat = ""
fix = ""
deps = "Updates or adds third-party dependencies"
```

Type names must be lowercase letters. A project's `types` replace your global ones.

`scopes` entries from both files are combined, and the project's entry wins when both set
the same path.

`base_url` can't be set in `.norn.toml`, so a repository you clone can't send your diffs
and API key somewhere else. Nor can `classifier`.

`norn install` also fits the built-in model to the repository's history, if it has at least
20 commits with conventional prefixes, so it learns how the project labels its changes,
including its own `types` and scopes. Run `norn learn` to update it. The built-in model
never marks a change breaking.

Run `norn config` to see the settings in effect and where each one comes from.
`TYPESAFE_API_KEY` overrides the key saved by `norn jev login`.

### Laya

[Laya](https://github.com/NandhaKishorM/laya) runs Jev-style models on your own machine and
serves the same API. Start its server, then point norn at it in your global config:

```sh
pip install "laya[serve]"
LAYA_HOST=127.0.0.1 laya-serve
```

```toml
# ~/.config/norn/config.toml
classifier = "jev"
base_url = "http://127.0.0.1:8000"
```

No API key is needed unless you set `LAYA_API_KEY`, in which case save it with
`norn jev login`. On a slow machine, raise `timeout_ms` and `deadline_ms`.

Laya's published checkpoints classify commits much less accurately than the built-in model,
and are often confidently wrong. Run `norn eval` in your repository before relying on one.

## Classification

1. **Skip** merges, squashes, amends, `fixup!`/`squash!` commits, and messages that already
   start with `type:` or `type(scope):`.
2. **Local rules.** If every changed file is documentation, a test or CI config, the type is
   `docs`, `test` or `ci`, and no request is made. This only applies when that type is in
   `types`.
3. **Classifier.** Otherwise, the classifier reads the changed file list, a trimmed patch
   (lockfiles and binaries omitted, about 24 KB at most) and your message. Jev is asked two
   questions:
   - `type`: a choice between your `types`, or by default `feat`, `fix`, `docs`, `style`,
     `refactor`, `perf`, `test`, `build`, `ci`, `chore` and `revert`
   - `breaking`: yes or no
4. **Scope.** With `scopes` configured, each changed file takes the scope of its longest
   matching path, so `packages/web/admin/users.ts` gets `(admin)` in the example above.
   Otherwise, once norn has learned from the repository's history, it uses the scope that
   history files those paths under, or none if the project doesn't use scopes. Before that,
   it's the directory every changed file shares, skipping container directories such as
   `src/` and `packages/`, so changes only in `src/config/` get `(config)`. If the files
   disagree, there's no scope. Lockfiles are ignored, because they change along with
   whatever they belong to.

## Privacy

With the built-in model, nothing leaves your machine.

With `classifier = "jev"`, when the local rules can't decide, norn sends the following to
TypeSafe (`api.typesafe.ai`, or your `base_url`):

- the paths of your staged files, except those matched by `exclude`
- the patch for those files, trimmed to about 24 KB, with lockfile and binary contents left
  out
- your commit message, if you've written one

The repository name, remote, branch and author are not sent. Diffs that only touch docs,
tests or CI config aren't sent anywhere.

Any secrets in your staged changes are sent along with the patch. Keep sensitive paths out
with `exclude`, catch secrets elsewhere with a secret scanner in a `pre-commit` hook, and
don't use Jev in repositories whose code can't leave your machine.

## Development

```sh
cargo test
cargo build --release
./target/release/norn install   # dogfood: classify this repo's own commits
```

To measure a change to classification, run `norn eval` in a repository whose history
already uses conventional prefixes. It replays the last 50 labelled commits (`--limit`)
through the classifier and reports accuracy and the most common mistakes. `--message` also
sends each commit's description, as `git commit -m` would, and `--out FILE` writes every
answer with the classifier's probabilities as JSON lines. Run `norn learn --skip 50` first,
so the model is scored on commits it hasn't learned from. With Jev, set `types` in that
repository's `.norn.toml` to match its conventions; many projects label dependency and CI
updates `chore`, which the default types would call `build` and `ci`.

`norn export` writes the same commits as training data for a local model; see
[`training/`](training/README.md).

### Releasing

Releases are built by [dist](https://opensource.axo.dev/cargo-dist/) in
`.github/workflows/release.yml`. Bump `version` in `Cargo.toml`, then push a matching tag:

```sh
git tag v0.1.0 && git push origin v0.1.0
```

This builds macOS, Linux (glibc and static musl) and Windows binaries, generates the
installers, and publishes a GitHub Release. After changing `dist-workspace.toml`, run
`dist generate` to regenerate the workflow.

## Status

Early development.
