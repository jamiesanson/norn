# norn

In Norse mythology, the [norns](https://en.wikipedia.org/wiki/Norns) decide the fate of
everything. This one just decides what kind of change your commit is.

norn is a git hook that prefixes your commit messages with
[Conventional Commits](https://www.conventionalcommits.org/) types and scopes, using a small
model that runs on your machine.

```console
$ git commit -m "handle empty config files"
$ git log -1 --format=%s
fix(config): handle empty config files
```

## Install

macOS and Linux:

```sh
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/jamiesanson/norn/releases/latest/download/norn-installer.sh | sh
```

Windows:

```powershell
powershell -ExecutionPolicy Bypass -c "irm https://github.com/jamiesanson/norn/releases/latest/download/norn-installer.ps1 | iex"
```

Or download an archive from the [releases page](https://github.com/jamiesanson/norn/releases),
or build from source with `cargo install --path .`. Then, in each repository:

```sh
norn install
```

`install` won't overwrite hooks it didn't write. If you have your own `prepare-commit-msg`
or `commit-msg`, add `norn <hook> "$@"` to it yourself; your own `post-commit` is left
alone. To try norn without committing, stage some changes and
run `norn classify "your message"`.

## How it works

- **You** write the description. With the editor, the first line is pre-filled (for example
  `fix(config): `) for you to finish.
- **norn** picks the type and scope. `norn install` fits its model to the repository's
  history, if there are at least 20 commits with conventional prefixes, so it learns the
  project's own types and scopes, including whether it uses scopes at all. Run `norn learn`
  to refresh it.
- Changes that only touch docs, tests or CI config get `docs`, `test` or `ci` directly.
- If norn isn't confident, your message is left unchanged. In the editor, it suggests its
  two best guesses in a comment.
- Commits norn labelled count for less when it learns, unless you changed its label.

## Configuration

Settings come from `~/.config/norn/config.toml`, then `.norn.toml` at the repository root
(for your team), then environment variables, each overriding the one before.

```toml
# .norn.toml
min_confidence = 0.75
types = ["feat", "fix", "docs", "refactor", "test", "chore"]

[scopes]
"packages/web" = "web"
"packages/api" = "api"
```

| Setting | Environment variable | Default | |
|---|---|---|---|
| `min_confidence` | `NORN_MIN_CONFIDENCE` | `0.6` | Minimum confidence to apply a type |
| `types` | | the built-in types | Types norn may choose from |
| `scopes` | | | Path prefixes mapped to scopes; these override learned scopes |
| `exclude` | | `[]` | [Pathspecs](https://git-scm.com/docs/gitglossary#Documentation/gitglossary.txt-aiddefpathspecapathspec) whose contents norn never reads, such as `**/*.pem` |
| `disable` | `NORN_DISABLE` | `false` | |
| `classifier` | `NORN_CLASSIFIER` | `local` | `local`, or `jev` (see below). Global config only |

For types of your own, use a table of names to descriptions, such as
`deps = "Updates or adds third-party dependencies"`. Run `norn config` to see the settings in
effect and where each comes from.

## Using Jev

[TypeSafe's Jev](https://docs.typesafe.ai/api) picks types more accurately than the
built-in model, and also marks breaking changes (`!`). Set `classifier = "jev"` in your
global config and run `norn jev login` once to save your API key (`TYPESAFE_API_KEY`
overrides it).

| Setting | Environment variable | Default | |
|---|---|---|---|
| `timeout_ms` | `NORN_TIMEOUT_MS` | `1000` | Request timeout |
| `deadline_ms` | `NORN_DEADLINE_MS` | `2000` | Time limit for the whole hook |
| `breaking_threshold` | `NORN_BREAKING_THRESHOLD` | `0.85` | Probability needed to add `!` |
| `base_url` | `NORN_BASE_URL` | `https://api.typesafe.ai` | Global config only |

With Jev, norn sends the paths of your staged files, their patch (trimmed to about 24 KB,
without lockfiles or binaries) and your message, but not the repository name, remote,
branch or author. Secrets in the patch are sent too, so keep sensitive paths out with
`exclude`, and don't use Jev where code can't leave your machine. A repository's `.norn.toml`
can't set `classifier` or `base_url`, so a repository you clone can't send your code
anywhere.

[Laya](https://github.com/NandhaKishorM/laya) serves Jev-style models locally: run
`laya-serve` and set `base_url = "http://127.0.0.1:8000"`. Its published models classify
commits less accurately than norn's own.

## Development

```sh
cargo test
cargo build --release
./target/release/norn install   # dogfood on this repository
```

To measure classification, run `norn learn --skip 50` then `norn eval` in a repository
whose history uses conventional prefixes. `eval` replays the 50 newest labelled commits
(`--limit`) and reports accuracy and common mistakes; `--message` includes each commit's
description, and `--out FILE` writes every answer as JSON lines. [`training/`](training/README.md)
builds the dataset and trains the built-in model.

Releases are built by [dist](https://opensource.axo.dev/cargo-dist/): bump `version` in
`Cargo.toml` and push a matching tag (`git tag v0.3.0 && git push origin v0.3.0`). After
changing `dist-workspace.toml`, run `dist generate`.
