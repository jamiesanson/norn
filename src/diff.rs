//! Collects the staged changes from git, trimmed so the request stays small and fast.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Upper bound on patch text sent to Jev. Well under the ~64k token input limit;
/// smaller inputs are also faster.
const PATCH_BUDGET_BYTES: usize = 24_000;
/// Per-file cap so one large file can't crowd out the rest of the change.
const FILE_BUDGET_BYTES: usize = 4_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChangedFile {
    /// Git status letter: A, M, D, R, C, T.
    pub status: char,
    pub path: String,
}

/// Which changes to read: what's staged, or what an existing commit changed (used by
/// `norn eval` to replay history).
#[derive(Debug, Clone, Copy)]
pub enum Changes<'a> {
    Staged,
    Commit(&'a str),
}

impl Changes<'_> {
    fn args(self) -> Vec<String> {
        match self {
            Changes::Staged => vec!["diff".into(), "--cached".into()],
            Changes::Commit(sha) => vec!["diff".into(), format!("{sha}^!")],
        }
    }
}

const NAME_STATUS: &[&str] = &["--name-status", "-M", "-z"];
const PATCH: &[&str] = &["-M", "--unified=1", "--no-color", "--no-ext-diff"];

/// Every changed file. Only used locally.
pub fn changed_files(changes: Changes) -> Result<Vec<ChangedFile>, String> {
    Ok(parse_name_status(&git(changes, NAME_STATUS, &[])?))
}

/// What's sent to Jev: the changed files not matched by an `exclude` pattern, and their
/// patch. Only built once the local rules can't decide, so they stay fast.
#[derive(Debug)]
pub struct Outgoing {
    pub files: Vec<ChangedFile>,
    pub patch: String,
}

/// `exclude` holds git glob pathspecs relative to the repository root. Git applies them,
/// so excluded files' paths and contents never reach the patch.
pub fn outgoing(
    changes: Changes,
    all: &[ChangedFile],
    exclude: &[String],
) -> Result<Outgoing, String> {
    let pathspecs = exclude_pathspecs(exclude);
    if pathspecs.is_empty() {
        return Ok(Outgoing {
            files: all.to_vec(),
            patch: trim_patch(&git(changes, PATCH, &[])?),
        });
    }
    // Both git runs are needed; run them side by side.
    let (files, patch) = std::thread::scope(|s| {
        let files = s.spawn(|| git(changes, NAME_STATUS, &pathspecs));
        let patch = git(changes, PATCH, &pathspecs);
        (files.join().expect("git thread panicked"), patch)
    });
    Ok(Outgoing {
        files: parse_name_status(&files?),
        patch: trim_patch(&patch?),
    })
}

/// `:(top)` includes everything, then each pattern is excluded relative to the repository
/// root, whatever the working directory.
fn exclude_pathspecs(exclude: &[String]) -> Vec<String> {
    if exclude.is_empty() {
        return Vec::new();
    }
    std::iter::once(":(top)".to_string())
        .chain(exclude.iter().map(|p| format!(":(top,glob,exclude){p}")))
        .collect()
}

/// Starts reading the staged patch in the background, so it's ready, or nearly, by the time
/// the classifier asks for it. The hook calls this before listing the changed files, so the
/// two run side by side.
pub fn prefetch_staged(exclude: &[String]) {
    let pathspecs = exclude_pathspecs(exclude);
    if !pathspecs.is_empty() {
        let pathspecs = pathspecs.clone();
        std::thread::spawn(move || git(Changes::Staged, NAME_STATUS, &pathspecs));
    }
    std::thread::spawn(move || git(Changes::Staged, PATCH, &pathspecs));
}

/// A staged diff that has been run, or is running, shared by everyone who asks for it.
type Shared = Arc<OnceLock<Result<String, String>>>;

/// Staged diffs are remembered for the rest of the process: what's staged can't change
/// while a hook runs. Diffs of commits aren't, because `norn learn` reads thousands.
fn git(changes: Changes, args: &[&str], pathspecs: &[String]) -> Result<String, String> {
    if !matches!(changes, Changes::Staged) {
        return run_git(changes, args, pathspecs);
    }
    static STAGED: OnceLock<Mutex<HashMap<Vec<String>, Shared>>> = OnceLock::new();
    let key: Vec<String> = args
        .iter()
        .map(|a| a.to_string())
        .chain(pathspecs.iter().cloned())
        .collect();
    let shared = STAGED
        .get_or_init(Default::default)
        .lock()
        .unwrap()
        .entry(key)
        .or_default()
        .clone();
    // A second caller waits for the first to finish rather than running git again.
    shared
        .get_or_init(|| run_git(changes, args, pathspecs))
        .clone()
}

fn run_git(changes: Changes, args: &[&str], pathspecs: &[String]) -> Result<String, String> {
    let mut command = crate::git::command();
    command.args(changes.args()).args(args);
    if !pathspecs.is_empty() {
        command.arg("--").args(pathspecs);
    }
    let output = command
        .output()
        .map_err(|e| format!("failed to run git: {e}"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_string());
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Parses `git diff --name-status -z` output. Renames and copies carry two paths;
/// the destination is kept.
fn parse_name_status(raw: &str) -> Vec<ChangedFile> {
    let mut fields = raw.split('\0').filter(|s| !s.is_empty());
    let mut files = Vec::new();
    while let Some(code) = fields.next() {
        let status = code.chars().next().unwrap_or('M');
        if matches!(status, 'R' | 'C') {
            fields.next();
        }
        if let Some(path) = fields.next() {
            files.push(ChangedFile {
                status,
                path: path.to_string(),
            });
        }
    }
    files
}

/// Drops noise (lockfiles, binaries) and caps each file and the total patch size.
fn trim_patch(patch: &str) -> String {
    let mut out = String::new();
    for section in split_files(patch) {
        if out.len() >= PATCH_BUDGET_BYTES {
            out.push_str("\n[remaining files omitted]\n");
            break;
        }
        let header = section.lines().next().unwrap_or_default();
        if is_lockfile(header) || section.contains("\nBinary files ") {
            out.push_str(header);
            out.push_str("\n[contents omitted]\n");
            continue;
        }
        let budget = FILE_BUDGET_BYTES.min(PATCH_BUDGET_BYTES - out.len());
        if section.len() > budget {
            out.push_str(&section[..floor_char_boundary(section, budget)]);
            out.push_str("\n[truncated]\n");
        } else {
            out.push_str(section);
        }
    }
    out
}

fn split_files(patch: &str) -> Vec<&str> {
    // Only headers at the start of a line; patch content lines begin with +, - or space.
    let mut starts: Vec<usize> = patch
        .match_indices("diff --git ")
        .map(|(i, _)| i)
        .filter(|&i| i == 0 || patch.as_bytes()[i - 1] == b'\n')
        .collect();
    starts.push(patch.len());
    starts.windows(2).map(|w| &patch[w[0]..w[1]]).collect()
}

const LOCKFILES: &[&str] = &[
    "Cargo.lock",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "bun.lockb",
    "Gemfile.lock",
    "poetry.lock",
    "uv.lock",
    "go.sum",
    "composer.lock",
];

/// Also accepts a `diff --git a/… b/…` header, which ends with the destination path.
pub fn is_lockfile(path: &str) -> bool {
    LOCKFILES.contains(&path.rsplit('/').next().unwrap_or(path))
}

fn floor_char_boundary(s: &str, mut i: usize) -> usize {
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_name_status_with_renames() {
        let raw = "M\0src/main.rs\0R100\0old.rs\0new.rs\0A\0README.md\0";
        let files = parse_name_status(raw);
        assert_eq!(
            files,
            vec![
                ChangedFile {
                    status: 'M',
                    path: "src/main.rs".into()
                },
                ChangedFile {
                    status: 'R',
                    path: "new.rs".into()
                },
                ChangedFile {
                    status: 'A',
                    path: "README.md".into()
                },
            ]
        );
    }

    #[test]
    fn omits_lockfile_contents() {
        let patch = "diff --git a/Cargo.lock b/Cargo.lock\n+lots\n+of\n+lines\n\
                     diff --git a/src/lib.rs b/src/lib.rs\n+fn x() {}\n";
        let trimmed = trim_patch(patch);
        assert!(trimmed.contains("Cargo.lock\n[contents omitted]"));
        assert!(!trimmed.contains("+lots"));
        assert!(trimmed.contains("+fn x() {}"));
    }

    #[test]
    fn truncates_large_files() {
        let patch = format!("diff --git a/big b/big\n{}", "+x\n".repeat(10_000));
        let trimmed = trim_patch(&patch);
        assert!(trimmed.len() < FILE_BUDGET_BYTES + 100);
        assert!(trimmed.ends_with("[truncated]\n"));
    }
}
