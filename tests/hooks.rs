//! End-to-end checks of the hooks, through real `git commit`s in throwaway repositories.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const NORN: &str = env!("CARGO_BIN_EXE_norn");

/// A repository with one commit, isolated from the machine's git and norn settings.
struct Repo {
    root: PathBuf,
    dir: PathBuf,
}

impl Repo {
    fn new(name: &str) -> Repo {
        let root = std::env::temp_dir().join(format!("norn-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let dir = root.join("repo");
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::create_dir_all(root.join("config")).unwrap();
        std::fs::write(root.join("gitconfig"), "").unwrap();
        let repo = Repo { root, dir };
        repo.git(&["init", "-q"]);
        repo.write("src/config.rs", "fn load() {}\n");
        repo.git(&["add", "."]);
        repo.git(&["commit", "-qm", "feat: start"]);
        repo
    }

    fn command(&self, program: &str) -> Command {
        let mut command = Command::new(program);
        command
            .current_dir(&self.dir)
            .env("GIT_CONFIG_GLOBAL", self.root.join("gitconfig"))
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .env("GIT_AUTHOR_NAME", "Test")
            .env("GIT_AUTHOR_EMAIL", "test@example.com")
            .env("GIT_COMMITTER_NAME", "Test")
            .env("GIT_COMMITTER_EMAIL", "test@example.com")
            .env("XDG_CONFIG_HOME", self.root.join("config"))
            .env("APPDATA", self.root.join("config"))
            .env_remove("TYPESAFE_API_KEY")
            .env_remove("GIT_EDITOR");
        for (key, _) in std::env::vars().filter(|(k, _)| k.starts_with("NORN_")) {
            command.env_remove(key);
        }
        command
    }

    fn run(&self, program: &str, args: &[&str], env: &[(&str, &str)]) -> Output {
        let mut command = self.command(program);
        command.args(args);
        for (key, value) in env {
            command.env(key, value);
        }
        command.output().unwrap()
    }

    /// Runs git, expecting success.
    fn git(&self, args: &[&str]) -> String {
        let output = self.run("git", args, &[]);
        assert!(
            output.status.success(),
            "git {args:?}: {}",
            text(&output.stderr)
        );
        text(&output.stdout)
    }

    fn norn(&self, args: &[&str]) -> Output {
        self.run(NORN, args, &[])
    }

    fn write(&self, path: &str, contents: &str) {
        std::fs::write(self.dir.join(path), contents).unwrap();
    }

    fn change(&self, path: &str) {
        let full = self.dir.join(path);
        let mut contents = std::fs::read_to_string(&full).unwrap_or_default();
        contents.push_str("// changed\n");
        std::fs::write(full, contents).unwrap();
        self.git(&["add", "."]);
    }

    fn subject(&self) -> String {
        self.git(&["log", "-1", "--format=%s"]).trim().to_string()
    }

    fn hook(&self, name: &str) -> PathBuf {
        self.dir.join(".git").join("hooks").join(name)
    }

    fn install(&self) {
        let output = self.norn(&["install"]);
        assert!(output.status.success(), "install: {}", text(&output.stderr));
    }

    /// A `GIT_EDITOR` that saves what it was shown to `seen`, then writes `message`.
    fn editor(&self, message: &str) -> (String, PathBuf) {
        let seen = self.root.join("seen.txt");
        let path = seen.to_string_lossy().replace('\\', "/");
        let editor = format!("sh -c 'cp \"$1\" \"{path}\"; printf \"{message}\\n\" > \"$1\"' -");
        (editor, seen)
    }
}

impl Drop for Repo {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

fn is_prefixed(subject: &str, description: &str) -> bool {
    subject
        .strip_suffix(description)
        .and_then(|head| head.strip_suffix(": "))
        .is_some_and(|head| {
            let kind = head.split('(').next().unwrap_or_default();
            !kind.is_empty() && kind.chars().all(|c| c.is_ascii_lowercase())
        })
}

#[test]
fn install_leaves_foreign_hooks_alone() {
    let repo = Repo::new("foreign-post-commit");
    std::fs::write(repo.hook("post-commit"), "#!/bin/sh\necho mine\n").unwrap();
    repo.install();
    assert!(repo.hook("prepare-commit-msg").is_file());
    assert!(repo.hook("commit-msg").is_file());
    let post_commit = std::fs::read_to_string(repo.hook("post-commit")).unwrap();
    assert_eq!(post_commit, "#!/bin/sh\necho mine\n");

    let repo = Repo::new("foreign-prepare");
    std::fs::write(repo.hook("prepare-commit-msg"), "#!/bin/sh\n").unwrap();
    let output = repo.norn(&["install"]);
    assert!(!output.status.success());
    assert!(
        !repo.hook("commit-msg").exists(),
        "nothing is written on refusal"
    );
}

#[test]
fn commits_get_a_prefix_and_a_label_record() {
    let repo = Repo::new("prefix");
    repo.install();
    repo.change("src/config.rs");
    let output = repo.run(
        "git",
        &["commit", "-qm", "handle empty config files"],
        &[("NORN_MIN_CONFIDENCE", "0")],
    );
    assert!(output.status.success(), "{}", text(&output.stderr));
    let subject = repo.subject();
    assert!(
        is_prefixed(&subject, "handle empty config files"),
        "{subject}"
    );

    let parent = repo.git(&["rev-parse", "HEAD~1"]);
    let labels = std::fs::read_to_string(repo.dir.join(".git/norn/labels")).unwrap();
    let expected = format!("{}\t{subject}\t", parent.trim());
    assert!(
        labels.starts_with(&expected) && labels.trim_end().ends_with("\tmessage\tkept"),
        "{labels}"
    );
}

#[test]
fn docs_only_changes_are_docs() {
    let repo = Repo::new("docs");
    repo.install();
    repo.change("README.md");
    repo.git(&["commit", "-qm", "explain installing"]);
    assert_eq!(repo.subject(), "docs: explain installing");
}

#[test]
fn quitting_the_editor_still_aborts_the_commit() {
    let repo = Repo::new("abort");
    repo.install();
    repo.change("src/config.rs");
    let before = repo.git(&["rev-parse", "HEAD"]);
    // `true` leaves the message as norn pre-filled it: just a prefix.
    let output = repo.run(
        "git",
        &["commit", "-q"],
        &[("GIT_EDITOR", "true"), ("NORN_MIN_CONFIDENCE", "0")],
    );
    assert!(!output.status.success());
    assert_eq!(repo.git(&["rev-parse", "HEAD"]), before);
}

#[test]
fn unsure_suggestions_appear_only_where_git_strips_them() {
    let repo = Repo::new("hint");
    repo.install();
    let (editor, seen) = repo.editor("chore: tidy");
    let commit = |repo: &Repo| {
        repo.change("src/config.rs");
        let output = repo.run(
            "git",
            &["commit", "-q"],
            &[
                ("GIT_EDITOR", editor.as_str()),
                ("NORN_MIN_CONFIDENCE", "1"),
            ],
        );
        assert!(output.status.success(), "{}", text(&output.stderr));
        std::fs::read_to_string(&seen).unwrap()
    };

    assert!(commit(&repo).contains("# norn isn't sure: "));
    assert_eq!(repo.subject(), "chore: tidy");

    repo.git(&["config", "commit.cleanup", "verbatim"]);
    assert!(!commit(&repo).contains("norn isn't sure"));
}

#[test]
fn a_slow_hook_never_blocks_the_commit() {
    let repo = Repo::new("deadline");
    repo.install();
    repo.change("src/config.rs");
    let output = repo.run(
        "git",
        &["commit", "-qm", "handle empty config files"],
        &[("NORN_DEADLINE_MS", "0")],
    );
    assert!(output.status.success(), "{}", text(&output.stderr));
    assert!(repo.subject().ends_with("handle empty config files"));
}

#[test]
fn shows_its_version() {
    let output = Command::new(NORN).arg("--version").output().unwrap();
    assert!(text(&output.stdout).starts_with("norn "));
    assert!(Path::new(NORN).is_file());
}

/// Two commits the hook wasn't sure about, then one it labelled, all after `feat: start`,
/// which is on a fake remote.
fn unsure_commits(repo: &Repo) -> (String, String) {
    repo.install();
    repo.git(&["update-ref", "refs/remotes/origin/main", "HEAD"]);
    let unsure = [("NORN_MIN_CONFIDENCE", "1")];
    for subject in ["handle empty files", "split the parser"] {
        repo.change("src/config.rs");
        let output = repo.run("git", &["commit", "-qm", subject], &unsure);
        assert!(output.status.success(), "{}", text(&output.stderr));
    }
    repo.change("README.md");
    repo.git(&["commit", "-qm", "docs: explain"]);
    let short = |rev: &str| {
        repo.git(&["rev-parse", "--short=7", rev])
            .trim()
            .to_string()
    };
    (short("HEAD~2"), short("HEAD~1"))
}

fn subjects(repo: &Repo) -> Vec<String> {
    repo.git(&["log", "--format=%s"])
        .lines()
        .map(str::to_string)
        .collect()
}

#[test]
fn reword_prefixes_unsure_commits_with_norns_guesses() {
    let repo = Repo::new("reword");
    unsure_commits(&repo);
    let identity = |repo: &Repo| repo.git(&["log", "--format=%T %an %ad"]);
    let before = identity(&repo);
    repo.write("src/config.rs", "fn load() { todo!() }\n");

    // `true` saves the todo list as norn wrote it.
    let output = repo.run(NORN, &["reword"], &[("GIT_EDITOR", "true")]);
    assert!(output.status.success(), "{}", text(&output.stderr));
    assert!(text(&output.stdout).contains("reworded 2 commits"));

    let subjects = subjects(&repo);
    assert_eq!(subjects[0], "docs: explain");
    assert!(
        is_prefixed(&subjects[1], "split the parser"),
        "{subjects:?}"
    );
    assert!(
        is_prefixed(&subjects[2], "handle empty files"),
        "{subjects:?}"
    );
    assert_eq!(subjects[3], "feat: start");
    assert_eq!(identity(&repo), before, "trees, authors and dates are kept");
    assert_eq!(
        std::fs::read_to_string(repo.dir.join("src/config.rs")).unwrap(),
        "fn load() { todo!() }\n",
        "uncommitted changes survive"
    );

    let parent = repo.git(&["rev-parse", "HEAD~2"]);
    let labels = std::fs::read_to_string(repo.dir.join(".git/norn/labels")).unwrap();
    let line = labels
        .lines()
        .find(|l| l.starts_with(&format!("{}\t{}\t", parent.trim(), subjects[1])))
        .unwrap_or_else(|| panic!("no label for the new commit: {labels}"));
    assert!(line.ends_with("\teditor\tkept"), "{line}");
}

#[test]
fn reword_uses_the_edited_list() {
    let repo = Repo::new("reword-edit");
    let (_, second) = unsure_commits(&repo);
    let (editor, seen) = repo.editor(&format!("{second} refactor: split the parser"));
    let output = repo.run(NORN, &["reword"], &[("GIT_EDITOR", editor.as_str())]);
    assert!(output.status.success(), "{}", text(&output.stderr));

    let shown = std::fs::read_to_string(seen).unwrap();
    assert!(shown.contains("# norn reword:"), "{shown}");
    assert!(
        shown.contains(" (") && shown.contains("%)"),
        "guesses: {shown}"
    );
    let subjects = subjects(&repo);
    assert_eq!(subjects[1], "refactor: split the parser");
    assert_eq!(
        subjects[2], "handle empty files",
        "deleted lines are left alone"
    );
}

#[test]
fn reword_changes_nothing_when_cancelled_or_dry_run() {
    let repo = Repo::new("reword-cancel");
    unsure_commits(&repo);
    let head = repo.git(&["rev-parse", "HEAD"]);

    let (editor, _) = repo.editor("# nothing");
    let output = repo.run(NORN, &["reword"], &[("GIT_EDITOR", editor.as_str())]);
    assert!(output.status.success(), "{}", text(&output.stderr));
    assert!(text(&output.stdout).contains("nothing reworded"));
    assert_eq!(repo.git(&["rev-parse", "HEAD"]), head);

    let output = repo.run(NORN, &["reword", "--dry-run"], &[("GIT_EDITOR", "false")]);
    assert!(output.status.success(), "{}", text(&output.stderr));
    assert_eq!(text(&output.stdout).lines().count(), 2);
    assert_eq!(repo.git(&["rev-parse", "HEAD"]), head);
}

#[test]
fn reword_needs_a_remote_or_a_base() {
    let repo = Repo::new("reword-base");
    unsure_commits(&repo);
    repo.git(&["update-ref", "-d", "refs/remotes/origin/main"]);
    let output = repo.run(NORN, &["reword"], &[("GIT_EDITOR", "true")]);
    assert!(!output.status.success());
    assert!(text(&output.stderr).contains("no remote"));

    let output = repo.run(NORN, &["reword", "HEAD~2"], &[("GIT_EDITOR", "true")]);
    assert!(output.status.success(), "{}", text(&output.stderr));
    assert!(text(&output.stdout).contains("reworded 1 commit;"));
    assert_eq!(subjects(&repo)[2], "handle empty files");
}
