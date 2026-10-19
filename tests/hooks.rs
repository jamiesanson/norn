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

    let sha = repo.git(&["rev-parse", "HEAD"]);
    let labels = std::fs::read_to_string(repo.dir.join(".git/norn/labels")).unwrap();
    assert!(
        labels.starts_with(sha.trim()) && labels.trim_end().ends_with("\tmessage\tkept"),
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
