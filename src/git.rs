//! Running git, which is most of what a commit spends in norn's hooks.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;

/// A git command. Inside a hook, git says where its own binary is (`GIT_EXEC_PATH`), and
/// running that directly skips wrappers such as macOS's `/usr/bin/git`, which hands off to
/// Xcode's copy and doubles the cost of every call.
pub fn command() -> Command {
    static GIT: OnceLock<PathBuf> = OnceLock::new();
    let git = GIT.get_or_init(|| {
        let name = if cfg!(windows) { "git.exe" } else { "git" };
        std::env::var_os("GIT_EXEC_PATH")
            .map(|dir| Path::new(&dir).join(name))
            .filter(|path| path.is_file())
            .unwrap_or_else(|| PathBuf::from("git"))
    });
    Command::new(git)
}

/// Trimmed standard output, if git succeeds.
pub fn output(args: &[&str]) -> Option<String> {
    let output = command().args(args).stderr(Stdio::null()).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// The repository's git directory, shared by all its worktrees. Found by reading `.git`
/// rather than asking git, which would cost a process on every commit, unless the
/// environment points somewhere else.
pub fn common_dir() -> Option<&'static Path> {
    static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();
    DIR.get_or_init(|| {
        let cwd = std::env::current_dir().ok()?;
        find_common_dir(&cwd)
            .or_else(|| output(&["rev-parse", "--git-common-dir"]).map(|d| cwd.join(d)))
    })
    .as_deref()
}

fn find_common_dir(start: &Path) -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("GIT_COMMON_DIR") {
        return Some(start.join(dir));
    }
    if std::env::var_os("GIT_DIR").is_some() {
        return None;
    }
    let mut dir = start.to_path_buf();
    loop {
        let dot_git = dir.join(".git");
        if dot_git.is_dir() {
            return Some(with_commondir(dot_git));
        }
        if dot_git.is_file() {
            // A worktree or submodule: `.git` names the real git directory.
            let text = std::fs::read_to_string(&dot_git).ok()?;
            let target = text.strip_prefix("gitdir:")?.trim();
            return Some(with_commondir(dir.join(target)));
        }
        if !dir.pop() {
            return None;
        }
    }
}

/// A worktree's git directory says where the shared one is in its `commondir` file.
fn with_commondir(git_dir: PathBuf) -> PathBuf {
    match std::fs::read_to_string(git_dir.join("commondir")) {
        Ok(common) => git_dir.join(common.trim()),
        Err(_) => git_dir,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("norn-git-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn finds_the_git_directory_from_a_subdirectory() {
        let root = temp("plain");
        std::fs::create_dir_all(root.join(".git")).unwrap();
        std::fs::create_dir_all(root.join("src/deep")).unwrap();
        assert_eq!(
            find_common_dir(&root.join("src/deep")),
            Some(root.join(".git"))
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn follows_a_worktree_to_the_shared_directory() {
        let root = temp("worktree");
        let main = root.join("main/.git");
        let linked = main.join("worktrees/feature");
        std::fs::create_dir_all(&linked).unwrap();
        std::fs::write(linked.join("commondir"), "../..\n").unwrap();
        let checkout = root.join("feature");
        std::fs::create_dir_all(&checkout).unwrap();
        std::fs::write(
            checkout.join(".git"),
            format!("gitdir: {}\n", linked.display()),
        )
        .unwrap();
        let found = find_common_dir(&checkout).unwrap();
        assert_eq!(found.canonicalize().unwrap(), main.canonicalize().unwrap());
        let _ = std::fs::remove_dir_all(&root);
    }
}
