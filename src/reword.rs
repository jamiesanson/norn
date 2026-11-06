//! `norn reword`: prefixes the commits norn wasn't sure enough to label.
//!
//! Every commit is classified up front, in parallel, and norn's guesses go into a todo
//! list for you to edit. Git then does the rewriting: an interactive rebase whose todo
//! norn writes, with an `exec` after each chosen commit to amend in its new subject.
//! Amending through `git commit` keeps signing as the user's git config has it, and
//! conflicts and `git rebase --abort` work as for any rebase.

use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

use crate::classify::Classification;
use crate::{eval, labels, message};

/// Where `__reword-amend` finds the chosen subjects.
const PLAN_VAR: &str = "NORN_REWORD_PLAN";

const TODO_HEADER: &str = "\
# norn reword: edit each subject, or delete its line to leave that commit alone.
# Oldest first. Save and quit to reword; delete every line to cancel.
";

pub struct Options {
    /// Commits after this one, up to HEAD, are considered. Defaults to the commits that
    /// aren't on any remote.
    pub base: Option<String>,
    pub dry_run: bool,
}

impl Options {
    pub fn parse(args: &[&str]) -> Result<Options, String> {
        let mut options = Options {
            base: None,
            dry_run: false,
        };
        for &arg in args {
            match arg {
                "--dry-run" | "-n" => options.dry_run = true,
                _ if arg.starts_with('-') => return Err(format!("unknown option {arg}")),
                _ if options.base.is_some() => return Err(format!("unexpected argument {arg}")),
                // `main..` reads naturally as "since main".
                _ => {
                    let base = arg.strip_suffix("..").unwrap_or(arg);
                    if base.is_empty() || base.contains("..") {
                        return Err(format!(
                            "{arg}: pass the commit to start after, such as main or HEAD~5"
                        ));
                    }
                    options.base = Some(base.to_string());
                }
            }
        }
        Ok(options)
    }
}

struct Commit {
    sha: String,
    message: String,
    merge: bool,
}

impl Commit {
    fn subject(&self) -> &str {
        self.message.lines().next().unwrap_or_default()
    }

    fn short(&self) -> &str {
        &self.sha[..7]
    }
}

/// `classify` answers for one commit: its sha and its message.
pub fn run<F>(options: &Options, classify: F) -> Result<(), String>
where
    F: Fn(&str, &str) -> Result<Classification, String> + Sync,
{
    if let Some(operation) = operation_in_progress()? {
        return Err(format!("a {operation} is in progress; finish it first"));
    }
    let head = git(&["rev-parse", "HEAD"])?.trim().to_string();
    let (commits, base) = match &options.base {
        Some(rev) => {
            let base = resolve(rev).ok_or(format!("unknown revision {rev}"))?;
            (commits(&["HEAD", "--not", &base])?, Some(base))
        }
        None => {
            if git(&["for-each-ref", "--count=1", "refs/remotes"])?.is_empty() {
                return Err(
                    "there's no remote to tell which commits are unpushed; pass the \
                     commit to start after, such as HEAD~5"
                        .into(),
                );
            }
            let commits = commits(&["HEAD", "--not", "--remotes"])?;
            // The parent of the oldest unpushed commit, or none to rebase from the root.
            let base = commits
                .first()
                .and_then(|c| resolve(&format!("{}^", c.sha)));
            (commits, base)
        }
    };
    let candidates: Vec<&Commit> = commits
        .iter()
        .filter(|c| !c.merge && !message::should_skip(c.subject()))
        .collect();
    if candidates.is_empty() {
        out!("nothing to reword: every commit already has a prefix");
        return Ok(());
    }
    note!("norn: classifying {} commits", candidates.len());
    let results = eval::map_commits(&candidates, |commit| {
        classify(&commit.sha, commit.message.trim())
    });

    if options.dry_run {
        for (commit, result) in candidates.iter().zip(&results) {
            match result {
                Ok(c) => out!(
                    "{} {}{}  ({})",
                    commit.short(),
                    c.prefix(),
                    commit.subject(),
                    c.top_guesses()
                ),
                Err(e) => out!("{} {}  (error: {e})", commit.short(), commit.subject()),
            }
        }
        return Ok(());
    }

    let todo_path = git_path("norn-reword")?;
    std::fs::write(&todo_path, todo(&candidates, &results)).map_err(|e| e.to_string())?;
    let edited = edit(&todo_path).and_then(|()| {
        std::fs::read_to_string(&todo_path).map_err(|e| format!("{todo_path}: {e}"))
    });
    let _ = std::fs::remove_file(&todo_path);
    let chosen: Vec<(usize, String)> = parse_todo(&edited?, &candidates)?
        .into_iter()
        .filter(|(i, subject)| subject != candidates[*i].subject())
        .collect();
    if chosen.is_empty() {
        out!("nothing reworded");
        return Ok(());
    }

    let mut published = None;
    for (i, _) in &chosen {
        published = published_on(&candidates[*i].sha)?;
        if published.is_some() {
            break;
        }
    }

    let plan_path = git_path("norn-reword-plan")?;
    let plan: String = chosen
        .iter()
        .map(|(i, subject)| plan_line(&candidates[*i].sha, subject, results[*i].as_ref().ok()))
        .collect();
    std::fs::write(&plan_path, plan).map_err(|e| e.to_string())?;
    let rebased = rebase(base.as_deref(), &plan_path);
    let _ = std::fs::remove_file(&plan_path);
    rebased?;

    let count = match chosen.len() {
        1 => "1 commit".to_string(),
        n => format!("{n} commits"),
    };
    out!(
        "reworded {count}; undo with `git reset --keep {}`",
        &head[..7]
    );
    if let Some(remote) = published {
        out!("some were already on {remote}; push with `git push --force-with-lease`");
    }
    Ok(())
}

/// The list you edit: each commit's suggested subject, with norn's guesses below it.
fn todo(candidates: &[&Commit], results: &[Result<Classification, String>]) -> String {
    let mut out = format!("{TODO_HEADER}\n");
    for (commit, result) in candidates.iter().zip(results) {
        match result {
            Ok(c) => {
                out.push_str(&format!(
                    "{} {}{}\n",
                    commit.short(),
                    c.prefix(),
                    commit.subject()
                ));
                out.push_str(&format!("#       {}\n", c.top_guesses()));
            }
            Err(e) => {
                out.push_str(&format!("{} {}\n", commit.short(), commit.subject()));
                out.push_str(&format!("#       couldn't classify: {e}\n"));
            }
        }
    }
    out
}

/// The subject chosen for each commit left in the list, by its index in `candidates`.
fn parse_todo(todo: &str, candidates: &[&Commit]) -> Result<Vec<(usize, String)>, String> {
    let mut chosen = Vec::new();
    for line in todo.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (short, subject) = line.split_once(' ').unwrap_or((line, ""));
        let subject = subject.trim().replace('\t', " ");
        let found: Vec<usize> = (0..candidates.len())
            .filter(|&i| short.len() >= 4 && candidates[i].sha.starts_with(short))
            .collect();
        let &[i] = found.as_slice() else {
            return Err(format!("{short} isn't one of the listed commits"));
        };
        if subject.is_empty() {
            return Err(format!("{short} has no subject"));
        }
        chosen.push((i, subject));
    }
    Ok(chosen)
}

/// `sha  subject  kind  scope  confidence`, tab-separated. The suggestion is empty when
/// norn couldn't classify the commit.
fn plan_line(sha: &str, subject: &str, suggested: Option<&Classification>) -> String {
    let (kind, scope, confidence) = match suggested {
        Some(c) => (
            c.kind.as_str(),
            c.scope.as_deref().unwrap_or_default(),
            format!("{:.2}", c.confidence),
        ),
        None => ("", "", String::new()),
    };
    format!("{sha}\t{subject}\t{kind}\t{scope}\t{confidence}\n")
}

/// Opens `path` in the editor git would use, as git runs it.
fn edit(path: &str) -> Result<(), String> {
    let editor = git(&["var", "GIT_EDITOR"])?;
    let editor = editor.trim();
    let status = Command::new("sh")
        .arg("-c")
        .arg(format!("{editor} \"$@\""))
        .arg(editor)
        .arg(path)
        .status()
        .map_err(|e| format!("failed to run {editor}: {e}"))?;
    if !status.success() {
        return Err(format!("{editor} failed; nothing reworded"));
    }
    Ok(())
}

/// Runs the rebase with norn as its sequence editor, from `base`, or the root without
/// one. The hook is disabled throughout, since the subjects are already decided.
fn rebase(base: Option<&str>, plan_path: &str) -> Result<(), String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let editor = format!("{} __reword-todo", shell_quote(&exe.to_string_lossy()));
    let mut args = vec!["rebase", "--interactive", "--autostash", "--rebase-merges"];
    args.push(base.unwrap_or("--root"));
    let output = crate::git::command()
        .args(&args)
        .env("GIT_SEQUENCE_EDITOR", editor)
        .env(PLAN_VAR, plan_path)
        .env("NORN_DISABLE", "1")
        .output()
        .map_err(|e| format!("failed to run git: {e}"))?;
    // Progress is noise, but anything else (a conflict, an autostash that didn't apply
    // cleanly) needs to be seen. Interactive rebases ignore --quiet.
    let stderr = String::from_utf8_lossy(&output.stderr);
    for line in stderr.split(['\r', '\n']) {
        let noise = [
            "Rebasing (",
            "Executing: ",
            "Successfully rebased",
            "Created autostash",
            "Applied autostash",
        ];
        if !line.trim().is_empty() && !noise.iter().any(|n| line.starts_with(n)) {
            note!("{line}");
        }
    }
    if !output.status.success() {
        return Err("rebase stopped; finish it or run `git rebase --abort`".into());
    }
    Ok(())
}

/// The sequence editor: adds an `exec` amending in the chosen subject after each chosen
/// commit's `pick`.
pub fn edit_rebase_todo(todo: &Path) -> Result<(), String> {
    let plan = read_plan()?;
    let shas: Vec<&str> = plan.lines().filter_map(|l| l.split('\t').next()).collect();
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let exe = shell_quote(&exe.to_string_lossy());

    let original = std::fs::read_to_string(todo).map_err(|e| e.to_string())?;
    std::fs::write(todo, add_amends(&original, &shas, &exe)).map_err(|e| e.to_string())
}

fn add_amends(todo: &str, shas: &[&str], exe: &str) -> String {
    let mut out = String::new();
    for line in todo.lines() {
        out.push_str(line);
        out.push('\n');
        let mut words = line.split_whitespace();
        let (Some("pick" | "p"), Some(abbrev)) = (words.next(), words.next()) else {
            continue;
        };
        if let Some(sha) = shas.iter().find(|sha| sha.starts_with(abbrev)) {
            out.push_str(&format!("exec {exe} __reword-amend {sha}\n"));
        }
    }
    out
}

/// Run by the rebase after each chosen commit: gives HEAD the subject chosen for `sha`,
/// and records whether you kept norn's suggestion.
pub fn amend(sha: &str) -> Result<(), String> {
    let plan = read_plan()?;
    let fields: Vec<&str> = plan
        .lines()
        .map(|l| l.split('\t').collect::<Vec<_>>())
        .find(|f| f.first() == Some(&sha))
        .ok_or(format!("{sha} isn't in the plan"))?;
    let [_, subject, kind, scope, confidence] = fields[..] else {
        return Err(format!("bad plan line for {sha}"));
    };

    let current = git(&["log", "-1", "--format=%B"])?;
    let mut lines = current.trim_end().lines();
    lines.next();
    let reworded: String = std::iter::once(subject)
        .chain(lines)
        .collect::<Vec<_>>()
        .join("\n");
    let mut child = crate::git::command()
        .args([
            "commit",
            "--amend",
            "--quiet",
            "--no-verify",
            "--allow-empty",
            "--cleanup=verbatim",
            "--file=-",
        ])
        .stdin(Stdio::piped())
        .spawn()
        .map_err(|e| format!("failed to run git: {e}"))?;
    child
        .stdin
        .take()
        .expect("stdin is piped")
        .write_all(format!("{reworded}\n").as_bytes())
        .map_err(|e| e.to_string())?;
    if !child.wait().map_err(|e| e.to_string())?.success() {
        return Err("git commit --amend failed".into());
    }

    if let Ok(confidence) = confidence.parse() {
        let parent = resolve("HEAD^").unwrap_or_default();
        let scope = (!scope.is_empty()).then_some(scope);
        labels::record_reworded(kind, scope, confidence, &parent, subject);
    }
    Ok(())
}

fn read_plan() -> Result<String, String> {
    let path = std::env::var(PLAN_VAR).map_err(|_| format!("{PLAN_VAR} isn't set"))?;
    std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))
}

fn shell_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', r"'\''"))
}

/// A rebase, merge or other operation git is in the middle of, which a rebase would
/// refuse to start over.
fn operation_in_progress() -> Result<Option<&'static str>, String> {
    for (path, operation) in [
        ("rebase-merge", "rebase"),
        ("rebase-apply", "rebase"),
        ("MERGE_HEAD", "merge"),
        ("CHERRY_PICK_HEAD", "cherry-pick"),
        ("REVERT_HEAD", "revert"),
    ] {
        if Path::new(&git_path(path)?).exists() {
            return Ok(Some(operation));
        }
    }
    Ok(None)
}

fn git_path(name: &str) -> Result<String, String> {
    Ok(git(&["rev-parse", "--git-path", name])?.trim().to_string())
}

fn resolve(rev: &str) -> Option<String> {
    crate::git::output(&[
        "rev-parse",
        "--verify",
        "--quiet",
        &format!("{rev}^{{commit}}"),
    ])
}

/// The commits `revs` selects, oldest first.
fn commits(revs: &[&str]) -> Result<Vec<Commit>, String> {
    let mut args = vec![
        "log",
        "--reverse",
        "--topo-order",
        "--format=%H%x00%P%x00%B%x1e",
    ];
    args.extend(revs);
    args.push("--");
    let log = git(&args)?;
    Ok(log
        .split('\x1e')
        .filter_map(|record| {
            let mut fields = record.trim_start_matches('\n').splitn(3, '\0');
            let (sha, parents, message) = (fields.next()?, fields.next()?, fields.next()?);
            Some(Commit {
                sha: sha.to_string(),
                message: message.to_string(),
                merge: parents.contains(' '),
            })
        })
        .collect())
}

/// A remote-tracking branch that already contains `sha`, if any.
fn published_on(sha: &str) -> Result<Option<String>, String> {
    let refs = git(&[
        "for-each-ref",
        "--contains",
        sha,
        "--format=%(refname:short)",
        "refs/remotes",
    ])?;
    Ok(refs.lines().next().map(str::to_string))
}

fn git(args: &[&str]) -> Result<String, String> {
    let output = crate::git::command()
        .args(args)
        .output()
        .map_err(|e| format!("failed to run git: {e}"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_string());
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classify::Source;
    use std::collections::BTreeMap;

    fn commit(sha: &str, subject: &str) -> Commit {
        Commit {
            sha: sha.into(),
            message: format!("{subject}\n\nThe body.\n"),
            merge: false,
        }
    }

    fn classification(kind: &str, scope: Option<&str>) -> Classification {
        Classification {
            kind: kind.into(),
            scope: scope.map(str::to_string),
            breaking: false,
            confidence: 0.48,
            probabilities: BTreeMap::from([(kind.into(), 0.48), ("feat".into(), 0.31)]),
            breaking_probability: None,
            source: Source::Model,
        }
    }

    #[test]
    fn parses_options() {
        let o = Options::parse(&["-n", "main.."]).unwrap();
        assert!(o.dry_run);
        assert_eq!(o.base.as_deref(), Some("main"));
        assert_eq!(Options::parse(&[]).unwrap().base, None);
        assert!(Options::parse(&["main..feature"]).is_err());
        assert!(Options::parse(&["main", "HEAD~2"]).is_err());
        assert!(Options::parse(&["--force"]).is_err());
    }

    #[test]
    fn todo_fills_in_guesses_and_errors() {
        let a = commit("1111111aaaa", "handle empty files");
        let b = commit("2222222bbbb", "split the parser");
        let results = [
            Ok(classification("fix", Some("config"))),
            Err("no diff".to_string()),
        ];
        let todo = todo(&[&a, &b], &results);
        assert!(todo.starts_with(TODO_HEADER));
        assert!(todo.ends_with(
            "1111111 fix(config): handle empty files\n\
             #       fix(config) (48%) or feat(config) (31%)\n\
             2222222 split the parser\n\
             #       couldn't classify: no diff\n"
        ));
    }

    #[test]
    fn parses_the_edited_todo() {
        let a = commit("1111111aaaa", "handle empty files");
        let b = commit("2222222bbbb", "split the parser");
        let candidates = [&a, &b];
        let edited = "# header\n\n2222222 refactor: split  the parser \n#   guesses\n";
        assert_eq!(
            parse_todo(edited, &candidates).unwrap(),
            vec![(1, "refactor: split  the parser".to_string())]
        );
        assert_eq!(
            parse_todo("1111 fix: handle empty files", &candidates).unwrap(),
            vec![(0, "fix: handle empty files".to_string())]
        );
        assert!(
            parse_todo("# all deleted\n", &candidates)
                .unwrap()
                .is_empty()
        );
        assert!(parse_todo("3333333 feat: who", &candidates).is_err());
        assert!(parse_todo("111 fix: too short", &candidates).is_err());
        assert!(parse_todo("1111111", &candidates).is_err());
    }

    #[test]
    fn plan_lines_carry_the_suggestion() {
        let c = classification("fix", Some("config"));
        assert_eq!(
            plan_line("abc", "feat(config): x", Some(&c)),
            "abc\tfeat(config): x\tfix\tconfig\t0.48\n"
        );
        assert_eq!(plan_line("abc", "chore: x", None), "abc\tchore: x\t\t\t\n");
    }

    #[test]
    fn adds_amends_after_chosen_picks() {
        let todo = "label onto\n\
                    reset onto\n\
                    pick 1111111 add a flag\n\
                    pick 2222222 docs: readme\n\
                    # pick 3333333 commented out\n";
        assert_eq!(
            add_amends(todo, &["1111111abcdef"], "'/bin/norn'"),
            "label onto\n\
             reset onto\n\
             pick 1111111 add a flag\n\
             exec '/bin/norn' __reword-amend 1111111abcdef\n\
             pick 2222222 docs: readme\n\
             # pick 3333333 commented out\n"
        );
    }

    #[test]
    fn quotes_for_the_shell() {
        assert_eq!(shell_quote("it's"), r"'it'\''s'");
    }
}
