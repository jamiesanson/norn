//! `norn eval`: replays commits that already have conventional prefixes and compares
//! what norn would have chosen with what the author wrote.
//!
//! `norn export`: writes the same commits as labelled training data, one request per
//! line with the author's answers, in the JSONL format `laya-train` reads.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::classify::{Classification, Source};
use crate::message;

/// Requests in flight at once.
const WORKERS: usize = 8;

pub struct Options {
    /// `usize::MAX` for every commit.
    pub limit: usize,
    /// Send each commit's description, as a user committing with `-m` would.
    pub with_message: bool,
    /// Writes one JSON object per commit, for analysis outside norn.
    pub out: Option<PathBuf>,
    pub rev: String,
}

impl Options {
    pub fn parse(args: &[&str], default_limit: usize) -> Result<Options, String> {
        let mut options = Options {
            limit: default_limit,
            with_message: false,
            out: None,
            rev: "HEAD".into(),
        };
        let mut args = args.iter();
        while let Some(&arg) = args.next() {
            let mut value = || args.next().copied().ok_or(format!("{arg} needs a value"));
            match arg {
                "--limit" => {
                    options.limit = value()?
                        .parse()
                        .map_err(|_| "--limit must be a number".to_string())?
                }
                "--message" => options.with_message = true,
                "--out" => options.out = Some(PathBuf::from(value()?)),
                _ if arg.starts_with('-') => return Err(format!("unknown option {arg}")),
                _ => options.rev = arg.to_string(),
            }
        }
        Ok(options)
    }
}

pub struct Labelled {
    pub sha: String,
    pub subject: String,
    /// Made by a bot (dependabot, renovate, ...), so never through norn's hook.
    pub bot: bool,
}

impl Labelled {
    pub fn label(&self) -> message::Prefix<'_> {
        message::parse_prefix(&self.subject).expect("only labelled commits are kept")
    }

    /// The description, as `git commit -m` would send it, or nothing for the editor flow.
    pub fn message(&self, with_message: bool) -> &str {
        if with_message {
            self.label().description
        } else {
            ""
        }
    }
}

/// `classify` answers for one commit: its sha and the message to send.
pub fn run<F>(options: &Options, types: &[String], classify: F) -> Result<(), String>
where
    F: Fn(&str, &str) -> Result<Classification, String> + Sync,
{
    let (commits, unknown_types) = labelled_commits(&options.rev, options.limit, types)?;
    if commits.is_empty() {
        return Err(format!(
            "no commits with conventional prefixes in {}",
            options.rev
        ));
    }
    eprintln!("norn: evaluating {} commits", commits.len());

    let results = map_commits(&commits, |commit| {
        classify(&commit.sha, commit.message(options.with_message))
    });

    if let Some(path) = &options.out {
        write_jsonl(path, &commits, &results)?;
    }
    for (commit, result) in commits.iter().zip(&results) {
        println!("{}", row(commit, result));
    }
    println!();
    print!("{}", summary(&commits, &results));
    if unknown_types > 0 {
        println!("skipped {unknown_types} commits whose type isn't in the configured types");
    }
    Ok(())
}

/// `request` gives the request norn would send for one commit (its sha and the message
/// to send), or `None` when the local rules would answer without one.
pub fn export<F>(options: &Options, types: &[String], request: F) -> Result<(), String>
where
    F: Fn(&str, &str) -> Result<Option<serde_json::Value>, String> + Sync,
{
    let (commits, unknown_types) = labelled_commits(&options.rev, options.limit, types)?;
    if commits.is_empty() {
        return Err(format!(
            "no commits with conventional prefixes in {}",
            options.rev
        ));
    }
    eprintln!("norn: exporting {} commits", commits.len());

    let results = map_commits(&commits, |commit| {
        request(&commit.sha, commit.message(options.with_message))
    });

    let mut out: Box<dyn Write> = match &options.out {
        Some(path) => Box::new(std::io::BufWriter::new(
            std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?,
        )),
        None => Box::new(std::io::BufWriter::new(std::io::stdout().lock())),
    };
    let (mut written, mut local, mut errors) = (0, 0, 0);
    for (commit, result) in commits.iter().zip(results) {
        match result {
            Ok(Some(request)) => {
                writeln!(out, "{}", training_row(commit, request)).map_err(|e| e.to_string())?;
                written += 1;
            }
            Ok(None) => local += 1,
            Err(e) => {
                eprintln!("!  {} error: {e}", &commit.sha[..7]);
                errors += 1;
            }
        }
    }
    out.flush().map_err(|e| e.to_string())?;

    eprintln!("norn: wrote {written} commits");
    if local > 0 {
        eprintln!("skipped {local} commits the local rules answer without a request");
    }
    if errors > 0 {
        eprintln!("skipped {errors} commits that couldn't be read");
    }
    if unknown_types > 0 {
        eprintln!("skipped {unknown_types} commits whose type isn't in the configured types");
    }
    Ok(())
}

/// `{state, questions, expected}`: the request's own state and questions, so training sees
/// exactly what the model is asked, and the author's answers. `sha` and `subject` are for
/// tracing a row back to its commit.
fn training_row(commit: &Labelled, request: serde_json::Value) -> serde_json::Value {
    let label = commit.label();
    serde_json::json!({
        "sha": commit.sha,
        "subject": commit.subject,
        "state": request["state"],
        "questions": request["questions"],
        "expected": { "type": label.kind, "breaking": label.breaking },
    })
}

/// Runs `f` over every commit, `WORKERS` at a time, keeping the commits' order.
pub fn map_commits<T, F>(commits: &[Labelled], f: F) -> Vec<T>
where
    T: Send,
    F: Fn(&Labelled) -> T + Sync,
{
    let results: Vec<Mutex<Option<T>>> = commits.iter().map(|_| Mutex::new(None)).collect();
    let next = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..WORKERS.min(commits.len()) {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(commit) = commits.get(i) else { break };
                    *results[i].lock().unwrap() = Some(f(commit));
                }
            });
        }
    });
    results
        .into_iter()
        .map(|r| r.into_inner().unwrap().expect("every commit is mapped"))
        .collect()
}

/// Non-merge, non-root commits in `rev` whose subject has a prefix with an allowed type,
/// newest first, and how many had a type that isn't allowed.
pub fn labelled_commits(
    rev: &str,
    limit: usize,
    types: &[String],
) -> Result<(Vec<Labelled>, usize), String> {
    let output = Command::new("git")
        .args([
            "log",
            "--no-merges",
            "--format=%H%x00%P%x00%an%x00%ae%x00%s",
            rev,
            "--",
        ])
        .output()
        .map_err(|e| format!("failed to run git: {e}"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_string());
    }
    let log = String::from_utf8_lossy(&output.stdout);

    let mut commits = Vec::new();
    let mut unknown_types = 0;
    for line in log.lines() {
        let mut fields = line.splitn(5, '\0');
        let (Some(sha), Some(parents), Some(name), Some(email), Some(subject)) = (
            fields.next(),
            fields.next(),
            fields.next(),
            fields.next(),
            fields.next(),
        ) else {
            continue;
        };
        if parents.is_empty() {
            continue;
        }
        let Some(label) = message::parse_prefix(subject) else {
            continue;
        };
        if !types.iter().any(|t| t == label.kind) {
            unknown_types += 1;
            continue;
        }
        commits.push(Labelled {
            sha: sha.to_string(),
            subject: subject.to_string(),
            bot: is_bot(name, email),
        });
        if commits.len() == limit {
            break;
        }
    }
    Ok((commits, unknown_types))
}

/// GitHub App bots (`dependabot[bot]`), the identity GitHub Actions commits as, and
/// accounts named like bots (`excalibot`, `GoReleaser Bot`).
fn is_bot(name: &str, email: &str) -> bool {
    let (name, email) = (name.to_ascii_lowercase(), email.to_ascii_lowercase());
    // GitHub's noreply addresses look like `12345+name@users.noreply.github.com`.
    let local = email.split('@').next().unwrap_or_default();
    let local = local.rsplit('+').next().unwrap_or(local);
    let named_bot = |s: &str| {
        s.ends_with("bot") && !["talbot", "abbot", "cabot"].iter().any(|n| s.ends_with(n))
    };
    [
        "[bot]",
        "dependabot",
        "renovate",
        "github-actions",
        "release-please",
    ]
    .iter()
    .any(|b| name.contains(b) || email.contains(b))
        || email == "actions@github.com"
        || named_bot(&name)
        || named_bot(local)
}

fn row(commit: &Labelled, result: &Result<Classification, String>) -> String {
    let label = commit.label();
    let short = &commit.sha[..7];
    match result {
        Err(e) => format!("!  {short} {:<9} error: {e}", label.kind),
        Ok(c) => {
            let mark = if c.kind == label.kind { "✓" } else { "✗" };
            let how = match c.source {
                Source::Rules => "rules".to_string(),
                Source::Jev | Source::Model => format!("{:>3.0}%", c.confidence * 100.0),
            };
            let runner_up = runner_up(c)
                .map(|(kind, p)| format!(" ({kind} {p:.2})"))
                .unwrap_or_default();
            format!(
                "{mark}  {short} {:<9} → {:<9} {how:>5}{runner_up}  {}",
                label.kind, c.kind, commit.subject
            )
        }
    }
}

/// The second most likely type, when it has a real share of the probability.
fn runner_up(c: &Classification) -> Option<(&str, f64)> {
    c.probabilities
        .iter()
        .filter(|(kind, _)| **kind != c.kind)
        .max_by(|a, b| a.1.total_cmp(b.1))
        .filter(|(_, p)| **p >= 0.05)
        .map(|(kind, p)| (kind.as_str(), *p))
}

fn summary(commits: &[Labelled], results: &[Result<Classification, String>]) -> String {
    let mut out = String::new();
    let mut line = |s: String| {
        out.push_str(&s);
        out.push('\n');
    };
    let pct = |n: usize, d: usize| {
        if d == 0 {
            "-".to_string()
        } else {
            format!("{n}/{d} ({:.0}%)", 100.0 * n as f64 / d as f64)
        }
    };

    let answered: Vec<(message::Prefix, &Classification)> = commits
        .iter()
        .zip(results)
        .filter_map(|(commit, r)| r.as_ref().ok().map(|c| (commit.label(), c)))
        .collect();
    let errors = results.len() - answered.len();
    let correct = |c: &(message::Prefix, &Classification)| c.0.kind == c.1.kind;

    line(format!("commits     {}", commits.len()));
    if errors > 0 {
        line(format!("errors      {errors}"));
    }
    line(format!(
        "type        {} correct",
        pct(
            answered.iter().filter(|a| correct(a)).count(),
            answered.len()
        )
    ));
    for (source, name) in [
        (Source::Rules, "rules"),
        (Source::Jev, "jev"),
        (Source::Model, "model"),
    ] {
        let by: Vec<_> = answered.iter().filter(|a| a.1.source == source).collect();
        if !by.is_empty() || source == Source::Rules {
            line(format!(
                "  {name:<9} {} correct",
                pct(by.iter().filter(|a| correct(a)).count(), by.len())
            ));
        }
    }

    let expected_breaking = answered.iter().filter(|a| a.0.breaking).count();
    let predicted_breaking = answered.iter().filter(|a| a.1.breaking).count();
    let both = answered
        .iter()
        .filter(|a| a.0.breaking && a.1.breaking)
        .count();
    line(format!(
        "breaking    {expected_breaking} expected, {predicted_breaking} marked, {both} both"
    ));
    let with_scope: Vec<_> = answered.iter().filter(|a| a.0.scope.is_some()).collect();
    line(format!(
        "scope       {} of labelled scopes matched",
        pct(
            with_scope
                .iter()
                .filter(|a| a.0.scope == a.1.scope.as_deref())
                .count(),
            with_scope.len()
        )
    ));

    let mut confusions: BTreeMap<(&str, &str), usize> = BTreeMap::new();
    for a in answered.iter().filter(|a| !correct(a)) {
        *confusions.entry((a.0.kind, a.1.kind.as_str())).or_default() += 1;
    }
    let mut confusions: Vec<_> = confusions.into_iter().collect();
    confusions.sort_by(|a, b| b.1.cmp(&a.1));
    if !confusions.is_empty() {
        line("mistakes    (labelled → chosen)".into());
        for ((expected, chosen), n) in confusions.iter().take(8) {
            line(format!("  {n:>3}  {expected} → {chosen}"));
        }
    }
    out
}

fn write_jsonl(
    path: &PathBuf,
    commits: &[Labelled],
    results: &[Result<Classification, String>],
) -> Result<(), String> {
    let mut file = std::fs::File::create(path).map_err(|e| e.to_string())?;
    for (commit, result) in commits.iter().zip(results) {
        let label = commit.label();
        let mut record = serde_json::json!({
            "sha": commit.sha,
            "subject": commit.subject,
            "label": { "type": label.kind, "scope": label.scope, "breaking": label.breaking },
        });
        match result {
            Ok(c) => {
                record["result"] = serde_json::json!({
                    "type": c.kind,
                    "scope": c.scope,
                    "breaking": c.breaking,
                    "breaking_probability": c.breaking_probability,
                    "confidence": c.confidence,
                    "probabilities": c.probabilities,
                    "source": format!("{:?}", c.source).to_lowercase(),
                });
            }
            Err(e) => record["error"] = serde_json::Value::from(e.as_str()),
        }
        writeln!(file, "{record}").map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_options() {
        let o = Options::parse(&["--limit", "20", "--message", "main~10"], 50).unwrap();
        assert_eq!(o.limit, 20);
        assert!(o.with_message);
        assert_eq!(o.rev, "main~10");
        assert_eq!(Options::parse(&[], usize::MAX).unwrap().limit, usize::MAX);
        assert!(Options::parse(&["--limit"], 50).is_err());
        assert!(Options::parse(&["--nope"], 50).is_err());
    }

    #[test]
    fn recognises_bots() {
        for (name, email) in [
            (
                "dependabot[bot]",
                "49699333+dependabot[bot]@users.noreply.github.com",
            ),
            ("renovate-bot", "bot@renovateapp.com"),
            ("actions-user", "actions@github.com"),
            (
                "GoReleaser Bot",
                "29843943+goreleaserbot@users.noreply.github.com",
            ),
            (
                "Excalidraw Bot",
                "77840495+excalibot@users.noreply.github.com",
            ),
        ] {
            assert!(is_bot(name, email), "{name}");
        }
        for (name, email) in [
            ("Carlos Becker", "caarlos0@users.noreply.github.com"),
            ("John Talbot", "jtalbot@example.com"),
            ("Copilot", "198982749+Copilot@users.noreply.github.com"),
        ] {
            assert!(!is_bot(name, email), "{name}");
        }
    }

    #[test]
    fn training_row_keeps_the_request_and_adds_the_label() {
        let commit = Labelled {
            sha: "abc1234".into(),
            subject: "fix(config)!: handle empty files".into(),
            bot: false,
        };
        let request = serde_json::json!({
            "model": "jev-latest",
            "state": { "files": ["M src/config.rs"], "patch": "..." },
            "questions": { "type": { "type": "choice" } },
        });
        let row = training_row(&commit, request);
        assert_eq!(row["state"]["files"][0], "M src/config.rs");
        assert_eq!(row["questions"]["type"]["type"], "choice");
        assert_eq!(
            row["expected"],
            serde_json::json!({ "type": "fix", "breaking": true })
        );
        assert!(row.get("model").is_none());
    }

    #[test]
    fn runner_up_needs_a_real_share() {
        let mut c = Classification {
            kind: "feat".into(),
            scope: None,
            breaking: false,
            confidence: 0.5,
            probabilities: BTreeMap::from([
                ("feat".into(), 0.6),
                ("fix".into(), 0.37),
                ("chore".into(), 0.03),
            ]),
            breaking_probability: None,
            source: Source::Jev,
        };
        assert_eq!(runner_up(&c), Some(("fix", 0.37)));
        c.probabilities = BTreeMap::from([("feat".into(), 0.98), ("fix".into(), 0.02)]);
        assert_eq!(runner_up(&c), None);
    }
}
