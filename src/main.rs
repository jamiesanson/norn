mod classify;
mod config;
mod credentials;
mod diff;
mod eval;
mod features;
mod jev;
mod lbfgs;
mod learn;
mod message;
mod model;

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};
use std::time::{Duration, Instant};

use classify::{Classification, Settings};
use config::Config;

const USAGE: &str = "\
norn: prefix commit messages with Conventional Commits types, decided on your machine or by Jev

USAGE:
    norn jev login                     Save your TypeSafe API key
    norn jev logout                    Remove the saved API key
    norn install                       Install the git hooks into the current repository
    norn classify [MESSAGE]            Classify the staged diff and print the prefix
    norn config                        Show the settings in effect and where each comes from
    norn eval [OPTIONS] [REV]          Compare norn's choices with the prefixes in REV's history
        --limit N      Commits to evaluate (default: 50)
        --message      Send each commit's description, as `git commit -m` would
        --out FILE     Also write each result as a JSON line
    norn learn [OPTIONS]               Fit the built-in model to this repository's history
        --limit N      Commits to learn from (default: 1000)
        --skip N       Leave out the newest N commits, to evaluate the model on them
    norn export [OPTIONS] [REV]        Write REV's labelled commits as training data (JSONL)
        --limit N      Commits to export (default: all)
        --message      Include each commit's description, as `git commit -m` would
        --out FILE     Write to FILE instead of standard output
    norn prepare-commit-msg FILE [SOURCE] [SHA]   (git hook)
    norn commit-msg FILE                          (git hook)

CONFIGURATION:
    Global:  ~/.config/norn/config.toml
    Project: .norn.toml at the repository root (can't set classifier or base_url)
    Environment variables override both:

    NORN_CLASSIFIER          `local` for the built-in model, or `jev` (default: local)
    TYPESAFE_API_KEY         API key for Jev; overrides the key saved by `norn jev login`
    NORN_BASE_URL            Jev API URL, or a Jev-compatible server (default: https://api.typesafe.ai)
    NORN_TIMEOUT_MS          Request timeout; the message is left as-is on timeout (default: 1000)
    NORN_DEADLINE_MS         Time limit for the whole hook (default: 2000)
    NORN_MIN_CONFIDENCE      Minimum confidence to apply a type (default: 0.6)
    NORN_BREAKING_THRESHOLD  Probability needed to mark a change breaking (default: 0.85)
    NORN_DISABLE             Set to 1 to skip the hook entirely
";

const HOOK_MARKER: &str = "# installed by norn";
const BIN_NAME: &str = if cfg!(windows) { "norn.exe" } else { "norn" };

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let args: Vec<&str> = args.iter().map(String::as_str).collect();

    match args.as_slice() {
        ["prepare-commit-msg", file, rest @ ..] => {
            let config = hook_config();
            start_deadline(Path::new(file), config.deadline_ms.value);
            let result = prepare_commit_msg(Path::new(file), rest.first().copied(), &config);
            if !config.disable.value {
                learn::refresh_if_stale();
            }
            hook_result(result)
        }
        ["commit-msg", file] => {
            // prepare-commit-msg already reported any config warnings for this commit.
            let config = config::load(std::env::current_dir().ok().as_deref());
            start_deadline(Path::new(file), config.deadline_ms.value);
            hook_result(commit_msg(Path::new(file)))
        }
        ["classify", message @ ..] => command_result(run_classify(&message.join(" "))),
        ["config"] => command_result(show_config()),
        ["eval", rest @ ..] => command_result(run_eval(rest)),
        ["export", rest @ ..] => command_result(run_export(rest)),
        ["learn", rest @ ..] => command_result(run_learn(rest)),
        // For `training/train_linear.py`; not in USAGE.
        ["features"] => command_result(run_features(None)),
        ["features", "--model", path] => command_result(run_features(Some(Path::new(path)))),
        ["install"] => command_result(install()),
        ["jev", "login"] => command_result(login()),
        ["jev", "logout"] => command_result(logout()),
        ["--version" | "-V"] => {
            println!("norn {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        _ => {
            eprint!("{USAGE}");
            ExitCode::from(2)
        }
    }
}

fn command_result(result: Result<(), String>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("norn: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Hooks never block a commit: failures are reported and the message is left alone.
fn hook_result(result: Result<(), String>) -> ExitCode {
    if let Err(e) = result {
        eprintln!("norn: message left unchanged ({e})");
    }
    ExitCode::SUCCESS
}

/// Git runs hooks from the repository root, so the project file is found without
/// spawning git to locate it.
fn hook_config() -> Config {
    let config = config::load(std::env::current_dir().ok().as_deref());
    print_warnings(&config);
    config
}

/// For commands run by hand, possibly from a subdirectory.
fn command_config() -> Config {
    let config = config::load(repo_root().as_deref());
    print_warnings(&config);
    config
}

fn print_warnings(config: &Config) {
    for warning in &config.warnings {
        eprintln!("norn: {warning}");
    }
}

fn repo_root() -> Option<PathBuf> {
    let output = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .stderr(Stdio::null())
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| PathBuf::from(String::from_utf8_lossy(&output.stdout).trim()))
}

/// Ends the hook after the deadline, whatever it's waiting on (git, DNS, the API), so
/// norn can never hang a commit. Message writes are atomic, so exiting at any point
/// leaves either the original message or the prefixed one.
fn start_deadline(message_file: &Path, deadline_ms: u64) {
    let deadline = Duration::from_millis(deadline_ms);
    let tmp = tmp_path(message_file);
    std::thread::spawn(move || {
        std::thread::sleep(deadline);
        let _ = std::fs::remove_file(&tmp);
        eprintln!(
            "norn: message left unchanged (took longer than {}ms)",
            deadline.as_millis()
        );
        std::process::exit(0);
    });
}

fn prepare_commit_msg(file: &Path, source: Option<&str>, config: &Config) -> Result<(), String> {
    if config.disable.value {
        return Ok(());
    }
    // merge, squash and commit (-c/-C/--amend) reuse an existing message.
    if matches!(source, Some("merge" | "squash" | "commit")) {
        return Ok(());
    }

    let content = std::fs::read_to_string(file).map_err(|e| e.to_string())?;
    let body = message::body(&content);
    let subject = body.lines().next().unwrap_or_default();
    if message::should_skip(subject) {
        return Ok(());
    }

    let files = diff::changed_files(diff::Changes::Staged)?;
    if files.is_empty() {
        return Ok(());
    }
    let classification = classify_changes(diff::Changes::Staged, &files, &body, config, None)?;
    let min_confidence = config.min_confidence.value;
    if classification.confidence < min_confidence {
        return Err(format!(
            "best guess {} at {:.0}% confidence, below {:.0}%",
            classification.kind,
            classification.confidence * 100.0,
            min_confidence * 100.0
        ));
    }

    write_atomic(
        file,
        &message::apply_prefix(&content, &classification.prefix()),
    )
}

/// Clears a message that is only the prefix we suggested, so quitting the editor
/// without writing anything still aborts the commit.
fn commit_msg(file: &Path) -> Result<(), String> {
    let content = std::fs::read_to_string(file).map_err(|e| e.to_string())?;
    if message::is_bare_prefix(&content) {
        write_atomic(file, "")?;
    }
    Ok(())
}

/// Writes via a temporary file and a rename, so an interrupted write can't leave git a
/// truncated commit message.
fn write_atomic(path: &Path, contents: &str) -> Result<(), String> {
    let tmp = tmp_path(path);
    std::fs::write(&tmp, contents)
        .and_then(|()| std::fs::rename(&tmp, path))
        .map_err(|e| {
            let _ = std::fs::remove_file(&tmp);
            e.to_string()
        })
}

fn tmp_path(path: &Path) -> PathBuf {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".norn.tmp");
    PathBuf::from(tmp)
}

fn run_classify(message: &str) -> Result<(), String> {
    let started = Instant::now();
    let config = command_config();
    let files = diff::changed_files(diff::Changes::Staged)?;
    if files.is_empty() {
        return Err("nothing staged".into());
    }
    let c = classify_changes(diff::Changes::Staged, &files, message, &config, None)?;
    println!("{}{message}", c.prefix());
    eprintln!(
        "  source: {:?}, confidence: {:.0}%, took {}ms",
        c.source,
        c.confidence * 100.0,
        started.elapsed().as_millis()
    );
    Ok(())
}

fn run_eval(args: &[&str]) -> Result<(), String> {
    let options = eval::Options::parse(args, 50)?;
    let config = command_config();
    if config.classifier.value == config::Classifier::Local
        && learn::model_path().is_some_and(|p| p.is_file())
    {
        eprintln!(
            "norn: using the model learned from this repository, which has seen these commits \
             unless it was learned with `norn learn --skip N`"
        );
    }
    let types: Vec<String> = config.types.value.iter().map(|t| t.name.clone()).collect();
    // Hook timeouts are tuned for commit latency; replaying history can wait longer.
    let timeout = Some(Duration::from_secs(20));
    eval::run(&options, &types, |sha, message| {
        let changes = diff::Changes::Commit(sha);
        let files = diff::changed_files(changes)?;
        classify_changes(changes, &files, message, &config, timeout)
    })
}

/// Reads `norn export` rows on stdin and writes each one's tokens as a JSON line, or with
/// a model, the model's probability for each class.
fn run_features(model: Option<&Path>) -> Result<(), String> {
    use std::io::{BufRead, Write};
    let model = model
        .map(|path| {
            let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
            model::Model::from_bytes(&bytes)
        })
        .transpose()?;
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    for (n, line) in std::io::stdin().lock().lines().enumerate() {
        let line = line.map_err(|e| e.to_string())?;
        if line.is_empty() {
            continue;
        }
        let row: serde_json::Value =
            serde_json::from_str(&line).map_err(|e| format!("line {}: {e}", n + 1))?;
        let state = &row["state"];
        let files: Vec<diff::ChangedFile> = state["files"]
            .as_array()
            .ok_or(format!("line {}: no state.files", n + 1))?
            .iter()
            .filter_map(|f| {
                let (status, path) = f.as_str()?.split_once(' ')?;
                Some(diff::ChangedFile {
                    status: status.chars().next()?,
                    path: path.to_string(),
                })
            })
            .collect();
        let tokens = features::tokens(
            &files,
            state["patch"].as_str().unwrap_or_default(),
            state["message"].as_str().unwrap_or_default(),
        );
        let value = match &model {
            Some(model) => serde_json::Value::from_iter(
                model.classes.iter().cloned().zip(model.predict(&tokens)),
            ),
            None => serde_json::to_value(&tokens).map_err(|e| e.to_string())?,
        };
        writeln!(out, "{value}").map_err(|e| e.to_string())?;
    }
    out.flush().map_err(|e| e.to_string())
}

fn run_learn(args: &[&str]) -> Result<(), String> {
    let options = learn::Options::parse(args)?;
    let _lock = options.background.then_some(learn::BackgroundLock);
    let config = command_config();
    let started = Instant::now();
    let learned = learn::run(&options, &config)?;
    print_learned(&learned);
    eprintln!("  took {}ms", started.elapsed().as_millis());
    Ok(())
}

fn print_learned(learned: &learn::Learned) {
    let types: Vec<String> = learned
        .types
        .iter()
        .map(|(kind, n)| format!("{kind} {n}"))
        .collect();
    println!(
        "learned from {} commits ({}), saved to {}",
        learned.commits,
        types.join(", "),
        learned.path.display()
    );
    let scoped: usize = learned.scopes.iter().map(|(_, n)| n).sum();
    if learned.scopes.is_empty() {
        println!("this repository doesn't use scopes, so norn won't add one");
    } else {
        let top: Vec<&str> = learned
            .scopes
            .iter()
            .take(6)
            .map(|(s, _)| s.as_str())
            .collect();
        let more = learned.scopes.len().saturating_sub(top.len());
        let more = if more > 0 {
            format!(" and {more} more")
        } else {
            String::new()
        };
        println!(
            "scopes: {}{more}, on {}% of commits",
            top.join(", "),
            100 * scoped / learned.scope_commits.max(1)
        );
    }
}

fn run_export(args: &[&str]) -> Result<(), String> {
    let options = eval::Options::parse(args, usize::MAX)?;
    let config = command_config();
    let types: Vec<String> = config.types.value.iter().map(|t| t.name.clone()).collect();
    let exclude: Vec<String> = config.exclude.iter().map(|s| s.value.clone()).collect();
    eval::export(&options, &types, |sha, message| {
        let changes = diff::Changes::Commit(sha);
        let files = diff::changed_files(changes)?;
        let request =
            classify::request_for(changes, &files, &exclude, message, &config.types.value)?;
        request
            .map(|r| serde_json::to_value(r).map_err(|e| e.to_string()))
            .transpose()
    })
}

/// `timeout` overrides the configured request timeout.
fn classify_changes(
    changes: diff::Changes,
    files: &[diff::ChangedFile],
    message: &str,
    config: &Config,
    timeout: Option<Duration>,
) -> Result<Classification, String> {
    let base_url = &config.base_url.value;
    let api_key = credentials::api_key();
    let client = (api_key.is_some() || !jev::requires_api_key(base_url)).then(|| {
        let timeout = timeout.unwrap_or(Duration::from_millis(config.timeout_ms.value));
        jev::Client::new(api_key, base_url, timeout)
    });
    let settings = Settings {
        breaking_threshold: config.breaking_threshold.value,
        types: &config.types.value,
        scopes: &config.scopes,
        classifier: config.classifier.value,
    };
    let exclude: Vec<String> = config.exclude.iter().map(|s| s.value.clone()).collect();
    classify::classify(
        client.as_ref(),
        changes,
        files,
        &exclude,
        message,
        &settings,
    )
}

fn show_config() -> Result<(), String> {
    let config = command_config();
    let describe = |path: &Option<PathBuf>| match path {
        Some(p) if p.is_file() => p.display().to_string(),
        Some(p) => format!("{} (not found)", p.display()),
        None => "(none)".to_string(),
    };
    println!("global   {}", describe(&config.global_path));
    println!("project  {}", describe(&config.project_path));
    println!();

    let rows = [
        (
            "disable",
            config.disable.value.to_string(),
            config.disable.source,
        ),
        (
            "classifier",
            config.classifier.value.to_string(),
            config.classifier.source,
        ),
        (
            "base_url",
            config.base_url.value.clone(),
            config.base_url.source,
        ),
        (
            "timeout_ms",
            config.timeout_ms.value.to_string(),
            config.timeout_ms.source,
        ),
        (
            "deadline_ms",
            config.deadline_ms.value.to_string(),
            config.deadline_ms.source,
        ),
        (
            "min_confidence",
            config.min_confidence.value.to_string(),
            config.min_confidence.source,
        ),
        (
            "breaking_threshold",
            config.breaking_threshold.value.to_string(),
            config.breaking_threshold.source,
        ),
    ];
    for (key, value, source) in rows {
        println!("{key:<20}{value:<30}  {source}");
    }
    match learn::model_path().filter(|p| p.is_file()) {
        Some(path) => {
            let age = learn::commits_since_learned()
                .map(|n| format!(", learned {n} commits ago"))
                .unwrap_or_default();
            println!(
                "{:<20}{:<30}  norn learn",
                "model",
                format!("{}{age}", path.display())
            );
        }
        None => println!(
            "{:<20}{:<30}  {}",
            "model",
            "built-in",
            config::Source::Default
        ),
    }
    if config.exclude.is_empty() {
        println!("{:<20}{:<30}  {}", "exclude", "[]", config::Source::Default);
    }
    for (i, pattern) in config.exclude.iter().enumerate() {
        let key = if i == 0 { "exclude" } else { "" };
        println!("{key:<20}{:<30}  {}", pattern.value, pattern.source);
    }
    let types: Vec<&str> = config.types.value.iter().map(|t| t.name.as_str()).collect();
    println!(
        "{:<20}{:<30}  {}",
        "types",
        types.join(", "),
        config.types.source
    );
    if config.scopes.is_empty() {
        let learned = learn::model_path()
            .map(|p| p.with_file_name("scopes.bin"))
            .is_some_and(|p| p.is_file());
        if learned {
            println!("{:<20}{:<30}  norn learn", "scopes", "(learned)");
        } else {
            let source = config::Source::Default;
            println!("{:<20}{:<30}  {source}", "scopes", "(shared directory)");
        }
    }
    for (i, rule) in config.scopes.iter().enumerate() {
        let key = if i == 0 { "scopes" } else { "" };
        let mapping = format!("{} → {}", rule.prefix, rule.scope);
        println!("{key:<20}{mapping:<30}  {}", rule.source);
    }
    Ok(())
}

fn login() -> Result<(), String> {
    let dir = config::dir().ok_or("couldn't find a config directory (HOME is not set)")?;
    let key = read_key()?;
    if key.is_empty() {
        return Err("no key entered".into());
    }
    let path = credentials::save(&dir, &key)?;
    println!("saved API key to {}", path.display());
    if std::env::var_os("TYPESAFE_API_KEY").is_some_and(|v| !v.is_empty()) {
        eprintln!("note: TYPESAFE_API_KEY is set in this shell and takes precedence");
    }
    Ok(())
}

fn logout() -> Result<(), String> {
    let dir = config::dir().ok_or("couldn't find a config directory (HOME is not set)")?;
    match credentials::delete(&dir)? {
        Some(path) => println!("removed {}", path.display()),
        None => println!("no saved API key"),
    }
    Ok(())
}

/// Reads the key from stdin: hidden when typed or pasted at a terminal, or piped in
/// (`echo "$KEY" | norn jev login`).
fn read_key() -> Result<String, String> {
    use std::io::{BufRead, IsTerminal};
    let stdin = std::io::stdin();
    let interactive = stdin.is_terminal();
    if interactive {
        // Echo off before the prompt appears, so fast pastes can't be echoed.
        set_echo(false);
        eprint!("Paste your TypeSafe API key: ");
    }
    let mut line = String::new();
    let result = stdin.lock().read_line(&mut line);
    if interactive {
        set_echo(true);
        eprintln!();
    }
    result.map_err(|e| e.to_string())?;
    Ok(line.trim().to_string())
}

#[cfg(unix)]
fn set_echo(on: bool) {
    let _ = Command::new("stty")
        .arg(if on { "echo" } else { "-echo" })
        .stdin(Stdio::inherit())
        .status();
}

#[cfg(not(unix))]
fn set_echo(_on: bool) {}

fn install() -> Result<(), String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let hooks_dir = git_hooks_dir()?;
    std::fs::create_dir_all(&hooks_dir).map_err(|e| e.to_string())?;

    for hook in ["prepare-commit-msg", "commit-msg"] {
        let path = hooks_dir.join(hook);
        if let Ok(existing) = std::fs::read_to_string(&path)
            && !existing.contains(HOOK_MARKER)
        {
            return Err(format!(
                "{} already exists and wasn't installed by norn; add `norn {hook} \"$@\"` to it instead",
                path.display()
            ));
        }
        std::fs::write(&path, hook_script(hook, &exe)).map_err(|e| e.to_string())?;
        make_executable(&path)?;
        println!("installed {}", path.display());
    }

    let config = command_config();
    // Learned scopes are used with either classifier.
    match learn::run(&learn::Options::default(), &config) {
        Ok(learned) => print_learned(&learned),
        Err(e) => println!("not learning from history yet: {e}"),
    }
    Ok(())
}

/// Hooks run the first candidate that exists and always exit 0, so a missing or broken
/// binary (wrong architecture, truncated download) never blocks a commit. It isn't
/// `exec`ed: a failed `exec` would end the hook with a non-zero status.
///
/// The first candidate is an absolute path, so hooks work from git GUIs that don't
/// inherit the shell's PATH. When the binary running `install` is the one on PATH, that
/// PATH entry is used (e.g. `/opt/homebrew/bin/norn`) rather than `current_exe()`,
/// which can resolve into a versioned directory that disappears on upgrade. The bare
/// name is the last resort for when that path moves.
fn hook_script(hook: &str, exe: &Path) -> String {
    let stable = match find_on_path(BIN_NAME) {
        Some(found) if same_file(&found, exe) => found,
        _ => exe.to_path_buf(),
    };
    let candidates = format!("\"{}\" {BIN_NAME}", stable.display());
    format!(
        "#!/bin/sh\n\
         {HOOK_MARKER}\n\
         for bin in {candidates}; do\n\
         \x20   if command -v \"$bin\" >/dev/null 2>&1; then\n\
         \x20       \"$bin\" {hook} \"$@\"\n\
         \x20       exit 0\n\
         \x20   fi\n\
         done\n\
         exit 0\n"
    )
}

fn find_on_path(name: &str) -> Option<PathBuf> {
    std::env::split_paths(&std::env::var_os("PATH")?)
        .map(|dir| dir.join(name))
        .find(|candidate| candidate.is_file())
}

fn same_file(a: &Path, b: &Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}

fn git_hooks_dir() -> Result<PathBuf, String> {
    let output = Command::new("git")
        .args(["rev-parse", "--git-path", "hooks"])
        .output()
        .map_err(|e| format!("failed to run git: {e}"))?;
    if !output.status.success() {
        return Err("not inside a git repository".into());
    }
    Ok(PathBuf::from(
        String::from_utf8_lossy(&output.stdout).trim(),
    ))
}

#[cfg(unix)]
fn make_executable(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755))
        .map_err(|e| e.to_string())
}

#[cfg(not(unix))]
fn make_executable(_path: &Path) -> Result<(), String> {
    Ok(())
}
