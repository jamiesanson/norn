mod classify;
mod config;
mod credentials;
mod diff;
mod eval;
mod jev;
mod message;

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};
use std::time::{Duration, Instant};

use classify::{Classification, Settings};
use config::Config;

const USAGE: &str = "\
jev-cc: prefix commit messages with Conventional Commits syntax using TypeSafe's Jev

USAGE:
    jev-cc login                         Save your TypeSafe API key
    jev-cc logout                        Remove the saved API key
    jev-cc install                       Install the git hooks into the current repository
    jev-cc classify [MESSAGE]            Classify the staged diff and print the prefix
    jev-cc config                        Show the settings in effect and where each comes from
    jev-cc eval [OPTIONS] [REV]          Compare jev-cc's choices with the prefixes in REV's history
        --limit N      Commits to evaluate (default: 50)
        --message      Send each commit's description, as `git commit -m` would
        --out FILE     Also write each result as a JSON line
    jev-cc prepare-commit-msg FILE [SOURCE] [SHA]   (git hook)
    jev-cc commit-msg FILE                          (git hook)

CONFIGURATION:
    Global:  ~/.config/jev-cc/config.toml
    Project: .jev-cc.toml at the repository root (can't set base_url)
    Environment variables override both:

    TYPESAFE_API_KEY             API key for Jev; overrides the key saved by `jev-cc login`
    JEV_CC_BASE_URL              API base URL, or a local Laya server (default: https://api.typesafe.ai)
    JEV_CC_TIMEOUT_MS            Request timeout; the message is left as-is on timeout (default: 1000)
    JEV_CC_DEADLINE_MS           Time limit for the whole hook (default: 2000)
    JEV_CC_MIN_CONFIDENCE        Minimum confidence to apply a type (default: 0.6)
    JEV_CC_BREAKING_THRESHOLD    Probability needed to mark a change breaking (default: 0.85)
    JEV_CC_DISABLE               Set to 1 to skip the hook entirely
";

const HOOK_MARKER: &str = "# installed by jev-cc";
const BIN_NAME: &str = if cfg!(windows) {
    "jev-cc.exe"
} else {
    "jev-cc"
};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let args: Vec<&str> = args.iter().map(String::as_str).collect();

    match args.as_slice() {
        ["prepare-commit-msg", file, rest @ ..] => {
            let config = hook_config();
            start_deadline(Path::new(file), config.deadline_ms.value);
            hook_result(prepare_commit_msg(
                Path::new(file),
                rest.first().copied(),
                &config,
            ))
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
        ["install"] => command_result(install()),
        ["login"] => command_result(login()),
        ["logout"] => command_result(logout()),
        ["--version" | "-V"] => {
            println!("jev-cc {}", env!("CARGO_PKG_VERSION"));
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
            eprintln!("jev-cc: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Hooks never block a commit: failures are reported and the message is left alone.
fn hook_result(result: Result<(), String>) -> ExitCode {
    if let Err(e) = result {
        eprintln!("jev-cc: message left unchanged ({e})");
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
        eprintln!("jev-cc: {warning}");
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
/// jev-cc can never hang a commit. Message writes are atomic, so exiting at any point
/// leaves either the original message or the prefixed one.
fn start_deadline(message_file: &Path, deadline_ms: u64) {
    let deadline = Duration::from_millis(deadline_ms);
    let tmp = tmp_path(message_file);
    std::thread::spawn(move || {
        std::thread::sleep(deadline);
        let _ = std::fs::remove_file(&tmp);
        eprintln!(
            "jev-cc: message left unchanged (took longer than {}ms)",
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
    tmp.push(".jev-cc.tmp");
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
    let options = eval::Options::parse(args)?;
    let config = command_config();
    let types: Vec<String> = config.types.value.iter().map(|t| t.name.clone()).collect();
    // Hook timeouts are tuned for commit latency; replaying history can wait longer.
    let timeout = Some(Duration::from_secs(20));
    eval::run(&options, &types, |sha, message| {
        let changes = diff::Changes::Commit(sha);
        let files = diff::changed_files(changes)?;
        classify_changes(changes, &files, message, &config, timeout)
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
        println!(
            "{:<20}{:<30}  {}",
            "scopes",
            "(shared directory)",
            config::Source::Default
        );
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
/// (`echo "$KEY" | jev-cc login`).
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
                "{} already exists and wasn't installed by jev-cc; add `jev-cc {hook} \"$@\"` to it instead",
                path.display()
            ));
        }
        std::fs::write(&path, hook_script(hook, &exe)).map_err(|e| e.to_string())?;
        make_executable(&path)?;
        println!("installed {}", path.display());
    }
    Ok(())
}

/// Hooks run the first candidate that exists and always exit 0, so a missing or broken
/// binary (wrong architecture, truncated download) never blocks a commit. It isn't
/// `exec`ed: a failed `exec` would end the hook with a non-zero status.
///
/// The first candidate is an absolute path, so hooks work from git GUIs that don't
/// inherit the shell's PATH. When the binary running `install` is the one on PATH, that
/// PATH entry is used (e.g. `/opt/homebrew/bin/jev-cc`) rather than `current_exe()`,
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
