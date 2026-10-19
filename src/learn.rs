//! `norn learn`: fits the built-in model to this repository's own history, so it picks up
//! the project's conventions: what it calls `chore`, its own types, its directory names.
//!
//! The fit starts from the built-in model and refits on the repository's labelled commits,
//! penalising distance from the built-in weights:
//!
//!     C * sum(cross-entropy) + 1/2 ||theta - theta_built_in||^2
//!
//! Terms the built-in model has never seen become new features, and types it doesn't know
//! become new classes, both starting at zero. The result is a complete model, saved in the
//! repository's git directory and used in place of the built-in one.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::time::Duration;

use crate::classify;
use crate::config::Config;
use crate::diff::{self, ChangedFile, Changes};
use crate::eval;
use crate::features::{self, Tokens};
use crate::labels;
use crate::lbfgs;
use crate::model::{self, Model};

/// Fewer usable commits than this aren't enough to learn from.
pub const MIN_COMMITS: usize = 20;
/// How strongly the repository's commits pull against the built-in weights.
const C: f64 = 15.0;
/// Bots don't commit through the hook, but their commits still show the project's
/// conventions, such as what it calls an update to generated files. The newest are kept,
/// up to this share of the commits learned from, so they can't swamp the rest.
const MAX_BOT_SHARE: usize = 10;
/// At most this many commits with the same subject (numbers aside), so templated commits
/// (`chore: release v1.2.3`, `chore: update translations`) don't drown out the rest.
const MAX_SAME_SUBJECT: usize = 3;
/// A term the built-in model lacks becomes a feature once this many commits use it.
const MIN_DF: usize = 2;
const MAX_ITERATIONS: usize = 300;
/// The scope model's class for commits without a scope. Not a valid scope itself.
const NO_SCOPE: &str = "(none)";
/// A commit this many commits older counts half as much towards the scope model.
const SCOPE_HALF_LIFE: f64 = 100.0;
const SCOPE_C: f64 = 10.0;
/// The hook re-learns in the background once this many commits have landed since the last
/// time, so the models keep up as a project's habits drift.
const REFRESH_AFTER: usize = 50;
/// A background learn that hasn't finished after this long is assumed to have died.
const LOCK_TIMEOUT: Duration = Duration::from_secs(600);

/// A commit's scope tokens and its scope, if it has one.
pub type ScopeDoc = (Vec<String>, Option<String>);

pub struct Options {
    pub limit: usize,
    /// Leave out the newest commits, so `norn eval` can be run on ones the model hasn't seen.
    pub skip: usize,
    /// Started by the hook; releases the background lock when done.
    pub background: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            limit: 1000,
            skip: 0,
            background: false,
        }
    }
}

impl Options {
    pub fn parse(args: &[&str]) -> Result<Options, String> {
        let mut options = Options::default();
        let mut args = args.iter();
        while let Some(&arg) = args.next() {
            let mut number = || -> Result<usize, String> {
                args.next()
                    .ok_or(format!("{arg} needs a value"))?
                    .parse()
                    .map_err(|_| format!("{arg} must be a number"))
            };
            match arg {
                "--limit" => options.limit = number()?,
                "--skip" => options.skip = number()?,
                "--background" => options.background = true,
                _ => return Err(format!("unknown option {arg}")),
            }
        }
        Ok(options)
    }
}

pub struct Learned {
    pub commits: usize,
    /// Each type and how many commits had it, most common first.
    pub types: Vec<(String, usize)>,
    /// Each scope and how many commits had it, most common first.
    pub scopes: Vec<(String, usize)>,
    /// Commits the scope model learned from, with or without a scope.
    pub scope_commits: usize,
    /// How many of those norn labelled itself.
    pub own_labels: usize,
    pub path: PathBuf,
}

pub fn run(options: &Options, config: &Config) -> Result<Learned, String> {
    let types: Vec<String> = config.types.value.iter().map(|t| t.name.clone()).collect();
    let exclude: Vec<String> = config.exclude.iter().map(|s| s.value.clone()).collect();
    let (commits, _) =
        eval::labelled_commits("HEAD", options.skip.saturating_add(options.limit), &types)?;
    let mut seen: HashMap<String, usize> = HashMap::new();
    let commits: Vec<eval::Labelled> = commits
        .into_iter()
        .skip(options.skip)
        .filter(|c| {
            let n = seen.entry(template(&c.subject)).or_default();
            *n += 1;
            *n <= MAX_SAME_SUBJECT
        })
        .collect();
    let commits = limit_bots(commits);

    // Commits norn labelled itself count for less, unless you corrected them.
    let label_weights = labels::weights();
    let read = eval::map_commits(&commits, |commit| {
        let changes = Changes::Commit(&commit.sha);
        let files = diff::changed_files(changes)?;
        let message = commit.message(with_message(&commit.sha));
        let tokens =
            classify::model_tokens(changes, &files, &exclude, message, &config.types.value)?;
        let label = commit.label();
        let weight = label_weights.get(&commit.sha).copied().unwrap_or(1.0);
        Ok::<_, String>((
            tokens.map(|t| (t, label.kind.to_string(), weight)),
            (
                features::scope_tokens(&files),
                label.scope.map(str::to_string),
                weight,
            ),
        ))
    });
    // Commits the local rules answer still say which scope their paths belong to.
    let (docs, scope_docs): (Vec<_>, Vec<_>) = read.into_iter().filter_map(Result::ok).unzip();
    let (docs, weights): (Vec<(Tokens, String)>, Vec<f64>) = docs
        .into_iter()
        .flatten()
        .map(|(t, kind, w)| ((t, kind), w))
        .unzip();
    let own_labels = scope_docs.iter().filter(|(_, _, w)| *w != 1.0).count();
    let (scope_docs, scope_weights): (Vec<ScopeDoc>, Vec<f64>) =
        scope_docs.into_iter().map(|(t, s, w)| ((t, s), w)).unzip();
    if docs.len() < MIN_COMMITS {
        // Recorded anyway, so the hook tries again once the history has grown.
        mark_learned(options.skip);
        let s = if docs.len() == 1 { "" } else { "s" };
        return Err(format!(
            "only {} conventional commit{s} to learn from, and norn needs {MIN_COMMITS}",
            docs.len()
        ));
    }

    let learned = finetune(model::shared()?, &docs, &weights);
    let path = model_path().ok_or("not in a git repository")?;
    save(&path, &learned.to_bytes())?;
    save(
        &scope_model_path(&path),
        &scope_model(&scope_docs, &scope_weights).to_bytes(),
    )?;
    mark_learned(options.skip);

    let mut counts: HashMap<&str, usize> = HashMap::new();
    for (_, kind) in &docs {
        *counts.entry(kind).or_default() += 1;
    }
    let mut types: Vec<(String, usize)> = counts
        .into_iter()
        .map(|(k, n)| (k.to_string(), n))
        .collect();
    types.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for scope in scope_docs.iter().filter_map(|(_, s)| s.as_deref()) {
        *counts.entry(scope).or_default() += 1;
    }
    let mut scopes: Vec<(String, usize)> = counts
        .into_iter()
        .map(|(k, n)| (k.to_string(), n))
        .collect();
    scopes.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    Ok(Learned {
        commits: docs.len(),
        types,
        scopes,
        scope_commits: scope_docs.len(),
        own_labels,
        path,
    })
}

/// `commits` with bot commits after the first `MAX_BOT_SHARE`% dropped.
fn limit_bots(commits: Vec<eval::Labelled>) -> Vec<eval::Labelled> {
    let humans = commits.iter().filter(|c| !c.bot).count();
    let allowed = humans * MAX_BOT_SHARE / (100 - MAX_BOT_SHARE);
    let mut bots = 0;
    commits
        .into_iter()
        .filter(|c| {
            if c.bot {
                bots += 1;
            }
            !c.bot || bots <= allowed
        })
        .collect()
}

/// A subject with its numbers and pull request reference blanked out, so
/// `chore: release v1.2.3 (#456)` and `chore: release v1.10.0 (#7)` match.
fn template(subject: &str) -> String {
    let mut out = String::new();
    for c in subject.trim().to_lowercase().chars() {
        if !c.is_ascii_digit() {
            out.push(c);
        } else if !out.ends_with('#') {
            out.push('#');
        }
    }
    out.trim_end_matches(" (##)")
        .trim_end_matches(" (#)")
        .to_string()
}

/// Half the commits are learned with their description, as `git commit -m` sends it, and
/// half without, as the editor flow does. The sha decides, so a rerun makes the same choice.
fn with_message(sha: &str) -> bool {
    sha.bytes().last().is_some_and(|b| b % 2 == 0)
}

/// Where the learned model lives: in the git directory, shared by every worktree.
pub fn model_path() -> Option<PathBuf> {
    // Asked several times per commit, and the answer can't change while norn runs.
    static GIT_DIR: OnceLock<Option<PathBuf>> = OnceLock::new();
    let git_dir =
        GIT_DIR.get_or_init(|| git_output(&["rev-parse", "--git-common-dir"]).map(PathBuf::from));
    Some(git_dir.as_ref()?.join("norn").join("model.bin"))
}

/// norn's directory inside the repository's git directory.
fn norn_dir() -> Option<PathBuf> {
    model_path()?.parent().map(Path::to_path_buf)
}

fn git_output(args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .args(args)
        .stderr(Stdio::null())
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Notes the commit norn last learned at, so the hook can tell when the models are behind.
/// A learn that skipped the newest commits, to evaluate on them, leaves models that are
/// behind on purpose, so the note is removed instead and the next commit re-learns.
fn mark_learned(skipped: usize) {
    let Some(dir) = norn_dir() else { return };
    let marker = dir.join("learned");
    if skipped > 0 {
        let _ = std::fs::remove_file(marker);
    } else if let Some(head) = git_output(&["rev-parse", "HEAD"]) {
        let _ = std::fs::create_dir_all(&dir);
        let _ = std::fs::write(marker, head + "\n");
    }
}

/// Commits on `HEAD` since norn last learned, or `None` if it never has here.
pub fn commits_since_learned() -> Option<usize> {
    let sha = std::fs::read_to_string(norn_dir()?.join("learned")).ok()?;
    git_output(&["rev-list", "--count", &format!("{}..HEAD", sha.trim())])?
        .parse()
        .ok()
}

/// Whether the hook should start a background learn: the models are behind, or have never
/// been learned, and no other learn has started in the last `LOCK_TIMEOUT`.
fn should_refresh(commits_behind: Option<usize>, lock_age: Option<Duration>) -> bool {
    commits_behind.is_none_or(|n| n >= REFRESH_AFTER)
        && lock_age.is_none_or(|age| age >= LOCK_TIMEOUT)
}

/// Starts `norn learn` in the background when the learned models are out of date, without
/// waiting for it, so the commit isn't slowed down. Its output goes to `learn.log`.
pub fn refresh_if_stale() {
    let Some(dir) = norn_dir() else { return };
    let lock = dir.join("learning");
    let lock_age = std::fs::metadata(&lock)
        .and_then(|m| m.modified())
        .ok()
        .map(|t| t.elapsed().unwrap_or_default());
    if !should_refresh(commits_since_learned(), lock_age) {
        return;
    }
    let (Ok(exe), Ok(())) = (std::env::current_exe(), std::fs::create_dir_all(&dir)) else {
        return;
    };
    let Ok(log) = std::fs::File::create(dir.join("learn.log")) else {
        return;
    };
    let Ok(log_err) = log.try_clone() else { return };
    if std::fs::write(&lock, "").is_err() {
        return;
    }
    let mut command = Command::new(exe);
    command
        .args(["learn", "--background"])
        .stdin(Stdio::null())
        .stdout(log)
        .stderr(log_err);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    if command.spawn().is_err() {
        let _ = std::fs::remove_file(&lock);
    }
}

/// Releases the background lock when a learn the hook started finishes, however it ends.
pub struct BackgroundLock;

impl Drop for BackgroundLock {
    fn drop(&mut self) {
        if let Some(dir) = norn_dir() {
            let _ = std::fs::remove_file(dir.join("learning"));
        }
    }
}

/// The learned scope model, beside the learned type model.
fn scope_model_path(model_path: &std::path::Path) -> PathBuf {
    model_path.with_file_name("scopes.bin")
}

/// This repository's learned scope model, if `norn learn` has made one.
pub fn learned_scopes() -> Option<&'static Model> {
    static SCOPES: OnceLock<Option<Model>> = OnceLock::new();
    SCOPES
        .get_or_init(|| {
            let bytes = std::fs::read(scope_model_path(&model_path()?)).ok()?;
            Model::from_bytes(&bytes).ok()
        })
        .as_ref()
}

/// The model `classifier = "local"` uses: this repository's learned one if there is one,
/// otherwise the built-in model.
pub fn active_model() -> Result<&'static Model, String> {
    static LEARNED: OnceLock<Option<Model>> = OnceLock::new();
    let learned = LEARNED.get_or_init(|| {
        let bytes = std::fs::read(model_path()?).ok()?;
        Model::from_bytes(&bytes).ok()
    });
    match learned {
        Some(model) => Ok(model),
        None => model::shared(),
    }
}

fn save(path: &std::path::Path, bytes: &[u8]) -> Result<(), String> {
    let dir = path.parent().expect("the path has a directory");
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let tmp = path.with_extension("bin.tmp");
    std::fs::write(&tmp, bytes)
        .and_then(|()| std::fs::rename(&tmp, path))
        .map_err(|e| {
            let _ = std::fs::remove_file(&tmp);
            format!("{}: {e}", path.display())
        })
}

/// `base` refitted on `docs`, each a commit's tokens and its type, counted by `weights`.
pub fn finetune(base: &Model, docs: &[(Tokens, String)], weights: &[f64]) -> Model {
    let mut model = extend(base, docs);
    let labels: Vec<usize> = docs
        .iter()
        .map(|(_, kind)| model.classes.iter().position(|c| c == kind).unwrap())
        .collect();
    let rows: Vec<Vec<(usize, f64)>> = docs.iter().map(|(t, _)| model.vectorize(t)).collect();
    fit(&mut model, &rows, &labels, weights, C, true);
    model
}

/// Fits `model`'s weights by L-BFGS, minimising
///
///     c * sum(weight * cross-entropy) + 1/2 ||W - W_now||^2
///
/// plus the same penalty on the intercepts when `penalise_intercepts`. The model's current
/// weights are the prior. `rows` are vectorised commits and `labels` their class indices.
fn fit(
    model: &mut Model,
    rows: &[Vec<(usize, f64)>],
    labels: &[usize],
    weights: &[f64],
    c: f64,
    penalise_intercepts: bool,
) {
    let k = model.classes.len();
    // A feature no commit uses gets no pull from the data, so the penalty holds it at its
    // prior value. Only the features that appear need fitting.
    let mut active: Vec<usize> = rows.iter().flatten().map(|&(f, _)| f).collect();
    active.sort_unstable();
    active.dedup();
    let slot: HashMap<usize, usize> = active.iter().enumerate().map(|(i, &f)| (f, i)).collect();
    let rows: Vec<Vec<(usize, f64)>> = rows
        .iter()
        .map(|r| r.iter().map(|&(f, v)| (slot[&f], v)).collect())
        .collect();

    // The parameters: each active feature's weight per class, then the intercepts.
    let a = active.len();
    let mut prior: Vec<f64> = Vec::with_capacity(a * k + k);
    for &f in &active {
        prior.extend(model.weights[f * k..(f + 1) * k].iter().map(|&w| w as f64));
    }
    prior.extend(model.intercepts.iter().map(|&b| b as f64));
    let penalised = if penalise_intercepts {
        a * k + k
    } else {
        a * k
    };

    let theta = lbfgs::minimize(
        prior.clone(),
        |theta, grad| {
            let mut value = 0.0;
            grad.fill(0.0);
            for i in 0..penalised {
                let d = theta[i] - prior[i];
                value += 0.5 * d * d;
                grad[i] = d;
            }
            let (w, b) = theta.split_at(a * k);
            let (gw, gb) = grad.split_at_mut(a * k);
            let mut scores = vec![0.0; k];
            for ((x, &y), &weight) in rows.iter().zip(labels).zip(weights) {
                scores.copy_from_slice(b);
                for &(s, v) in x {
                    for c in 0..k {
                        scores[c] += v * w[s * k + c];
                    }
                }
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let log_total = scores.iter().map(|s| (s - max).exp()).sum::<f64>().ln() + max;
                let scale = c * weight;
                value -= scale * (scores[y] - log_total);
                for class in 0..k {
                    let error = scale * ((scores[class] - log_total).exp() - f64::from(class == y));
                    gb[class] += error;
                    for &(s, v) in x {
                        gw[s * k + class] += error * v;
                    }
                }
            }
            value
        },
        MAX_ITERATIONS,
    );

    for (i, &f) in active.iter().enumerate() {
        for class in 0..k {
            model.weights[f * k + class] = theta[i * k + class] as f32;
        }
    }
    for class in 0..k {
        model.intercepts[class] = theta[a * k + class] as f32;
    }
}

/// A model of which scope this repository files changes under, from `docs`: each commit's
/// scope tokens and scope, newest first. Newer commits count for more, because projects'
/// habits drift. Scopes are the project's own, so it starts from nothing.
pub fn scope_model(docs: &[ScopeDoc], weights: &[f64]) -> Model {
    let label = |s: &Option<String>| s.clone().unwrap_or_else(|| NO_SCOPE.to_string());
    let mut counts: HashMap<String, usize> = HashMap::new();
    for (_, scope) in docs {
        *counts.entry(label(scope)).or_default() += 1;
    }
    let mut classes: Vec<(String, usize)> = counts.into_iter().collect();
    classes.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let classes: Vec<String> = classes.into_iter().map(|(c, _)| c).collect();

    let mut df: HashMap<&str, usize> = HashMap::new();
    for (tokens, _) in docs {
        for t in tokens.iter().map(String::as_str).collect::<HashSet<_>>() {
            *df.entry(t).or_default() += 1;
        }
    }
    let mut terms: Vec<(&str, usize)> = df.into_iter().collect();
    terms.sort_unstable();
    let n = docs.len() as f64;
    let idf = terms
        .iter()
        .map(|&(_, d)| (((1.0 + n) / (1.0 + d as f64)).ln() + 1.0) as f32)
        .collect();
    let k = classes.len();
    let mut model = Model::new(
        classes,
        vec![(
            "scope".to_string(),
            1.0,
            terms.iter().map(|(t, _)| t.to_string()).collect(),
        )],
        idf,
        vec![0.0; terms.len() * k],
        vec![0.0; k],
    )
    .expect("sizes are consistent");

    let rows: Vec<Vec<(usize, f64)>> = docs
        .iter()
        .map(|(scope, _)| {
            model.vectorize(&Tokens {
                scope: scope.clone(),
                ..Tokens::default()
            })
        })
        .collect();
    let labels: Vec<usize> = docs
        .iter()
        .map(|(_, s)| model.classes.iter().position(|c| *c == label(s)).unwrap())
        .collect();
    let weights: Vec<f64> = weights
        .iter()
        .enumerate()
        .map(|(i, w)| w * 0.5f64.powf(i as f64 / SCOPE_HALF_LIFE))
        .collect();
    fit(&mut model, &rows, &labels, &weights, SCOPE_C, false);
    model
}

/// The scope `model` picks for a change, if any.
pub fn predict_scope(model: &Model, files: &[ChangedFile]) -> Option<String> {
    let p = model.predict(&Tokens {
        scope: features::scope_tokens(files),
        ..Tokens::default()
    });
    let best = p
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| &model.classes[i])?;
    (best != NO_SCOPE).then(|| best.clone())
}

/// `base` with the types and terms `docs` use that it doesn't know, all weighted zero.
fn extend(base: &Model, docs: &[(Tokens, String)]) -> Model {
    let mut classes = base.classes.clone();
    for (_, kind) in docs {
        if !classes.contains(kind) {
            classes.push(kind.clone());
        }
    }
    let (k0, k) = (base.classes.len(), classes.len());
    let n = docs.len() as f64;

    let mut blocks = Vec::new();
    let mut idf = Vec::new();
    let mut weights = Vec::new();
    for block in &base.blocks {
        let mut df: HashMap<&str, usize> = HashMap::new();
        for (tokens, _) in docs {
            let unknown: HashSet<&str> = tokens
                .block(&block.name)
                .iter()
                .map(String::as_str)
                .filter(|t| block.feature(t).is_none())
                .collect();
            for t in unknown {
                *df.entry(t).or_default() += 1;
            }
        }
        let mut new: Vec<(&str, usize)> = df.into_iter().filter(|&(_, d)| d >= MIN_DF).collect();
        new.sort_unstable();

        let mut terms = block.terms.clone();
        for f in block.first..block.first + block.terms.len() {
            idf.push(base.idf[f]);
            weights.extend(&base.weights[f * k0..(f + 1) * k0]);
            weights.extend(std::iter::repeat_n(0.0, k - k0));
        }
        for (term, d) in new {
            terms.push(term.to_string());
            // scikit-learn's smoothed idf, over this repository's commits.
            idf.push((((1.0 + n) / (1.0 + d as f64)).ln() + 1.0) as f32);
            weights.extend(std::iter::repeat_n(0.0, k));
        }
        blocks.push((block.name.clone(), block.weight, terms));
    }
    let mut intercepts = base.intercepts.clone();
    intercepts.resize(k, 0.0);
    Model::new(classes, blocks, idf, weights, intercepts).expect("extend keeps sizes consistent")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(msg: &[&str], paths: &[&str]) -> Tokens {
        Tokens {
            msg: msg.iter().map(|s| s.to_string()).collect(),
            paths: paths.iter().map(|s| s.to_string()).collect(),
            ..Tokens::default()
        }
    }

    fn tiny() -> Model {
        Model::new(
            vec!["feat".into(), "fix".into()],
            vec![
                ("msg".into(), 1.0, vec!["m:add".into(), "m:fix".into()]),
                ("paths".into(), 1.0, vec!["dir:src".into()]),
            ],
            vec![1.0, 1.0, 1.0],
            vec![1.0, -1.0, -1.0, 1.0, 0.5, 0.5],
            vec![0.0, 0.0],
        )
        .unwrap()
    }

    #[test]
    fn learns_new_terms_and_types() {
        // This repository calls dependency bumps `deps`, under a `bump` verb the base
        // model has never seen.
        let mut docs = Vec::new();
        for _ in 0..10 {
            docs.push((tokens(&["m:bump"], &["dir:src"]), "deps".to_string()));
            docs.push((tokens(&["m:fix"], &["dir:src"]), "fix".to_string()));
        }
        let learned = finetune(&tiny(), &docs, &vec![1.0; docs.len()]);
        assert_eq!(learned.classes, ["feat", "fix", "deps"]);
        let p = learned.predict(&tokens(&["m:bump"], &["dir:src"]));
        assert!(p[2] > 0.8, "{p:?}");
        let p = learned.predict(&tokens(&["m:fix"], &[]));
        assert!(p[1] > p[0] && p[1] > p[2], "{p:?}");
    }

    #[test]
    fn unused_features_keep_their_weights() {
        let docs: Vec<_> = (0..5)
            .map(|_| (tokens(&["m:fix"], &[]), "fix".to_string()))
            .collect();
        let base = tiny();
        let learned = finetune(&base, &docs, &vec![1.0; docs.len()]);
        let add = base.blocks[0].feature("m:add").unwrap();
        let k = base.classes.len();
        assert_eq!(
            learned.weights[add * k..(add + 1) * k],
            base.weights[add * k..(add + 1) * k]
        );
    }

    #[test]
    fn rare_unknown_terms_stay_out() {
        let docs = vec![
            (tokens(&["m:once"], &[]), "fix".to_string()),
            (tokens(&["m:twice"], &[]), "fix".to_string()),
            (tokens(&["m:twice"], &[]), "fix".to_string()),
        ];
        let extended = extend(&tiny(), &docs);
        assert!(extended.blocks[0].feature("m:twice").is_some());
        assert!(extended.blocks[0].feature("m:once").is_none());
    }

    fn file(path: &str) -> ChangedFile {
        ChangedFile {
            status: 'M',
            path: path.into(),
        }
    }

    #[test]
    fn learns_scopes_and_their_absence() {
        let mut docs = Vec::new();
        for _ in 0..10 {
            for (path, scope) in [
                (
                    "packages/runtime-core/src/apiWatch.ts",
                    Some("runtime-core"),
                ),
                ("packages/compiler-sfc/src/parse.ts", Some("compiler-sfc")),
                ("scripts/release.js", None),
            ] {
                docs.push((
                    features::scope_tokens(&[file(path)]),
                    scope.map(String::from),
                ));
            }
        }
        let model = scope_model(&docs, &vec![1.0; docs.len()]);
        let scope = |path| predict_scope(&model, &[file(path)]);
        assert_eq!(
            scope("packages/runtime-core/src/hydration.ts").as_deref(),
            Some("runtime-core")
        );
        assert_eq!(
            scope("packages/compiler-sfc/src/compileScript.ts").as_deref(),
            Some("compiler-sfc")
        );
        assert_eq!(scope("scripts/build.js"), None);
    }

    #[test]
    fn newer_scope_habits_win() {
        // The newest 80 of 200 commits use a scope that older ones didn't.
        let mut docs = Vec::new();
        for i in 0..200 {
            let scope = (i < 80).then(|| "editor".to_string());
            docs.push((
                features::scope_tokens(&[file("packages/editor/src/App.tsx")]),
                scope,
            ));
        }
        let model = scope_model(&docs, &vec![1.0; docs.len()]);
        assert_eq!(
            predict_scope(&model, &[file("packages/editor/src/Canvas.tsx")]).as_deref(),
            Some("editor")
        );
    }

    #[test]
    fn bots_are_a_small_share() {
        let commits: Vec<eval::Labelled> = (0..100)
            .map(|i| eval::Labelled {
                sha: i.to_string(),
                subject: format!("fix: {i}"),
                bot: i % 2 == 0,
            })
            .collect();
        let kept = limit_bots(commits);
        let bots = kept.iter().filter(|c| c.bot).count();
        assert_eq!((kept.len() - bots, bots), (50, 5));
        assert!(
            kept.iter()
                .filter(|c| c.bot)
                .all(|c| c.sha.parse::<u32>().unwrap() < 10)
        );
    }

    #[test]
    fn templates_ignore_numbers() {
        assert_eq!(
            template("chore: release v1.2.3 (#456)"),
            template("chore: Release v1.10.0 (#7)")
        );
        assert_ne!(template("fix: a"), template("fix: b"));
    }

    #[test]
    fn refreshes_when_behind_and_not_already_learning() {
        let minute = Duration::from_secs(60);
        assert!(should_refresh(None, None), "never learned");
        assert!(should_refresh(Some(REFRESH_AFTER), None));
        assert!(!should_refresh(Some(REFRESH_AFTER - 1), None));
        assert!(
            !should_refresh(Some(500), Some(minute)),
            "a learn is running"
        );
        assert!(
            should_refresh(Some(500), Some(LOCK_TIMEOUT + minute)),
            "that learn died"
        );
    }

    #[test]
    fn parses_options() {
        let o = Options::parse(&["--skip", "60", "--limit", "200"]).unwrap();
        assert_eq!((o.limit, o.skip), (200, 60));
        assert_eq!(Options::parse(&[]).unwrap().limit, 1000);
        assert!(Options::parse(&["--skip"]).is_err());
        assert!(Options::parse(&["--nope"]).is_err());
    }
}
