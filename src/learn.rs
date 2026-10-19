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
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::OnceLock;

use crate::classify;
use crate::config::Config;
use crate::diff::{self, Changes};
use crate::eval;
use crate::features::Tokens;
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

pub struct Options {
    pub limit: usize,
    /// Leave out the newest commits, so `norn eval` can be run on ones the model hasn't seen.
    pub skip: usize,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            limit: 1000,
            skip: 0,
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

    let docs = eval::map_commits(&commits, |commit| {
        let changes = Changes::Commit(&commit.sha);
        let files = diff::changed_files(changes)?;
        let message = commit.message(with_message(&commit.sha));
        let tokens =
            classify::model_tokens(changes, &files, &exclude, message, &config.types.value)?;
        Ok::<_, String>(tokens.map(|t| (t, commit.label().kind.to_string())))
    });
    let docs: Vec<(Tokens, String)> = docs.into_iter().filter_map(|d| d.ok().flatten()).collect();
    if docs.len() < MIN_COMMITS {
        let s = if docs.len() == 1 { "" } else { "s" };
        return Err(format!(
            "only {} conventional commit{s} to learn from, and norn needs {MIN_COMMITS}",
            docs.len()
        ));
    }

    let learned = finetune(model::shared()?, &docs);
    let path = model_path().ok_or("not in a git repository")?;
    save(&path, &learned.to_bytes())?;

    let mut counts: HashMap<&str, usize> = HashMap::new();
    for (_, kind) in &docs {
        *counts.entry(kind).or_default() += 1;
    }
    let mut types: Vec<(String, usize)> = counts
        .into_iter()
        .map(|(k, n)| (k.to_string(), n))
        .collect();
    types.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    Ok(Learned {
        commits: docs.len(),
        types,
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
    let output = Command::new("git")
        .args(["rev-parse", "--git-common-dir"])
        .stderr(Stdio::null())
        .output()
        .ok()?;
    output.status.success().then(|| {
        PathBuf::from(String::from_utf8_lossy(&output.stdout).trim())
            .join("norn")
            .join("model.bin")
    })
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

/// `base` refitted on `docs`, each a commit's tokens and its type.
pub fn finetune(base: &Model, docs: &[(Tokens, String)]) -> Model {
    let mut model = extend(base, docs);
    let k = model.classes.len();
    let labels: Vec<usize> = docs
        .iter()
        .map(|(_, kind)| model.classes.iter().position(|c| c == kind).unwrap())
        .collect();
    let rows: Vec<Vec<(usize, f64)>> = docs.iter().map(|(t, _)| model.vectorize(t)).collect();

    // A feature no commit uses gets no pull from the data, so the penalty holds it at its
    // built-in value. Only the features that appear need fitting.
    let mut active: Vec<usize> = rows.iter().flatten().map(|&(f, _)| f).collect();
    active.sort_unstable();
    active.dedup();
    let slot: HashMap<usize, usize> = active.iter().enumerate().map(|(i, &f)| (f, i)).collect();
    let rows: Vec<Vec<(usize, f64)>> = rows
        .into_iter()
        .map(|r| r.into_iter().map(|(f, v)| (slot[&f], v)).collect())
        .collect();

    // The parameters: each active feature's weight per class, then the intercepts.
    let a = active.len();
    let mut prior: Vec<f64> = Vec::with_capacity(a * k + k);
    for &f in &active {
        prior.extend(model.weights[f * k..(f + 1) * k].iter().map(|&w| w as f64));
    }
    prior.extend(model.intercepts.iter().map(|&b| b as f64));

    let theta = lbfgs::minimize(
        prior.clone(),
        |theta, grad| {
            let mut value = 0.0;
            for i in 0..theta.len() {
                let d = theta[i] - prior[i];
                value += 0.5 * d * d;
                grad[i] = d;
            }
            let (w, b) = theta.split_at(a * k);
            let (gw, gb) = grad.split_at_mut(a * k);
            let mut scores = vec![0.0; k];
            for (x, &y) in rows.iter().zip(&labels) {
                scores.copy_from_slice(b);
                for &(s, v) in x {
                    for c in 0..k {
                        scores[c] += v * w[s * k + c];
                    }
                }
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let log_total = scores.iter().map(|s| (s - max).exp()).sum::<f64>().ln() + max;
                value -= C * (scores[y] - log_total);
                for c in 0..k {
                    let error = C * ((scores[c] - log_total).exp() - f64::from(c == y));
                    gb[c] += error;
                    for &(s, v) in x {
                        gw[s * k + c] += error * v;
                    }
                }
            }
            value
        },
        MAX_ITERATIONS,
    );

    for (i, &f) in active.iter().enumerate() {
        for c in 0..k {
            model.weights[f * k + c] = theta[i * k + c] as f32;
        }
    }
    for c in 0..k {
        model.intercepts[c] = theta[a * k + c] as f32;
    }
    model
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
        let learned = finetune(&tiny(), &docs);
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
        let learned = finetune(&base, &docs);
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
    fn parses_options() {
        let o = Options::parse(&["--skip", "60", "--limit", "200"]).unwrap();
        assert_eq!((o.limit, o.skip), (200, 60));
        assert_eq!(Options::parse(&[]).unwrap().limit, 1000);
        assert!(Options::parse(&["--skip"]).is_err());
        assert!(Options::parse(&["--nope"]).is_err());
    }
}
