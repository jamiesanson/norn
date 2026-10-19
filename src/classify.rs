//! Decides the conventional commit type, scope and breaking flag for a staged diff.
//!
//! Local rules answer obvious cases for free. Everything else goes to the classifier: one
//! Jev call, or the model built into norn.

use std::collections::BTreeMap;

use crate::config::{Classifier, ScopeRule, TypeDef};
use crate::diff::{self, ChangedFile, Changes, Outgoing};
use crate::jev::{self, Answer, NoulCriteria, Question};
use crate::{features, model};

const TYPE_INSTRUCTIONS: &str =
    "Which Conventional Commits type best describes the primary purpose of this staged change?";
const BREAKING_INSTRUCTIONS: &str = "Does this change break backwards compatibility for users of \
     the code, such as removing or renaming a public API, changing a signature, or changing \
     default behaviour?";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// The path rules (docs, tests or CI only).
    Rules,
    Jev,
    /// The model built into norn.
    Model,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Classification {
    pub kind: String,
    pub scope: Option<String>,
    pub breaking: bool,
    pub confidence: f64,
    /// The classifier's probability for every allowed type. Empty for rule answers.
    pub probabilities: BTreeMap<String, f64>,
    /// Jev's probability that the change is breaking. `None` when Jev didn't answer.
    pub breaking_probability: Option<f64>,
    pub source: Source,
}

impl Classification {
    /// The `type(scope)!: ` prefix.
    pub fn prefix(&self) -> String {
        let scope = self
            .scope
            .as_ref()
            .map(|s| format!("({s})"))
            .unwrap_or_default();
        let bang = if self.breaking { "!" } else { "" };
        format!("{}{scope}{bang}: ", self.kind)
    }
}

pub struct Settings<'a> {
    /// Probability above which a change is marked breaking.
    pub breaking_threshold: f64,
    /// The types Jev may choose from. Local rules only answer with types in this list.
    pub types: &'a [TypeDef],
    /// When non-empty, replaces the shared-directory guess for the scope.
    pub scopes: &'a [ScopeRule],
    pub classifier: Classifier,
}

pub fn classify(
    client: Option<&jev::Client>,
    changes: Changes,
    files: &[ChangedFile],
    exclude: &[String],
    message: &str,
    settings: &Settings,
) -> Result<Classification, String> {
    let scope = scope_for(files, settings.scopes);

    if let Some(kind) = local_type(files, settings.types) {
        return Ok(Classification {
            kind: kind.to_string(),
            scope,
            breaking: false,
            confidence: 1.0,
            probabilities: BTreeMap::new(),
            breaking_probability: None,
            source: Source::Rules,
        });
    }

    if settings.classifier == Classifier::Local {
        let outgoing = outgoing_for(changes, files, exclude)?;
        return local_model(&outgoing, message, settings.types, scope);
    }

    let client = client.ok_or_else(|| jev::Error::MissingApiKey.to_string())?;
    let request = model_request(changes, files, exclude, message, settings.types)?;
    let mut response = client.system_one(&request).map_err(|e| e.to_string())?;

    let (kind, confidence, probabilities) = match response.answers.remove("type") {
        Some(Answer::Choice {
            choice,
            confidence,
            probabilities,
            x_jev_confidence,
        }) => {
            if !settings.types.iter().any(|t| t.name == choice) {
                return Err(format!("unexpected type from Jev: {choice}"));
            }
            (
                choice,
                x_jev_confidence.unwrap_or(confidence),
                probabilities,
            )
        }
        _ => return Err("response had no type answer".into()),
    };
    let breaking_probability = match response.answers.get("breaking") {
        Some(Answer::Noul { noul }) => Some(*noul),
        _ => None,
    };
    let breaking = breaking_probability.is_some_and(|p| p >= settings.breaking_threshold);

    Ok(Classification {
        kind,
        scope,
        breaking,
        confidence,
        probabilities,
        breaking_probability,
        source: Source::Jev,
    })
}

/// The request `classify` would send, or `None` when the local rules answer without one.
/// `norn export` writes these as training data, so a model learns from what it will see.
pub fn request_for<'a>(
    changes: Changes,
    files: &[ChangedFile],
    exclude: &[String],
    message: &str,
    types: &'a [TypeDef],
) -> Result<Option<jev::Request<'a>>, String> {
    if local_type(files, types).is_some() {
        return Ok(None);
    }
    model_request(changes, files, exclude, message, types).map(Some)
}

fn model_request<'a>(
    changes: Changes,
    files: &[ChangedFile],
    exclude: &[String],
    message: &str,
    types: &'a [TypeDef],
) -> Result<jev::Request<'a>, String> {
    let outgoing = outgoing_for(changes, files, exclude)?;
    Ok(request(&outgoing, message, types))
}

/// What either classifier sees: the files and trimmed patch, without excluded paths.
fn outgoing_for(
    changes: Changes,
    files: &[ChangedFile],
    exclude: &[String],
) -> Result<Outgoing, String> {
    let outgoing = diff::outgoing(changes, files, exclude)?;
    if outgoing.files.is_empty() {
        return Err("every changed file is excluded".into());
    }
    Ok(outgoing)
}

/// Picks among the allowed types the model knows, with probabilities renormalised over
/// them, and confidence computed as Jev computes it, so `min_confidence` means the same.
fn local_model(
    outgoing: &Outgoing,
    message: &str,
    types: &[TypeDef],
    scope: Option<String>,
) -> Result<Classification, String> {
    let model = model::shared()?;
    let p = model.predict(&features::tokens(&outgoing.files, &outgoing.patch, message));
    let allowed: Vec<(&str, f64)> = model
        .classes
        .iter()
        .zip(p)
        .filter(|(class, _)| types.iter().any(|t| &t.name == *class))
        .map(|(class, p)| (class.as_str(), p))
        .collect();
    if allowed.is_empty() {
        return Err("the built-in model doesn't know any of the configured types".into());
    }
    let total: f64 = allowed.iter().map(|(_, p)| p).sum();
    let probabilities: BTreeMap<String, f64> = allowed
        .iter()
        .map(|(class, p)| (class.to_string(), p / total))
        .collect();
    let (kind, best) = probabilities
        .iter()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(k, p)| (k.clone(), *p))
        .expect("allowed isn't empty");
    Ok(Classification {
        kind,
        scope,
        breaking: false,
        confidence: jev_confidence(best, probabilities.len()),
        probabilities,
        breaking_probability: None,
        source: Source::Model,
    })
}

/// Jev's confidence for a choice: how far the top probability is above uniform, scaled to
/// 0..1.
fn jev_confidence(top: f64, options: usize) -> f64 {
    if options < 2 {
        return 1.0;
    }
    let uniform = 1.0 / options as f64;
    (top - uniform) / (1.0 - uniform)
}

fn request<'a>(outgoing: &Outgoing, message: &str, types: &'a [TypeDef]) -> jev::Request<'a> {
    let files: Vec<String> = outgoing
        .files
        .iter()
        .map(|f| format!("{} {}", f.status, f.path))
        .collect();
    let mut state = serde_json::json!({ "files": files, "patch": outgoing.patch });
    if !message.is_empty() {
        state["message"] = serde_json::Value::from(message);
    }

    let mut questions = BTreeMap::new();
    questions.insert(
        "type",
        Question::Choice {
            instructions: TYPE_INSTRUCTIONS,
            criteria: types
                .iter()
                .map(|t| (t.name.as_str(), t.description.as_str()))
                .collect(),
        },
    );
    questions.insert(
        "breaking",
        Question::Noul {
            instructions: BREAKING_INSTRUCTIONS,
            criteria: NoulCriteria {
                yes: "Existing callers or users must change something",
                no: "Existing callers and users are unaffected",
            },
        },
    );

    jev::Request {
        model: jev::DEFAULT_MODEL,
        state,
        questions,
    }
}

/// Answers without the model when every changed file points the same way, and that
/// type is allowed.
fn local_type(files: &[ChangedFile], types: &[TypeDef]) -> Option<&'static str> {
    if files.is_empty() {
        return None;
    }
    let all = |f: fn(&str) -> bool| files.iter().all(|c| f(&c.path));
    let kind = if all(is_docs) {
        "docs"
    } else if all(is_test) {
        "test"
    } else if all(is_ci) {
        "ci"
    } else {
        return None;
    };
    types.iter().any(|t| t.name == kind).then_some(kind)
}

fn is_docs(path: &str) -> bool {
    let name = file_name(path).to_ascii_lowercase();
    path.starts_with("docs/")
        || [".md", ".mdx", ".rst", ".adoc"]
            .iter()
            .any(|ext| name.ends_with(ext))
        || name.starts_with("license")
}

fn is_test(path: &str) -> bool {
    let name = file_name(path);
    path.split('/')
        .any(|dir| matches!(dir, "tests" | "test" | "__tests__" | "spec"))
        || name.starts_with("test_")
        || name.contains("_test.")
        || name.contains(".test.")
        || name.contains(".spec.")
}

fn is_ci(path: &str) -> bool {
    path.starts_with(".github/workflows/")
        || path.starts_with(".circleci/")
        || path.starts_with(".buildkite/")
        || path == ".gitlab-ci.yml"
}

fn file_name(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

/// The scope every changed file agrees on, ignoring lockfiles, which follow changes
/// elsewhere. With configured `scopes`, each file's longest matching path prefix decides;
/// otherwise it's the directory the files share. Disagreement means no scope.
fn scope_for(files: &[ChangedFile], rules: &[ScopeRule]) -> Option<String> {
    let files: Vec<&ChangedFile> = files
        .iter()
        .filter(|f| !diff::is_lockfile(&f.path))
        .collect();
    if rules.is_empty() {
        shared_directory(&files)
    } else {
        let mut scopes = files.iter().map(|f| mapped_scope(&f.path, rules));
        let first = scopes.next()??;
        scopes.all(|s| s == Some(first)).then(|| first.to_string())
    }
}

/// `rules` is sorted longest prefix first. Prefixes match whole path components.
fn mapped_scope<'a>(path: &str, rules: &'a [ScopeRule]) -> Option<&'a str> {
    rules
        .iter()
        .find(|r| {
            path.strip_prefix(r.prefix.as_str())
                .is_some_and(|rest| rest.is_empty() || rest.starts_with('/'))
        })
        .map(|r| r.scope.as_str())
}

/// The directory every file shares, skipping container directories like `src/`. Files at
/// the repository root, or spread across directories, get no scope.
fn shared_directory(files: &[&ChangedFile]) -> Option<String> {
    const CONTAINERS: &[&str] = &[
        "src", "lib", "crates", "packages", "apps", "pkg", "internal",
    ];

    let mut scopes = files.iter().map(|f| {
        let mut parts = f.path.split('/').collect::<Vec<_>>();
        parts.pop(); // file name
        parts.into_iter().find(|p| !CONTAINERS.contains(p))
    });
    let first = scopes.next()??;
    scopes.all(|s| s == Some(first)).then(|| first.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{self, Source as ConfigSource};

    fn files(paths: &[&str]) -> Vec<ChangedFile> {
        paths
            .iter()
            .map(|p| ChangedFile {
                status: 'M',
                path: p.to_string(),
            })
            .collect()
    }

    fn types(names: &[&str]) -> Vec<TypeDef> {
        config::built_in_types()
            .into_iter()
            .filter(|t| names.contains(&t.name.as_str()))
            .collect()
    }

    fn rules(pairs: &[(&str, &str)]) -> Vec<ScopeRule> {
        let mut rules: Vec<ScopeRule> = pairs
            .iter()
            .map(|(prefix, scope)| ScopeRule {
                prefix: prefix.to_string(),
                scope: scope.to_string(),
                source: ConfigSource::Project,
            })
            .collect();
        rules.sort_by_key(|r| std::cmp::Reverse(r.prefix.len()));
        rules
    }

    #[test]
    fn docs_only_is_local() {
        let all = config::built_in_types();
        assert_eq!(
            local_type(&files(&["README.md", "docs/setup.txt"]), &all),
            Some("docs")
        );
    }

    #[test]
    fn tests_only_is_local() {
        let all = config::built_in_types();
        assert_eq!(
            local_type(
                &files(&["tests/cli.rs", "src/app.test.ts", "pkg/x_test.go"]),
                &all
            ),
            Some("test")
        );
    }

    #[test]
    fn mixed_change_needs_model() {
        let all = config::built_in_types();
        assert_eq!(
            local_type(&files(&["README.md", "src/main.rs"]), &all),
            None
        );
    }

    #[test]
    fn local_rules_respect_allowed_types() {
        let no_docs = types(&["feat", "fix", "chore"]);
        assert_eq!(local_type(&files(&["README.md"]), &no_docs), None);
    }

    #[test]
    fn scope_from_shared_directory() {
        assert_eq!(
            scope_for(&files(&["src/config/load.rs", "src/config/mod.rs"]), &[]),
            Some("config".into())
        );
        assert_eq!(
            scope_for(&files(&["packages/api/src/index.ts"]), &[]),
            Some("api".into())
        );
    }

    #[test]
    fn no_scope_for_root_or_spread_changes() {
        assert_eq!(scope_for(&files(&["src/main.rs"]), &[]), None);
        assert_eq!(scope_for(&files(&["src/a/x.rs", "src/b/y.rs"]), &[]), None);
        assert_eq!(scope_for(&files(&["Cargo.toml", "src/a/x.rs"]), &[]), None);
    }

    #[test]
    fn lockfiles_dont_affect_scope() {
        assert_eq!(
            scope_for(&files(&["src/config/load.rs", "Cargo.lock"]), &[]),
            Some("config".into())
        );
        let web = rules(&[("packages/web", "web")]);
        assert_eq!(
            scope_for(&files(&["packages/web/app.ts", "pnpm-lock.yaml"]), &web),
            Some("web".into())
        );
        assert_eq!(scope_for(&files(&["Cargo.lock"]), &[]), None);
    }

    #[test]
    fn configured_scopes_use_longest_prefix() {
        let r = rules(&[("packages/web", "web"), ("packages/web/admin", "admin")]);
        assert_eq!(
            scope_for(&files(&["packages/web/admin/users.ts"]), &r),
            Some("admin".into())
        );
        assert_eq!(
            scope_for(&files(&["packages/web/index.ts"]), &r),
            Some("web".into())
        );
    }

    #[test]
    fn configured_scopes_match_whole_path_components() {
        let r = rules(&[("packages/web", "web")]);
        assert_eq!(scope_for(&files(&["packages/website/a.ts"]), &r), None);
        assert_eq!(scope_for(&files(&["packages/web"]), &r), Some("web".into()));
    }

    #[test]
    fn configured_scopes_replace_shared_directory() {
        let r = rules(&[("packages/web", "web")]);
        // Would be `config` from the shared directory, but no rule matches.
        assert_eq!(scope_for(&files(&["src/config/load.rs"]), &r), None);
        // Files mapping to different scopes, or to none, get no scope.
        let r = rules(&[("packages/web", "web"), ("packages/api", "api")]);
        assert_eq!(
            scope_for(&files(&["packages/web/a.ts", "packages/api/b.ts"]), &r),
            None
        );
        assert_eq!(
            scope_for(&files(&["packages/web/a.ts", "README.md"]), &r),
            None
        );
    }

    #[test]
    fn prefix_formatting() {
        let c = Classification {
            kind: "feat".into(),
            scope: Some("api".into()),
            breaking: true,
            confidence: 0.9,
            probabilities: BTreeMap::new(),
            breaking_probability: None,
            source: Source::Jev,
        };
        assert_eq!(c.prefix(), "feat(api)!: ");
    }
}
