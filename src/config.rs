//! Layered settings. Each layer overrides the one before:
//!
//! 1. built-in defaults
//! 2. the global file, `~/.config/norn/config.toml`
//! 3. the project file, `.norn.toml` at the repository root
//! 4. `NORN_*` environment variables
//!
//! The project file comes with whatever repository you clone, so it can't set anything
//! that changes where your diffs and API key are sent (`base_url`, `classifier`), and its
//! `exclude` patterns add to the global ones rather than replacing them.
//!
//! `types` is a whole set, so the project's list replaces the global one. `scopes` entries
//! combine, with the project winning on the same prefix.

use std::collections::BTreeMap;
use std::fmt::Display;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use serde::Deserialize;
use serde::de::IgnoredAny;

use crate::jev::DEFAULT_BASE_URL;

pub const PROJECT_FILE: &str = ".norn.toml";
const GLOBAL_FILE: &str = "config.toml";

/// `$XDG_CONFIG_HOME/norn`, `~/.config/norn`, or `%APPDATA%\norn` on Windows.
pub fn dir() -> Option<PathBuf> {
    let base = if cfg!(windows) {
        std::env::var_os("APPDATA").map(PathBuf::from)
    } else {
        std::env::var_os("XDG_CONFIG_HOME")
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| Path::new(&h).join(".config")))
    };
    Some(base?.join("norn"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    Default,
    Global,
    Project,
    Env(&'static str),
}

impl Display for Source {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Source::Default => write!(f, "default"),
            Source::Global => write!(f, "global"),
            Source::Project => write!(f, "project"),
            Source::Env(name) => write!(f, "{name}"),
        }
    }
}

/// Conventional commit types and the criteria Jev uses to tell them apart.
pub const BUILT_IN_TYPES: &[(&str, &str)] = &[
    ("feat", "Adds new user-facing behaviour or capability"),
    ("fix", "Corrects a bug or wrong behaviour"),
    ("docs", "Changes documentation only"),
    (
        "style",
        "Formatting or whitespace only; no change in behaviour",
    ),
    ("refactor", "Restructures code without changing behaviour"),
    ("perf", "Improves performance without changing behaviour"),
    ("test", "Adds or changes tests only"),
    (
        "build",
        "Changes the build system, packaging or dependencies",
    ),
    ("ci", "Changes CI configuration or pipelines"),
    ("chore", "Maintenance that doesn't fit another type"),
    ("revert", "Reverts an earlier commit"),
];

/// Jev's limit on options for a single question.
const MAX_TYPES: usize = 255;

#[derive(Debug, Clone, PartialEq)]
pub struct TypeDef {
    pub name: String,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScopeRule {
    /// Repository-relative directory or file, without leading or trailing slashes.
    pub prefix: String,
    pub scope: String,
    pub source: Source,
}

/// What picks the type when the local rules can't.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Classifier {
    /// TypeSafe's Jev, over the network.
    Jev,
    /// The model built into norn. Nothing leaves your machine.
    Local,
}

impl Display for Classifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Classifier::Jev => write!(f, "jev"),
            Classifier::Local => write!(f, "local"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Setting<T> {
    pub value: T,
    pub source: Source,
}

#[derive(Debug)]
pub struct Config {
    pub disable: Setting<bool>,
    pub classifier: Setting<Classifier>,
    pub base_url: Setting<String>,
    pub timeout_ms: Setting<u64>,
    pub deadline_ms: Setting<u64>,
    pub min_confidence: Setting<f64>,
    pub breaking_threshold: Setting<f64>,
    /// Git glob pathspecs, relative to the repository root. Global and project patterns
    /// combined.
    pub exclude: Vec<Setting<String>>,
    pub types: Setting<Vec<TypeDef>>,
    /// Longest prefix first, so the first match is the most specific.
    pub scopes: Vec<ScopeRule>,
    pub global_path: Option<PathBuf>,
    pub project_path: Option<PathBuf>,
    /// Problems found while loading. Loading never fails; bad values are skipped.
    pub warnings: Vec<String>,
}

#[derive(Debug, Default, Deserialize)]
struct File {
    disable: Option<bool>,
    classifier: Option<Classifier>,
    base_url: Option<String>,
    timeout_ms: Option<u64>,
    deadline_ms: Option<u64>,
    min_confidence: Option<f64>,
    breaking_threshold: Option<f64>,
    #[serde(default)]
    exclude: Vec<String>,
    types: Option<TypesField>,
    #[serde(default)]
    scopes: BTreeMap<String, String>,
    #[serde(flatten)]
    unknown: BTreeMap<String, IgnoredAny>,
}

#[derive(Debug, Deserialize)]
#[serde(
    untagged,
    expecting = "a list of built-in type names, or a table of type names to descriptions"
)]
enum TypesField {
    Names(Vec<String>),
    Described(BTreeMap<String, String>),
}

/// Loads every layer. `project_root` is the repository's top-level directory, if any.
pub fn load(project_root: Option<&Path>) -> Config {
    let global_path = dir().map(|d| d.join(GLOBAL_FILE));
    let project_path = project_root.map(|r| r.join(PROJECT_FILE));
    let mut warnings = Vec::new();
    let global = global_path.as_deref().and_then(|p| read(p, &mut warnings));
    let project = project_path.as_deref().and_then(|p| read(p, &mut warnings));
    let mut config = resolve(global, project, &|name| std::env::var(name).ok(), warnings);
    config.global_path = global_path;
    config.project_path = project_path;
    config
}

/// A missing file is fine. A file that can't be read or parsed is skipped with a warning.
fn read(path: &Path, warnings: &mut Vec<String>) -> Option<File> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return None,
        Err(e) => {
            warnings.push(format!("ignoring {}: {e}", path.display()));
            return None;
        }
    };
    match toml::from_str::<File>(&text) {
        Ok(file) => {
            for key in file.unknown.keys() {
                warnings.push(format!("unknown setting `{key}` in {}", path.display()));
            }
            Some(file)
        }
        Err(e) => {
            let reason = e.message().trim().to_string();
            warnings.push(format!("ignoring {}: {reason}", path.display()));
            None
        }
    }
}

fn resolve(
    global: Option<File>,
    project: Option<File>,
    env: &dyn Fn(&str) -> Option<String>,
    mut warnings: Vec<String>,
) -> Config {
    let global = global.unwrap_or_default();
    let mut project = project.unwrap_or_default();

    if project.base_url.take().is_some() {
        warnings.push(format!(
            "ignoring `base_url` in {PROJECT_FILE}: it can only be set in the global config"
        ));
    }
    if project.classifier.take().is_some() {
        warnings.push(format!(
            "ignoring `classifier` in {PROJECT_FILE}: it can only be set in the global config"
        ));
    }

    let mut w = |msg: String| warnings.push(msg);
    let disable = layer(
        false,
        global.disable,
        project.disable,
        "NORN_DISABLE",
        env,
        &mut w,
    );
    let classifier = layer(
        Classifier::Local,
        global.classifier,
        None,
        "NORN_CLASSIFIER",
        env,
        &mut w,
    );
    let base_url = layer(
        DEFAULT_BASE_URL.to_string(),
        global.base_url,
        None,
        "NORN_BASE_URL",
        env,
        &mut w,
    );
    let timeout_ms = layer(
        1000,
        global.timeout_ms,
        project.timeout_ms,
        "NORN_TIMEOUT_MS",
        env,
        &mut w,
    );
    let deadline_ms = layer(
        2000,
        global.deadline_ms,
        project.deadline_ms,
        "NORN_DEADLINE_MS",
        env,
        &mut w,
    );
    let min_confidence = layer(
        0.6,
        global.min_confidence,
        project.min_confidence,
        "NORN_MIN_CONFIDENCE",
        env,
        &mut w,
    );
    let breaking_threshold = layer(
        0.85,
        global.breaking_threshold,
        project.breaking_threshold,
        "NORN_BREAKING_THRESHOLD",
        env,
        &mut w,
    );

    let exclude = global
        .exclude
        .into_iter()
        .map(|value| Setting {
            value,
            source: Source::Global,
        })
        .chain(project.exclude.into_iter().map(|value| Setting {
            value,
            source: Source::Project,
        }))
        .collect();

    let types = match (
        project
            .types
            .and_then(|t| type_defs(t, PROJECT_FILE, &mut warnings)),
        global
            .types
            .and_then(|t| type_defs(t, "the global config", &mut warnings)),
    ) {
        (Some(value), _) => Setting {
            value,
            source: Source::Project,
        },
        (None, Some(value)) => Setting {
            value,
            source: Source::Global,
        },
        (None, None) => Setting {
            value: built_in_types(),
            source: Source::Default,
        },
    };

    let mut scopes: BTreeMap<String, ScopeRule> = BTreeMap::new();
    for (entries, source, label) in [
        (global.scopes, Source::Global, "the global config"),
        (project.scopes, Source::Project, PROJECT_FILE),
    ] {
        for (prefix, scope) in entries {
            match scope_rule(&prefix, scope, source) {
                Ok(rule) => {
                    scopes.insert(rule.prefix.clone(), rule);
                }
                Err(e) => warnings.push(format!("ignoring scope for `{prefix}` in {label}: {e}")),
            }
        }
    }
    let mut scopes: Vec<ScopeRule> = scopes.into_values().collect();
    scopes.sort_by_key(|rule| std::cmp::Reverse(rule.prefix.len()));

    Config {
        disable,
        classifier,
        base_url,
        timeout_ms,
        deadline_ms,
        min_confidence,
        breaking_threshold,
        exclude,
        types,
        scopes,
        global_path: None,
        project_path: None,
        warnings,
    }
}

pub fn built_in_types() -> Vec<TypeDef> {
    BUILT_IN_TYPES
        .iter()
        .map(|(name, description)| TypeDef {
            name: name.to_string(),
            description: description.to_string(),
        })
        .collect()
}

/// Type names must be lowercase letters, so norn recognises its own prefixes later.
fn valid_type_name(name: &str) -> bool {
    !name.is_empty() && name.chars().all(|c| c.is_ascii_lowercase())
}

/// Returns `None`, with a warning, when no usable types remain.
fn type_defs(field: TypesField, label: &str, warnings: &mut Vec<String>) -> Option<Vec<TypeDef>> {
    let built_in = |name: &str| {
        BUILT_IN_TYPES
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, d)| d.to_string())
    };
    let entries: Vec<(String, Option<String>)> = match field {
        TypesField::Names(names) => names.into_iter().map(|n| (n, None)).collect(),
        TypesField::Described(map) => map
            .into_iter()
            .map(|(n, d)| (n, Some(d).filter(|d| !d.trim().is_empty())))
            .collect(),
    };

    let mut defs: Vec<TypeDef> = Vec::new();
    for (name, description) in entries {
        if !valid_type_name(&name) {
            warnings.push(format!(
                "ignoring type `{name}` in {label}: type names must be lowercase letters"
            ));
            continue;
        }
        let Some(description) = description.or_else(|| built_in(&name)) else {
            warnings.push(format!(
                "ignoring type `{name}` in {label}: it isn't built in, so it needs a description \
                 (`[types]` table, e.g. {name} = \"...\")"
            ));
            continue;
        };
        if !defs.iter().any(|d| d.name == name) {
            defs.push(TypeDef { name, description });
        }
    }

    if defs.is_empty() {
        warnings.push(format!("ignoring `types` in {label}: no usable types"));
        return None;
    }
    if defs.len() > MAX_TYPES {
        warnings.push(format!(
            "ignoring `types` in {label}: at most {MAX_TYPES} types are allowed"
        ));
        return None;
    }
    Some(defs)
}

fn scope_rule(prefix: &str, scope: String, source: Source) -> Result<ScopeRule, String> {
    let prefix = prefix
        .trim()
        .trim_start_matches("./")
        .trim_matches('/')
        .to_string();
    if prefix.is_empty() {
        return Err("the path is empty".into());
    }
    if scope.is_empty() || scope.contains(|c: char| c.is_whitespace() || c == '(' || c == ')') {
        return Err(format!(
            "`{scope}` isn't a valid scope (no spaces or parentheses)"
        ));
    }
    Ok(ScopeRule {
        prefix,
        scope,
        source,
    })
}

fn layer<T: EnvValue>(
    default: T,
    global: Option<T>,
    project: Option<T>,
    env_name: &'static str,
    env: &dyn Fn(&str) -> Option<String>,
    warn: &mut dyn FnMut(String),
) -> Setting<T> {
    let mut setting = Setting {
        value: default,
        source: Source::Default,
    };
    if let Some(value) = global {
        setting = Setting {
            value,
            source: Source::Global,
        };
    }
    if let Some(value) = project {
        setting = Setting {
            value,
            source: Source::Project,
        };
    }
    if let Some(raw) = env(env_name).filter(|v| !v.is_empty()) {
        match T::parse_env(&raw) {
            Some(value) => {
                setting = Setting {
                    value,
                    source: Source::Env(env_name),
                }
            }
            None => warn(format!("ignoring {env_name}={raw}: not a valid value")),
        }
    }
    setting
}

trait EnvValue: Sized {
    fn parse_env(raw: &str) -> Option<Self>;
}

impl EnvValue for bool {
    fn parse_env(raw: &str) -> Option<Self> {
        match raw.to_ascii_lowercase().as_str() {
            "1" | "true" => Some(true),
            "0" | "false" => Some(false),
            _ => None,
        }
    }
}

macro_rules! env_value_from_str {
    ($($t:ty),*) => {$(
        impl EnvValue for $t {
            fn parse_env(raw: &str) -> Option<Self> {
                <$t>::from_str(raw).ok()
            }
        }
    )*};
}
env_value_from_str!(u64, f64, String);

impl EnvValue for Classifier {
    fn parse_env(raw: &str) -> Option<Self> {
        match raw.to_ascii_lowercase().as_str() {
            "jev" => Some(Classifier::Jev),
            "local" => Some(Classifier::Local),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(toml: &str) -> File {
        toml::from_str(toml).unwrap()
    }

    fn no_env(_: &str) -> Option<String> {
        None
    }

    #[test]
    fn defaults_when_nothing_is_set() {
        let c = resolve(None, None, &no_env, vec![]);
        assert_eq!(
            c.timeout_ms,
            Setting {
                value: 1000,
                source: Source::Default
            }
        );
        assert_eq!(c.base_url.value, DEFAULT_BASE_URL);
        assert!(!c.disable.value);
        assert!(c.exclude.is_empty());
        assert!(c.warnings.is_empty());
    }

    #[test]
    fn project_overrides_global_and_env_overrides_both() {
        let global = file("timeout_ms = 500\nmin_confidence = 0.5\ndeadline_ms = 3000");
        let project = file("timeout_ms = 800\nmin_confidence = 0.75");
        let env = |name: &str| (name == "NORN_MIN_CONFIDENCE").then(|| "0.9".to_string());
        let c = resolve(Some(global), Some(project), &env, vec![]);
        assert_eq!(
            c.deadline_ms,
            Setting {
                value: 3000,
                source: Source::Global
            }
        );
        assert_eq!(
            c.timeout_ms,
            Setting {
                value: 800,
                source: Source::Project
            }
        );
        assert_eq!(
            c.min_confidence,
            Setting {
                value: 0.9,
                source: Source::Env("NORN_MIN_CONFIDENCE")
            }
        );
    }

    #[test]
    fn project_cannot_set_base_url() {
        let project = file("base_url = \"https://attacker.example\"");
        let c = resolve(None, Some(project), &no_env, vec![]);
        assert_eq!(c.base_url.value, DEFAULT_BASE_URL);
        assert!(c.warnings[0].contains("base_url"));
    }

    #[test]
    fn global_and_env_can_set_base_url() {
        let global = file("base_url = \"https://proxy.internal\"");
        let c = resolve(Some(global), None, &no_env, vec![]);
        assert_eq!(c.base_url.value, "https://proxy.internal");
        let env = |name: &str| (name == "NORN_BASE_URL").then(|| "http://localhost:1".into());
        let c = resolve(None, None, &env, vec![]);
        assert_eq!(c.base_url.source, Source::Env("NORN_BASE_URL"));
    }

    #[test]
    fn classifier_is_global_or_env_only() {
        let c = resolve(None, None, &no_env, vec![]);
        assert_eq!(c.classifier.value, Classifier::Local);
        let global = file("classifier = \"jev\"");
        let c = resolve(Some(global), None, &no_env, vec![]);
        assert_eq!(c.classifier.value, Classifier::Jev);

        let project = file("classifier = \"jev\"");
        let c = resolve(None, Some(project), &no_env, vec![]);
        assert_eq!(c.classifier.value, Classifier::Local);
        assert!(c.warnings[0].contains("classifier"));

        let env = |name: &str| (name == "NORN_CLASSIFIER").then(|| "Jev".to_string());
        let c = resolve(None, None, &env, vec![]);
        assert_eq!(c.classifier.value, Classifier::Jev);
        assert_eq!(c.classifier.source, Source::Env("NORN_CLASSIFIER"));
    }

    #[test]
    fn exclude_patterns_combine() {
        let global = file("exclude = [\"**/*.pem\"]");
        let project = file("exclude = [\"secrets/**\"]");
        let c = resolve(Some(global), Some(project), &no_env, vec![]);
        let patterns: Vec<_> = c
            .exclude
            .iter()
            .map(|s| (s.value.as_str(), s.source))
            .collect();
        assert_eq!(
            patterns,
            vec![
                ("**/*.pem", Source::Global),
                ("secrets/**", Source::Project)
            ]
        );
    }

    #[test]
    fn default_types_are_built_in() {
        let c = resolve(None, None, &no_env, vec![]);
        assert_eq!(c.types.source, Source::Default);
        assert_eq!(c.types.value.len(), BUILT_IN_TYPES.len());
    }

    #[test]
    fn project_types_replace_global_types() {
        let global = file("types = [\"feat\", \"fix\", \"chore\"]");
        let project = file("types = [\"feat\", \"fix\"]");
        let c = resolve(Some(global), Some(project), &no_env, vec![]);
        assert_eq!(c.types.source, Source::Project);
        let names: Vec<_> = c.types.value.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["feat", "fix"]);
        assert_eq!(c.types.value[0].description, BUILT_IN_TYPES[0].1);
    }

    #[test]
    fn custom_types_need_descriptions() {
        let project = file("[types]\nfeat = \"\"\ndeps = \"Updates dependencies\"\n");
        let c = resolve(None, Some(project), &no_env, vec![]);
        let names: Vec<_> = c.types.value.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["deps", "feat"]);
        assert_eq!(c.types.value[0].description, "Updates dependencies");
        assert!(c.warnings.is_empty(), "{:?}", c.warnings);

        let project = file("types = [\"feat\", \"deps\", \"Bad-Name\"]");
        let c = resolve(None, Some(project), &no_env, vec![]);
        assert_eq!(c.types.value.len(), 1);
        assert_eq!(c.warnings.len(), 2, "{:?}", c.warnings);
        assert!(c.warnings[0].contains("`deps`"));
        assert!(c.warnings[1].contains("`Bad-Name`"));
    }

    #[test]
    fn unusable_project_types_fall_back_to_global() {
        let global = file("types = [\"feat\", \"fix\"]");
        let project = file("types = [\"Nope\"]");
        let c = resolve(Some(global), Some(project), &no_env, vec![]);
        assert_eq!(c.types.source, Source::Global);
        assert!(c.warnings.iter().any(|w| w.contains("no usable types")));
    }

    #[test]
    fn scopes_combine_and_sort_longest_first() {
        let global = file("[scopes]\n\"packages/web\" = \"frontend\"\n\"docs\" = \"docs\"\n");
        let project =
            file("[scopes]\n\"./packages/web/\" = \"web\"\n\"packages/web/admin\" = \"admin\"\n");
        let c = resolve(Some(global), Some(project), &no_env, vec![]);
        let rules: Vec<_> = c
            .scopes
            .iter()
            .map(|r| (r.prefix.as_str(), r.scope.as_str(), r.source))
            .collect();
        assert_eq!(
            rules,
            vec![
                ("packages/web/admin", "admin", Source::Project),
                ("packages/web", "web", Source::Project),
                ("docs", "docs", Source::Global),
            ]
        );
    }

    #[test]
    fn invalid_scopes_are_skipped() {
        let project = file("[scopes]\n\"/\" = \"root\"\n\"api\" = \"public api\"\n");
        let c = resolve(None, Some(project), &no_env, vec![]);
        assert!(c.scopes.is_empty());
        assert_eq!(c.warnings.len(), 2, "{:?}", c.warnings);
    }

    #[test]
    fn invalid_env_value_is_ignored_with_warning() {
        let env = |name: &str| (name == "NORN_TIMEOUT_MS").then(|| "soon".to_string());
        let c = resolve(None, None, &env, vec![]);
        assert_eq!(c.timeout_ms.source, Source::Default);
        assert!(c.warnings[0].contains("NORN_TIMEOUT_MS=soon"));
    }

    #[test]
    fn env_disable_accepts_common_spellings() {
        for (raw, expected) in [("1", true), ("TRUE", true), ("0", false), ("false", false)] {
            let env = move |name: &str| (name == "NORN_DISABLE").then(|| raw.to_string());
            assert_eq!(resolve(None, None, &env, vec![]).disable.value, expected);
        }
    }

    #[test]
    fn unknown_keys_and_bad_files_warn() {
        let dir = std::env::temp_dir().join(format!("norn-config-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let unknown = dir.join("unknown.toml");
        std::fs::write(&unknown, "timeout_ms = 5\ntypo_setting = 1\n").unwrap();
        let broken = dir.join("broken.toml");
        std::fs::write(&broken, "timeout_ms = \"not a number\"\n").unwrap();

        let mut warnings = Vec::new();
        assert_eq!(read(&unknown, &mut warnings).unwrap().timeout_ms, Some(5));
        assert!(read(&broken, &mut warnings).is_none());
        assert!(read(&dir.join("missing.toml"), &mut warnings).is_none());
        assert_eq!(warnings.len(), 2, "{warnings:?}");
        assert!(warnings[0].contains("typo_setting"));
        assert!(warnings[1].contains("broken.toml"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
