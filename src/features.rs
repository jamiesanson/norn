//! Turns a change into tokens for the local model, in four blocks that are weighted
//! separately: the message, the changed paths, the patch, and the change's shape.
//!
//! This is the only implementation. `norn features` runs it over training data, so the
//! model is trained on exactly the tokens it sees at commit time.

use crate::diff::{self, ChangedFile};

#[derive(Debug, Default, Clone, PartialEq, serde::Serialize)]
pub struct Tokens {
    pub msg: Vec<String>,
    pub paths: Vec<String>,
    pub patch: Vec<String>,
    pub meta: Vec<String>,
}

impl Tokens {
    pub fn block(&self, name: &str) -> &[String] {
        match name {
            "msg" => &self.msg,
            "paths" => &self.paths,
            "patch" => &self.patch,
            "meta" => &self.meta,
            _ => &[],
        }
    }
}

/// `patch` is the trimmed patch norn would send to Jev, so both see the same change.
pub fn tokens(files: &[ChangedFile], patch: &str, message: &str) -> Tokens {
    let mut t = Tokens::default();

    let mw = words(message);
    if let Some(first) = mw.first() {
        t.msg.push(format!("m0:{first}"));
        t.msg.extend(mw.iter().map(|w| format!("m:{w}")));
        t.msg
            .extend(mw.windows(2).map(|p| format!("m2:{}_{}", p[0], p[1])));
        t.meta.push("has_msg".into());
    }

    t.meta.push(format!("nfiles:{}", bucket(files.len())));
    let mut kinds: Vec<(&str, Vec<char>)> = Vec::new();
    for file in files {
        let lower = file.path.to_lowercase();
        let parts: Vec<&str> = lower.split('/').collect();
        let (dirs, name) = parts.split_at(parts.len() - 1);
        let name = name[0];
        let ext = name.rsplit_once('.').map_or("-", |(_, e)| e);
        let s = file.status;
        t.paths.extend([
            format!("st:{s}"),
            format!("ext:{ext}"),
            format!("stext:{s}{ext}"),
            format!("top:{}", parts[0]),
            format!("name:{name}"),
        ]);
        t.paths.extend(dirs.iter().map(|d| format!("dir:{d}")));
        t.paths
            .extend(words(&file.path).into_iter().map(|w| format!("pw:{w}")));

        let kind = category(&file.path);
        match kinds.iter_mut().find(|(k, _)| *k == kind) {
            Some((_, statuses)) if !statuses.contains(&s) => statuses.push(s),
            Some(_) => {}
            None => kinds.push((kind, vec![s])),
        }
    }
    kinds.sort();
    let names: Vec<&str> = kinds.iter().map(|(k, _)| *k).collect();
    t.meta.push(format!("cats:{}", names.join("+")));
    for (kind, statuses) in &kinds {
        t.meta.push(format!("cat:{kind}"));
        t.meta
            .extend(statuses.iter().map(|s| format!("catst:{kind}{s}")));
    }
    if !names.is_empty() && names.iter().all(|k| matches!(*k, "lock" | "manifest")) {
        t.meta.push("deps-only".into());
    }

    let (mut added, mut removed) = (0, 0);
    for line in patch.lines() {
        if line.starts_with("+++")
            || line.starts_with("---")
            || line.starts_with("diff --git")
            || line.starts_with("index ")
        {
            continue;
        }
        if let Some(rest) = line.strip_prefix('+') {
            added += 1;
            t.patch
                .extend(words(rest).into_iter().map(|w| format!("+:{w}")));
        } else if let Some(rest) = line.strip_prefix('-') {
            removed += 1;
            t.patch
                .extend(words(rest).into_iter().map(|w| format!("-:{w}")));
        } else if line.starts_with("@@") {
            let context = line.rsplit("@@").next().unwrap_or_default();
            t.patch
                .extend(words(context).into_iter().map(|w| format!("h:{w}")));
        }
    }
    let shape = if added > 2 * removed {
        "add"
    } else if removed > 2 * added {
        "del"
    } else {
        "mix"
    };
    t.meta.extend([
        format!("add:{}", bucket(added)),
        format!("del:{}", bucket(removed)),
        format!("shape:{shape}"),
    ]);
    t
}

/// ASCII words and numbers, split at camelCase boundaries, lowercased, at least two long:
/// `parseHTTPResponse2` gives `parse`, `http`, `response`.
pub fn words(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for run in text.split(|c: char| !c.is_ascii_alphanumeric()) {
        let b = run.as_bytes();
        let mut start = 0;
        for i in 1..=b.len() {
            let split = i == b.len() || {
                let (p, c) = (b[i - 1], b[i]);
                p.is_ascii_digit() != c.is_ascii_digit()
                    || (p.is_ascii_lowercase() && c.is_ascii_uppercase())
                    || (p.is_ascii_uppercase()
                        && c.is_ascii_uppercase()
                        && b.get(i + 1).is_some_and(u8::is_ascii_lowercase))
            };
            if split {
                if i - start > 1 {
                    out.push(run[start..i].to_ascii_lowercase());
                }
                start = i;
            }
        }
    }
    out
}

/// `0` for none, otherwise 1 + floor(log2 n), capped at 10.
fn bucket(n: usize) -> u32 {
    if n == 0 { 0 } else { (n.ilog2() + 1).min(10) }
}

/// A rough kind for a path, so a change's shape (source with tests, only manifests, ...)
/// can be a feature.
fn category(original: &str) -> &'static str {
    const MANIFESTS: &[&str] = &[
        "package.json",
        "cargo.toml",
        "go.mod",
        "pyproject.toml",
        "setup.py",
        "requirements.txt",
        "pom.xml",
        "build.gradle",
        "gemfile",
        "composer.json",
        "pnpm-workspace.yaml",
    ];
    if diff::is_lockfile(original) {
        return "lock";
    }
    let path = &original.to_lowercase();
    let name = path.rsplit('/').next().unwrap_or(path);
    if MANIFESTS.contains(&name) {
        return "manifest";
    }
    if [".github/", ".circleci/", ".gitlab", ".buildkite/"]
        .iter()
        .any(|p| path.starts_with(p))
    {
        return "ci";
    }
    if path.starts_with("docs/")
        || [".md", ".mdx", ".rst", ".adoc"]
            .iter()
            .any(|e| name.ends_with(e))
    {
        return "docs";
    }
    let test_dir = path.split('/').rev().skip(1).any(|d| {
        matches!(
            d,
            "test"
                | "tests"
                | "__tests__"
                | "spec"
                | "e2e"
                | "fixture"
                | "fixtures"
                | "snapshot"
                | "snapshots"
        )
    });
    if test_dir
        || name.starts_with("test_")
        || ["_test.", ".test.", ".spec."]
            .iter()
            .any(|p| name.contains(p))
    {
        return "test";
    }
    if [".yml", ".yaml", ".toml", ".json", ".ini", ".cfg"]
        .iter()
        .any(|e| name.ends_with(e))
        || name.starts_with('.')
    {
        return "config";
    }
    "src"
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(status: char, path: &str) -> ChangedFile {
        ChangedFile {
            status,
            path: path.into(),
        }
    }

    #[test]
    fn words_split_identifiers() {
        assert_eq!(
            words("parseHTTPResponse2 snake_case x-y a"),
            ["parse", "http", "response", "snake", "case"]
        );
        assert_eq!(words("v1.2.30 IOError"), ["30", "io", "error"]);
        assert!(words("").is_empty());
    }

    #[test]
    fn tokens_cover_every_block() {
        let files = [
            file('M', "src/Config.rs"),
            file('A', "tests/config_test.rs"),
        ];
        let patch = "diff --git a/src/Config.rs b/src/Config.rs\n--- a/src/Config.rs\n\
                     +++ b/src/Config.rs\n@@ -1,2 +1,2 @@ fn loadFile\n-old line\n+newValue here\n";
        let t = tokens(&files, patch, "handle empty files");
        assert_eq!(t.msg[0], "m0:handle");
        assert!(t.msg.contains(&"m2:empty_files".to_string()));
        assert!(t.paths.contains(&"stext:Mrs".to_string()));
        assert!(t.paths.contains(&"dir:tests".to_string()));
        assert!(t.patch.contains(&"+:new".to_string()));
        assert!(t.patch.contains(&"-:old".to_string()));
        assert!(t.patch.contains(&"h:load".to_string()));
        assert!(t.meta.contains(&"cats:src+test".to_string()));
        assert!(t.meta.contains(&"catst:testA".to_string()));
        assert!(t.meta.contains(&"has_msg".to_string()));
        assert!(t.meta.contains(&"shape:mix".to_string()));
    }

    #[test]
    fn dependency_changes_are_marked() {
        let t = tokens(&[file('M', "Cargo.toml"), file('M', "Cargo.lock")], "", "");
        assert!(t.meta.contains(&"deps-only".to_string()));
        assert!(t.msg.is_empty());
    }

    #[test]
    fn buckets_are_logarithmic() {
        assert_eq!(
            [0, 1, 2, 3, 4, 1000, 1 << 20].map(bucket),
            [0, 1, 2, 2, 3, 10, 10]
        );
    }
}
