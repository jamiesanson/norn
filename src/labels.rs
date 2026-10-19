//! Which commits norn labelled, and whether you kept its label, so `norn learn` doesn't
//! learn its own guesses back as if you'd written them.
//!
//! `prepare-commit-msg` saves its suggestion to `.git/norn/pending`. `commit-msg`, which
//! sees the message you actually committed, compares the two and appends a line to
//! `.git/norn/labels`:
//!
//!     <parent>  <subject>  <suggested prefix>  <confidence>  <editor|message>  <kept|changed>
//!
//! The commit doesn't exist yet at that point, so it's identified by its parent and
//! subject, which `norn learn` can match. Only commits made here are recorded, and
//! rebased commits lose their record; both then count as your own labels.

use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;

use crate::learn;
use crate::message;

/// A correction is the most useful thing to learn from: it's where the model was wrong.
const CHANGED: f64 = 2.0;
/// Seen in the editor and accepted: weak evidence.
const KEPT_IN_EDITOR: f64 = 0.5;
/// Applied to `git commit -m` without being seen: it only says what norn guessed.
const KEPT_UNSEEN: f64 = 0.2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flow {
    /// The message was written in the editor, so the prefix was seen before committing.
    Editor,
    /// `-m` or `-F`: committed without being seen.
    Message,
}

impl Flow {
    pub fn from_source(source: Option<&str>) -> Flow {
        if matches!(source, Some("message")) {
            Flow::Message
        } else {
            Flow::Editor
        }
    }

    fn name(self) -> &'static str {
        match self {
            Flow::Editor => "editor",
            Flow::Message => "message",
        }
    }
}

/// How `norn learn` finds a recorded commit: its first parent and its subject.
pub type Key = (String, String);

fn path(name: &str) -> Option<PathBuf> {
    Some(learn::model_path()?.with_file_name(name))
}

/// Forgets a saved suggestion, so a commit that never happened can't be matched to the
/// next one.
pub fn clear_pending() {
    if let Some(p) = path("pending") {
        let _ = std::fs::remove_file(p);
    }
}

/// `parent` is `HEAD` before the commit, or empty for a repository's first commit.
pub fn save_pending(kind: &str, scope: Option<&str>, confidence: f64, flow: Flow, parent: &str) {
    let Some(p) = path("pending") else { return };
    if let Some(dir) = p.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let line = format!(
        "{kind}\t{}\t{confidence:.2}\t{}\t{parent}\n",
        scope.unwrap_or_default(),
        flow.name()
    );
    let _ = std::fs::write(p, line);
}

/// For `commit-msg`: records the saved suggestion against the message being committed.
pub fn record(content: &str) {
    let Some(pending) = path("pending") else {
        return;
    };
    let Ok(saved) = std::fs::read_to_string(&pending) else {
        return;
    };
    let _ = std::fs::remove_file(&pending);
    let body = message::body(content);
    let subject = body.lines().next().unwrap_or_default();
    let Some(line) = describe(saved.trim_end_matches('\n'), subject) else {
        return;
    };
    let Some(labels) = path("labels") else { return };
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(labels)
    {
        let _ = writeln!(file, "{line}");
    }
}

/// The labels line for a saved suggestion and the subject it became.
fn describe(saved: &str, subject: &str) -> Option<String> {
    let mut fields = saved.split('\t');
    let (kind, scope, confidence, flow, parent) = (
        fields.next()?,
        fields.next()?,
        fields.next()?,
        fields.next()?,
        fields.next()?,
    );
    let scope = (!scope.is_empty()).then_some(scope);
    let subject = subject.trim_end().replace('\t', " ");
    let committed = message::parse_prefix(&subject);
    let kept = committed.is_some_and(|p| p.kind == kind && p.scope == scope);
    let suggested = match scope {
        Some(s) => format!("{kind}({s})"),
        None => kind.to_string(),
    };
    let outcome = if kept { "kept" } else { "changed" };
    Some(format!(
        "{parent}\t{subject}\t{suggested}\t{confidence}\t{flow}\t{outcome}"
    ))
}

/// How much each recorded commit counts towards learning. Commits norn didn't label aren't
/// listed, and count as 1.
pub fn weights() -> HashMap<Key, f64> {
    path("labels")
        .and_then(|p| std::fs::read_to_string(p).ok())
        .map(|text| parse_weights(&text))
        .unwrap_or_default()
}

fn parse_weights(text: &str) -> HashMap<Key, f64> {
    text.lines()
        .filter_map(|line| {
            let fields: Vec<&str> = line.split('\t').collect();
            let weight = match (fields.get(4)?, fields.get(5)?) {
                (_, &"changed") => CHANGED,
                (&"editor", &"kept") => KEPT_IN_EDITOR,
                (_, &"kept") => KEPT_UNSEEN,
                _ => return None,
            };
            Some(((fields[0].to_string(), fields[1].to_string()), weight))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn notices_kept_and_changed_labels() {
        assert_eq!(
            describe(
                "fix\tconfig\t0.87\teditor\tabc",
                "fix(config): handle empty files"
            ),
            Some("abc\tfix(config): handle empty files\tfix(config)\t0.87\teditor\tkept".into())
        );
        assert_eq!(
            describe("fix\tconfig\t0.87\teditor\tabc", "refactor(config): tidy"),
            Some("abc\trefactor(config): tidy\tfix(config)\t0.87\teditor\tchanged".into())
        );
        assert_eq!(
            describe("feat\t\t0.64\tmessage\t", "feat(api): add a thing"),
            Some("\tfeat(api): add a thing\tfeat\t0.64\tmessage\tchanged".into()),
            "adding a scope is a change"
        );
        assert_eq!(describe("garbage", "fix: x"), None);
    }

    #[test]
    fn corrections_count_most_and_unseen_labels_least() {
        let weights = parse_weights(
            "a\tfix: x\tfix\t0.87\teditor\tkept\n\
             b\tfeat: y\tfeat\t0.64\tmessage\tkept\n\
             c\trefactor: z\tfix\t0.58\teditor\tchanged\n\
             garbage\n",
        );
        let key = |p: &str, s: &str| (p.to_string(), s.to_string());
        assert_eq!(weights[&key("a", "fix: x")], KEPT_IN_EDITOR);
        assert_eq!(weights[&key("b", "feat: y")], KEPT_UNSEEN);
        assert_eq!(weights[&key("c", "refactor: z")], CHANGED);
        assert_eq!(weights.len(), 3);
    }

    #[test]
    fn flow_comes_from_the_hook_source() {
        assert_eq!(Flow::from_source(Some("message")), Flow::Message);
        assert_eq!(Flow::from_source(None), Flow::Editor);
        assert_eq!(Flow::from_source(Some("template")), Flow::Editor);
    }
}
