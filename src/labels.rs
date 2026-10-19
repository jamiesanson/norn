//! Which commits norn labelled, and whether you kept its label, so `norn learn` doesn't
//! learn its own guesses back as if you'd written them.
//!
//! `prepare-commit-msg` saves its suggestion to `.git/norn/pending`. Once the commit exists,
//! `post-commit` compares it with what was committed and appends a line to
//! `.git/norn/labels`:
//!
//!     <sha>  <suggested prefix>  <confidence>  <editor|message>  <kept|changed>
//!
//! Only commits made here are recorded, and rebased commits lose their record; both then
//! count as your own labels.

use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

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

fn path(name: &str) -> Option<PathBuf> {
    Some(learn::model_path()?.with_file_name(name))
}

/// Forgets any suggestion a commit that never happened left behind, so it can't be matched
/// to the next one.
pub fn clear_pending() {
    if let Some(p) = path("pending") {
        let _ = std::fs::remove_file(p);
    }
}

pub fn save_pending(kind: &str, scope: Option<&str>, confidence: f64, flow: Flow) {
    let Some(p) = path("pending") else { return };
    if let Some(dir) = p.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let line = format!(
        "{kind}\t{}\t{confidence:.2}\t{}\n",
        scope.unwrap_or_default(),
        flow.name()
    );
    let _ = std::fs::write(p, line);
}

/// For `post-commit`: records the pending suggestion against the commit just made.
pub fn record() {
    let Some(pending) = path("pending") else {
        return;
    };
    let Ok(saved) = std::fs::read_to_string(&pending) else {
        return;
    };
    let _ = std::fs::remove_file(&pending);
    let Some(line) = describe(saved.trim_end(), &head()) else {
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

/// `HEAD`'s sha and subject.
fn head() -> Option<(String, String)> {
    let output = Command::new("git")
        .args(["log", "-1", "--format=%H%x00%s"])
        .stderr(Stdio::null())
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&output.stdout);
    let (sha, subject) = text.trim_end().split_once('\0')?;
    Some((sha.to_string(), subject.to_string()))
}

/// The labels line for a saved suggestion and the commit it became.
fn describe(saved: &str, head: &Option<(String, String)>) -> Option<String> {
    let (sha, subject) = head.as_ref()?;
    let mut fields = saved.split('\t');
    let (kind, scope, confidence, flow) = (
        fields.next()?,
        fields.next()?,
        fields.next()?,
        fields.next()?,
    );
    let scope = (!scope.is_empty()).then_some(scope);
    let committed = message::parse_prefix(subject);
    let kept = committed.is_some_and(|p| p.kind == kind && p.scope == scope);
    let suggested = match scope {
        Some(s) => format!("{kind}({s})"),
        None => kind.to_string(),
    };
    let outcome = if kept { "kept" } else { "changed" };
    Some(format!(
        "{sha}\t{suggested}\t{confidence}\t{flow}\t{outcome}"
    ))
}

/// How much each recorded commit counts towards learning, by sha. Commits norn didn't
/// label aren't listed, and count as 1.
pub fn weights() -> HashMap<String, f64> {
    path("labels")
        .and_then(|p| std::fs::read_to_string(p).ok())
        .map(|text| parse_weights(&text))
        .unwrap_or_default()
}

fn parse_weights(text: &str) -> HashMap<String, f64> {
    text.lines()
        .filter_map(|line| {
            let fields: Vec<&str> = line.split('\t').collect();
            let weight = match (fields.get(3)?, fields.get(4)?) {
                (_, &"changed") => CHANGED,
                (&"editor", &"kept") => KEPT_IN_EDITOR,
                (_, &"kept") => KEPT_UNSEEN,
                _ => return None,
            };
            Some((fields[0].to_string(), weight))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn head(subject: &str) -> Option<(String, String)> {
        Some(("abc123".to_string(), subject.to_string()))
    }

    #[test]
    fn notices_kept_and_changed_labels() {
        assert_eq!(
            describe(
                "fix\tconfig\t0.87\teditor",
                &head("fix(config): handle empty files")
            ),
            Some("abc123\tfix(config)\t0.87\teditor\tkept".to_string())
        );
        assert_eq!(
            describe("fix\tconfig\t0.87\teditor", &head("refactor(config): tidy")),
            Some("abc123\tfix(config)\t0.87\teditor\tchanged".to_string())
        );
        assert_eq!(
            describe("feat\t\t0.64\tmessage", &head("feat(api): add a thing")),
            Some("abc123\tfeat\t0.64\tmessage\tchanged".to_string()),
            "adding a scope is a change"
        );
        assert_eq!(describe("fix\t\t0.9\teditor", &None), None);
    }

    #[test]
    fn corrections_count_most_and_unseen_labels_least() {
        let weights = parse_weights(
            "a\tfix\t0.87\teditor\tkept\n\
             b\tfeat\t0.64\tmessage\tkept\n\
             c\tfix\t0.58\teditor\tchanged\n\
             garbage\n",
        );
        assert_eq!(weights["a"], KEPT_IN_EDITOR);
        assert_eq!(weights["b"], KEPT_UNSEEN);
        assert_eq!(weights["c"], CHANGED);
        assert_eq!(weights.len(), 3);
    }

    #[test]
    fn flow_comes_from_the_hook_source() {
        assert_eq!(Flow::from_source(Some("message")), Flow::Message);
        assert_eq!(Flow::from_source(None), Flow::Editor);
        assert_eq!(Flow::from_source(Some("template")), Flow::Editor);
    }
}
