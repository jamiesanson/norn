//! Reading and rewriting commit message files.

const COMMENT_CHAR: char = '#';

/// Messages git or the user produced that shouldn't be prefixed.
const SKIP_PREFIXES: &[&str] = &["Merge ", "Revert \"", "fixup! ", "squash! ", "amend! "];

/// The message a user wrote: every non-comment line, trimmed.
pub fn body(content: &str) -> String {
    content
        .lines()
        .take_while(|l| !l.starts_with("# ------------------------ >8"))
        .filter(|l| !l.starts_with(COMMENT_CHAR))
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

/// The parts of a `type(scope)!: description` subject.
#[derive(Debug, PartialEq, Eq)]
pub struct Prefix<'a> {
    pub kind: &'a str,
    pub scope: Option<&'a str>,
    pub breaking: bool,
    pub description: &'a str,
}

pub fn parse_prefix(subject: &str) -> Option<Prefix<'_>> {
    let (head, description) = subject.split_once(": ")?;
    let (head, breaking) = match head.strip_suffix('!') {
        Some(head) => (head, true),
        None => (head, false),
    };
    let (kind, scope) = match head.split_once('(') {
        Some((kind, rest)) => match rest.strip_suffix(')') {
            Some(scope) if !scope.is_empty() && !scope.contains(['(', ')']) => (kind, Some(scope)),
            _ => return None,
        },
        None => (head, None),
    };
    let valid = !kind.is_empty()
        && kind.chars().all(|c| c.is_ascii_lowercase())
        && scope.is_none_or(|s| !s.contains(' '));
    valid.then_some(Prefix {
        kind,
        scope,
        breaking,
        description,
    })
}

/// Whether `subject` already starts with `type(scope)!: `.
pub fn is_conventional(subject: &str) -> bool {
    parse_prefix(subject).is_some()
}

pub fn should_skip(subject: &str) -> bool {
    is_conventional(subject) || SKIP_PREFIXES.iter().any(|p| subject.starts_with(p))
}

/// Puts `prefix` in front of the first non-comment line, or adds it as the first line
/// when the user hasn't written anything yet.
pub fn apply_prefix(content: &str, prefix: &str) -> String {
    let mut lines: Vec<String> = content.lines().map(str::to_string).collect();
    match lines
        .iter()
        .position(|l| !l.starts_with(COMMENT_CHAR) && !l.trim().is_empty())
    {
        Some(i) => lines[i] = format!("{prefix}{}", lines[i].trim_start()),
        None => lines.insert(0, prefix.trim_end().to_string() + " "),
    }
    let mut out = lines.join("\n");
    if content.ends_with('\n') || content.is_empty() {
        out.push('\n');
    }
    out
}

/// True when the message is nothing but a prefix this tool added, so the user
/// never wrote a description. Clearing it lets git abort as for any empty message.
/// `content` with `hint` as a comment line, just above git's own comments, where the editor
/// shows it and git leaves it out of the commit.
pub fn add_hint(content: &str, hint: &str) -> String {
    let mut lines: Vec<&str> = content.lines().collect();
    let line = format!("{COMMENT_CHAR} {hint}");
    let at = lines
        .iter()
        .position(|l| l.starts_with(COMMENT_CHAR))
        .unwrap_or(lines.len());
    lines.insert(at, &line);
    let mut out = lines.join("\n");
    out.push('\n');
    out
}

pub fn is_bare_prefix(content: &str) -> bool {
    let body = body(content);
    !body.contains('\n') && body.ends_with(':') && is_conventional(&format!("{body} x"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_prefix_parts() {
        assert_eq!(
            parse_prefix("feat(api)!: drop v1"),
            Some(Prefix {
                kind: "feat",
                scope: Some("api"),
                breaking: true,
                description: "drop v1",
            })
        );
        assert_eq!(parse_prefix("fix: typo").map(|p| p.scope), Some(None));
        assert_eq!(parse_prefix("Update README"), None);
    }

    #[test]
    fn detects_conventional_subjects() {
        assert!(is_conventional("feat: add thing"));
        assert!(is_conventional("fix(parser): handle eof"));
        assert!(is_conventional("feat(api)!: drop v1"));
        assert!(is_conventional("refactor!: rename module"));
        assert!(!is_conventional("Add thing"));
        assert!(!is_conventional("Note: this is a sentence"));
        assert!(!is_conventional("fix(): empty scope"));
        assert!(!is_conventional("feat(two words): nope"));
    }

    #[test]
    fn prefixes_first_written_line() {
        let content = "handle empty files\n\nMore detail.\n# Please enter...\n";
        assert_eq!(
            apply_prefix(content, "fix(config): "),
            "fix(config): handle empty files\n\nMore detail.\n# Please enter...\n"
        );
    }

    #[test]
    fn inserts_prefix_into_empty_editor_template() {
        let content = "\n# Please enter the commit message\n";
        assert_eq!(
            apply_prefix(content, "feat: "),
            "feat: \n\n# Please enter the commit message\n"
        );
    }

    #[test]
    fn body_ignores_comments_and_scissors() {
        let content =
            "subject\n# comment\n# ------------------------ >8 ------------------------\ndiff";
        assert_eq!(body(content), "subject");
    }

    #[test]
    fn recognises_bare_prefix() {
        assert!(is_bare_prefix("feat(api)!: \n\n# comment\n"));
        assert!(is_bare_prefix("docs:"));
        assert!(!is_bare_prefix("feat: real description\n"));
        assert!(!is_bare_prefix("\n# only comments\n"));
    }

    #[test]
    fn skips_git_generated_messages() {
        assert!(should_skip("Merge branch 'main'"));
        assert!(should_skip("fixup! feat: thing"));
        assert!(!should_skip("add thing"));
    }

    #[test]
    fn hints_go_above_gits_comments() {
        let template = "\n# Please enter the commit message for your changes.\n#\n";
        let hinted = add_hint(template, "norn isn't sure: fix (48%) or feat (31%)");
        assert_eq!(
            hinted,
            "\n# norn isn't sure: fix (48%) or feat (31%)\n# Please enter the commit message for your changes.\n#\n"
        );
        assert_eq!(body(&hinted), "");
        assert_eq!(add_hint("wip", "h"), "wip\n# h\n");
    }
}
