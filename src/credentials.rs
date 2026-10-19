//! Stores the TypeSafe API key in a user config file, so hooks run from git GUIs (which
//! don't inherit the shell's environment) can still find it.
//!
//! A file readable only by the user, as `gh` and `cargo` do, rather than the OS keychain:
//! reading the keychain means spawning a process on every commit.

use std::path::{Path, PathBuf};

const ENV_VAR: &str = "TYPESAFE_API_KEY";
const FILE_NAME: &str = "api-key";

/// `TYPESAFE_API_KEY` if set, otherwise the stored key.
pub fn api_key() -> Option<String> {
    std::env::var(ENV_VAR)
        .ok()
        .filter(|k| !k.trim().is_empty())
        .or_else(|| load(&crate::config::dir()?))
}

pub fn load(dir: &Path) -> Option<String> {
    let key = std::fs::read_to_string(dir.join(FILE_NAME)).ok()?;
    let key = key.trim();
    (!key.is_empty()).then(|| key.to_string())
}

/// Returns the path the key was written to.
pub fn save(dir: &Path, key: &str) -> Result<PathBuf, String> {
    create_private_dir(dir).map_err(|e| format!("couldn't create {}: {e}", dir.display()))?;
    let path = dir.join(FILE_NAME);
    let tmp = dir.join(format!("{FILE_NAME}.tmp"));
    write_private(&tmp, key.trim())
        .and_then(|()| std::fs::rename(&tmp, &path))
        .map_err(|e| {
            let _ = std::fs::remove_file(&tmp);
            format!("couldn't write {}: {e}", path.display())
        })?;
    Ok(path)
}

/// Returns the removed path, or `None` if there was no stored key.
pub fn delete(dir: &Path) -> Result<Option<PathBuf>, String> {
    let path = dir.join(FILE_NAME);
    match std::fs::remove_file(&path) {
        Ok(()) => Ok(Some(path)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(format!("couldn't remove {}: {e}", path.display())),
    }
}

#[cfg(unix)]
fn create_private_dir(dir: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::DirBuilderExt;
    std::fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(dir)
}

#[cfg(not(unix))]
fn create_private_dir(dir: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)
}

/// Creates the file with owner-only permissions from the start, so the key is never
/// briefly readable by others.
#[cfg(unix)]
fn write_private(path: &Path, contents: &str) -> std::io::Result<()> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(path)?;
    file.write_all(contents.as_bytes())?;
    file.write_all(b"\n")
}

#[cfg(not(unix))]
fn write_private(path: &Path, contents: &str) -> std::io::Result<()> {
    std::fs::write(path, format!("{contents}\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("norn-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn save_load_delete_round_trip() {
        let dir = temp_dir("round-trip").join("norn");
        assert_eq!(load(&dir), None);
        save(&dir, "  ts_key_123\n").unwrap();
        assert_eq!(load(&dir).as_deref(), Some("ts_key_123"));
        assert!(delete(&dir).unwrap().is_some());
        assert_eq!(load(&dir), None);
        assert!(delete(&dir).unwrap().is_none());
        let _ = std::fs::remove_dir_all(dir.parent().unwrap());
    }

    #[cfg(unix)]
    #[test]
    fn key_file_is_private() {
        use std::os::unix::fs::PermissionsExt;
        let dir = temp_dir("perms").join("norn");
        let path = save(&dir, "secret").unwrap();
        let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode(&path), 0o600);
        assert_eq!(mode(&dir), 0o700);
        let _ = std::fs::remove_dir_all(dir.parent().unwrap());
    }

    #[test]
    fn blank_file_is_no_key() {
        let dir = temp_dir("blank");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(FILE_NAME), "\n").unwrap();
        assert_eq!(load(&dir), None);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
