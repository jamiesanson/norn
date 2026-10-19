//! The local classifier: TF-IDF over each block of `features::Tokens`, then a linear
//! layer and softmax. `training/train_linear.py` trains it and writes the file format
//! read here.
//!
//! Format, little-endian:
//!   b"JCCM", u32 version
//!   u32 classes, then each class name
//!   u32 blocks, then each: name, f32 weight, u32 terms, each term, then f32 idf per term
//!   f32 weights, feature-major (all classes for feature 0, then feature 1, ...), features
//!   numbered through the blocks in order
//!   f32 intercept per class
//! Strings are a u16 byte length and UTF-8 bytes.

use std::collections::HashMap;
use std::sync::LazyLock;

use crate::features::Tokens;

const MAGIC: &[u8; 4] = b"JCCM";
const VERSION: u32 = 1;

/// The model trained on public conventional-commit history, built into norn.
pub fn shared() -> Result<&'static Model, String> {
    static SHARED: LazyLock<Result<Model, String>> =
        LazyLock::new(|| Model::from_bytes(include_bytes!("../models/shared.bin")));
    SHARED.as_ref().map_err(|e| format!("built-in model: {e}"))
}

pub struct Model {
    pub classes: Vec<String>,
    blocks: Vec<Block>,
    idf: Vec<f32>,
    /// `weights[feature * classes + class]`.
    weights: Vec<f32>,
    intercepts: Vec<f32>,
}

struct Block {
    name: String,
    weight: f32,
    /// Term to feature number.
    terms: HashMap<String, usize>,
}

impl Model {
    pub fn from_bytes(bytes: &[u8]) -> Result<Model, String> {
        let mut r = Reader { bytes, at: 0 };
        if r.take(4)? != MAGIC {
            return Err("not a norn model".into());
        }
        let version = r.u32()?;
        if version != VERSION {
            return Err(format!("model format {version} isn't supported"));
        }
        let classes = (0..r.u32()?)
            .map(|_| r.str())
            .collect::<Result<Vec<_>, _>>()?;
        let mut blocks = Vec::new();
        let mut idf = Vec::new();
        for _ in 0..r.u32()? {
            let name = r.str()?;
            let weight = r.f32()?;
            let n = r.u32()? as usize;
            let mut terms = HashMap::with_capacity(n);
            for _ in 0..n {
                terms.insert(r.str()?, idf.len() + terms.len());
            }
            for _ in 0..n {
                idf.push(r.f32()?);
            }
            blocks.push(Block {
                name,
                weight,
                terms,
            });
        }
        let weights = (0..idf.len() * classes.len())
            .map(|_| r.f32())
            .collect::<Result<Vec<_>, _>>()?;
        let intercepts = (0..classes.len())
            .map(|_| r.f32())
            .collect::<Result<Vec<_>, _>>()?;
        if r.at != bytes.len() {
            return Err("model file has trailing data".into());
        }
        Ok(Model {
            classes,
            blocks,
            idf,
            weights,
            intercepts,
        })
    }

    /// The probability of each class, in `classes` order.
    pub fn predict(&self, tokens: &Tokens) -> Vec<f64> {
        let k = self.classes.len();
        let mut scores: Vec<f64> = self.intercepts.iter().map(|&b| b as f64).collect();
        for block in &self.blocks {
            let mut counts: HashMap<usize, u32> = HashMap::new();
            for token in tokens.block(&block.name) {
                if let Some(&f) = block.terms.get(token) {
                    *counts.entry(f).or_default() += 1;
                }
            }
            let values: Vec<(usize, f64)> = counts
                .into_iter()
                .map(|(f, n)| (f, (1.0 + (n as f64).ln()) * self.idf[f] as f64))
                .collect();
            let norm = values.iter().map(|(_, v)| v * v).sum::<f64>().sqrt();
            if norm == 0.0 {
                continue;
            }
            let scale = block.weight as f64 / norm;
            for (f, v) in values {
                let w = &self.weights[f * k..(f + 1) * k];
                for (score, &w) in scores.iter_mut().zip(w) {
                    *score += v * scale * w as f64;
                }
            }
        }
        softmax(&scores)
    }
}

fn softmax(scores: &[f64]) -> Vec<f64> {
    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
    let total: f64 = exp.iter().sum();
    exp.into_iter().map(|e| e / total).collect()
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], String> {
        let end = self.at + n;
        let slice = self
            .bytes
            .get(self.at..end)
            .ok_or("model file is truncated")?;
        self.at = end;
        Ok(slice)
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn str(&mut self) -> Result<String, String> {
        let len = u16::from_le_bytes(self.take(2)?.try_into().unwrap()) as usize;
        String::from_utf8(self.take(len)?.to_vec())
            .map_err(|_| "model file has invalid text".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A block's name, weight, and terms with their idf.
    type TestBlock<'a> = (&'a str, f32, &'a [(&'a str, f32)]);

    /// A model file, written the way `training/train_linear.py` writes one.
    fn write(
        classes: &[&str],
        blocks: &[TestBlock],
        weights: &[f32],
        intercepts: &[f32],
    ) -> Vec<u8> {
        let mut out = MAGIC.to_vec();
        let s = |out: &mut Vec<u8>, s: &str| {
            out.extend((s.len() as u16).to_le_bytes());
            out.extend(s.as_bytes());
        };
        out.extend(VERSION.to_le_bytes());
        out.extend((classes.len() as u32).to_le_bytes());
        for c in classes {
            s(&mut out, c);
        }
        out.extend((blocks.len() as u32).to_le_bytes());
        for (name, weight, terms) in blocks {
            s(&mut out, name);
            out.extend(weight.to_le_bytes());
            out.extend((terms.len() as u32).to_le_bytes());
            for (term, _) in *terms {
                s(&mut out, term);
            }
            for (_, idf) in *terms {
                out.extend(idf.to_le_bytes());
            }
        }
        for v in weights.iter().chain(intercepts) {
            out.extend(v.to_le_bytes());
        }
        out
    }

    #[test]
    fn predicts_like_tfidf_and_softmax() {
        // One block, two terms. "m:fix" twice and "m:add" once.
        let bytes = write(
            &["feat", "fix"],
            &[("msg", 2.0, &[("m:add", 1.5), ("m:fix", 1.0)])],
            &[1.0, -1.0, -2.0, 3.0],
            &[0.1, 0.0],
        );
        let model = Model::from_bytes(&bytes).unwrap();
        let tokens = Tokens {
            msg: vec![
                "m:fix".into(),
                "m:fix".into(),
                "m:add".into(),
                "m:unknown".into(),
            ],
            ..Tokens::default()
        };
        let p = model.predict(&tokens);

        let add = 1.5;
        let fix = (1.0 + 2f64.ln()) * 1.0;
        let norm = (add * add + fix * fix).sqrt();
        let (add, fix) = (add / norm * 2.0, fix / norm * 2.0);
        let feat = 0.1 + add * 1.0 + fix * -2.0;
        let fixs = -add + fix * 3.0;
        let expected = 1.0 / (1.0 + (feat - fixs).exp());
        assert!((p[1] - expected).abs() < 1e-6, "{p:?}");
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn empty_input_gives_the_intercepts() {
        let bytes = write(
            &["a", "b"],
            &[("msg", 1.0, &[("m:x", 1.0)])],
            &[5.0, 5.0],
            &[0.0, 0.0],
        );
        let p = Model::from_bytes(&bytes)
            .unwrap()
            .predict(&Tokens::default());
        assert_eq!(p, [0.5, 0.5]);
    }

    #[test]
    fn built_in_model_knows_the_built_in_types() {
        let model = shared().unwrap();
        for (name, _) in crate::config::BUILT_IN_TYPES {
            assert!(model.classes.iter().any(|c| c == name), "{name}");
        }
    }

    #[test]
    fn rejects_bad_files() {
        assert!(Model::from_bytes(b"nope").is_err());
        let mut bytes = write(&["a"], &[], &[], &[0.0]);
        bytes.push(0);
        assert!(Model::from_bytes(&bytes).is_err());
        assert!(Model::from_bytes(&bytes[..bytes.len() - 3]).is_err());
    }
}
