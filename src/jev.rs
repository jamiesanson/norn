//! Minimal blocking client for TypeSafe's System One API.
//!
//! One endpoint, one request per commit: `POST {base}/v1/systemone`.
//! See <https://docs.typesafe.ai/api>. Servers that speak the same protocol, such as
//! [Laya](https://github.com/NandhaKishorM/laya)'s `laya-serve`, work too.

use std::collections::BTreeMap;
use std::time::Duration;

use serde::{Deserialize, Serialize};

pub const DEFAULT_BASE_URL: &str = "https://api.typesafe.ai";
pub const DEFAULT_MODEL: &str = "jev-latest";

#[derive(Debug, Serialize)]
pub struct Request<'a> {
    pub model: &'a str,
    pub state: serde_json::Value,
    pub questions: BTreeMap<&'a str, Question<'a>>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Question<'a> {
    /// Yes/no. Answered with a probability of "yes".
    Noul {
        instructions: &'a str,
        criteria: NoulCriteria<'a>,
    },
    /// Pick one label from up to 255.
    Choice {
        instructions: &'a str,
        criteria: BTreeMap<&'a str, &'a str>,
    },
}

#[derive(Debug, Serialize)]
pub struct NoulCriteria<'a> {
    #[serde(rename = "true")]
    pub yes: &'a str,
    #[serde(rename = "false")]
    pub no: &'a str,
}

#[derive(Debug, Deserialize)]
pub struct Response {
    pub answers: BTreeMap<String, Answer>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    Noul {
        noul: f64,
    },
    Choice {
        choice: String,
        confidence: f64,
        #[serde(default)]
        probabilities: BTreeMap<String, f64>,
        /// Laya's `confidence` is normalised entropy, on a different scale from Jev's. It
        /// also sends Jev's formula under this name, so the same thresholds work on both.
        #[serde(default)]
        x_jev_confidence: Option<f64>,
    },
    /// Question types this tool doesn't ask, kept so parsing never fails on them.
    #[serde(other)]
    Other,
}

#[derive(Debug)]
pub enum Error {
    MissingApiKey,
    Http(ureq::Error),
    Status(u16, String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::MissingApiKey => write!(f, "no API key; run `norn jev login`"),
            Error::Http(e) => write!(f, "request failed: {e}"),
            Error::Status(code, body) => write!(f, "API returned {code}: {body}"),
        }
    }
}

/// Whether `base_url` is TypeSafe's API, which needs a key. Other servers may not.
pub fn requires_api_key(base_url: &str) -> bool {
    base_url.trim_end_matches('/') == DEFAULT_BASE_URL
}

pub struct Client {
    agent: ureq::Agent,
    url: String,
    api_key: Option<String>,
}

impl Client {
    pub fn new(api_key: Option<String>, base_url: &str, timeout: Duration) -> Self {
        let config = ureq::Agent::config_builder()
            .timeout_global(Some(timeout))
            .http_status_as_error(false)
            .build();
        Client {
            agent: ureq::Agent::new_with_config(config),
            url: format!("{}/v1/systemone", base_url.trim_end_matches('/')),
            api_key,
        }
    }

    pub fn system_one(&self, request: &Request) -> Result<Response, Error> {
        let mut post = self.agent.post(&self.url);
        if let Some(key) = &self.api_key {
            post = post.header("Authorization", &format!("Bearer {key}"));
        }
        let mut response = post.send_json(request).map_err(Error::Http)?;

        let status = response.status().as_u16();
        if status != 200 {
            let body = response.body_mut().read_to_string().unwrap_or_default();
            return Err(Error::Status(status, body));
        }
        response.body_mut().read_json().map_err(Error::Http)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_questions_with_type_tags() {
        let mut questions = BTreeMap::new();
        questions.insert(
            "breaking",
            Question::Noul {
                instructions: "Is it breaking?",
                criteria: NoulCriteria {
                    yes: "Yes",
                    no: "No",
                },
            },
        );
        let request = Request {
            model: DEFAULT_MODEL,
            state: serde_json::json!("diff"),
            questions,
        };
        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(json["questions"]["breaking"]["type"], "noul");
        assert_eq!(json["questions"]["breaking"]["criteria"]["true"], "Yes");
    }

    #[test]
    fn parses_documented_response() {
        let body = r#"{
            "model": "jev-1.13.0",
            "answers": {
                "is_urgent": { "type": "noul", "noul": 0.95 },
                "department": {
                    "type": "choice",
                    "choice": "billing",
                    "probabilities": { "billing": 0.88, "technical": 0.12 },
                    "confidence": 0.81
                },
                "frustration": { "type": "score", "score": 1.05, "confidence": 0.92 }
            },
            "usage": { "input_tokens": 304, "output_tokens": 18 }
        }"#;
        let response: Response = serde_json::from_str(body).unwrap();
        assert!(matches!(response.answers["is_urgent"], Answer::Noul { noul } if noul == 0.95));
        assert!(
            matches!(&response.answers["department"], Answer::Choice { choice, .. } if choice == "billing")
        );
        assert!(matches!(response.answers["frustration"], Answer::Other));
    }

    #[test]
    fn parses_laya_response() {
        let body = r#"{
            "model": "laya-rl-agent",
            "answers": {
                "queue": {"type": "choice", "choice": "billing",
                          "probabilities": {"billing": 0.9519, "tech": 0.0327, "other": 0.0154},
                          "confidence": 0.797, "answer_confidence": 0.9519,
                          "action": {"act_probability": 1.0}, "x_jev_confidence": 0.9278},
                "urgent": {"type": "noul", "noul": 0.12, "confidence": 0.6}
            },
            "usage": {"input_tokens": 83, "output_tokens": 0, "truncated": false},
            "routing": {"model": "english", "reason": "English Latin text"}
        }"#;
        let response: Response = serde_json::from_str(body).unwrap();
        assert!(matches!(
            response.answers["queue"],
            Answer::Choice { x_jev_confidence: Some(c), .. } if c == 0.9278
        ));
        assert!(matches!(response.answers["urgent"], Answer::Noul { noul } if noul == 0.12));
    }

    #[test]
    fn only_typesafe_requires_a_key() {
        assert!(requires_api_key(DEFAULT_BASE_URL));
        assert!(requires_api_key("https://api.typesafe.ai/"));
        assert!(!requires_api_key("http://localhost:8000"));
    }
}
