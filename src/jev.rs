//! Minimal blocking client for TypeSafe's System One API.
//!
//! One endpoint, one request per commit: `POST {base}/v1/systemone`.
//! See <https://docs.typesafe.ai/api>.

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

pub struct Client {
    agent: ureq::Agent,
    url: String,
    api_key: String,
}

impl Client {
    pub fn new(api_key: String, base_url: &str, timeout: Duration) -> Self {
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
        let mut response = self
            .agent
            .post(&self.url)
            .header("Authorization", &format!("Bearer {}", self.api_key))
            .send_json(request)
            .map_err(Error::Http)?;

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
}
