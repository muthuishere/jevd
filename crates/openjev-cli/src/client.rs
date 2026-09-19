//! The HTTP client the one-shot commands use — and the reference implementation of the
//! discovery order every other client should follow (design 02 §6.1).
//!
//! Organised against: a user waiting 90 s for a one-line answer while a warm server sits
//! idle on the same box.

use crate::api::*;
use crate::exit::{CliError, CliResult};
use std::path::Path;
use std::time::Duration;

pub struct Client {
    base: String,
    token: Option<String>,
    agent: ureq::Agent,
}

/// Ordered, and it stops at the first hit. Every step is confirmed with `/healthz`,
/// because a state file can outlive its process.
pub fn discover(explicit: Option<&str>, state_file: Option<&Path>) -> Option<(String, Discovery)> {
    if let Some(url) = explicit {
        return Some((url.trim_end_matches('/').to_string(), Discovery::Explicit));
    }
    let sf = state_file
        .map(Path::to_path_buf)
        .unwrap_or_else(crate::state::default_state_file);
    if let Some(state) = crate::state::read_state(&sf)
        && probe(&state.url)
    {
        return Some((state.url, Discovery::StateFile));
    }
    let default = format!("http://127.0.0.1:{}", crate::cli::DEFAULT_PORT);
    if probe(&default) {
        return Some((default, Discovery::DefaultPort));
    }
    None
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Discovery {
    Explicit,
    StateFile,
    DefaultPort,
}

impl Discovery {
    pub fn describe(self) -> &'static str {
        match self {
            Discovery::Explicit => "--server / OPENJEV_URL",
            Discovery::StateFile => "the state file",
            Discovery::DefaultPort => "the default port",
        }
    }
}

/// Reachability, not success: any HTTP answer at all means the host is there. Used for
/// the hub check, where a 401 or a 404 still proves the network works.
pub fn reachable(url: &str) -> bool {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(3)))
        .build()
        .into();
    !matches!(agent.get(url).call(), Err(e) if !matches!(e, ureq::Error::StatusCode(_)))
}

pub fn probe(base: &str) -> bool {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_millis(1500)))
        .build()
        .into();
    agent
        .get(&format!("{}/healthz", base.trim_end_matches('/')))
        .call()
        .is_ok()
}

impl Client {
    pub fn new(base: impl Into<String>) -> Self {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(300)))
            .build()
            .into();
        Client {
            base: base.into().trim_end_matches('/').to_string(),
            // A client that cannot be told a token cannot talk to a secured server, and
            // a flag would put it in `ps`.
            token: std::env::var("OPENJEV_TOKEN")
                .ok()
                .filter(|t| !t.is_empty()),
            agent,
        }
    }

    fn post<Req: serde::Serialize, Res: serde::de::DeserializeOwned>(
        &self,
        path: &str,
        body: &Req,
    ) -> CliResult<Res> {
        let mut req = self.agent.post(format!("{}{path}", self.base));
        if let Some(t) = &self.token {
            req = req.header("authorization", format!("Bearer {t}"));
        }
        match req.send_json(body) {
            Ok(mut resp) => resp
                .body_mut()
                .read_json::<Res>()
                .map_err(|e| CliError::other(format!("{path}: malformed response: {e}"))),
            Err(ureq::Error::StatusCode(code)) => {
                Err(CliError::other(format!("{path}: server returned {code}")))
            }
            Err(e) => Err(CliError::NoServer(format!(" at {}: {e}", self.base))),
        }
    }

    pub fn get_json<Res: serde::de::DeserializeOwned>(&self, path: &str) -> CliResult<Res> {
        let mut req = self.agent.get(format!("{}{path}", self.base));
        if let Some(t) = &self.token {
            req = req.header("authorization", format!("Bearer {t}"));
        }
        let mut resp = req
            .call()
            .map_err(|e| CliError::NoServer(format!(" at {}: {e}", self.base)))?;
        resp.body_mut()
            .read_json::<Res>()
            .map_err(|e| CliError::other(format!("{path}: malformed response: {e}")))
    }

    /// `/readyz` answers 503 while loading, which is information, not an error — so this
    /// reads the body either way.
    pub fn readyz(&self) -> CliResult<ReadyResponse> {
        let mut resp = self
            .agent
            .get(format!("{}/readyz", self.base))
            .call()
            .map_err(|e| match e {
                ureq::Error::StatusCode(_) => e,
                other => other,
            });
        match &mut resp {
            Ok(r) => r
                .body_mut()
                .read_json::<ReadyResponse>()
                .map_err(|e| CliError::other(format!("/readyz: {e}"))),
            Err(ureq::Error::StatusCode(503)) => Ok(ReadyResponse {
                ready: false,
                phase: "unknown".into(),
                detail: None,
                since: crate::util::rfc3339_now(),
                error: None,
            }),
            Err(e) => Err(CliError::NoServer(format!(" at {}: {e}", self.base))),
        }
    }

    pub fn predict(&self, req: &PredictRequest) -> CliResult<PredictResponse> {
        self.post("/v1/predict", req)
    }
    pub fn rerank(&self, req: &RerankRequest) -> CliResult<RerankResponse> {
        self.post("/v1/rerank", req)
    }
    pub fn grade(&self, req: &GradeRequest) -> CliResult<GradeResponse> {
        self.post("/v1/grade", req)
    }
    pub fn info(&self) -> CliResult<InfoResponse> {
        self.get_json("/v1/info")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_explicit_url_short_circuits_discovery_and_is_not_probed() {
        let (url, how) = discover(Some("http://example.invalid:9/"), None).expect("explicit");
        assert_eq!(url, "http://example.invalid:9");
        assert_eq!(how, Discovery::Explicit);
    }

    #[test]
    fn a_state_file_pointing_at_a_dead_process_is_not_believed() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("server.json");
        crate::state::write_state(
            &p,
            &crate::state::ServerState {
                schema: 1,
                pid: 999_999,
                // Port 1 on loopback: reserved, never listening.
                url: "http://127.0.0.1:1".into(),
                bound_addr: "127.0.0.1:1".into(),
                api_version: 1,
                server_version: "0".into(),
                model: "m".into(),
                revision: "r".into(),
                device: "cpu".into(),
                auth: "none".into(),
                started_at: crate::util::rfc3339_now(),
            },
        )
        .unwrap();
        assert!(!probe("http://127.0.0.1:1"));
    }
}
