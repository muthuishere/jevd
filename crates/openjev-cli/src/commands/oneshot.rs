//! `predict`, `rerank`, `grade`: the same code path as the HTTP handlers, so
//! `openjev` composes in a pipeline without anyone standing up a server.

use crate::api::*;
use crate::cli::{GradeArgs, PredictArgs, RerankArgs, Target};
use crate::client::{Client, discover};
use crate::exit::{self, CliError, CliResult};
use crate::runtime::LoadSpec;
use openjev_core::Session;
use std::io::BufRead;

fn load_spec(t: &Target, cfg: &crate::config::Layered) -> LoadSpec {
    let mut load = LoadSpec::from_config(cfg);
    if t.model.is_some() {
        load.model = t.model.clone();
    }
    if t.device.is_some() {
        load.device = t.device;
    }
    load.offline = t.offline;
    load
}

/// Attach to a warm server when there is one, unless told not to. Says which route it
/// took, on stderr, because a silent choice between two very different latencies is a
/// mystery waiting to be filed as a bug.
fn connect(t: &Target) -> Option<Client> {
    if t.local {
        return None;
    }
    let (url, how) = discover(t.server.as_deref(), None)?;
    eprintln!("using server {url} (found via {})", how.describe());
    Some(Client::new(url))
}

pub fn predict(
    args: &PredictArgs,
    cfg: &crate::config::Layered,
    json: bool,
    assume_yes: bool,
) -> CliResult<i32> {
    let inputs = read_pairs(args)?;
    if inputs.is_empty() {
        return Err(CliError::bad_input(
            "no pairs: pass --premise/--hypothesis or NDJSON on stdin",
        ));
    }
    // One malformed line must not kill line 90,000 of a batch job, so bad lines are
    // carried through as error objects and settled at the end.
    let mut had_bad_input = inputs.iter().any(|i| i.is_err());

    let good: Vec<Pair> = inputs
        .iter()
        .filter_map(|i| i.as_ref().ok().cloned())
        .collect();
    let results = match connect(&args.target) {
        Some(client) => {
            let resp = client.predict(&PredictRequest {
                pairs: good.clone(),
                truncate: match args.truncate {
                    crate::cli::TruncateArg::Error => Truncate::Error,
                    crate::cli::TruncateArg::Tail => Truncate::Tail,
                },
            })?;
            resp.results
        }
        None => {
            let session =
                crate::commands::local_session(&load_spec(&args.target, cfg), assume_yes)?;
            local_predict(&session, &good)?
        }
    };

    // Output order is the input order, always: batching happens internally and must not
    // show through.
    let mut out = std::io::stdout().lock();
    let mut ri = results.into_iter();
    for input in &inputs {
        match input {
            Ok(_) => {
                let Some(r) = ri.next() else { break };
                write_predict(&mut out, &r, json)?;
            }
            Err(msg) => {
                had_bad_input = true;
                let line =
                    serde_json::json!({"error": {"code": "invalid_request", "message": msg}});
                writeln_line(&mut out, &line.to_string())?;
            }
        }
    }
    Ok(if had_bad_input {
        exit::BAD_INPUT
    } else {
        exit::OK
    })
}

fn writeln_line(out: &mut impl std::io::Write, s: &str) -> CliResult<()> {
    writeln!(out, "{s}").map_err(CliError::Io)
}

fn write_predict(out: &mut impl std::io::Write, r: &PredictResult, json: bool) -> CliResult<()> {
    if json {
        writeln_line(out, &serde_json::to_string(r).unwrap_or_default())
    } else {
        let mut scores: Vec<String> = r
            .scores
            .iter()
            .map(|(k, v)| format!("{k}={v:.3}"))
            .collect();
        scores.sort();
        writeln_line(
            out,
            &format!(
                "{:<4} {:<14} {}",
                r.id.clone().unwrap_or_else(|| r.index.to_string()),
                r.label,
                scores.join("  ")
            ),
        )
    }
}

fn local_predict(session: &Session, pairs: &[Pair]) -> CliResult<Vec<PredictResult>> {
    let refs: Vec<(&str, &str)> = pairs
        .iter()
        .map(|p| (p.premise.as_str(), p.hypothesis.as_str()))
        .collect();
    let preds = session.predict(&refs)?;
    Ok(preds
        .iter()
        .enumerate()
        .map(|(index, p)| PredictResult {
            id: pairs[index].id.clone(),
            index,
            label: p.label.clone(),
            scores: scores_map(session.labels(), &p.probs),
        })
        .collect())
}

type PairLine = Result<Pair, String>;

fn read_pairs(args: &PredictArgs) -> CliResult<Vec<PairLine>> {
    if let (Some(p), Some(h)) = (&args.premise, &args.hypothesis) {
        return Ok(vec![Ok(Pair {
            premise: p.clone(),
            hypothesis: h.clone(),
            id: None,
        })]);
    }
    if crate::util::stdin_is_tty() {
        return Ok(Vec::new());
    }
    Ok(parse_ndjson(std::io::stdin().lock()))
}

pub fn parse_ndjson(reader: impl BufRead) -> Vec<PairLine> {
    reader
        .lines()
        .map_while(Result::ok)
        .filter(|l| !l.trim().is_empty())
        .map(|l| match serde_json::from_str::<Pair>(&l) {
            Ok(p) => Ok(p),
            Err(e) => Err(format!("malformed input line: {e}")),
        })
        .collect()
}

pub fn rerank(
    args: &RerankArgs,
    cfg: &crate::config::Layered,
    json: bool,
    assume_yes: bool,
) -> CliResult<i32> {
    let options = if args.options.is_empty() {
        if crate::util::stdin_is_tty() {
            return Err(CliError::bad_input(
                "no options: pass --option or pipe one per line",
            ));
        }
        std::io::stdin()
            .lock()
            .lines()
            .map_while(Result::ok)
            .filter(|l| !l.trim().is_empty())
            .collect()
    } else {
        args.options.clone()
    };
    if options.is_empty() {
        return Err(CliError::bad_input("no options to rank"));
    }

    let results = match connect(&args.target) {
        Some(client) => {
            client
                .rerank(&RerankRequest {
                    question: args.question.clone(),
                    options: options.clone(),
                    top_k: args.top_k,
                    return_documents: args.return_documents,
                })?
                .results
        }
        None => {
            let session =
                crate::commands::local_session(&load_spec(&args.target, cfg), assume_yes)?;
            let refs: Vec<&str> = options.iter().map(String::as_str).collect();
            let ranked = session.rerank(&args.question, &refs)?;
            let take = args.top_k.unwrap_or(ranked.len()).min(ranked.len());
            ranked[..take]
                .iter()
                .enumerate()
                .map(|(rank, r)| RerankResult {
                    rank,
                    index: r.index,
                    score: r.score,
                    text: args.return_documents.then(|| options[r.index].clone()),
                })
                .collect()
        }
    };

    let mut out = std::io::stdout().lock();
    for r in &results {
        if json {
            writeln_line(&mut out, &serde_json::to_string(r).unwrap_or_default())?;
        } else {
            writeln_line(
                &mut out,
                &format!(
                    "{:<4} {:<8.4} {}",
                    r.rank,
                    r.score,
                    r.text.clone().unwrap_or_else(|| format!("#{}", r.index))
                ),
            )?;
        }
    }
    Ok(exit::OK)
}

pub fn grade(
    args: &GradeArgs,
    cfg: &crate::config::Layered,
    json: bool,
    assume_yes: bool,
) -> CliResult<i32> {
    let (label, scores, entail) = match connect(&args.target) {
        Some(client) => {
            let r = client.grade(&GradeRequest {
                answer: args.answer.clone(),
                reference: args.reference.clone(),
                threshold: args.threshold,
            })?;
            let e = r
                .scores
                .get("entailment")
                .copied()
                .unwrap_or(if r.pass { 1.0 } else { 0.0 });
            (r.label, r.scores, e)
        }
        None => {
            let session =
                crate::commands::local_session(&load_spec(&args.target, cfg), assume_yes)?;
            // premise = reference, hypothesis = answer: does the ground truth entail what
            // was said?
            let g = session.grade(&args.reference, &args.answer)?;
            (
                g.prediction.label.clone(),
                scores_map(session.labels(), &g.prediction.probs),
                g.score,
            )
        }
    };
    let pass = entail >= args.threshold;
    let mut out = std::io::stdout().lock();
    if json {
        writeln_line(
            &mut out,
            &serde_json::json!({
                "object": "grade", "label": label, "scores": scores,
                "pass": pass, "threshold": args.threshold
            })
            .to_string(),
        )?;
    } else {
        let mut s: Vec<String> = scores.iter().map(|(k, v)| format!("{k}={v:.3}")).collect();
        s.sort();
        writeln_line(
            &mut out,
            &format!(
                "{:<14} {}  {}",
                label,
                s.join("  "),
                if pass { "PASS" } else { "FAIL" }
            ),
        )?;
    }
    // The predicate, not an error: this is what makes `openjev grade` a CI assertion.
    Ok(if pass {
        exit::OK
    } else {
        exit::ASSERTION_FAILED
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn one_malformed_line_does_not_discard_the_good_ones() {
        let input = "{\"premise\":\"a\",\"hypothesis\":\"b\"}\nnot json\n{\"premise\":\"c\",\"hypothesis\":\"d\",\"id\":\"x\"}\n";
        let parsed = parse_ndjson(Cursor::new(input));
        assert_eq!(parsed.len(), 3);
        assert!(parsed[0].is_ok() && parsed[2].is_ok());
        assert!(parsed[1].is_err());
        assert_eq!(parsed[2].as_ref().unwrap().id.as_deref(), Some("x"));
    }

    #[test]
    fn blank_lines_are_not_input_errors() {
        let parsed = parse_ndjson(Cursor::new(
            "\n\n{\"premise\":\"a\",\"hypothesis\":\"b\"}\n\n",
        ));
        assert_eq!(parsed.len(), 1);
        assert!(parsed[0].is_ok());
    }

    #[test]
    fn json_output_is_one_object_per_line_and_human_output_is_a_table() {
        let r = PredictResult {
            id: Some("a1".into()),
            index: 0,
            label: "entailment".into(),
            scores: scores_map(
                &["contradiction".into(), "entailment".into()],
                &[0.02, 0.98],
            ),
        };
        let mut buf = Vec::new();
        write_predict(&mut buf, &r, true).unwrap();
        let line = String::from_utf8(buf).unwrap();
        assert!(line.ends_with('\n') && line.lines().count() == 1);
        let v: serde_json::Value = serde_json::from_str(line.trim()).unwrap();
        assert_eq!(v["label"], "entailment");

        let mut buf = Vec::new();
        write_predict(&mut buf, &r, false).unwrap();
        let text = String::from_utf8(buf).unwrap();
        assert!(text.contains("entailment") && text.contains("a1"));
        assert!(serde_json::from_str::<serde_json::Value>(text.trim()).is_err());
    }
}
