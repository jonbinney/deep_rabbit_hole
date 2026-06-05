//! HTTP routing for the play server.
//!
//! Lives in the library (not the binary) so both `bin/play_server.rs` and the
//! end-to-end integration test (`tests/play_server_e2e.rs`) can share one
//! source of truth for request dispatch + response shaping.

use anyhow::{Context, Result};
use serde_json::{Value, json};
use tiny_http::{Header, Method, Request, Response};

use crate::play_server::config::ServerConfig;
use crate::play_server::handlers::{
    HandlerError, MoveRequest, NewGameRequest, apply_move, create_game, get_config, get_game,
};
use crate::play_server::session::GameRegistry;

pub const INDEX_HTML: &str = include_str!("assets/index.html");
pub const APP_CSS: &str = include_str!("assets/app.css");
pub const APP_JS: &str = include_str!("assets/app.js");

/// Route a single request to the appropriate handler and write the response.
pub fn handle_request(
    mut req: Request,
    cfg: &ServerConfig,
    registry: &GameRegistry,
    default_mcts_n: u32,
) -> Result<()> {
    let method = req.method().clone();
    let url = req.url().to_string();
    // Strip query string if any.
    let path = url.split('?').next().unwrap_or(&url).to_string();

    let result: Result<Response<std::io::Cursor<Vec<u8>>>, HandlerError> =
        (|| match (&method, path.as_str()) {
            (&Method::Get, "/") => Ok(html_response(INDEX_HTML)),
            (&Method::Get, "/static/app.css") => Ok(text_response("text/css", APP_CSS)),
            (&Method::Get, "/static/app.js") => Ok(text_response("application/javascript", APP_JS)),
            (&Method::Get, "/api/config") => {
                let view = get_config(cfg, default_mcts_n);
                Ok(json_response(serde_json::to_value(view).map_err(|e| {
                    HandlerError::Internal(format!("serializing config: {e}"))
                })?))
            }
            (&Method::Post, "/api/games") => {
                let body = read_body(&mut req)
                    .map_err(|e| HandlerError::BadRequest(format!("reading body: {e}")))?;
                let parsed: NewGameRequest = serde_json::from_str(&body)
                    .map_err(|e| HandlerError::BadRequest(format!("invalid JSON: {e}")))?;
                let resp = create_game(cfg, registry, parsed)?;
                Ok(json_response(serde_json::to_value(resp).map_err(|e| {
                    HandlerError::Internal(format!("serializing response: {e}"))
                })?))
            }
            (&Method::Post, p) if p.starts_with("/api/games/") && p.ends_with("/move") => {
                let id = &p["/api/games/".len()..p.len() - "/move".len()];
                let body = read_body(&mut req)
                    .map_err(|e| HandlerError::BadRequest(format!("reading body: {e}")))?;
                let parsed: MoveRequest = serde_json::from_str(&body)
                    .map_err(|e| HandlerError::BadRequest(format!("invalid JSON: {e}")))?;
                let resp = apply_move(registry, id, parsed)?;
                Ok(json_response(serde_json::to_value(resp).map_err(|e| {
                    HandlerError::Internal(format!("serializing response: {e}"))
                })?))
            }
            (&Method::Get, p)
                if p.starts_with("/api/games/")
                    && !p[("/api/games/".len())..].is_empty()
                    && !p.ends_with("/move") =>
            {
                let id = &p["/api/games/".len()..];
                let resp = get_game(registry, id)?;
                Ok(json_response(serde_json::to_value(resp).map_err(|e| {
                    HandlerError::Internal(format!("serializing response: {e}"))
                })?))
            }
            _ => Err(HandlerError::NotFound(format!(
                "no route for {} {}",
                method, path
            ))),
        })();

    let response = match result {
        Ok(r) => r,
        Err(e) => error_response(&e),
    };
    req.respond(response).context("writing HTTP response")
}

fn read_body(req: &mut Request) -> std::io::Result<String> {
    let mut buf = String::new();
    req.as_reader().read_to_string(&mut buf)?;
    Ok(buf)
}

fn html_response(body: &'static str) -> Response<std::io::Cursor<Vec<u8>>> {
    text_response("text/html; charset=utf-8", body)
}

fn text_response(content_type: &str, body: &'static str) -> Response<std::io::Cursor<Vec<u8>>> {
    Response::from_string(body).with_header(
        Header::from_bytes(&b"Content-Type"[..], content_type.as_bytes())
            .expect("content-type header"),
    )
}

fn json_response(value: Value) -> Response<std::io::Cursor<Vec<u8>>> {
    let body = value.to_string();
    Response::from_string(body).with_header(
        Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..])
            .expect("content-type header"),
    )
}

fn error_response(err: &HandlerError) -> Response<std::io::Cursor<Vec<u8>>> {
    let status = match err {
        HandlerError::BadRequest(_) => 400,
        HandlerError::NotFound(_) => 404,
        HandlerError::Internal(_) => 500,
    };
    let body = json!({ "error": err.message() }).to_string();
    Response::from_string(body)
        .with_status_code(status)
        .with_header(
            Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..])
                .expect("content-type header"),
        )
}
