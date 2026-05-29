//! Local web server for playing Quoridor against the AlphaZero agent.
//!
//! Architecture and HTTP API are documented in
//! `docs/superpowers/specs/2026-05-29-quoridor-play-server-design.md`.
//!
//! Threading model: one `tiny_http::Server` accepts requests in the main
//! thread and dispatches each to a fresh worker thread that holds a clone of
//! the `Arc`-backed `GameRegistry`. Per-session locking keeps games
//! independent.

use std::path::PathBuf;
use std::sync::Arc;
use std::thread;

use anyhow::{Context, Result};
use clap::Parser;
use serde_json::{json, Value};
use tiny_http::{Header, Method, Request, Response, Server};

use quoridor_rs::play_server::config::ServerConfig;
use quoridor_rs::play_server::handlers::{
    apply_move, create_game, get_config, get_game, HandlerError, MoveRequest, NewGameRequest,
};
use quoridor_rs::play_server::session::GameRegistry;

const INDEX_HTML: &str = include_str!("../play_server/assets/index.html");
const APP_CSS: &str = include_str!("../play_server/assets/app.css");
const APP_JS: &str = include_str!("../play_server/assets/app.js");

#[derive(Parser)]
#[command(
    name = "play_server",
    about = "Local Quoridor play server (browser vs AlphaZero)"
)]
struct Cli {
    /// Directory containing `config.yaml` and `models/*.onnx`.
    #[arg(long)]
    play_dir: PathBuf,

    /// TCP port to listen on.
    #[arg(long, default_value_t = 8080)]
    port: u16,

    /// Bind address. Use `0.0.0.0` for LAN access.
    #[arg(long, default_value = "127.0.0.1")]
    bind: String,

    /// Default MCTS simulations per move shown in the UI slider.
    #[arg(long, default_value_t = 400)]
    default_mcts_n: u32,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let cfg = Arc::new(
        ServerConfig::load(&cli.play_dir).context("loading server config")?,
    );
    let registry = GameRegistry::new();
    let bind = format!("{}:{}", cli.bind, cli.port);
    let server = Server::http(&bind)
        .map_err(|e| anyhow::anyhow!("failed to bind {bind}: {e}"))?;
    eprintln!(
        "play_server listening on http://{bind}  (board {}x{}, {} model(s))",
        cfg.board_size,
        cfg.board_size,
        cfg.models.len()
    );

    for request in server.incoming_requests() {
        let cfg = Arc::clone(&cfg);
        let registry = registry.clone();
        let default_mcts_n = cli.default_mcts_n;
        thread::spawn(move || {
            if let Err(e) = handle_request(request, &cfg, &registry, default_mcts_n) {
                eprintln!("request handler error: {e:#}");
            }
        });
    }
    Ok(())
}

fn handle_request(
    mut req: Request,
    cfg: &ServerConfig,
    registry: &GameRegistry,
    default_mcts_n: u32,
) -> Result<()> {
    let method = req.method().clone();
    let url = req.url().to_string();
    // Strip query string if any.
    let path = url.split('?').next().unwrap_or(&url).to_string();

    let result: Result<Response<std::io::Cursor<Vec<u8>>>, HandlerError> = (|| {
        match (&method, path.as_str()) {
            (&Method::Get, "/") => Ok(html_response(INDEX_HTML)),
            (&Method::Get, "/static/app.css") => Ok(text_response("text/css", APP_CSS)),
            (&Method::Get, "/static/app.js") => {
                Ok(text_response("application/javascript", APP_JS))
            }
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
            (&Method::Post, p)
                if p.starts_with("/api/games/") && p.ends_with("/move") =>
            {
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
            _ => Err(HandlerError::NotFound(format!("no route for {} {}", method, path))),
        }
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

fn text_response(
    content_type: &str,
    body: &'static str,
) -> Response<std::io::Cursor<Vec<u8>>> {
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
