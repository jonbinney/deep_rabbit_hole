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
use tiny_http::Server;

use quoridor_rs::play_server::config::ServerConfig;
use quoridor_rs::play_server::http::handle_request;
use quoridor_rs::play_server::session::GameRegistry;

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
    #[arg(long, default_value_t = 1000)]
    default_mcts_n: u32,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let cfg = Arc::new(ServerConfig::load(&cli.play_dir).context("loading server config")?);
    let registry = GameRegistry::new();
    let bind = format!("{}:{}", cli.bind, cli.port);
    let server = Server::http(&bind).map_err(|e| anyhow::anyhow!("failed to bind {bind}: {e}"))?;
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
