//! End-to-end test: boot the play server in-process, play a few moves
//! against the real B5W2 ONNX fixture, assert state transitions.
//!
//! Run: cargo test --no-default-features --features binary --test play_server_e2e -- --nocapture
#![cfg(feature = "binary")]

use std::net::TcpStream;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use quoridor_rs::play_server::config::ServerConfig;
use quoridor_rs::play_server::http::handle_request;
use quoridor_rs::play_server::session::GameRegistry;

fn fixture_onnx_path() -> Option<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let fallback = root.join("fixtures").join("alphazero_B5W2_mv1.onnx");
    let p = std::env::var("DEEP_QUORIDOR_ONNX_MODEL")
        .map(PathBuf::from)
        .unwrap_or(fallback);
    if p.exists() { Some(p) } else { None }
}

#[test]
fn play_server_serves_a_few_moves_against_real_model() {
    let onnx = match fixture_onnx_path() {
        Some(p) => p,
        None => {
            eprintln!("skipping: B5W2 ONNX fixture not available");
            return;
        }
    };

    // Build temp play_dir: config.yaml + models/alphazero_B5W2_mv1.onnx
    let temp = tempfile::tempdir().expect("tempdir");
    let models_dir = temp.path().join("models");
    std::fs::create_dir_all(&models_dir).unwrap();
    let model_dst = models_dir.join("alphazero_B5W2_mv1.onnx");
    std::fs::copy(&onnx, &model_dst).expect("copy model");
    std::fs::write(
        temp.path().join("config.yaml"),
        "quoridor:\n  board_size: 5\n  max_walls: 2\n  max_steps: 50\n",
    )
    .unwrap();

    let cfg = Arc::new(ServerConfig::load(temp.path()).expect("config load"));
    let registry = GameRegistry::new();

    // Bind to port 0; read assigned port.
    let server = Arc::new(tiny_http::Server::http("127.0.0.1:0").expect("bind tiny_http"));
    let addr = server.server_addr();
    let port = match addr {
        tiny_http::ListenAddr::IP(sock) => sock.port(),
        _ => panic!("expected IP listen addr"),
    };

    // Worker thread runs the handler loop until the server is unblocked.
    let worker_server = Arc::clone(&server);
    let worker_cfg = Arc::clone(&cfg);
    let worker_registry = registry.clone();
    let worker = thread::spawn(move || {
        for request in worker_server.incoming_requests() {
            if let Err(e) = handle_request(request, &worker_cfg, &worker_registry, 8) {
                eprintln!("test worker error: {e:#}");
            }
        }
    });

    let base = format!("http://127.0.0.1:{port}");
    let http = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(30))
        .build();

    // 1. GET /api/config
    let cfg_view: serde_json::Value = http
        .get(&format!("{base}/api/config"))
        .call()
        .expect("GET /api/config")
        .into_json()
        .expect("config json");
    assert_eq!(cfg_view["board_size"], 5);
    assert_eq!(cfg_view["max_walls"], 2);
    let models = cfg_view["models"].as_array().expect("models array");
    assert!(
        models
            .iter()
            .any(|m| m.as_str() == Some("alphazero_B5W2_mv1.onnx"))
    );

    // 2. POST /api/games
    let new_game: serde_json::Value = http
        .post(&format!("{base}/api/games"))
        .send_json(serde_json::json!({
            "model": "alphazero_B5W2_mv1.onnx",
            "mcts_n": 8,
            "human_player": 0,
        }))
        .expect("POST /api/games")
        .into_json()
        .expect("new-game json");
    let game_id = new_game["game_id"].as_str().expect("game_id").to_string();
    let mut state = new_game["state"].clone();
    // human_player == 0 means human moves first.
    assert_eq!(state["current_player"].as_i64(), Some(0));

    // 3. GET /api/games/<id> to confirm the registry stored the game.
    let fetched: serde_json::Value = http
        .get(&format!("{base}/api/games/{game_id}"))
        .call()
        .expect("GET /api/games/<id>")
        .into_json()
        .expect("game json");
    assert_eq!(fetched["state"]["current_player"].as_i64(), Some(0));

    // 4. Play up to 10 human moves, picking the first legal move action.
    let mut moves = 0;
    while state["winner"].is_null() && moves < 10 {
        let legal = state["legal_actions"].as_array().expect("legal_actions");
        let move_idx = legal
            .iter()
            .find_map(|a| {
                if a["kind"] == "move" {
                    a["index"].as_u64()
                } else {
                    None
                }
            })
            .expect("at least one legal pawn move");
        let resp: serde_json::Value = http
            .post(&format!("{base}/api/games/{game_id}/move"))
            .send_json(serde_json::json!({ "action_index": move_idx }))
            .expect("POST /move")
            .into_json()
            .expect("move json");
        state = resp["state"].clone();
        moves += 1;
        // Each round-trip adds at least the human's move; if the game isn't
        // over, the handler also runs an AI step (so +2).
        let history_len = state["move_history"].as_array().unwrap().len();
        assert!(
            history_len >= 2 * moves - 1,
            "after {moves} human move(s), move_history len = {history_len}",
        );
    }

    assert!(moves > 0, "played zero moves");

    // 5. Unblock the listener and join the worker.
    server.unblock();
    // tiny_http's unblock won't fire until the next incoming_requests poll
    // returns, so prod the listener with a stray TCP connect.
    let _ = TcpStream::connect(("127.0.0.1", port));
    worker.join().expect("worker join");
}
