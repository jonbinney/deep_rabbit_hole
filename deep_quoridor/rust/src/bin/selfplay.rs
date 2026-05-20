//! Self-play binary for Quoridor.
//!
//! Reads game parameters from a YAML config (same format as the Python pipeline),
//! loads an ONNX model, plays games, and writes `.npz` + `.yaml` replay files.
//!
//! By default this runs the multi-threaded path: one eval coordinator thread
//! owns the ORT session and serves batched inference for `threads_per_process ×
//! games_per_thread` worker threads that play games concurrently. The
//! coordinator maintains a shared `DashMap` eval cache. Defaults reproduce
//! sequential behaviour (1 worker, batch-of-1, no cache).
//!
//! `--use-raw-onnx-agent` selects the legacy single-threaded greedy ONNX path,
//! unchanged from before.

use std::path::Path;
use std::process;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicUsize, Ordering};
use std::sync::mpsc::{sync_channel, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use clap::Parser;
use dashmap::DashMap;

use quoridor_rs::agents::alphazero::eval_coordinator::{
    load_session, run_coordinator, CoordinatorConfig, CtrlMsg, EvalCache, EvalRequest,
};
use quoridor_rs::agents::alphazero::evaluator::BatchingEvaluator;
use quoridor_rs::agents::alphazero::AlphaZeroAgent;
use quoridor_rs::agents::onnx_agent::OnnxAgent;
use quoridor_rs::agents::random_agent::RandomAgent;
use quoridor_rs::agents::ActionSelector;
use quoridor_rs::game_runner::{play_game, GameResult};
use quoridor_rs::replay_writer::{write_game_npz, write_game_yaml, GameMetadata};
use quoridor_rs::selfplay_config::{
    load_config, load_latest_model, AlphaZeroConfig, QuoridorConfig, SelfPlayWorkerConfig,
};

/// Convert a `.pt` model path to its corresponding `.onnx` path.
/// If the path doesn't end in `.pt`, returns it unchanged.
fn pt_to_onnx_path(path: &str) -> String {
    if let Some(stem) = path.strip_suffix(".pt") {
        format!("{}.onnx", stem)
    } else {
        path.to_string()
    }
}

#[derive(Parser)]
#[command(about = "Quoridor self-play data generator")]
struct Cli {
    /// Path to the YAML configuration file (same format as Python pipeline).
    #[arg(long)]
    config: String,

    /// Path to the ONNX model file (required unless --continuous is set).
    #[arg(long)]
    model_path: Option<String>,

    /// Directory to write replay output files.
    #[arg(long)]
    output_dir: String,

    /// Number of games to play (ignored in --continuous mode).
    #[arg(long, default_value = "100")]
    num_games: usize,

    /// Use the legacy single-threaded raw ONNX greedy agent (skips MCTS, no
    /// eval coordinator). New batched/multi-threaded settings are ignored.
    #[arg(long, default_value = "false")]
    use_raw_onnx_agent: bool,

    /// Agent for player 2. Omit to use the same agent as P1. Use "random" for a random agent.
    #[arg(long)]
    p2: Option<String>,

    /// Print a step-by-step trace of each game (whose turn, action, board).
    #[arg(long, default_value = "false")]
    trace: bool,

    /// Model version number to record in replay metadata (batch mode only).
    #[arg(long, default_value = "0")]
    model_version: i64,

    /// Run in continuous mode: play games indefinitely, polling for new models.
    #[arg(long, default_value = "false")]
    continuous: bool,

    /// Path to `latest.yaml` for model hot-reload (required with --continuous).
    #[arg(long)]
    latest_model_yaml: Option<String>,

    /// Path to shutdown sentinel file. When this file exists, exit gracefully.
    #[arg(long)]
    shutdown_file: Option<String>,

    /// Number of worker threads (default: 1, or value from YAML self_play.threads_per_process).
    #[arg(long)]
    threads_per_process: Option<usize>,

    /// Games per worker thread (default: 1, or value from YAML self_play.games_per_thread).
    #[arg(long)]
    games_per_thread: Option<usize>,

    /// Max eval batch size at the coordinator (default: 1).
    #[arg(long)]
    eval_batch_size: Option<usize>,

    /// Max wait (ms) for batch to fill after first request (default: 0).
    #[arg(long)]
    eval_max_wait_ms: Option<u64>,

    /// Max entries in the shared eval cache; 0 disables caching (default: 0).
    #[arg(long)]
    eval_cache_max_size: Option<usize>,
}

/// Resolved runtime config (CLI overrides > YAML > defaults).
#[derive(Debug, Clone, Copy)]
struct ResolvedRustConfig {
    threads_per_process: usize,
    games_per_thread: usize,
    eval_batch_size: usize,
    eval_max_wait_ms: u64,
    eval_cache_max_size: usize,
}

impl ResolvedRustConfig {
    fn resolve(cli: &Cli, yaml: Option<&SelfPlayWorkerConfig>) -> Self {
        let pick_usize = |c: Option<usize>, y: Option<usize>, d: usize| c.or(y).unwrap_or(d).max(1);
        let pick_usize_zero_ok =
            |c: Option<usize>, y: Option<usize>, d: usize| c.or(y).unwrap_or(d);
        let pick_u64 = |c: Option<u64>, y: Option<u64>, d: u64| c.or(y).unwrap_or(d);
        Self {
            threads_per_process: pick_usize(
                cli.threads_per_process,
                None, // SelfPlayWorkerConfig uses games_per_process (not split)
                1,
            ),
            games_per_thread: pick_usize(
                cli.games_per_thread,
                yaml.and_then(|c| c.games_per_process),
                1,
            ),
            eval_batch_size: pick_usize(
                cli.eval_batch_size,
                yaml.and_then(|c| c.eval_batch_size),
                1,
            ),
            eval_max_wait_ms: pick_u64(
                cli.eval_max_wait_ms,
                yaml.and_then(|c| c.eval_max_wait_ms),
                0,
            ),
            eval_cache_max_size: pick_usize_zero_ok(
                cli.eval_cache_max_size,
                yaml.and_then(|c| c.eval_cache_max_size),
                0,
            ),
        }
    }

    fn total_workers(&self) -> usize {
        self.threads_per_process * self.games_per_thread
    }
}

/// Boxed agent trait object for dynamic dispatch.
enum BoxedAgent {
    Onnx(OnnxAgent),
    AlphaZero(AlphaZeroAgent),
    Random(RandomAgent),
}

impl BoxedAgent {
    fn as_mut(&mut self) -> &mut dyn ActionSelector {
        match self {
            BoxedAgent::Onnx(a) => a,
            BoxedAgent::AlphaZero(a) => a,
            BoxedAgent::Random(a) => a,
        }
    }

    fn reset_game(&mut self) {
        if let BoxedAgent::AlphaZero(a) = self {
            a.reset_game();
        }
    }
}

/// Build a P1 AlphaZero agent that submits eval requests to the shared coordinator.
fn build_p1_agent_batched(
    az_config: &AlphaZeroConfig,
    board_size: i32,
    max_walls: i32,
    sender: SyncSender<EvalRequest>,
    cache: Arc<EvalCache>,
) -> BoxedAgent {
    let agent_cfg = az_config.to_agent_config(board_size, max_walls);
    let eval = Box::new(BatchingEvaluator::new(sender, cache));
    BoxedAgent::AlphaZero(AlphaZeroAgent::with_evaluator(eval, agent_cfg))
}

/// Build a P2 agent: AlphaZero (default, shared coordinator) or "random".
fn build_p2_agent_batched(
    p2_override: Option<&str>,
    az_config: &AlphaZeroConfig,
    board_size: i32,
    max_walls: i32,
    sender: SyncSender<EvalRequest>,
    cache: Arc<EvalCache>,
) -> Result<BoxedAgent> {
    match p2_override {
        Some("random") => Ok(BoxedAgent::Random(RandomAgent::new())),
        Some(other) => anyhow::bail!("Unknown --p2 agent: '{}'. Valid: random", other),
        None => Ok(build_p1_agent_batched(
            az_config, board_size, max_walls, sender, cache,
        )),
    }
}

/// Legacy single-threaded agent factory (used only with --use-raw-onnx-agent).
fn create_agent_legacy(
    use_raw_onnx: bool,
    p2_override: Option<&str>,
    model_path: &str,
    az_config: &AlphaZeroConfig,
    board_size: i32,
    max_walls: i32,
) -> Result<BoxedAgent> {
    if let Some("random") = p2_override {
        return Ok(BoxedAgent::Random(RandomAgent::new()));
    }
    if let Some(other) = p2_override {
        anyhow::bail!("Unknown --p2 agent: '{}'. Valid: random", other);
    }
    if use_raw_onnx {
        Ok(BoxedAgent::Onnx(OnnxAgent::new(model_path)?))
    } else {
        Ok(BoxedAgent::AlphaZero(AlphaZeroAgent::new(
            model_path,
            az_config.to_agent_config(board_size, max_walls),
        )?))
    }
}

#[derive(Default, Debug)]
struct Stats {
    wins: [u32; 2],
    draws: u32,
    total_turns: u64,
    completed: usize,
}

/// Write a single game's replay files. If `tmp_dir` is Some, npz is written
/// there first and atomically renamed into `output_dir` (used by continuous
/// mode so the trainer never sees a partial file).
fn write_replay(
    output_dir: &str,
    tmp_dir: Option<&str>,
    result: &GameResult,
    model_version: i64,
    game_idx: usize,
    pid: u32,
) -> Result<()> {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let base_name = format!("game_{}_{:06}_{}", ts, game_idx, pid);
    let yaml_path = format!("{}/{}.yaml", output_dir, base_name);
    let npz_final = format!("{}/{}.npz", output_dir, base_name);
    let metadata = GameMetadata {
        model_version,
        game_length: result.replay_items.len(),
        creator: format!("{}", pid),
    };
    if let Some(tmp) = tmp_dir {
        // Atomic order matches the legacy continuous mode: yaml first (in the
        // ready dir), then npz via tmp + rename. The trainer triggers on npz.
        write_game_yaml(&yaml_path, &metadata)?;
        let npz_tmp = format!("{}/{}.npz", tmp, base_name);
        write_game_npz(&npz_tmp, result)?;
        std::fs::rename(&npz_tmp, &npz_final)?;
    } else {
        write_game_yaml(&yaml_path, &metadata)?;
        write_game_npz(&npz_final, result)?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let config = load_config(&cli.config)?;
    let q = config.quoridor;

    let base_az = config.alphazero.clone().unwrap_or_default();
    let az_config = if let Some(ref sp) = config.self_play {
        if let Some(ref sp_az) = sp.alphazero {
            base_az.merge_self_play(sp_az)
        } else {
            base_az
        }
    } else {
        base_az
    };

    let rust_cfg = ResolvedRustConfig::resolve(&cli, config.self_play.as_ref());

    if cli.use_raw_onnx_agent {
        if cli.continuous {
            return run_continuous_legacy(&cli, &q, &az_config);
        } else {
            return run_batch_legacy(&cli, &q, &az_config);
        }
    }

    if cli.continuous {
        run_continuous_batched(&cli, &q, &az_config, rust_cfg)
    } else {
        run_batch_batched(&cli, &q, &az_config, rust_cfg)
    }
}

fn spawn_coordinator(
    model_path: &str,
    cache: Arc<EvalCache>,
    rust_cfg: ResolvedRustConfig,
    req_rx: std::sync::mpsc::Receiver<EvalRequest>,
    ctrl_rx: std::sync::mpsc::Receiver<CtrlMsg>,
) -> Result<thread::JoinHandle<()>> {
    let session = load_session(model_path)?;
    let coord_cfg = CoordinatorConfig {
        eval_batch_size: rust_cfg.eval_batch_size,
        eval_max_wait_ms: rust_cfg.eval_max_wait_ms,
        eval_cache_max_size: rust_cfg.eval_cache_max_size,
    };
    let handle = thread::Builder::new()
        .name("eval-coordinator".to_string())
        .spawn(move || run_coordinator(session, cache, coord_cfg, req_rx, ctrl_rx))?;
    Ok(handle)
}

fn run_batch_batched(
    cli: &Cli,
    q: &QuoridorConfig,
    az_config: &AlphaZeroConfig,
    rust_cfg: ResolvedRustConfig,
) -> Result<()> {
    let model_path = cli
        .model_path
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--model-path is required in batch mode"))?;

    let p2_desc = match cli.p2.as_deref() {
        Some(p2) => p2.to_string(),
        None => "alphazero (same as P1)".to_string(),
    };
    println!(
        "Self-play (batched): board_size={}, max_walls={}, max_steps={}, num_games={}",
        q.board_size, q.max_walls, q.max_steps, cli.num_games,
    );
    println!("P1: alphazero ({})", model_path);
    println!("P2: {}", p2_desc);
    println!(
        "Multi-threading: threads_per_process={}, games_per_thread={} (total in-flight={}), eval_batch_size={}, eval_max_wait_ms={}, eval_cache_max_size={}",
        rust_cfg.threads_per_process,
        rust_cfg.games_per_thread,
        rust_cfg.total_workers(),
        rust_cfg.eval_batch_size,
        rust_cfg.eval_max_wait_ms,
        rust_cfg.eval_cache_max_size,
    );
    println!(
        "MCTS config: n={:?}, k={:?}, c_puct={}, noise_epsilon={}",
        az_config.mcts_n, az_config.mcts_k, az_config.mcts_c_puct, az_config.mcts_noise_epsilon
    );
    println!("Output: {}", cli.output_dir);

    let cache: Arc<EvalCache> = Arc::new(DashMap::new());
    let total_workers = rust_cfg.total_workers();
    let (req_tx, req_rx) = sync_channel::<EvalRequest>(total_workers.max(1) * 4);
    let (ctrl_tx, ctrl_rx) = sync_channel::<CtrlMsg>(4);

    let coord_handle =
        spawn_coordinator(model_path, Arc::clone(&cache), rust_cfg, req_rx, ctrl_rx)?;

    println!("Model loaded.");

    let counter = Arc::new(AtomicUsize::new(0));
    let stats = Arc::new(Mutex::new(Stats::default()));
    let model_version_atomic = Arc::new(AtomicI64::new(cli.model_version));
    let shutdown = Arc::new(AtomicBool::new(false));
    let pid = process::id();
    let start = Instant::now();

    let mut worker_handles = Vec::with_capacity(total_workers);
    for tid in 0..total_workers {
        let az_config = az_config.clone();
        let p2_override = cli.p2.clone();
        let board_size = q.board_size;
        let max_walls = q.max_walls;
        let max_steps = q.max_steps as i32;
        let trace = cli.trace;
        let req_tx = req_tx.clone();
        let cache = Arc::clone(&cache);
        let counter = Arc::clone(&counter);
        let stats = Arc::clone(&stats);
        let mv = Arc::clone(&model_version_atomic);
        let shutdown = Arc::clone(&shutdown);
        let output_dir = cli.output_dir.clone();
        let num_games = cli.num_games;

        let handle = thread::Builder::new()
            .name(format!("selfplay-worker-{}", tid))
            .spawn(move || -> Result<()> {
                let mut agent_p1 = build_p1_agent_batched(
                    &az_config,
                    board_size,
                    max_walls,
                    req_tx.clone(),
                    Arc::clone(&cache),
                );
                let mut agent_p2 = build_p2_agent_batched(
                    p2_override.as_deref(),
                    &az_config,
                    board_size,
                    max_walls,
                    req_tx.clone(),
                    Arc::clone(&cache),
                )?;

                loop {
                    if shutdown.load(Ordering::Relaxed) {
                        break;
                    }
                    let game_idx = counter.fetch_add(1, Ordering::Relaxed);
                    if game_idx >= num_games {
                        break;
                    }

                    let game_mv = mv.load(Ordering::Relaxed);
                    agent_p1.reset_game();
                    agent_p2.reset_game();
                    let result = play_game(
                        agent_p1.as_mut(),
                        agent_p2.as_mut(),
                        board_size,
                        max_walls,
                        max_steps,
                        trace,
                        None,
                    )?;

                    write_replay(&output_dir, None, &result, game_mv, game_idx, pid)?;

                    let mut s = stats.lock().unwrap();
                    match result.winner {
                        Some(0) => s.wins[0] += 1,
                        Some(1) => s.wins[1] += 1,
                        _ => s.draws += 1,
                    }
                    s.total_turns += result.num_turns as u64;
                    s.completed += 1;
                    let done = s.completed;
                    let p1w = s.wins[0];
                    let p2w = s.wins[1];
                    let draws = s.draws;
                    let avg_turns = s.total_turns as f64 / done.max(1) as f64;
                    drop(s);

                    if done % 10 == 0 || done == num_games {
                        let elapsed = start.elapsed().as_secs_f64();
                        let gps = done as f64 / elapsed;
                        println!(
                            "[{}/{}] P1 wins: {}, P2 wins: {}, draws: {}, avg turns: {:.1}, {:.1} games/s",
                            done, num_games, p1w, p2w, draws, avg_turns, gps,
                        );
                    }
                }
                Ok(())
            })?;
        worker_handles.push(handle);
    }

    // Drop our handle to the request sender so the coordinator sees disconnect
    // once all workers finish.
    drop(req_tx);

    for h in worker_handles {
        h.join()
            .map_err(|e| anyhow::anyhow!("worker thread panicked: {:?}", e))??;
    }

    let _ = ctrl_tx.send(CtrlMsg::Shutdown);
    let _ = coord_handle.join();

    println!(
        "Done. {} games written to {}",
        cli.num_games, cli.output_dir
    );
    Ok(())
}

fn run_continuous_batched(
    cli: &Cli,
    q: &QuoridorConfig,
    az_config: &AlphaZeroConfig,
    rust_cfg: ResolvedRustConfig,
) -> Result<()> {
    let latest_yaml_path = cli
        .latest_model_yaml
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--latest-model-yaml is required with --continuous"))?
        .to_string();
    let shutdown_path = cli
        .shutdown_file
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--shutdown-file is required with --continuous"))?
        .to_string();

    let tmp_dir = format!("{}/tmp", cli.output_dir);
    std::fs::create_dir_all(&tmp_dir)?;

    println!(
        "Continuous self-play (batched): board_size={}, max_walls={}, max_steps={}",
        q.board_size, q.max_walls, q.max_steps,
    );
    println!(
        "Multi-threading: threads_per_process={}, games_per_thread={} (total in-flight={}), eval_batch_size={}, eval_max_wait_ms={}, eval_cache_max_size={}",
        rust_cfg.threads_per_process,
        rust_cfg.games_per_thread,
        rust_cfg.total_workers(),
        rust_cfg.eval_batch_size,
        rust_cfg.eval_max_wait_ms,
        rust_cfg.eval_cache_max_size,
    );
    println!("Polling: {}", latest_yaml_path);
    println!("Shutdown: {}", shutdown_path);
    println!("Output: {}", cli.output_dir);

    println!("Waiting for initial model...");
    loop {
        if Path::new(&shutdown_path).exists() {
            println!("Shutdown signal detected before model was available. Exiting.");
            return Ok(());
        }
        if Path::new(&latest_yaml_path).exists() {
            let onnx_path = pt_to_onnx_path(
                &load_latest_model(&latest_yaml_path)
                    .map(|m| m.filename)
                    .unwrap_or_default(),
            );
            if Path::new(&onnx_path).exists() {
                break;
            }
        }
        thread::sleep(Duration::from_secs(1));
    }

    let latest = load_latest_model(&latest_yaml_path)?;
    let initial_model_version = latest.version;
    let initial_model_path = pt_to_onnx_path(&latest.filename);
    println!(
        "Loading initial model: version={}, path={}",
        initial_model_version, initial_model_path
    );

    let cache: Arc<EvalCache> = Arc::new(DashMap::new());
    let total_workers = rust_cfg.total_workers();
    let (req_tx, req_rx) = sync_channel::<EvalRequest>(total_workers.max(1) * 4);
    let (ctrl_tx, ctrl_rx) = sync_channel::<CtrlMsg>(4);
    let coord_handle = spawn_coordinator(
        &initial_model_path,
        Arc::clone(&cache),
        rust_cfg,
        req_rx,
        ctrl_rx,
    )?;

    let counter = Arc::new(AtomicUsize::new(0));
    let model_version_atomic = Arc::new(AtomicI64::new(initial_model_version));
    let shutdown = Arc::new(AtomicBool::new(false));
    let pid = process::id();

    let mut worker_handles = Vec::with_capacity(total_workers);
    for tid in 0..total_workers {
        let az_config = az_config.clone();
        let p2_override = cli.p2.clone();
        let board_size = q.board_size;
        let max_walls = q.max_walls;
        let max_steps = q.max_steps as i32;
        let req_tx = req_tx.clone();
        let cache = Arc::clone(&cache);
        let counter = Arc::clone(&counter);
        let mv = Arc::clone(&model_version_atomic);
        let shutdown = Arc::clone(&shutdown);
        let output_dir = cli.output_dir.clone();
        let tmp_dir = tmp_dir.clone();

        let handle = thread::Builder::new()
            .name(format!("selfplay-worker-{}", tid))
            .spawn(move || -> Result<()> {
                let mut agent_p1 = build_p1_agent_batched(
                    &az_config,
                    board_size,
                    max_walls,
                    req_tx.clone(),
                    Arc::clone(&cache),
                );
                let mut agent_p2 = build_p2_agent_batched(
                    p2_override.as_deref(),
                    &az_config,
                    board_size,
                    max_walls,
                    req_tx.clone(),
                    Arc::clone(&cache),
                )?;

                loop {
                    if shutdown.load(Ordering::Relaxed) {
                        break;
                    }
                    let game_idx = counter.fetch_add(1, Ordering::Relaxed);
                    let game_mv = mv.load(Ordering::Relaxed);

                    agent_p1.reset_game();
                    agent_p2.reset_game();
                    let game_start = Instant::now();
                    let result = play_game(
                        agent_p1.as_mut(),
                        agent_p2.as_mut(),
                        board_size,
                        max_walls,
                        max_steps,
                        false,
                        None,
                    )?;
                    let game_elapsed = game_start.elapsed().as_secs_f64();
                    println!("{}-{} - selfplay finished in {:.4}", pid, tid, game_elapsed);

                    write_replay(&output_dir, Some(&tmp_dir), &result, game_mv, game_idx, pid)?;
                }
                Ok(())
            })?;
        worker_handles.push(handle);
    }
    drop(req_tx);

    // Main thread: poll latest.yaml and shutdown sentinel.
    let mut current_version = initial_model_version;
    loop {
        if Path::new(&shutdown_path).exists() {
            println!("Shutdown signal detected. Stopping workers...");
            shutdown.store(true, Ordering::Relaxed);
            break;
        }
        if let Ok(new_latest) = load_latest_model(&latest_yaml_path) {
            if new_latest.version != current_version {
                let new_path = pt_to_onnx_path(&new_latest.filename);
                if Path::new(&new_path).exists() {
                    println!(
                        "New model detected: version {} -> {} ({})",
                        current_version, new_latest.version, new_path
                    );
                    current_version = new_latest.version;
                    model_version_atomic.store(new_latest.version, Ordering::Relaxed);
                    let _ = ctrl_tx.send(CtrlMsg::ReloadModel(new_path));
                }
            }
        }
        thread::sleep(Duration::from_millis(500));
    }

    for h in worker_handles {
        let _ = h.join();
    }
    let _ = ctrl_tx.send(CtrlMsg::Shutdown);
    let _ = coord_handle.join();

    Ok(())
}

// -- Legacy single-thread paths used only when `--use-raw-onnx-agent` is set. --

fn run_batch_legacy(cli: &Cli, q: &QuoridorConfig, az_config: &AlphaZeroConfig) -> Result<()> {
    let model_path = cli
        .model_path
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--model-path is required in batch mode"))?;

    println!(
        "Self-play (legacy raw-onnx): board_size={}, max_walls={}, max_steps={}, num_games={}",
        q.board_size, q.max_walls, q.max_steps, cli.num_games,
    );

    let mut agent_p1 = create_agent_legacy(
        cli.use_raw_onnx_agent,
        None,
        model_path,
        az_config,
        q.board_size,
        q.max_walls,
    )?;
    let mut agent_p2 = create_agent_legacy(
        cli.use_raw_onnx_agent,
        cli.p2.as_deref(),
        model_path,
        az_config,
        q.board_size,
        q.max_walls,
    )?;

    println!("Model loaded.");
    let pid = process::id();
    let mut wins = [0u32; 2];
    let mut draws = 0u32;
    let mut total_turns = 0u64;
    let start = Instant::now();

    for game_idx in 0..cli.num_games {
        agent_p1.reset_game();
        agent_p2.reset_game();
        let result = play_game(
            agent_p1.as_mut(),
            agent_p2.as_mut(),
            q.board_size,
            q.max_walls,
            q.max_steps as i32,
            cli.trace,
            None,
        )?;

        match result.winner {
            Some(0) => wins[0] += 1,
            Some(1) => wins[1] += 1,
            _ => draws += 1,
        }
        total_turns += result.num_turns as u64;

        write_replay(
            &cli.output_dir,
            None,
            &result,
            cli.model_version,
            game_idx,
            pid,
        )?;

        if (game_idx + 1) % 10 == 0 || game_idx + 1 == cli.num_games {
            let elapsed = start.elapsed().as_secs_f64();
            let gps = (game_idx + 1) as f64 / elapsed;
            println!(
                "[{}/{}] P1 wins: {}, P2 wins: {}, draws: {}, avg turns: {:.1}, {:.1} games/s",
                game_idx + 1,
                cli.num_games,
                wins[0],
                wins[1],
                draws,
                total_turns as f64 / (game_idx + 1) as f64,
                gps,
            );
        }
    }

    println!(
        "Done. {} games written to {}",
        cli.num_games, cli.output_dir
    );
    Ok(())
}

fn run_continuous_legacy(cli: &Cli, q: &QuoridorConfig, az_config: &AlphaZeroConfig) -> Result<()> {
    let latest_yaml_path = cli
        .latest_model_yaml
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--latest-model-yaml is required with --continuous"))?;
    let shutdown_path = cli
        .shutdown_file
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--shutdown-file is required with --continuous"))?;

    let tmp_dir = format!("{}/tmp", cli.output_dir);
    std::fs::create_dir_all(&tmp_dir)?;

    println!("Waiting for initial model...");
    loop {
        if Path::new(shutdown_path).exists() {
            return Ok(());
        }
        if Path::new(latest_yaml_path).exists() {
            let onnx_path = pt_to_onnx_path(
                &load_latest_model(latest_yaml_path)
                    .map(|m| m.filename)
                    .unwrap_or_default(),
            );
            if Path::new(&onnx_path).exists() {
                break;
            }
        }
        thread::sleep(Duration::from_secs(1));
    }

    let latest = load_latest_model(latest_yaml_path)?;
    let mut model_version = latest.version;
    let mut model_path = pt_to_onnx_path(&latest.filename);

    let mut agent_p1 = create_agent_legacy(
        true,
        None,
        &model_path,
        az_config,
        q.board_size,
        q.max_walls,
    )?;
    let mut agent_p2 = create_agent_legacy(
        true,
        cli.p2.as_deref(),
        &model_path,
        az_config,
        q.board_size,
        q.max_walls,
    )?;

    let pid = process::id();
    let mut game_idx: usize = 0;
    loop {
        if Path::new(shutdown_path).exists() {
            break;
        }
        if let Ok(new_latest) = load_latest_model(latest_yaml_path) {
            if new_latest.version != model_version {
                let new_path = pt_to_onnx_path(&new_latest.filename);
                if Path::new(&new_path).exists() {
                    model_version = new_latest.version;
                    model_path = new_path;
                    agent_p1 = create_agent_legacy(
                        true,
                        None,
                        &model_path,
                        az_config,
                        q.board_size,
                        q.max_walls,
                    )?;
                    agent_p2 = create_agent_legacy(
                        true,
                        cli.p2.as_deref(),
                        &model_path,
                        az_config,
                        q.board_size,
                        q.max_walls,
                    )?;
                }
            }
        }

        agent_p1.reset_game();
        agent_p2.reset_game();
        let result = play_game(
            agent_p1.as_mut(),
            agent_p2.as_mut(),
            q.board_size,
            q.max_walls,
            q.max_steps as i32,
            false,
            None,
        )?;
        write_replay(
            &cli.output_dir,
            Some(&tmp_dir),
            &result,
            model_version,
            game_idx,
            pid,
        )?;
        game_idx += 1;
    }
    Ok(())
}
