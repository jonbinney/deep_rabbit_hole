//! Self-play binary for Quoridor.
//!
//! Reads game parameters from a YAML config (same format as the Python pipeline),
//! loads an ONNX model, plays games, and writes `.npz` + `.yaml` replay files.
//!
//! By default this runs the async path: one eval coordinator thread
//! owns the ORT session and serves batched inference for `games_per_process`
//! async game tasks running leaf-parallel MCTS with `leaf_parallelism`
//! in-flight evals each. The coordinator maintains a shared `DashMap` eval
//! cache. Defaults reproduce sequential behaviour (1 game, 1 parallelism, no cache).
//!
//! `--use-raw-onnx-agent` selects the legacy single-threaded greedy ONNX path,
//! unchanged from before.

use std::path::Path;
use std::process;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use clap::Parser;

use quoridor_rs::agents::ActionSelector;
use quoridor_rs::agents::alphazero::AlphaZeroAgent;
use quoridor_rs::agents::onnx_agent::OnnxAgent;
use quoridor_rs::agents::random_agent::RandomAgent;
use quoridor_rs::game_runner::{GameResult, play_game};
use quoridor_rs::replay_writer::{GameMetadata, write_game_npz, write_game_yaml};
use quoridor_rs::selfplay_config::{
    AlphaZeroConfig, QuoridorConfig, SelfPlayWorkerConfig, load_config, load_latest_model,
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

    /// Number of concurrent game tasks per process (default: 1, or YAML self_play.games_per_process).
    #[arg(long)]
    games_per_process: Option<usize>,

    /// Leaf-parallel batch size: number of in-flight evals per game per outer iteration.
    #[arg(long)]
    leaf_parallelism: Option<usize>,

    /// Virtual loss magnitude applied during descent.
    #[arg(long)]
    virtual_loss: Option<u32>,

    /// Disable tree reuse across moves (default: enabled).
    #[arg(long, default_value = "false")]
    no_tree_reuse: bool,

    /// Tokio worker threads (default: hardware threads, or YAML self_play.mcts_worker_threads).
    #[arg(long)]
    mcts_worker_threads: Option<usize>,

    /// Max eval batch size at the coordinator (default: 1).
    #[arg(long)]
    eval_batch_size: Option<usize>,

    /// Max wait (ms) for batch to fill after first request (default: 0).
    #[arg(long)]
    eval_max_wait_ms: Option<u64>,

    /// Max entries in the shared eval cache; 0 disables caching (default: 0).
    #[arg(long)]
    eval_cache_max_size: Option<usize>,

    /// Periodically print pipeline counters (GPU time, batcher wait, postprocess time).
    #[arg(long, default_value = "false")]
    profile_counters: bool,

    /// Directory to write per-model-version MCTS metric JSON records. When omitted,
    /// metric collection is disabled.
    #[arg(long)]
    metrics_dir: Option<String>,
}

/// Resolved runtime config (CLI overrides > YAML > defaults).
#[derive(Debug, Clone, Copy)]
struct ResolvedRustConfig {
    games_per_process: usize,
    leaf_parallelism: usize,
    virtual_loss: u32,
    enable_tree_reuse: bool,
    mcts_worker_threads: usize,
    eval_batch_size: usize,
    eval_max_wait_ms: u64,
    eval_cache_max_size: usize,
}

impl ResolvedRustConfig {
    fn resolve(cli: &Cli, yaml: Option<&SelfPlayWorkerConfig>) -> Self {
        let pick_usize = |c: Option<usize>, y: Option<usize>, d: usize| c.or(y).unwrap_or(d).max(1);
        let pick_usize_zero_ok =
            |c: Option<usize>, y: Option<usize>, d: usize| c.or(y).unwrap_or(d);
        let pick_u32 = |c: Option<u32>, y: Option<u32>, d: u32| c.or(y).unwrap_or(d);
        let pick_u64 = |c: Option<u64>, y: Option<u64>, d: u64| c.or(y).unwrap_or(d);
        let pick_bool = |yaml_v: Option<bool>, d: bool| yaml_v.unwrap_or(d);
        let default_workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        Self {
            games_per_process: pick_usize(
                cli.games_per_process,
                yaml.and_then(|c| c.games_per_process),
                1,
            ),
            leaf_parallelism: pick_usize(
                cli.leaf_parallelism,
                yaml.and_then(|c| c.leaf_parallelism),
                1,
            ),
            virtual_loss: pick_u32(cli.virtual_loss, yaml.and_then(|c| c.virtual_loss), 3),
            enable_tree_reuse: if cli.no_tree_reuse {
                false
            } else {
                pick_bool(yaml.and_then(|c| c.enable_tree_reuse), true)
            },
            mcts_worker_threads: pick_usize(
                cli.mcts_worker_threads,
                yaml.and_then(|c| c.mcts_worker_threads),
                default_workers,
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
                100000,
            ),
        }
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

fn run_batch_batched(
    cli: &Cli,
    q: &QuoridorConfig,
    az_config: &AlphaZeroConfig,
    rust_cfg: ResolvedRustConfig,
) -> Result<()> {
    use quoridor_rs::agents::alphazero::eval_pipeline::{self, EvalCache, FrontMsg};
    use quoridor_rs::agents::alphazero::selfplay_game::{GameSettings, P2, play_game_async};
    use quoridor_rs::agents::alphazero::selfplay_mcts::{LeafParallelConfig, LeafParallelMCTS};
    use tokio::sync::mpsc as tokio_mpsc;

    let model_path = cli
        .model_path
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--model-path is required in batch mode"))?
        .to_string();
    let num_games = cli.num_games;
    let output_dir = cli.output_dir.clone();
    let p2_kind = cli.p2.clone();

    println!(
        "Self-play (leaf-parallel): board_size={}, max_walls={}, max_steps={}, num_games={}",
        q.board_size, q.max_walls, q.max_steps, num_games,
    );
    println!(
        "games_per_process={}, leaf_parallelism={}, virtual_loss={}, tree_reuse={}, eval_batch_size={}, eval_max_wait_ms={}, eval_cache_max_size={}, mcts_worker_threads={}",
        rust_cfg.games_per_process,
        rust_cfg.leaf_parallelism,
        rust_cfg.virtual_loss,
        rust_cfg.enable_tree_reuse,
        rust_cfg.eval_batch_size,
        rust_cfg.eval_max_wait_ms,
        rust_cfg.eval_cache_max_size,
        rust_cfg.mcts_worker_threads,
    );

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(rust_cfg.mcts_worker_threads)
        .enable_time()
        .build()?;

    let profile_counters = cli.profile_counters;
    rt.block_on(async move {
        let cache = std::sync::Arc::new(EvalCache::new());
        let (front_tx, front_rx) = tokio_mpsc::channel::<FrontMsg>(1024);
        let session = eval_pipeline::load_session(&model_path)?;
        let counters = std::sync::Arc::new(eval_pipeline::PipelineCounters::default());
        let coord = eval_pipeline::spawn_coordinator(session, std::sync::Arc::clone(&cache),
            eval_pipeline::CoordinatorConfig {
                eval_batch_size: rust_cfg.eval_batch_size,
                eval_max_wait_ms: rust_cfg.eval_max_wait_ms,
                eval_cache_max_size: rust_cfg.eval_cache_max_size,
            },
            front_rx,
            std::sync::Arc::clone(&counters),
        );

        let print_shutdown = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let print_task = if profile_counters {
            let counters = std::sync::Arc::clone(&counters);
            let shutdown = std::sync::Arc::clone(&print_shutdown);
            Some(tokio::spawn(async move {
                let mut prev_gpu = 0u64; let mut prev_wait = 0u64; let mut prev_post = 0u64;
                let mut prev_batches = 0u64; let mut prev_items = 0u64;
                loop {
                    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                    if shutdown.load(std::sync::atomic::Ordering::Relaxed) { break; }
                    let gpu = counters.gpu_ns.load(std::sync::atomic::Ordering::Relaxed);
                    let wait = counters.batcher_wait_ns.load(std::sync::atomic::Ordering::Relaxed);
                    let post = counters.postprocess_ns.load(std::sync::atomic::Ordering::Relaxed);
                    let batches = counters.batches.load(std::sync::atomic::Ordering::Relaxed);
                    let items = counters.items.load(std::sync::atomic::Ordering::Relaxed);
                    let d_gpu = gpu - prev_gpu;
                    let d_wait = wait - prev_wait;
                    let d_post = post - prev_post;
                    let d_batches = batches - prev_batches;
                    let d_items = items - prev_items;
                    let avg_batch = if d_batches > 0 { d_items as f64 / d_batches as f64 } else { 0.0 };
                    let gpu_busy_pct = (d_gpu as f64 / 5_000_000_000.0) * 100.0;
                    let batch_wait_ms = if d_batches > 0 { (d_wait as f64 / d_batches as f64) / 1_000_000.0 } else { 0.0 };
                    let post_ms = if d_batches > 0 { (d_post as f64 / d_batches as f64) / 1_000_000.0 } else { 0.0 };
                    println!(
                        "[pipe] batches={} items={} avg_batch={:.1} gpu_busy={:.1}% batch_wait_ms={:.1} post_ms={:.1}",
                        d_batches, d_items, avg_batch, gpu_busy_pct, batch_wait_ms, post_ms,
                    );
                    prev_gpu = gpu; prev_wait = wait; prev_post = post;
                    prev_batches = batches; prev_items = items;
                }
            }))
        } else { None };

        let mcts_cfg = az_config.to_agent_config(q.board_size, q.max_walls).mcts;
        let lp_cfg = LeafParallelConfig {
            leaf_parallelism: rust_cfg.leaf_parallelism as u32,
            virtual_loss: rust_cfg.virtual_loss,
            enable_tree_reuse: rust_cfg.enable_tree_reuse,
        };
        let settings = GameSettings {
            temperature: az_config.temperature.unwrap_or(1.0),
            drop_t_on_step: az_config.drop_t_on_step,
            deterministic_tie_break: false,
        };
        let counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let pid = std::process::id();
        let start = std::time::Instant::now();
        let stats = std::sync::Arc::new(std::sync::Mutex::new(Stats::default()));

        let mut handles = Vec::with_capacity(rust_cfg.games_per_process);
        for _ in 0..rust_cfg.games_per_process {
            let front_tx = front_tx.clone();
            let cache = std::sync::Arc::clone(&cache);
            let counter = std::sync::Arc::clone(&counter);
            let stats = std::sync::Arc::clone(&stats);
            let output_dir = output_dir.clone();
            let p2_kind = p2_kind.clone();
            let mcts_cfg = mcts_cfg.clone();
            let board_size = q.board_size;
            let max_walls = q.max_walls;
            let max_steps = q.max_steps as i32;
            let model_version = cli.model_version;
            handles.push(tokio::spawn(async move {
                let mut p1 = LeafParallelMCTS::new(mcts_cfg.clone(), lp_cfg, front_tx.clone(), std::sync::Arc::clone(&cache));
                let mut p2: P2 = match p2_kind.as_deref() {
                    Some("random") => P2::Random,
                    Some(other) => return Err(anyhow::anyhow!("Unknown --p2 agent: '{}'", other)),
                    None => P2::AlphaZero(LeafParallelMCTS::new(mcts_cfg, lp_cfg, front_tx.clone(), std::sync::Arc::clone(&cache))),
                };
                loop {
                    let idx = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if idx >= num_games { break; }
                    p1.reset_tree();
                    if let P2::AlphaZero(m) = &mut p2 { m.reset_tree(); }
                    let (result, _game_metrics) = play_game_async(&mut p1, &mut p2, settings, board_size, max_walls, max_steps).await?;
                    write_replay(&output_dir, None, &result, model_version, idx, pid)?;
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
                        println!("[{}/{}] P1 wins: {}, P2 wins: {}, draws: {}, avg turns: {:.1}, {:.1} games/s",
                            done, num_games, p1w, p2w, draws, avg_turns, gps);
                    }
                }
                Ok::<(), anyhow::Error>(())
            }));
        }
        drop(front_tx);
        for h in handles {
            h.await??;
        }
        let _ = coord.batcher.join();
        let _ = coord.inference.join();
        let _ = coord.post.join();
        print_shutdown.store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(t) = print_task { let _ = t.await; }
        Ok::<(), anyhow::Error>(())
    })?;

    println!("Done. {} games written to {}", num_games, cli.output_dir);
    Ok(())
}

fn run_continuous_batched(
    cli: &Cli,
    q: &QuoridorConfig,
    az_config: &AlphaZeroConfig,
    rust_cfg: ResolvedRustConfig,
) -> Result<()> {
    use quoridor_rs::agents::alphazero::eval_pipeline::{self, EvalCache, FrontMsg};
    use quoridor_rs::agents::alphazero::selfplay_game::{GameSettings, P2, play_game_async};
    use quoridor_rs::agents::alphazero::selfplay_mcts::{LeafParallelConfig, LeafParallelMCTS};
    use quoridor_rs::selfplay_config::load_latest_model;
    use tokio::sync::mpsc as tokio_mpsc;

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
        "Continuous self-play (leaf-parallel): board_size={}, max_walls={}, max_steps={}",
        q.board_size, q.max_walls, q.max_steps,
    );
    println!(
        "games_per_process={}, leaf_parallelism={}, virtual_loss={}, tree_reuse={}, eval_batch_size={}, eval_max_wait_ms={}, eval_cache_max_size={}, mcts_worker_threads={}",
        rust_cfg.games_per_process,
        rust_cfg.leaf_parallelism,
        rust_cfg.virtual_loss,
        rust_cfg.enable_tree_reuse,
        rust_cfg.eval_batch_size,
        rust_cfg.eval_max_wait_ms,
        rust_cfg.eval_cache_max_size,
        rust_cfg.mcts_worker_threads,
    );
    println!(
        "Polling: {}\nShutdown: {}\nOutput: {}",
        latest_yaml_path, shutdown_path, cli.output_dir
    );

    println!("Waiting for initial model...");
    loop {
        if std::path::Path::new(&shutdown_path).exists() {
            println!("Shutdown signal detected before model was available. Exiting.");
            return Ok(());
        }
        if std::path::Path::new(&latest_yaml_path).exists() {
            let onnx_path = pt_to_onnx_path(
                &load_latest_model(&latest_yaml_path)
                    .map(|m| m.filename)
                    .unwrap_or_default(),
            );
            if std::path::Path::new(&onnx_path).exists() {
                break;
            }
        }
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
    let latest = load_latest_model(&latest_yaml_path)?;
    let initial_version = latest.version;
    let initial_path = pt_to_onnx_path(&latest.filename);

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(rust_cfg.mcts_worker_threads)
        .enable_time()
        .build()?;

    let profile_counters = cli.profile_counters;
    rt.block_on(async move {
        let cache = std::sync::Arc::new(EvalCache::new());
        let (front_tx, front_rx) = tokio_mpsc::channel::<FrontMsg>(1024);
        let session = eval_pipeline::load_session(&initial_path)?;
        let counters = std::sync::Arc::new(eval_pipeline::PipelineCounters::default());
        let coord = eval_pipeline::spawn_coordinator(session, std::sync::Arc::clone(&cache),
            eval_pipeline::CoordinatorConfig {
                eval_batch_size: rust_cfg.eval_batch_size,
                eval_max_wait_ms: rust_cfg.eval_max_wait_ms,
                eval_cache_max_size: rust_cfg.eval_cache_max_size,
            },
            front_rx,
            std::sync::Arc::clone(&counters),
        );

        let mcts_cfg = az_config.to_agent_config(q.board_size, q.max_walls).mcts;
        let lp_cfg = LeafParallelConfig {
            leaf_parallelism: rust_cfg.leaf_parallelism as u32,
            virtual_loss: rust_cfg.virtual_loss,
            enable_tree_reuse: rust_cfg.enable_tree_reuse,
        };
        let settings = GameSettings {
            temperature: az_config.temperature.unwrap_or(1.0),
            drop_t_on_step: az_config.drop_t_on_step,
            deterministic_tie_break: false,
        };
        let counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let model_version = std::sync::Arc::new(std::sync::atomic::AtomicI64::new(initial_version));
        let shutdown = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let pid = std::process::id();

        use quoridor_rs::agents::alphazero::selfplay_metrics::SelfPlayAccumulator;
        let metrics_dir = cli.metrics_dir.clone();
        if let Some(ref d) = metrics_dir {
            std::fs::create_dir_all(d)?;
        }
        let metrics = std::sync::Arc::new(std::sync::Mutex::new(SelfPlayAccumulator::new(
            initial_version,
        )));

        let print_task = if profile_counters {
            let counters = std::sync::Arc::clone(&counters);
            let shutdown = std::sync::Arc::clone(&shutdown);
            Some(tokio::spawn(async move {
                let mut prev_gpu = 0u64; let mut prev_wait = 0u64; let mut prev_post = 0u64;
                let mut prev_batches = 0u64; let mut prev_items = 0u64;
                loop {
                    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                    if shutdown.load(std::sync::atomic::Ordering::Relaxed) { break; }
                    let gpu = counters.gpu_ns.load(std::sync::atomic::Ordering::Relaxed);
                    let wait = counters.batcher_wait_ns.load(std::sync::atomic::Ordering::Relaxed);
                    let post = counters.postprocess_ns.load(std::sync::atomic::Ordering::Relaxed);
                    let batches = counters.batches.load(std::sync::atomic::Ordering::Relaxed);
                    let items = counters.items.load(std::sync::atomic::Ordering::Relaxed);
                    let d_gpu = gpu - prev_gpu;
                    let d_wait = wait - prev_wait;
                    let d_post = post - prev_post;
                    let d_batches = batches - prev_batches;
                    let d_items = items - prev_items;
                    let avg_batch = if d_batches > 0 { d_items as f64 / d_batches as f64 } else { 0.0 };
                    let gpu_busy_pct = (d_gpu as f64 / 5_000_000_000.0) * 100.0;
                    let batch_wait_ms = if d_batches > 0 { (d_wait as f64 / d_batches as f64) / 1_000_000.0 } else { 0.0 };
                    let post_ms = if d_batches > 0 { (d_post as f64 / d_batches as f64) / 1_000_000.0 } else { 0.0 };
                    println!(
                        "[pipe] batches={} items={} avg_batch={:.1} gpu_busy={:.1}% batch_wait_ms={:.1} post_ms={:.1}",
                        d_batches, d_items, avg_batch, gpu_busy_pct, batch_wait_ms, post_ms,
                    );
                    prev_gpu = gpu; prev_wait = wait; prev_post = post;
                    prev_batches = batches; prev_items = items;
                }
            }))
        } else { None };

        let mut handles = Vec::with_capacity(rust_cfg.games_per_process);
        for _tid in 0..rust_cfg.games_per_process {
            let front_tx = front_tx.clone();
            let cache = std::sync::Arc::clone(&cache);
            let counter = std::sync::Arc::clone(&counter);
            let model_version = std::sync::Arc::clone(&model_version);
            let shutdown = std::sync::Arc::clone(&shutdown);
            let output_dir = cli.output_dir.clone();
            let tmp_dir = tmp_dir.clone();
            let p2_kind = cli.p2.clone();
            let mcts_cfg = mcts_cfg.clone();
            let board_size = q.board_size;
            let max_walls = q.max_walls;
            let max_steps = q.max_steps as i32;
            let metrics = std::sync::Arc::clone(&metrics);
            let metrics_enabled = metrics_dir.is_some();
            handles.push(tokio::spawn(async move {
                let mut p1 = LeafParallelMCTS::new(mcts_cfg.clone(), lp_cfg, front_tx.clone(), std::sync::Arc::clone(&cache));
                let mut p2: P2 = match p2_kind.as_deref() {
                    Some("random") => P2::Random,
                    Some(other) => return Err(anyhow::anyhow!("Unknown --p2 agent: '{}'", other)),
                    None => P2::AlphaZero(LeafParallelMCTS::new(mcts_cfg, lp_cfg, front_tx.clone(), std::sync::Arc::clone(&cache))),
                };
                loop {
                    if shutdown.load(std::sync::atomic::Ordering::Relaxed) { break; }
                    let mv = model_version.load(std::sync::atomic::Ordering::Relaxed);
                    p1.note_model_version(mv);
                    if let P2::AlphaZero(m) = &mut p2 { m.note_model_version(mv); }
                    p1.reset_tree();
                    if let P2::AlphaZero(m) = &mut p2 { m.reset_tree(); }
                    let idx = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let (result, game_metrics) = play_game_async(&mut p1, &mut p2, settings, board_size, max_walls, max_steps).await?;
                    write_replay(&output_dir, Some(&tmp_dir), &result, mv, idx, pid)?;
                    if metrics_enabled {
                        metrics.lock().unwrap().fold_game(&game_metrics);
                    }
                }
                Ok::<(), anyhow::Error>(())
            }));
        }

        // Main coordinator-poll loop: watches latest.yaml + shutdown sentinel.
        let main_handle = {
            let front_tx = front_tx.clone();
            let shutdown = std::sync::Arc::clone(&shutdown);
            let model_version = std::sync::Arc::clone(&model_version);
            let metrics = std::sync::Arc::clone(&metrics);
            let metrics_dir = metrics_dir.clone();
            let pid_for_metrics = pid;
            tokio::spawn(async move {
                let mut current = initial_version;
                loop {
                    if std::path::Path::new(&shutdown_path).exists() {
                        println!("Shutdown signal detected. Stopping workers...");
                        shutdown.store(true, std::sync::atomic::Ordering::Relaxed);
                        if let Some(ref d) = metrics_dir {
                            let mut m = metrics.lock().unwrap();
                            if let Err(e) = m.flush_and_reset(d, pid_for_metrics) {
                                eprintln!("selfplay-metrics: final flush failed: {:#}", e);
                            }
                        }
                        break;
                    }
                    if let Ok(latest) = load_latest_model(&latest_yaml_path) {
                        if latest.version != current {
                            let new_path = pt_to_onnx_path(&latest.filename);
                            if std::path::Path::new(&new_path).exists() {
                                println!("New model detected: version {} -> {} ({})", current, latest.version, new_path);
                                current = latest.version;
                                model_version.store(latest.version, std::sync::atomic::Ordering::Relaxed);
                                if let Some(ref d) = metrics_dir {
                                    let mut m = metrics.lock().unwrap();
                                    if let Err(e) = m.flush_and_reset(d, pid_for_metrics) {
                                        eprintln!("selfplay-metrics: flush failed: {:#}", e);
                                    }
                                    m.set_version(latest.version);
                                }
                                let _ = front_tx.send(FrontMsg::Reload(new_path)).await;
                            }
                        }
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                }
                Ok::<(), anyhow::Error>(())
            })
        };

        drop(front_tx);
        let _ = main_handle.await?;
        for h in handles { h.await??; }
        let _ = coord.batcher.join();
        let _ = coord.inference.join();
        let _ = coord.post.join();
        if let Some(t) = print_task { let _ = t.await; }
        Ok::<(), anyhow::Error>(())
    })?;

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
