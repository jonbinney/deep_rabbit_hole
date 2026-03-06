//! Self-play binary for Quoridor.
//!
//! Reads game parameters from a YAML config (same format as the Python pipeline),
//! loads an ONNX model, plays `num_games` games, and writes `.npz` + `.yaml`
//! replay files to `output_dir`.
//!
//! Usage:
//!   # Default: AlphaZero MCTS agent
//!   selfplay --config experiments/ci.yaml \
//!            --model-path experiments/onnx/B5W3_resnet_sample.onnx \
//!            --output-dir /tmp/replays \
//!            --num-games 100
//!
//!   # Continuous mode (used by Python trainer):
//!   selfplay --config experiments/ci.yaml \
//!            --output-dir /tmp/replays \
//!            --continuous \
//!            --latest-model-yaml /tmp/run/models/latest.yaml \
//!            --shutdown-file /tmp/run/.shutdown
//!
//!   # Use legacy raw ONNX greedy agent:
//!   selfplay --config experiments/ci.yaml \
//!            --model-path experiments/onnx/B5W3_resnet_sample.onnx \
//!            --output-dir /tmp/replays \
//!            --num-games 100 --use-raw-onnx-agent
//!
//!   # AlphaZero vs Random:
//!   selfplay --config experiments/ci.yaml \
//!            --model-path experiments/onnx/B5W3_resnet_sample.onnx \
//!            --output-dir /tmp/replays \
//!            --num-games 100 --p2 random

use anyhow::Result;
use clap::Parser;
use std::path::Path;
use std::process;
use std::time::Instant;

use quoridor_rs::agents::alphazero::AlphaZeroAgent;
use quoridor_rs::agents::onnx_agent::OnnxAgent;
use quoridor_rs::agents::random_agent::RandomAgent;
use quoridor_rs::agents::ActionSelector;
use quoridor_rs::game_runner::play_game;
use quoridor_rs::replay_writer::{write_game_npz, write_game_yaml, GameMetadata};
use quoridor_rs::selfplay_config::{load_config, load_latest_model, AlphaZeroConfig};

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

    /// Use the legacy raw ONNX greedy agent instead of the default AlphaZero MCTS agent.
    #[arg(long, default_value = "false")]
    use_raw_onnx_agent: bool,

    /// Agent for player 2. Omit to use the same agent as P1. Use "random" for a random agent.
    #[arg(long)]
    p2: Option<String>,

    /// Print a step-by-step trace of each game (whose turn, action, board).
    #[arg(long, default_value = "false")]
    trace: bool,

    /// Model version number to record in replay metadata.
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

fn create_agent(
    use_raw_onnx: bool,
    p2_override: Option<&str>,
    model_path: &str,
    az_config: &AlphaZeroConfig,
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
            az_config.to_agent_config(),
        )?))
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let config = load_config(&cli.config)?;
    let q = &config.quoridor;

    // Build AlphaZero config from YAML, merging self_play overrides
    let base_az = config.alphazero.unwrap_or_default();
    let az_config = if let Some(ref sp) = config.self_play {
        if let Some(ref sp_az) = sp.alphazero {
            base_az.merge(sp_az)
        } else {
            base_az
        }
    } else {
        base_az
    };

    if cli.continuous {
        run_continuous(&cli, q, &az_config)
    } else {
        run_batch(&cli, q, &az_config)
    }
}

/// Batch mode: play a fixed number of games and exit.
fn run_batch(
    cli: &Cli,
    q: &quoridor_rs::selfplay_config::QuoridorConfig,
    az_config: &AlphaZeroConfig,
) -> Result<()> {
    let model_path = cli
        .model_path
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--model-path is required in batch mode"))?;

    let agent_label = if cli.use_raw_onnx_agent {
        "raw-onnx"
    } else {
        "alphazero"
    };
    let p2_desc = match cli.p2.as_deref() {
        Some(p2) => p2.to_string(),
        None => format!("{} (same as P1)", agent_label),
    };

    println!(
        "Self-play: board_size={}, max_walls={}, max_steps={}, num_games={}",
        q.board_size, q.max_walls, q.max_steps, cli.num_games,
    );
    println!("P1: {} ({})", agent_label, model_path);
    println!("P2: {}", p2_desc);
    println!("Output: {}", cli.output_dir);

    if !cli.use_raw_onnx_agent {
        println!(
            "MCTS config: n={:?}, k={:?}, c_puct={}, noise_epsilon={}",
            az_config.mcts_n, az_config.mcts_k, az_config.mcts_c_puct, az_config.mcts_noise_epsilon
        );
    }

    let mut agent_p1 = create_agent(cli.use_raw_onnx_agent, None, model_path, az_config)?;
    let mut agent_p2 = create_agent(
        cli.use_raw_onnx_agent,
        cli.p2.as_deref(),
        model_path,
        az_config,
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
        )?;

        match result.winner {
            Some(0) => wins[0] += 1,
            Some(1) => wins[1] += 1,
            _ => draws += 1,
        }
        total_turns += result.num_turns as u64;

        write_game_files(&cli.output_dir, &result, cli.model_version, game_idx, pid)?;

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

/// Continuous mode: play games indefinitely, polling `latest.yaml` for new
/// model versions and checking the shutdown sentinel file between games.
fn run_continuous(
    cli: &Cli,
    q: &quoridor_rs::selfplay_config::QuoridorConfig,
    az_config: &AlphaZeroConfig,
) -> Result<()> {
    let latest_yaml_path = cli
        .latest_model_yaml
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--latest-model-yaml is required with --continuous"))?;
    let shutdown_path = cli
        .shutdown_file
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--shutdown-file is required with --continuous"))?;

    // Create a tmp subdirectory for atomic writes
    let tmp_dir = format!("{}/tmp", cli.output_dir);
    std::fs::create_dir_all(&tmp_dir)?;

    println!(
        "Continuous self-play: board_size={}, max_walls={}, max_steps={}",
        q.board_size, q.max_walls, q.max_steps,
    );
    println!("Polling: {}", latest_yaml_path);
    println!("Shutdown: {}", shutdown_path);
    println!("Output: {}", cli.output_dir);

    // Wait for the initial latest.yaml to appear
    println!("Waiting for initial model...");
    loop {
        if Path::new(shutdown_path).exists() {
            println!("Shutdown signal detected before model was available. Exiting.");
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
        std::thread::sleep(std::time::Duration::from_secs(1));
    }

    let latest = load_latest_model(latest_yaml_path)?;
    let mut model_version = latest.version;
    let mut model_path = pt_to_onnx_path(&latest.filename);

    println!(
        "Loading initial model: version={}, path={}",
        model_version, model_path
    );

    let mut agent_p1 = create_agent(false, None, &model_path, az_config)?;
    let mut agent_p2 = create_agent(false, cli.p2.as_deref(), &model_path, az_config)?;

    let pid = process::id();
    let mut game_idx: usize = 0;

    loop {
        // Check for shutdown
        if Path::new(shutdown_path).exists() {
            println!(
                "Shutdown signal detected. Exiting after {} games.",
                game_idx
            );
            break;
        }

        // Check for new model version
        if let Ok(new_latest) = load_latest_model(latest_yaml_path) {
            if new_latest.version != model_version {
                let new_path = pt_to_onnx_path(&new_latest.filename);
                // Wait for the ONNX file to appear before loading
                if Path::new(&new_path).exists() {
                    println!(
                        "New model detected: version {} -> {} ({})",
                        model_version, new_latest.version, new_path
                    );
                    model_version = new_latest.version;
                    model_path = new_path;
                    agent_p1 = create_agent(false, None, &model_path, az_config)?;
                    agent_p2 = create_agent(false, cli.p2.as_deref(), &model_path, az_config)?;
                }
            }
        }

        agent_p1.reset_game();
        agent_p2.reset_game();

        let game_start = std::time::Instant::now();
        let result = play_game(
            agent_p1.as_mut(),
            agent_p2.as_mut(),
            q.board_size,
            q.max_walls,
            q.max_steps as i32,
            false,
        )?;
        let game_elapsed = game_start.elapsed().as_secs_f64();
        println!("{} - selfplay finished in {:.4}", pid, game_elapsed);

        // Atomic write: write to tmp dir, then rename to output dir
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis();
        let base_name = format!("game_{}_{:04}_{}", ts, game_idx, pid);

        let yaml_ready = format!("{}/{}.yaml", cli.output_dir, base_name);
        let npz_tmp = format!("{}/{}.npz", tmp_dir, base_name);
        let npz_ready = format!("{}/{}.npz", cli.output_dir, base_name);

        let metadata = GameMetadata {
            model_version,
            game_length: result.replay_items.len(),
            creator: format!("{}", pid),
        };
        // Write YAML first to output dir (trainer looks for npz to know game is ready)
        write_game_yaml(&yaml_ready, &metadata)?;
        // Write npz to tmp, then atomically rename
        write_game_npz(&npz_tmp, &result)?;
        std::fs::rename(&npz_tmp, &npz_ready)?;

        game_idx += 1;
    }

    Ok(())
}

/// Write game replay files (YAML metadata + npz data) to the output directory.
fn write_game_files(
    output_dir: &str,
    result: &quoridor_rs::game_runner::GameResult,
    model_version: i64,
    game_idx: usize,
    pid: u32,
) -> Result<()> {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let base_name = format!("game_{}_{:04}_{}", ts, game_idx, pid);
    let yaml_path = format!("{}/{}.yaml", output_dir, base_name);
    let npz_path = format!("{}/{}.npz", output_dir, base_name);

    let metadata = GameMetadata {
        model_version,
        game_length: result.replay_items.len(),
        creator: format!("{}", pid),
    };
    write_game_yaml(&yaml_path, &metadata)?;
    write_game_npz(&npz_path, result)?;
    Ok(())
}
