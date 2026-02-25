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
use std::process;
use std::time::Instant;

use quoridor_rs::agents::alphazero::AlphaZeroAgent;
use quoridor_rs::agents::onnx_agent::OnnxAgent;
use quoridor_rs::agents::random_agent::RandomAgent;
use quoridor_rs::agents::ActionSelector;
use quoridor_rs::game_runner::play_game;
use quoridor_rs::replay_writer::{write_game_npz, write_game_yaml, GameMetadata};
use quoridor_rs::selfplay_config::{load_config, AlphaZeroConfig};

#[derive(Parser)]
#[command(about = "Quoridor self-play data generator")]
struct Cli {
    /// Path to the YAML configuration file (same format as Python pipeline).
    #[arg(long)]
    config: String,

    /// Path to the ONNX model file.
    #[arg(long)]
    model_path: String,

    /// Directory to write replay output files.
    #[arg(long)]
    output_dir: String,

    /// Number of games to play.
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
    println!("P1: {} ({})", agent_label, cli.model_path);
    println!("P2: {}", p2_desc);
    println!("Output: {}", cli.output_dir);

    if !cli.use_raw_onnx_agent {
        println!(
            "MCTS config: n={:?}, k={:?}, c_puct={}, noise_epsilon={}",
            az_config.mcts_n, az_config.mcts_k, az_config.mcts_c_puct, az_config.mcts_noise_epsilon
        );
    }

    let mut agent_p1 = create_agent(cli.use_raw_onnx_agent, None, &cli.model_path, &az_config)?;
    let mut agent_p2 = create_agent(
        cli.use_raw_onnx_agent,
        cli.p2.as_deref(),
        &cli.model_path,
        &az_config,
    )?;

    println!("Model loaded.");

    let pid = process::id();
    let mut wins = [0u32; 2];
    let mut draws = 0u32;
    let mut total_turns = 0u64;

    let start = Instant::now();

    for game_idx in 0..cli.num_games {
        // Reset visited states between games for AlphaZero agents
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

        // Update stats
        match result.winner {
            Some(0) => wins[0] += 1,
            Some(1) => wins[1] += 1,
            _ => draws += 1,
        }
        total_turns += result.num_turns as u64;

        // Write YAML first (so trainer never sees .npz without metadata)
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis();
        let base_name = format!("game_{}_{:04}_{}", ts, game_idx, pid);
        let yaml_path = format!("{}/{}.yaml", cli.output_dir, base_name);
        let npz_path = format!("{}/{}.npz", cli.output_dir, base_name);

        let metadata = GameMetadata {
            model_version: cli.model_version,
            game_length: result.replay_items.len(),
            creator: format!("{}", pid),
        };
        write_game_yaml(&yaml_path, &metadata)?;
        write_game_npz(&npz_path, &result)?;

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
