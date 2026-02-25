//! Self-play binary for Quoridor.
//!
//! Reads game parameters from a YAML config (same format as the Python pipeline),
//! loads an ONNX model, plays `num_games` games, and writes `.npz` + `.yaml`
//! replay files to `output_dir`.
//!
//! Usage:
//!   selfplay --config experiments/ci.yaml \
//!            --model-path experiments/onnx/B5W3_resnet_sample.onnx \
//!            --output-dir /tmp/replays \
//!            --num-games 100
//!
//!   # Play ONNX (P1) vs Random (P2):
//!   selfplay --config experiments/ci.yaml \
//!            --model-path experiments/onnx/B5W3_resnet_sample.onnx \
//!            --output-dir /tmp/replays \
//!            --num-games 100 --p2 random
//!
//!   # Use AlphaZero MCTS agent:
//!   selfplay --config experiments/ci.yaml \
//!            --model-path experiments/onnx/B5W3_resnet_sample.onnx \
//!            --output-dir /tmp/replays \
//!            --num-games 100 --agent-type alphazero

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

    /// Agent type for both players. "onnx" (default) or "alphazero".
    #[arg(long, default_value = "onnx")]
    agent_type: String,

    /// Agent type for player 2. Omit to use the same type as P1.
    /// Use "random" for a random agent.
    #[arg(long)]
    p2: Option<String>,

    /// Print a step-by-step trace of each game (whose turn, action, board).
    #[arg(long, default_value = "false")]
    trace: bool,
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
    agent_type: &str,
    model_path: &str,
    az_config: &AlphaZeroConfig,
) -> Result<BoxedAgent> {
    match agent_type {
        "onnx" => Ok(BoxedAgent::Onnx(OnnxAgent::new(model_path)?)),
        "alphazero" => Ok(BoxedAgent::AlphaZero(AlphaZeroAgent::new(
            model_path,
            az_config.to_agent_config(),
        )?)),
        "random" => Ok(BoxedAgent::Random(RandomAgent::new())),
        other => anyhow::bail!("Unknown agent type: '{}'. Valid: onnx, alphazero, random", other),
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

    // Determine P2 agent type
    let p2_type = cli.p2.as_deref().unwrap_or(&cli.agent_type);
    let p2_desc = if cli.p2.is_some() {
        p2_type.to_string()
    } else {
        format!("{} (same as P1)", p2_type)
    };

    println!(
        "Self-play: board_size={}, max_walls={}, max_steps={}, num_games={}",
        q.board_size, q.max_walls, q.max_steps, cli.num_games,
    );
    println!("P1: {} ({})", cli.agent_type, cli.model_path);
    println!("P2: {}", p2_desc);
    println!("Output: {}", cli.output_dir);

    if cli.agent_type == "alphazero" {
        println!(
            "MCTS config: n={:?}, k={:?}, c_puct={}, noise_epsilon={}",
            az_config.mcts_n, az_config.mcts_k, az_config.mcts_c_puct, az_config.mcts_noise_epsilon
        );
    }

    let mut agent_p1 = create_agent(&cli.agent_type, &cli.model_path, &az_config)?;
    let mut agent_p2 = create_agent(p2_type, &cli.model_path, &az_config)?;

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
            model_version: 0,
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
