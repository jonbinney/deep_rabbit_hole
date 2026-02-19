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

use anyhow::Result;
use clap::Parser;
use std::process;
use std::time::Instant;

use quoridor_rs::agents::onnx_agent::OnnxAgent;
use quoridor_rs::agents::random_agent::RandomAgent;
use quoridor_rs::agents::ActionSelector;
use quoridor_rs::game_runner::play_game;
use quoridor_rs::replay_writer::{write_game_npz, write_game_yaml, GameMetadata};
use quoridor_rs::selfplay_config::load_config;

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

    /// Agent type for player 2. Omit to use the same ONNX model as P1.
    /// Use "random" for a random agent.
    #[arg(long)]
    p2: Option<String>,

    /// Print a step-by-step trace of each game (whose turn, action, board).
    #[arg(long, default_value = "false")]
    trace: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let config = load_config(&cli.config)?;
    let q = &config.quoridor;

    let p2_desc = cli.p2.as_deref().unwrap_or("onnx (same model)");
    println!(
        "Self-play: board_size={}, max_walls={}, max_steps={}, num_games={}",
        q.board_size, q.max_walls, q.max_steps, cli.num_games,
    );
    println!("P1: ONNX {}", cli.model_path);
    println!("P2: {}", p2_desc);
    println!("Output: {}", cli.output_dir);

    let mut agent_p1 = OnnxAgent::new(&cli.model_path)?;

    // Build P2 agent: either another OnnxAgent (same model) or a RandomAgent
    let mut onnx_p2: Option<OnnxAgent> = None;
    let mut random_p2: Option<RandomAgent> = None;
    match cli.p2.as_deref() {
        Some("random") => {
            random_p2 = Some(RandomAgent::new());
        }
        Some(other) => {
            anyhow::bail!(
                "Unknown --p2 agent type: '{}'. Valid options: random",
                other
            );
        }
        None => {
            onnx_p2 = Some(OnnxAgent::new(&cli.model_path)?);
        }
    }

    let agent_p2: &mut dyn ActionSelector = match (&mut onnx_p2, &mut random_p2) {
        (Some(ref mut a), _) => a,
        (_, Some(ref mut a)) => a,
        _ => unreachable!(),
    };

    println!("Model loaded.");

    let pid = process::id();
    let mut wins = [0u32; 2];
    let mut draws = 0u32;
    let mut total_turns = 0u64;

    let start = Instant::now();

    for game_idx in 0..cli.num_games {
        let result = play_game(
            &mut agent_p1,
            agent_p2,
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
