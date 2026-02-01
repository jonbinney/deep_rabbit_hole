#!/usr/bin/env rust
//! Create a policy database from minimax evaluations.
//!
//! This binary initializes a Quoridor game and uses the Rust q_minimax implementation
//! to evaluate all possible actions from the initial state, logging the results to
//! a SQLite database for later analysis or training.

use clap::Parser;
use quoridor_rs::compact::{q_game_mechanics::QGameMechanics, q_minimax};
use quoridor_rs::q_log_entries_to_sqlite;
use std::env;

#[derive(Parser, Debug)]
#[command(
    name = "create_policy_db",
    about = "Create a policy database from Quoridor minimax evaluations",
    version
)]
struct Args {
    /// Size of the Quoridor board
    #[arg(long, default_value_t = 3)]
    board_size: usize,

    /// Maximum walls each player can place
    #[arg(long, default_value_t = 0)]
    max_walls: usize,

    /// Maximum steps before game is terminated
    #[arg(long, default_value_t = 8)]
    max_steps: usize,

    /// How many actions to consider at each minimax stage
    #[arg(long, default_value_t = 10000)]
    branching_factor: usize,

    /// Discount factor for future rewards
    #[arg(long, default_value_t = 1.0)]
    discount_factor: f32,

    /// Heuristic to use (0=none, 1=distance+walls)
    #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(i32).range(0..=1))]
    heuristic: i32,

    /// Number of threads to use for computation
    #[arg(long, default_value_t = 1)]
    num_threads: usize,

    /// Output SQLite database file path
    #[arg(short, long, default_value = "policy_db.sqlite")]
    output: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Set number of threads for Rayon
    env::set_var("RAYON_NUM_THREADS", args.num_threads.to_string());

    println!("Initializing Quoridor game...");
    println!("  Board size: {}x{}", args.board_size, args.board_size);
    println!("  Max walls: {} per player", args.max_walls);
    println!("  Max steps: {}", args.max_steps);
    println!("  Branching factor: {}", args.branching_factor);
    println!("  Discount factor: {}", args.discount_factor);
    println!("  Heuristic: {}", args.heuristic);
    println!("  Number of threads: {}", args.num_threads);

    // Create game mechanics and initial state
    let mechanics = QGameMechanics::new(args.board_size, args.max_walls, args.max_steps);
    let initial_state = mechanics.create_initial_state();

    // Print game state info
    let current_player = mechanics.repr().get_current_player(&initial_state);
    let (p0_row, p0_col) = mechanics.repr().get_player_position(&initial_state, 0);
    let (p1_row, p1_col) = mechanics.repr().get_player_position(&initial_state, 1);
    let p0_walls = mechanics.repr().get_walls_remaining(&initial_state, 0);
    let p1_walls = mechanics.repr().get_walls_remaining(&initial_state, 1);

    println!("\nGame state:");
    println!("  State size: {} bytes", initial_state.len());
    println!(
        "  Player positions: [({}, {}), ({}, {})]",
        p0_row, p0_col, p1_row, p1_col
    );
    println!("  Walls remaining: [{}, {}]", p0_walls, p1_walls);
    println!("  Current player: {}", current_player);

    // Call q_minimax to evaluate actions and generate log entries
    println!("\nEvaluating actions and creating policy database...");
    println!("  Output file: {}", args.output);

    let (_actions, _values, log_entries) = q_minimax::evaluate_actions(
        &mechanics,
        &initial_state,
        args.max_steps,
        args.branching_factor,
        args.discount_factor,
        args.heuristic,
        true, // enable logging
    );

    // Write log entries to SQLite database
    if let Some(entries) = log_entries {
        let num_entries = entries.len();
        println!("  Collected {} log entries", num_entries);

        q_log_entries_to_sqlite(
            entries,
            &args.output,
            args.board_size,
            args.max_steps,
            args.max_walls,
        )?;

        println!(
            "\n✓ Successfully created policy database with {} entries",
            num_entries
        );
        println!("  Saved to: {}", args.output);
    } else {
        println!("\n✗ No log entries were generated");
    }

    Ok(())
}
