#!/usr/bin/env rust
//! Create a policy database from minimax evaluations.

use clap::Parser;
use quoridor_rs::compact::{
    policy_db::{self, PolicyDb, TranspositionTable},
    q_game_mechanics::QGameMechanics,
};
use std::time::Instant;

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

    /// Number of threads for Lazy SMP parallel search
    #[arg(long, default_value_t = 1)]
    num_threads: usize,

    /// Only save states where the number of completed steps is a multiple of this value
    #[arg(long, default_value_t = 1)]
    step_interval: usize,

    /// Output SQLite database file path
    #[arg(short, long, default_value = "policy_db.sqlite")]
    output: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("Initializing Quoridor game...");
    println!("  Board size: {}x{}", args.board_size, args.board_size);
    println!("  Max walls: {} per player", args.max_walls);
    println!("  Max steps: {}", args.max_steps);

    // Create game mechanics and initial state
    let mechanics = QGameMechanics::new(args.board_size, args.max_walls, args.max_steps);
    let mut initial_state = mechanics.create_initial_state();

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

    println!("\nRunning minimax and creating policy database...");
    println!("  Output file: {}", args.output);
    println!("  Threads: {}", args.num_threads);

    let eval_start = Instant::now();
    let transposition_table = TranspositionTable::new();
    let value = if args.num_threads > 1 {
        policy_db::minimax_lazy_smp(
            &mechanics,
            &mut initial_state,
            &transposition_table,
            args.num_threads,
        )
    } else {
        policy_db::minimax(&mechanics, &mut initial_state, &transposition_table)
    };

    let eval_elapsed = eval_start.elapsed();
    println!("  minimax took {:.3}s", eval_elapsed.as_secs_f64());
    println!("  Root value: {:?}", value);

    let num_entries = transposition_table.len();
    println!("  Collected {} transposition table entries", num_entries);

    let write_start = Instant::now();
    PolicyDb::write(
        &mechanics,
        transposition_table,
        &args.output,
        args.board_size,
        args.max_steps,
        args.max_walls,
        args.step_interval,
    )?;
    let write_elapsed = write_start.elapsed();
    println!(
        "  Writing database took {:.3}s",
        write_elapsed.as_secs_f64()
    );

    println!(
        "\nSuccessfully created policy database with {} entries",
        num_entries
    );
    println!("  Saved to: {}", args.output);

    Ok(())
}
