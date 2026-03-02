#!/usr/bin/env rust
//! Create a policy database from minimax evaluations.

use clap::Parser;
use quoridor_rs::compact::{q_game_mechanics::QGameMechanics, q_minimax};
use rusqlite::{params, Connection};
use std::env;
use std::path::Path;
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

#[allow(dead_code)]
pub fn save_policy_to_sqlite(
    entries: Vec<q_minimax::MinimaxLogEntry>,
    filename: &str,
    board_size: usize,
    max_steps: usize,
    max_walls: usize,
) -> Result<usize, Box<dyn std::error::Error>> {
    let mut conn = Connection::open(Path::new(filename))?;

    // Drop existing tables to avoid schema conflicts
    // This ensures we always have the correct schema for QBitRepr-based data
    conn.execute("DROP TABLE IF EXISTS policy", [])?;
    conn.execute("DROP TABLE IF EXISTS metadata", [])?;

    // Create metadata table for global parameters
    conn.execute(
        "CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value FLOAT NOT NULL
        )",
        [],
    )?;

    // Insert metadata
    conn.execute(
        "INSERT INTO metadata (key, value) VALUES ('board_size', ?1)",
        params![board_size as f32],
    )?;
    conn.execute(
        "INSERT INTO metadata (key, value) VALUES ('max_steps', ?1)",
        params![max_steps as f32],
    )?;
    conn.execute(
        "INSERT INTO metadata (key, value) VALUES ('max_walls', ?1)",
        params![max_walls as f32],
    )?;

    // Create table for policy entries
    conn.execute(
        "CREATE TABLE policy (
            id INTEGER PRIMARY KEY,
            state BLOB NOT NULL,
            agent_player INTEGER NOT NULL,
            num_actions INTEGER NOT NULL,
            actions BLOB NOT NULL,
            action_values BLOB NOT NULL
        )",
        [],
    )?;

    // Create index for fast lookups by state
    conn.execute("CREATE INDEX idx_state ON policy (state)", [])?;

    let num_entries = entries.len();

    // Insert entries in a transaction for better performance
    let tx = conn.transaction()?;
    {
        let mut stmt = tx.prepare(
            "INSERT INTO policy (state, agent_player, num_actions, actions, action_values)
             VALUES (?1, ?2, ?3, ?4, ?5)",
        )?;

        for entry in entries {
            // State is already packed as Vec<u8>
            let state_blob = entry.data;

            // Flatten actions into a single vector: each action is (row, col, action_type)
            let actions_flat: Vec<usize> = entry
                .actions
                .into_iter()
                .flat_map(|(r, c, t)| vec![r, c, t])
                .collect();
            let actions_blob: Vec<u8> = actions_flat
                .iter()
                .flat_map(|&x| (x as u32).to_le_bytes())
                .collect();

            // Convert values to bytes
            let values_blob: Vec<u8> = entry.values.iter().flat_map(|&x| x.to_le_bytes()).collect();

            let num_actions = entry.values.len() as i32;

            stmt.execute(params![
                state_blob,
                entry.agent_player as i32,
                num_actions,
                actions_blob,
                values_blob,
            ])?;
        }
        // Explicitly drop statement before committing
        drop(stmt);
    }
    tx.commit()?;

    Ok(num_entries)
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

    let eval_start = Instant::now();
    let (_actions, _values, log_entries) = q_minimax::evaluate_actions(
        &mechanics,
        &initial_state,
        args.max_steps,
        args.branching_factor,
        args.discount_factor,
        args.heuristic,
        true, // enable logging
    );
    let eval_elapsed = eval_start.elapsed();
    println!("  evaluate_actions took {:.3}s", eval_elapsed.as_secs_f64());

    // Write log entries to SQLite database
    if let Some(entries) = log_entries {
        let num_entries = entries.len();
        println!("  Collected {} log entries", num_entries);

        let write_start = Instant::now();
        save_policy_to_sqlite(
            entries,
            &args.output,
            args.board_size,
            args.max_steps,
            args.max_walls,
        )?;
        let write_elapsed = write_start.elapsed();
        println!(
            "  Writing database took {:.3}s",
            write_elapsed.as_secs_f64()
        );

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
