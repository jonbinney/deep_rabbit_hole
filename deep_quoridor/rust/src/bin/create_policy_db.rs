#!/usr/bin/env rust
//! Create a policy database from minimax evaluations.

use clap::Parser;
use quoridor_rs::compact::{
    policy_db::{self, TranspositionTable},
    q_game_mechanics::QGameMechanics,
};
use rusqlite::{params, Connection};
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

    /// Number of threads for Lazy SMP parallel search
    #[arg(long, default_value_t = 1)]
    num_threads: usize,

    /// Output SQLite database file path
    #[arg(short, long, default_value = "policy_db.sqlite")]
    output: String,
}

#[allow(dead_code)]
pub fn save_policy_to_sqlite(
    _mechanics: &QGameMechanics,
    entries: TranspositionTable,
    filename: &str,
    board_size: usize,
    max_steps: usize,
    max_walls: usize,
) -> Result<usize, Box<dyn std::error::Error>> {
    let mut conn = Connection::open(Path::new(filename))?;

    // Performance pragmas: disable journaling and syncs during bulk insert
    conn.execute_batch(
        "PRAGMA journal_mode = OFF;
         PRAGMA synchronous = OFF;
         PRAGMA locking_mode = EXCLUSIVE;
         PRAGMA temp_store = MEMORY;
         PRAGMA cache_size = -64000;",
    )?;

    // Drop existing tables to avoid schema conflicts
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

    // Create table WITHOUT primary key index — insert first, index later
    conn.execute(
        "CREATE TABLE policy (
            state BLOB,
            value INTEGER NOT NULL
        )",
        [],
    )?;

    let num_entries = entries.len();

    // Bulk insert in a transaction
    let tx = conn.transaction()?;
    {
        let mut stmt = tx.prepare("INSERT INTO policy (state, value) VALUES (?1, ?2)")?;

        for item in entries.into_iter() {
            let (key, value) = item;

            // Transposition table already stores P0-absolute values
            // (1=P0 wins, -1=P0 loses). Store directly.
            stmt.execute(params![key.as_slice(), value as i32])?;
        }
        drop(stmt);
    }
    tx.commit()?;

    // Create index after all inserts (much faster than maintaining during insert)
    conn.execute("CREATE UNIQUE INDEX idx_policy_state ON policy(state)", [])?;

    Ok(num_entries)
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

    // Write transposition table entries to SQLite database
    let num_entries = transposition_table.len();
    println!("  Collected {} transposition table entries", num_entries);

    let write_start = Instant::now();
    save_policy_to_sqlite(
        &mechanics,
        transposition_table,
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
        "\nSuccessfully created policy database with {} entries",
        num_entries
    );
    println!("  Saved to: {}", args.output);

    Ok(())
}
