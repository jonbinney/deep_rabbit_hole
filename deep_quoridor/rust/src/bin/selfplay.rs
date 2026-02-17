#!/usr/bin/env rust
//! Self-play executable using ONNX inference for Quoridor.
//!
//! This binary loads a trained ONNX model and uses it to evaluate actions
//! on a Quoridor game board, applying the selected action and displaying the result.

use anyhow::{Context, Result};
use ndarray::Array1;
use ort::session::Session;

use quoridor_rs::actions::{get_valid_move_actions, get_valid_wall_actions};
use quoridor_rs::game_state::apply_action;
use quoridor_rs::grid_helpers::{board_to_resnet_input, create_initial_board};

/// Convert 4D array to 1D vector for ONNX input
fn array4d_to_vec(arr: &ndarray::Array4<f32>) -> Vec<f32> {
    arr.iter().copied().collect()
}

/// Compute softmax values for policy logits
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_values: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exp_values.iter().sum();
    exp_values.iter().map(|&x| x / sum).collect()
}

/// Evaluate an action using the ONNX model
///
/// Returns the chosen action as [row, col, action_type]
fn evaluate_action(
    session: &mut Session,
    grid: &ndarray::ArrayView2<i8>,
    player_positions: &ndarray::ArrayView2<i32>,
    walls_remaining: &ndarray::ArrayView1<i32>,
    goal_rows: &ndarray::ArrayView1<i32>,
    current_player: i32,
) -> Result<Array1<i32>> {
    // Convert game state to ResNet input format
    let resnet_input_tensor = board_to_resnet_input(
        grid,
        player_positions,
        walls_remaining,
        current_player,
    );
    
    // Convert to ONNX input format
    let shape = resnet_input_tensor.shape().to_vec();
    let data = array4d_to_vec(&resnet_input_tensor);
    let input_value = ort::value::Value::from_array((shape.as_slice(), data))
        .context("Failed to create ResNet input value")?;
    
    // Run inference
    let outputs = session
        .run(ort::inputs!["input" => input_value])
        .context("Failed to run ResNet inference")?;
    
    // Extract policy logits
    let policy_logits_tuple = outputs["policy_logits"]
        .try_extract_tensor::<f32>()
        .context("Failed to extract policy logits")?;
    
    // Convert to probabilities
    let policy_probs = softmax(policy_logits_tuple.1);
    
    // Get all valid actions
    let move_actions = get_valid_move_actions(grid, player_positions, current_player);
    let wall_actions = get_valid_wall_actions(
        grid,
        player_positions,
        walls_remaining,
        goal_rows,
        current_player,
    );
    
    // Calculate action sizes
    let grid_width = grid.ncols() as i32;
    let board_size = (grid_width - 4) / 2 + 1;
    let num_move_actions = board_size * board_size;
    let wall_size = board_size - 1;
    let num_wall_actions = wall_size * wall_size;
    
    // Find best valid action
    let mut best_action_idx = 0;
    let mut best_prob = f32::NEG_INFINITY;
    
    // Check move actions
    for i in 0..move_actions.nrows() {
        let row = move_actions[[i, 0]];
        let col = move_actions[[i, 1]];
        let action_idx = (row * board_size + col) as usize;
        
        if action_idx < policy_probs.len() && policy_probs[action_idx] > best_prob {
            best_prob = policy_probs[action_idx];
            best_action_idx = i;
        }
    }
    
    // Check wall actions
    for i in 0..wall_actions.nrows() {
        let row = wall_actions[[i, 0]];
        let col = wall_actions[[i, 1]];
        let action_type = wall_actions[[i, 2]];
        
        // Calculate action index
        let wall_base_idx = if action_type == 1 {
            // Horizontal wall
            (num_move_actions + row * wall_size + col) as usize
        } else {
            // Vertical wall
            (num_move_actions + num_wall_actions + row * wall_size + col) as usize
        };
        
        if wall_base_idx < policy_probs.len() && policy_probs[wall_base_idx] > best_prob {
            best_prob = policy_probs[wall_base_idx];
            best_action_idx = move_actions.nrows() + i;
        }
    }
    
    // Return the chosen action
    if best_action_idx < move_actions.nrows() {
        Ok(Array1::from_vec(vec![
            move_actions[[best_action_idx, 0]],
            move_actions[[best_action_idx, 1]],
            move_actions[[best_action_idx, 2]],
        ]))
    } else {
        let wall_idx = best_action_idx - move_actions.nrows();
        Ok(Array1::from_vec(vec![
            wall_actions[[wall_idx, 0]],
            wall_actions[[wall_idx, 1]],
            wall_actions[[wall_idx, 2]],
        ]))
    }
}

/// Print the game board
fn print_board(
    grid: &ndarray::ArrayView2<i8>,
    player_positions: &ndarray::ArrayView2<i32>,
    walls_remaining: &ndarray::ArrayView1<i32>,
) {
    let grid_width = grid.ncols() as i32;
    let board_size = (grid_width - 4) / 2 + 1;
    
    println!("\n=== Game Board ({}x{}) ===", board_size, board_size);
    println!("Player 0 (P0): Position ({}, {}), Walls remaining: {}",
        player_positions[[0, 0]], player_positions[[0, 1]], walls_remaining[0]);
    println!("Player 1 (P1): Position ({}, {}), Walls remaining: {}",
        player_positions[[1, 0]], player_positions[[1, 1]], walls_remaining[1]);
    println!();
    
    // Print the board (showing only player positions and walls)
    for row in 0..board_size {
        for col in 0..board_size {
            let grid_row = (row * 2 + 2) as usize;
            let grid_col = (col * 2 + 2) as usize;
            
            let cell = grid[[grid_row, grid_col]];
            if cell == 0 {
                print!("P0 ");
            } else if cell == 1 {
                print!("P1 ");
            } else {
                print!(" . ");
            }
        }
        println!();
    }
    println!();
}

fn main() -> Result<()> {
    println!("=== Quoridor Self-Play with ONNX Inference ===\n");
    
    // Hardcoded model path (relative to rust directory)
    let model_path = "../../experiments/onnx/B5W3_resnet_sample.onnx";
    
    println!("Loading ONNX model from: {}", model_path);
    
    // Load ONNX model
    let mut session = Session::builder()
        .context("Failed to create session builder")?
        .commit_from_file(model_path)
        .context("Failed to load ONNX model")?;
    
    println!("✓ Model loaded successfully!\n");
    
    // Game configuration (must match the trained model)
    let board_size = 5;
    let walls_per_player = 3;
    
    println!("Game configuration: {}x{} board, {} walls per player\n", 
        board_size, board_size, walls_per_player);
    
    // Create initial board
    let (mut grid, mut player_positions, mut walls_remaining, goal_rows) = 
        create_initial_board(board_size, walls_per_player);
    let current_player = 0;
    
    println!("Initial board:");
    print_board(&grid.view(), &player_positions.view(), &walls_remaining.view());
    
    // Evaluate action using ONNX model
    println!("Evaluating action for Player {}...", current_player);
    let action = evaluate_action(
        &mut session,
        &grid.view(),
        &player_positions.view(),
        &walls_remaining.view(),
        &goal_rows.view(),
        current_player,
    )?;
    
    println!("Selected action: row={}, col={}, type={}", 
        action[0], action[1], action[2]);
    
    // Apply the action
    apply_action(
        &mut grid.view_mut(),
        &mut player_positions.view_mut(),
        &mut walls_remaining.view_mut(),
        current_player,
        &action.view(),
    );
    
    println!("\nBoard after applying action:");
    print_board(&grid.view(), &player_positions.view(), &walls_remaining.view());
    
    println!("✓ Self-play demonstration completed successfully!");
    
    Ok(())
}
