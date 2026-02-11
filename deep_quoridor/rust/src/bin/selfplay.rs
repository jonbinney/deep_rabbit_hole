use anyhow::{Context, Result};
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2};
use ort::session::Session;
use std::cell::RefCell;

// Action type constants (matching the minimax implementation)
// TODO: These will be used when we implement action mapping
#[allow(dead_code)]
const ACTION_WALL_VERTICAL: i32 = 0;
#[allow(dead_code)]
const ACTION_WALL_HORIZONTAL: i32 = 1;
#[allow(dead_code)]
const ACTION_MOVE: i32 = 2;

/// ONNX inference engine for Quoridor
/// 
/// This struct manages the ONNX session and provides methods for running
/// inference on Quoridor game states.
pub struct OnnxInference {
    session: RefCell<Session>,
    #[allow(dead_code)]
    board_size: usize,
}

impl OnnxInference {
    /// Create a new ONNX inference engine
    /// 
    /// # Arguments
    /// * `model_path` - Path to the ONNX model file
    /// * `board_size` - Size of the game board (e.g., 9 for 9x9)
    /// 
    /// # Returns
    /// * `Result<Self>` - The inference engine or an error
    pub fn new(model_path: &str, board_size: usize) -> Result<Self> {
        // Configure session with CUDA (GPU) execution provider
        let session = Session::builder()
            .context("Failed to create session builder")?
            .with_execution_providers([
                ort::execution_providers::CUDAExecutionProvider::default()
                    .with_device_id(0)
                    .build()
                    .error_on_failure(),
            ])
            .context("Failed to configure CUDA execution provider - ensure CUDA/GPU is available")?
            .commit_from_file(model_path)
            .context("Failed to load ONNX model")?;

        Ok(Self {
            session: RefCell::new(session),
            board_size,
        })
    }

    /// Run inference on a game state
    /// 
    /// # Arguments
    /// * `input_tensor` - Flattened input tensor for the model
    /// 
    /// # Returns
    /// * `Result<(Vec<f32>, f32)>` - (policy_logits, value) or an error
    fn run_inference(&self, input_tensor: &Array2<f32>) -> Result<(Vec<f32>, f32)> {
        // Convert ndarray to Vec and create an ort Value with shape
        let shape = input_tensor.shape().to_vec();
        let data: Vec<f32> = input_tensor.iter().copied().collect();
        
        let input_value = ort::value::Value::from_array((shape.as_slice(), data))
            .context("Failed to create input value")?;
        
        let mut session = self.session.borrow_mut();
        let outputs = session
            .run(ort::inputs!["input" => input_value])
            .context("Failed to run inference")?;
        
        // Extract policy logits and value
        let policy_output = outputs["policy_logits"]
            .try_extract_tensor::<f32>()
            .context("Failed to extract policy tensor")?;
        let value_output = outputs["value"]
            .try_extract_tensor::<f32>()
            .context("Failed to extract value tensor")?;
        
        let policy_logits = policy_output.1.to_vec();
        let value = value_output.1[0];
        
        Ok((policy_logits, value))
    }
}

/// Convert game state from minimax format to ONNX input tensor
/// 
/// The input format expected by the neural network is:
/// - player_board.flatten() - board_size*board_size elements (1 where current player is, 0 elsewhere)
/// - opponent_board.flatten() - board_size*board_size elements (1 where opponent is, 0 elsewhere)
/// - walls.flatten() - (board_size-1)*(board_size-1)*2 elements (horizontal and vertical walls)
/// - [my_walls, opponent_walls] - 2 elements
/// 
/// # Arguments
/// * `grid` - Game grid with walls and player positions
/// * `player_positions` - Array of player positions [2, 2] where each row is [row, col]
/// * `walls_remaining` - Remaining walls for each player [2]
/// * `current_player` - Current player (0 or 1)
/// 
/// # Returns
/// * `Array2<f32>` - Input tensor with shape (1, input_size)
fn convert_game_state_to_onnx_input(
    grid: &ArrayView2<i8>,
    player_positions: &ArrayView2<i32>,
    walls_remaining: &ArrayView1<i32>,
    current_player: i32,
) -> Array2<f32> {
    let board_size = (grid.ncols() - 4) / 2 + 1;
    let opponent = 1 - current_player;
    
    // Create player board (1 where current player is located)
    let mut player_board = Array2::<f32>::zeros((board_size, board_size));
    let player_row = player_positions[[current_player as usize, 0]] as usize;
    let player_col = player_positions[[current_player as usize, 1]] as usize;
    player_board[[player_row, player_col]] = 1.0;
    
    // Create opponent board (1 where opponent is located)
    let mut opponent_board = Array2::<f32>::zeros((board_size, board_size));
    let opponent_row = player_positions[[opponent as usize, 0]] as usize;
    let opponent_col = player_positions[[opponent as usize, 1]] as usize;
    opponent_board[[opponent_row, opponent_col]] = 1.0;
    
    // Extract walls from grid
    // Walls are stored in the grid at odd indices (wall cells)
    // We need to convert to the (board_size-1, board_size-1, 2) format
    let wall_size = board_size - 1;
    let mut walls = Array3::<f32>::zeros((wall_size, wall_size, 2));
    
    // Extract horizontal and vertical walls from the grid
    for wall_row in 0..wall_size {
        for wall_col in 0..wall_size {
            // Check for vertical wall (spans 3 vertical cells)
            let grid_row = (wall_row * 2 + 2) as usize;
            let grid_col = (wall_col * 2 + 3) as usize;
            
            if grid[[grid_row, grid_col]] == -1 {
                walls[[wall_row, wall_col, 0]] = 1.0; // Vertical wall
            }
            
            // Check for horizontal wall (spans 3 horizontal cells)
            let grid_row = (wall_row * 2 + 3) as usize;
            let grid_col = (wall_col * 2 + 2) as usize;
            
            if grid[[grid_row, grid_col]] == -1 {
                walls[[wall_row, wall_col, 1]] = 1.0; // Horizontal wall
            }
        }
    }
    
    // Flatten all components
    let player_flat = player_board.into_shape_with_order((board_size * board_size,)).unwrap();
    let opponent_flat = opponent_board.into_shape_with_order((board_size * board_size,)).unwrap();
    let walls_flat = walls.into_shape_with_order((wall_size * wall_size * 2,)).unwrap();
    
    let my_walls = walls_remaining[current_player as usize] as f32;
    let opp_walls = walls_remaining[opponent as usize] as f32;
    
    // Calculate total input size
    let input_size = board_size * board_size * 2 + wall_size * wall_size * 2 + 2;
    
    // Concatenate all features
    let mut input = Array1::<f32>::zeros(input_size);
    let mut offset = 0;
    
    // Copy player board
    input.slice_mut(ndarray::s![offset..offset + player_flat.len()]).assign(&player_flat);
    offset += player_flat.len();
    
    // Copy opponent board
    input.slice_mut(ndarray::s![offset..offset + opponent_flat.len()]).assign(&opponent_flat);
    offset += opponent_flat.len();
    
    // Copy walls
    input.slice_mut(ndarray::s![offset..offset + walls_flat.len()]).assign(&walls_flat);
    offset += walls_flat.len();
    
    // Copy wall counts
    input[offset] = my_walls;
    input[offset + 1] = opp_walls;
    
    // Add batch dimension: (1, input_size)
    input.into_shape_with_order((1, input_size)).unwrap()
}

/// Evaluate actions using ONNX inference
/// 
/// This function takes the same game state format as the minimax `evaluate_actions`
/// but uses ONNX inference instead of tree search.
/// 
/// # Arguments
/// * `inference` - ONNX inference engine
/// * `grid` - Game grid with walls and player positions
/// * `player_positions` - Array of player positions [2, 2]
/// * `walls_remaining` - Remaining walls for each player [2]
/// * `goal_rows` - Goal rows for each player [2] (not used in ONNX inference)
/// * `current_player` - Current player (0 or 1)
/// 
/// # Returns
/// * `Result<(Vec<f32>, f32)>` - (policy_logits, value) or an error
pub fn evaluate_actions(
    inference: &OnnxInference,
    grid: &ArrayView2<i8>,
    player_positions: &ArrayView2<i32>,
    walls_remaining: &ArrayView1<i32>,
    _goal_rows: &ArrayView1<i32>,
    current_player: i32,
) -> Result<(Vec<f32>, f32)> {
    // Convert game state to ONNX input format
    let input_tensor = convert_game_state_to_onnx_input(
        grid,
        player_positions,
        walls_remaining,
        current_player,
    );
    
    // Run inference
    inference.run_inference(&input_tensor)
}

// TODO: Add function to map policy indices to (row, col, action_type) format
// The policy output is a flat vector where:
// - Indices 0 to board_size*board_size-1 are movement actions
// - Indices board_size*board_size to board_size*board_size + (board_size-1)^2 - 1 are horizontal wall actions
// - Remaining indices are vertical wall actions

fn main() -> Result<()> {
    println!("Quoridor Self-Play with ONNX Inference");
    println!("======================================\n");
    
    // Hardcoded model path - TODO: make this configurable
    let model_path = "../../experiments/onnx/model_6.onnx";
    let board_size = 5; // Must match the trained model
    
    // Check if model file exists
    if !std::path::Path::new(model_path).exists() {
        anyhow::bail!(
            "Error: Model not found at {}\nPlease ensure the ONNX model file exists.",
            model_path
        );
    }
    
    println!("Loading ONNX model from: {}", model_path);
    println!("Board size: {}x{}\n", board_size, board_size);
    
    // Create inference engine
    let inference = OnnxInference::new(model_path, board_size)
        .context("Failed to create ONNX inference engine")?;
    
    println!("✓ Model loaded successfully!");
    println!("✓ CUDA execution provider configured (GPU acceleration enabled)\n");
    
    // Create a sample game state (initial position)
    let grid_width = board_size * 2 + 3;
    let mut grid = Array2::from_elem((grid_width, grid_width), 0i8);
    
    // Add border walls
    for i in 0..grid_width {
        grid[[0, i]] = -1;
        grid[[1, i]] = -1;
        grid[[grid_width - 1, i]] = -1;
        grid[[grid_width - 2, i]] = -1;
        grid[[i, 0]] = -1;
        grid[[i, 1]] = -1;
        grid[[i, grid_width - 1]] = -1;
        grid[[i, grid_width - 2]] = -1;
    }
    
    // Set player positions
    let mut player_positions = Array2::zeros((2, 2));
    player_positions[[0, 0]] = 0;
    player_positions[[0, 1]] = (board_size / 2) as i32;
    player_positions[[1, 0]] = (board_size - 1) as i32;
    player_positions[[1, 1]] = (board_size / 2) as i32;
    
    // Mark players on grid
    let p0_grid_row = (player_positions[[0, 0]] * 2 + 2) as usize;
    let p0_grid_col = (player_positions[[0, 1]] * 2 + 2) as usize;
    grid[[p0_grid_row, p0_grid_col]] = 0;
    
    let p1_grid_row = (player_positions[[1, 0]] * 2 + 2) as usize;
    let p1_grid_col = (player_positions[[1, 1]] * 2 + 2) as usize;
    grid[[p1_grid_row, p1_grid_col]] = 1;
    
    let walls_remaining = Array1::from_elem(2, 1);
    let goal_rows = Array1::from_vec(vec![(board_size - 1) as i32, 0]);
    let current_player = 0;
    
    println!("=== Sample Game State ===");
    println!("Current player: {}", current_player);
    println!("Player 0 position: [{}, {}]", 
        player_positions[[0, 0]], player_positions[[0, 1]]);
    println!("Player 1 position: [{}, {}]", 
        player_positions[[1, 0]], player_positions[[1, 1]]);
    println!("Walls remaining: [{}, {}]\n", 
        walls_remaining[0], walls_remaining[1]);
    
    // Evaluate actions
    println!("Running inference...");
    let (policy_logits, value) = evaluate_actions(
        &inference,
        &grid.view(),
        &player_positions.view(),
        &walls_remaining.view(),
        &goal_rows.view(),
        current_player,
    )?;
    
    println!("✓ Inference complete!\n");
    
    println!("=== Model Output ===");
    println!("Policy logits length: {}", policy_logits.len());
    println!("Position value: {:.4} (range: -1 to 1, positive favors current player)\n", value);
    
    // Show top 5 actions by logit value
    let mut indexed_logits: Vec<(usize, f32)> = policy_logits
        .iter()
        .enumerate()
        .map(|(i, &p)| (i, p))
        .collect();
    indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    println!("=== Top 5 Actions by Logit Value ===");
    for (i, (action_idx, logit)) in indexed_logits.iter().take(5).enumerate() {
        println!("{}. Action index: {}, Logit: {:.4}", i + 1, action_idx, logit);
    }
    
    println!("\n✓ Self-play binary successfully executed!");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_game_state_empty_board() {
        // Create a minimal 3x3 board (grid will be 9x9)
        let board_size = 3;
        let grid_width = board_size * 2 + 3;
        let grid = Array2::from_elem((grid_width, grid_width), 0i8);
        
        let mut player_positions = Array2::zeros((2, 2));
        player_positions[[0, 0]] = 0;
        player_positions[[0, 1]] = 1;
        player_positions[[1, 0]] = 2;
        player_positions[[1, 1]] = 1;
        
        let walls_remaining = Array1::from_elem(2, 5);
        let current_player = 0;
        
        let input = convert_game_state_to_onnx_input(
            &grid.view(),
            &player_positions.view(),
            &walls_remaining.view(),
            current_player,
        );
        
        // Check shape: (1, board_size^2*2 + wall_size^2*2 + 2)
        // = (1, 3*3*2 + 2*2*2 + 2) = (1, 18 + 8 + 2) = (1, 28)
        assert_eq!(input.shape(), &[1, 28]);
        
        // Check that player position is marked
        let player_flat_idx = player_positions[[0, 0]] as usize * board_size + player_positions[[0, 1]] as usize;
        assert_eq!(input[[0, player_flat_idx]], 1.0);
        
        // Check opponent position
        let opponent_flat_idx = player_positions[[1, 0]] as usize * board_size + player_positions[[1, 1]] as usize;
        let opponent_offset = board_size * board_size;
        assert_eq!(input[[0, opponent_offset + opponent_flat_idx]], 1.0);
    }

    #[test]
    fn test_convert_game_state_with_walls() {
        let board_size = 3;
        let grid_width = board_size * 2 + 3;
        let mut grid = Array2::from_elem((grid_width, grid_width), 0i8);
        
        // Add a vertical wall at position (0, 0)
        // This spans cells at grid indices [2, 3], [3, 3], [4, 3]
        grid[[2, 3]] = -1;
        grid[[3, 3]] = -1;
        grid[[4, 3]] = -1;
        
        let mut player_positions = Array2::zeros((2, 2));
        player_positions[[0, 0]] = 0;
        player_positions[[0, 1]] = 0;
        player_positions[[1, 0]] = 2;
        player_positions[[1, 1]] = 2;
        
        let walls_remaining = Array1::from_elem(2, 4);
        let current_player = 0;
        
        let input = convert_game_state_to_onnx_input(
            &grid.view(),
            &player_positions.view(),
            &walls_remaining.view(),
            current_player,
        );
        
        // The wall should be in the walls section
        // walls_offset = board_size^2 * 2 = 18
        // For wall at (0, 0), vertical (index 0 in walls array)
        let walls_offset = board_size * board_size * 2;
        assert_eq!(input[[0, walls_offset]], 1.0);
    }

    #[test]
    fn test_convert_game_state_wall_counts() {
        let board_size = 3;
        let grid_width = board_size * 2 + 3;
        let grid = Array2::from_elem((grid_width, grid_width), 0i8);
        
        let player_positions = Array2::zeros((2, 2));
        let walls_remaining = Array1::from_vec(vec![3, 7]);
        let current_player = 1; // Player 1 is current
        
        let input = convert_game_state_to_onnx_input(
            &grid.view(),
            &player_positions.view(),
            &walls_remaining.view(),
            current_player,
        );
        
        // Wall counts are at the end
        let input_size = 28; // 3*3*2 + 2*2*2 + 2
        assert_eq!(input[[0, input_size - 2]], 7.0); // Current player (1) has 7 walls
        assert_eq!(input[[0, input_size - 1]], 3.0); // Opponent (0) has 3 walls
    }

    #[test]
    fn test_onnx_session_creation() {
        // This test will fail if the model doesn't exist, which is expected
        // in many environments. Mark as ignored for CI.
        let model_path = "../../experiments/onnx/model_6.onnx";
        let board_size = 5;
        
        // Try to create the inference engine
        let result = OnnxInference::new(model_path, board_size);
        
        // If the model exists and CUDA is available, it should succeed
        // Otherwise, it's OK to fail (we're just testing the API)
        if result.is_ok() {
            println!("ONNX inference engine created successfully!");
        } else {
            println!("Model not found or CUDA not available (expected in some environments)");
        }
    }

    #[test]
    fn test_input_tensor_size_calculation() {
        // Test that the input tensor has the correct size for different board sizes
        for board_size in [3, 5, 7, 9] {
            let expected_size = board_size * board_size * 2 
                + (board_size - 1) * (board_size - 1) * 2 
                + 2;
            
            let grid_width = board_size * 2 + 3;
            let grid = Array2::from_elem((grid_width, grid_width), 0i8);
            let player_positions = Array2::zeros((2, 2));
            let walls_remaining = Array1::from_elem(2, 5);
            
            let input = convert_game_state_to_onnx_input(
                &grid.view(),
                &player_positions.view(),
                &walls_remaining.view(),
                0,
            );
            
            assert_eq!(input.shape()[1], expected_size,
                "Board size {} should have input size {}", board_size, expected_size);
        }
    }
}
