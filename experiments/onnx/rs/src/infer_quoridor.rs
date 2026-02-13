use anyhow::{Context, Result};
use ndarray::{Array, Array1, Array2};
use ort::session::Session;
use std::path::Path;
use std::time::Instant;

/// Represents a Quoridor game state for a 5x5 board with 1 wall per player
struct QuoridorState {
    board_size: usize,
    player_positions: Array2<i32>,  // [2, 2] - [player_id, [row, col]]
    walls: ndarray::Array3<i32>,    // [board_size-1, board_size-1, 2] - horizontal and vertical walls
    walls_remaining: Array1<i32>,   // [2] - walls remaining for each player
    current_player: usize,
}

impl QuoridorState {
    /// Create a new initial Quoridor game state
    fn new(board_size: usize, max_walls: usize) -> Self {
        let mut player_positions = Array2::zeros((2, 2));
        
        // Player 0 starts at top center
        player_positions[[0, 0]] = 0;
        player_positions[[0, 1]] = (board_size / 2) as i32;
        
        // Player 1 starts at bottom center
        player_positions[[1, 0]] = (board_size - 1) as i32;
        player_positions[[1, 1]] = (board_size / 2) as i32;
        
        // Walls array: [board_size-1, board_size-1, 2]
        // Third dimension: 0=horizontal, 1=vertical
        let walls = ndarray::Array3::zeros((board_size - 1, board_size - 1, 2));
        let walls_remaining = Array1::from_elem(2, max_walls as i32);
        
        QuoridorState {
            board_size,
            player_positions,
            walls,
            walls_remaining,
            current_player: 0,
        }
    }
    
    /// Convert the game state to ResNet input format
    /// 
    /// ResNet expects input of shape (batch_size, 5, M, M) where M = board_size * 2 + 3
    /// The 5 channels are:
    /// 1. Walls (1 where there is a wall, 0 otherwise)
    /// 2. Current player's position (1-hot encoding)
    /// 3. Opponent's position (1-hot encoding)
    /// 4. Current player walls remaining (same value for entire plane)
    /// 5. Opponent walls remaining (same value for entire plane)
    fn to_resnet_input_tensor(&self) -> ndarray::Array4<f32> {
        let opponent = 1 - self.current_player;
        let grid_size = self.board_size * 2 + 3; // Combined grid representation
        
        let mut input = ndarray::Array4::<f32>::zeros((1, 5, grid_size, grid_size));
        
        // Channel 0: Walls - we need to map the wall array to the combined grid
        // For a 5x5 board, the combined grid is 13x13
        // Walls are placed at odd positions (1, 3, 5, 7, 9, 11) in both dimensions
        for i in 0..(self.board_size - 1) {
            for j in 0..(self.board_size - 1) {
                let grid_i = i * 2 + 1;
                let grid_j = j * 2 + 1;
                
                // Horizontal wall
                if self.walls[[i, j, 0]] == 1 {
                    input[[0, 0, grid_i, grid_j]] = 1.0;
                    input[[0, 0, grid_i, grid_j + 1]] = 1.0;
                    input[[0, 0, grid_i, grid_j + 2]] = 1.0;
                }
                
                // Vertical wall
                if self.walls[[i, j, 1]] == 1 {
                    input[[0, 0, grid_i, grid_j]] = 1.0;
                    input[[0, 0, grid_i + 1, grid_j]] = 1.0;
                    input[[0, 0, grid_i + 2, grid_j]] = 1.0;
                }
            }
        }
        
        // Channel 1: Current player position (1-hot encoding)
        let player_row = self.player_positions[[self.current_player, 0]] as usize;
        let player_col = self.player_positions[[self.current_player, 1]] as usize;
        let player_grid_row = player_row * 2;
        let player_grid_col = player_col * 2;
        input[[0, 1, player_grid_row, player_grid_col]] = 1.0;
        
        // Channel 2: Opponent position (1-hot encoding)
        let opponent_row = self.player_positions[[opponent, 0]] as usize;
        let opponent_col = self.player_positions[[opponent, 1]] as usize;
        let opponent_grid_row = opponent_row * 2;
        let opponent_grid_col = opponent_col * 2;
        input[[0, 2, opponent_grid_row, opponent_grid_col]] = 1.0;
        
        // Channel 3: Current player walls remaining (same value for entire plane)
        let my_walls = self.walls_remaining[self.current_player] as f32;
        input.slice_mut(ndarray::s![0, 3, .., ..]).fill(my_walls);
        
        // Channel 4: Opponent walls remaining (same value for entire plane)
        let opp_walls = self.walls_remaining[opponent] as f32;
        input.slice_mut(ndarray::s![0, 4, .., ..]).fill(opp_walls);
        
        input
    }
    
    /// Convert the game state to the input format expected by the neural network
    /// 
    /// The input format is:
    /// - player_board.flatten() - board_size*board_size elements (1 where current player is, 0 elsewhere)
    /// - opponent_board.flatten() - board_size*board_size elements (1 where opponent is, 0 elsewhere)
    /// - walls.flatten() - (board_size-1)*(board_size-1)*2 elements (horizontal and vertical walls)
    /// - [my_walls, opponent_walls] - 2 elements
    fn to_input_tensor(&self) -> Array<f32, ndarray::Ix2> {
        let opponent = 1 - self.current_player;
        
        // Create player board (1 where current player is located)
        let mut player_board = Array2::<f32>::zeros((self.board_size, self.board_size));
        let player_row = self.player_positions[[self.current_player, 0]] as usize;
        let player_col = self.player_positions[[self.current_player, 1]] as usize;
        player_board[[player_row, player_col]] = 1.0;
        
        // Create opponent board (1 where opponent is located)
        let mut opponent_board = Array2::<f32>::zeros((self.board_size, self.board_size));
        let opponent_row = self.player_positions[[opponent, 0]] as usize;
        let opponent_col = self.player_positions[[opponent, 1]] as usize;
        opponent_board[[opponent_row, opponent_col]] = 1.0;
        
        // Convert walls to f32
        let walls_f32 = self.walls.mapv(|x| x as f32);
        
        // Flatten all components
        let player_flat = player_board.into_shape_with_order((self.board_size * self.board_size,)).unwrap();
        let opponent_flat = opponent_board.into_shape_with_order((self.board_size * self.board_size,)).unwrap();
        let walls_flat = walls_f32.into_shape_with_order(((self.board_size - 1) * (self.board_size - 1) * 2,)).unwrap();
        
        let my_walls = self.walls_remaining[self.current_player] as f32;
        let opp_walls = self.walls_remaining[opponent] as f32;
        
        // Calculate total input size
        let input_size = self.board_size * self.board_size * 2 
            + (self.board_size - 1) * (self.board_size - 1) * 2
            + 2;
        
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
}

/// Convert 2D array to 1D vector for ONNX input
fn array2d_to_vec(arr: &Array<f32, ndarray::Ix2>) -> Vec<f32> {
    arr.iter().copied().collect()
}

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

fn main() -> Result<()> {
    // Enable ONNX Runtime verbose logging
    std::env::set_var("ORT_LOG_SEVERITY_LEVEL", "1"); // 1 = INFO level

    let mlp_model_path = "../B5W3_mlp_sample.onnx";
    let resnet_model_path = "../B5W3_resnet_sample.onnx";
    
    // Board configuration (must match the trained model)
    let board_size = 5;
    let max_walls = 3;

    // --- 1. Load the MLP ONNX model ---
    if !Path::new(mlp_model_path).exists() {
        anyhow::bail!(
            "Error: MLP model not found at {}\nPlease ensure the ONNX model file exists.",
            mlp_model_path
        );
    }

    println!("Loading Quoridor MLP model from {}...", mlp_model_path);
    println!("Board size: {}x{}, Max walls: {}\n", board_size, board_size, max_walls);

    // Configure session with CUDA (GPU) execution provider
    let mut mlp_session = Session::builder()
        .context("Failed to create session builder")?
        .with_execution_providers([
            ort::execution_providers::CUDAExecutionProvider::default()
                .with_device_id(0)
                .build()
                .error_on_failure(),
        ])
        .context("Failed to configure CUDA execution provider - ensure CUDA/GPU is available")?
        .commit_from_file(mlp_model_path)
        .context("Failed to load MLP ONNX model")?;
    
    println!("✓ MLP Model loaded successfully!");
    println!("✓ CUDA execution provider configured (GPU acceleration enabled)\n");

    // --- 2. Create an initial game state ---
    let game_state = QuoridorState::new(board_size, max_walls);
    
    println!("=== Initial Game State ===");
    println!("Current player: {}", game_state.current_player);
    println!("Player 0 position: [{}, {}]", 
        game_state.player_positions[[0, 0]], 
        game_state.player_positions[[0, 1]]);
    println!("Player 1 position: [{}, {}]", 
        game_state.player_positions[[1, 0]], 
        game_state.player_positions[[1, 1]]);
    println!("Walls remaining: [{}, {}]\n", 
        game_state.walls_remaining[0], 
        game_state.walls_remaining[1]);

    // --- 3. Convert game state to input tensor ---
    let input_tensor = game_state.to_input_tensor();
    println!("Input tensor shape: {:?}", input_tensor.shape());
    println!("Input tensor size: {}\n", input_tensor.len());

    // --- 4. Run MLP inference multiple times for benchmarking ---
    println!("=== MLP Model Inference ===");
    let num_runs = 1000;
    println!("Running MLP inference {} times...", num_runs);
    
    let start = Instant::now();
    let mut last_policy_logits = None;
    let mut last_value = None;
    
    for _ in 0..num_runs {
        // Convert ndarray to Vec and create an ort Value with shape
        let shape = input_tensor.shape().to_vec();
        let data = array2d_to_vec(&input_tensor);
        let input_value = ort::value::Value::from_array((shape.as_slice(), data))
            .context("Failed to create input value")?;
        
        let outputs = mlp_session
            .run(ort::inputs!["input" => input_value])
            .context("Failed to run inference")?;
        
        // Extract and store the output data
        let policy_output = outputs["policy_logits"]
            .try_extract_tensor::<f32>()
            .context("Failed to extract policy tensor")?;
        let value_output = outputs["value"]
            .try_extract_tensor::<f32>()
            .context("Failed to extract value tensor")?;
        
        last_policy_logits = Some(policy_output.1.to_vec());
        last_value = Some(value_output.1.to_vec());
    }
    
    let duration = start.elapsed();
    let avg_time = duration.as_secs_f64() / num_runs as f64;
    
    println!("✓ Completed {} inferences", num_runs);
    println!("Total time: {:.4}s", duration.as_secs_f64());
    println!("Average time per inference: {:.6}s ({:.2} inferences/sec)\n", 
        avg_time, 1.0 / avg_time);

    // --- 5. Process the last inference output ---
    if let (Some(policy_logits), Some(value)) = (last_policy_logits, last_value) {
        println!("=== Model Output ===");
        println!("Policy logits length: {}", policy_logits.len());
        println!("Value length: {}", value.len());
        println!("Position value: {:.4} (range: -1 to 1, positive favors current player)\n", 
            value[0]);
        
        // Convert policy logits to probabilities using softmax
        let policy_probs = softmax(&policy_logits);
        
        // Find top 5 actions
        let mut indexed_probs: Vec<(usize, f32)> = policy_probs.iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        println!("=== Top 5 Recommended Actions ===");
        for (i, (action_idx, prob)) in indexed_probs.iter().take(5).enumerate() {
            println!("{}. Action index: {}, Probability: {:.4}", i + 1, action_idx, prob);
        }
        
        // Calculate action space info
        let num_move_actions = board_size * board_size;
        let num_wall_actions = (board_size - 1) * (board_size - 1) * 2; // horiz + vert
        let total_actions = num_move_actions + num_wall_actions;
        
        println!("\n=== Action Space ===");
        println!("Total action space size: {}", total_actions);
        println!("- Movement actions: {} (indices 0-{})", 
            num_move_actions, num_move_actions - 1);
        println!("- Horizontal wall actions: {} (indices {}-{})", 
            (board_size - 1) * (board_size - 1),
            num_move_actions,
            num_move_actions + (board_size - 1) * (board_size - 1) - 1);
        println!("- Vertical wall actions: {} (indices {}-{})", 
            (board_size - 1) * (board_size - 1),
            num_move_actions + (board_size - 1) * (board_size - 1),
            total_actions - 1);
    }

    // --- 6. Load ResNet model and run inference ---
    println!("\n\n=== ResNet Model Inference ===");
    
    if !Path::new(resnet_model_path).exists() {
        anyhow::bail!(
            "Error: ResNet model not found at {}\nPlease ensure the ONNX model file exists.",
            resnet_model_path
        );
    }

    println!("Loading Quoridor ResNet model from {}...", resnet_model_path);
    
    let mut resnet_session = Session::builder()
        .context("Failed to create ResNet session builder")?
        .with_execution_providers([
            ort::execution_providers::CUDAExecutionProvider::default()
                .with_device_id(0)
                .build()
                .error_on_failure(),
        ])
        .context("Failed to configure CUDA execution provider for ResNet")?
        .commit_from_file(resnet_model_path)
        .context("Failed to load ResNet ONNX model")?;
    
    println!("✓ ResNet Model loaded successfully!\n");

    // Convert game state to ResNet input format
    let resnet_input_tensor = game_state.to_resnet_input_tensor();
    let grid_size = board_size * 2 + 3;
    println!("ResNet input tensor shape: {:?}", resnet_input_tensor.shape());
    println!("Expected: (1, 5, {}, {})\n", grid_size, grid_size);

    // Run ResNet inference
    println!("Running ResNet inference {} times...", num_runs);
    
    let start = Instant::now();
    let mut last_policy_logits = None;
    let mut last_value = None;
    
    for _ in 0..num_runs {
        // Convert ndarray to Vec and create an ort Value with shape (1, 5, M, M)
        let shape = resnet_input_tensor.shape().to_vec();
        let data = array4d_to_vec(&resnet_input_tensor);
        let input_value = ort::value::Value::from_array((shape.as_slice(), data))
            .context("Failed to create ResNet input value")?;
        
        let outputs = resnet_session
            .run(ort::inputs!["input" => input_value])
            .context("Failed to run ResNet inference")?;
        
        // Extract and store the output data
        let policy_output = outputs["policy_logits"]
            .try_extract_tensor::<f32>()
            .context("Failed to extract ResNet policy tensor")?;
        let value_output = outputs["value"]
            .try_extract_tensor::<f32>()
            .context("Failed to extract ResNet value tensor")?;
        
        last_policy_logits = Some(policy_output.1.to_vec());
        last_value = Some(value_output.1.to_vec());
    }
    
    let duration = start.elapsed();
    let avg_time = duration.as_secs_f64() / num_runs as f64;
    
    println!("✓ Completed {} ResNet inferences", num_runs);
    println!("Total time: {:.4}s", duration.as_secs_f64());
    println!("Average time per inference: {:.6}s ({:.2} inferences/sec)\n", 
        avg_time, 1.0 / avg_time);

    // Process ResNet output
    if let (Some(policy_logits), Some(value)) = (last_policy_logits, last_value) {
        println!("=== ResNet Model Output ===");
        println!("Policy logits length: {}", policy_logits.len());
        println!("Value length: {}", value.len());
        println!("Position value: {:.4} (range: -1 to 1, positive favors current player)\n", 
            value[0]);
        
        // Convert policy logits to probabilities using softmax
        let policy_probs = softmax(&policy_logits);
        
        // Find top 5 actions
        let mut indexed_probs: Vec<(usize, f32)> = policy_probs.iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        println!("=== Top 5 Recommended Actions (ResNet) ===");
        for (i, (action_idx, prob)) in indexed_probs.iter().take(5).enumerate() {
            println!("{}. Action index: {}, Probability: {:.4}", i + 1, action_idx, prob);
        }
    }

    Ok(())
}
