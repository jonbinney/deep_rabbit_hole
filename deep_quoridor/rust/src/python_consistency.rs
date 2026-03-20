use std::io::ErrorKind;
use std::path::PathBuf;
use std::process::Command;

use crate::actions::{action_index_to_action, policy_size};
#[cfg(feature = "binary")]
use crate::agents::alphazero::agent::apply_temperature_and_sample;
#[cfg(feature = "binary")]
use crate::agents::alphazero::evaluator::UniformMockEvaluator;
#[cfg(feature = "binary")]
use crate::agents::alphazero::mcts::{search, ChildInfo, MCTSConfig};
use crate::game_state::GameState;
use crate::grid_helpers::grid_game_state_to_resnet_input;
#[cfg(feature = "binary")]
use crate::rotation::rotate_action_coords;
use crate::rotation::{rotate_goal_rows, rotate_grid_180, rotate_player_positions};

fn python_reference(board_size: i32, max_walls: i32) -> (Vec<[i32; 3]>, Vec<bool>) {
    let src_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("rust crate should live under deep_quoridor/")
        .join("src");

    let script_path = src_dir.join("action_reference.py");

    let args = [
        src_dir.to_string_lossy().into_owned(),
        board_size.to_string(),
        max_walls.to_string(),
    ];

    let output = run_python(&script_path.to_string_lossy(), &args);
    parse_reference_output(&output, board_size)
}

fn run_python(script_path: &str, args: &[String]) -> String {
    let mut candidates = Vec::new();
    if let Ok(python) = std::env::var("PYTHON") {
        candidates.push(python);
    }
    candidates.push("python".to_string());
    candidates.push("python3".to_string());

    for candidate in candidates {
        let output = Command::new(&candidate)
            .arg(script_path)
            .args(args)
            .output();

        match output {
            Ok(output) if output.status.success() => {
                return String::from_utf8(output.stdout).expect("python stdout should be utf-8");
            }
            Ok(output) => {
                panic!(
                    "python command '{}' failed:\nstdout:\n{}\nstderr:\n{}",
                    candidate,
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            Err(err) if err.kind() == ErrorKind::NotFound => continue,
            Err(err) => panic!("failed to run python command '{}': {}", candidate, err),
        }
    }

    panic!("no python interpreter found for cross-language consistency test");
}

fn parse_reference_output(output: &str, board_size: i32) -> (Vec<[i32; 3]>, Vec<bool>) {
    let expected_actions = policy_size(board_size);
    let mut actions = vec![[0, 0, 0]; expected_actions];
    let mut mask = None;

    for line in output.lines() {
        if let Some(rest) = line.strip_prefix("A,") {
            let parts: Vec<_> = rest.split(',').collect();
            assert_eq!(parts.len(), 4, "unexpected python action line: {line}");
            let idx: usize = parts[0].parse().expect("valid action index");
            actions[idx] = [
                parts[1].parse().expect("valid row"),
                parts[2].parse().expect("valid col"),
                parts[3].parse().expect("valid action type"),
            ];
        } else if let Some(rest) = line.strip_prefix("M,") {
            mask = Some(rest.chars().map(|c| c == '1').collect::<Vec<_>>());
        }
    }

    let mask = mask.expect("python output should include action mask");
    assert_eq!(actions.len(), expected_actions);
    assert_eq!(mask.len(), expected_actions);
    (actions, mask)
}

#[test]
fn test_action_encoding_matches_python() {
    for (board_size, max_walls) in [(5, 2), (9, 10)] {
        let (python_actions, _) = python_reference(board_size, max_walls);
        for (idx, python_action) in python_actions.into_iter().enumerate() {
            let rust_action = action_index_to_action(board_size, idx);
            assert_eq!(
                rust_action, python_action,
                "board_size={board_size}, idx={idx}"
            );
        }
    }
}

#[test]
fn test_initial_action_mask_matches_python() {
    for (board_size, max_walls) in [(5, 2), (9, 10)] {
        let (_, python_mask) = python_reference(board_size, max_walls);
        let rust_mask = GameState::new(board_size, max_walls).get_action_mask();
        assert_eq!(rust_mask, python_mask, "board_size={board_size}");
    }
}

// ---------------------------------------------------------------------------
// Step-trace consistency test
// ---------------------------------------------------------------------------

/// One snapshot of game state at a particular step, as reported by Python.
#[derive(Debug)]
#[allow(dead_code)]
struct StepSnapshot {
    step: usize,
    grid_hex: String,
    player_positions: [i32; 4], // p0r, p0c, p1r, p1c
    walls_remaining: [i32; 2],
    current_player: i32,
    action_mask: Vec<bool>,
    tensor_hex: String,
    /// Rotated mask (Some only when current_player == 1)
    rotated_mask: Option<Vec<bool>>,
    /// Rotated tensor (Some only when current_player == 1)
    rotated_tensor_hex: Option<String>,
    /// Root value hex-encoded float32 bytes (Some only when a move is selected)
    root_value_hex: Option<String>,
    /// Root policy hex-encoded float32 bytes (Some only when a move is selected)
    root_policy_hex: Option<String>,
    /// Selected action index in the working frame (Some only when a move is selected)
    selected_action_index: Option<usize>,
}

/// Call the step_trace_reference.py script and return its raw stdout.
fn run_step_trace_python(board_size: i32, max_walls: i32, action_indices: &[usize]) -> String {
    let src_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("rust crate should live under deep_quoridor/")
        .join("src");

    let script_path = src_dir.join("step_trace_reference.py");

    let mut args = vec![
        src_dir.to_string_lossy().into_owned(),
        board_size.to_string(),
        max_walls.to_string(),
    ];
    for &idx in action_indices {
        args.push(idx.to_string());
    }

    run_python(&script_path.to_string_lossy(), &args)
}

/// Call the mcts_game_reference.py script and return its raw stdout.
#[cfg(feature = "binary")]
fn run_mcts_game_python(board_size: i32, max_walls: i32, max_steps: i32, mcts_n: u32) -> String {
    let src_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("rust crate should live under deep_quoridor/")
        .join("src");

    let script_path = src_dir.join("mcts_game_reference.py");

    let args = [
        src_dir.to_string_lossy().into_owned(),
        board_size.to_string(),
        max_walls.to_string(),
        max_steps.to_string(),
        mcts_n.to_string(),
    ];

    run_python(&script_path.to_string_lossy(), &args)
}

/// Parse step_trace_reference.py output into a vector of snapshots.
fn parse_step_trace_output(output: &str) -> Vec<StepSnapshot> {
    // Collect lines by step, then build snapshots.
    use std::collections::BTreeMap;

    struct RawStep {
        grid_hex: Option<String>,
        positions: Option<[i32; 4]>,
        walls: Option<[i32; 2]>,
        current_player: Option<i32>,
        mask: Option<Vec<bool>>,
        tensor_hex: Option<String>,
        rotated_mask: Option<Vec<bool>>,
        rotated_tensor_hex: Option<String>,
        root_value_hex: Option<String>,
        root_policy_hex: Option<String>,
        selected_action_index: Option<usize>,
    }

    let mut steps: BTreeMap<usize, RawStep> = BTreeMap::new();

    for line in output.lines() {
        let parts: Vec<&str> = line.splitn(3, ',').collect();
        if parts.len() < 3 {
            continue;
        }
        let tag = parts[0];
        let step: usize = parts[1].parse().expect("valid step number");
        let rest = parts[2];

        let entry = steps.entry(step).or_insert_with(|| RawStep {
            grid_hex: None,
            positions: None,
            walls: None,
            current_player: None,
            mask: None,
            tensor_hex: None,
            rotated_mask: None,
            rotated_tensor_hex: None,
            root_value_hex: None,
            root_policy_hex: None,
            selected_action_index: None,
        });

        match tag {
            "G" => entry.grid_hex = Some(rest.to_string()),
            "P" => {
                let nums: Vec<i32> = rest.split(',').map(|s| s.parse().unwrap()).collect();
                entry.positions = Some([nums[0], nums[1], nums[2], nums[3]]);
            }
            "W" => {
                let nums: Vec<i32> = rest.split(',').map(|s| s.parse().unwrap()).collect();
                entry.walls = Some([nums[0], nums[1]]);
            }
            "C" => {
                entry.current_player = Some(rest.parse().unwrap());
            }
            "M" => {
                entry.mask = Some(rest.chars().map(|c| c == '1').collect());
            }
            "T" => entry.tensor_hex = Some(rest.to_string()),
            "RM" => {
                entry.rotated_mask = Some(rest.chars().map(|c| c == '1').collect());
            }
            "RT" => entry.rotated_tensor_hex = Some(rest.to_string()),
            "V" => entry.root_value_hex = Some(rest.to_string()),
            "Q" => entry.root_policy_hex = Some(rest.to_string()),
            "A" => entry.selected_action_index = Some(rest.parse().unwrap()),
            _ => {} // ignore unknown tags
        }
    }

    steps
        .into_iter()
        .map(|(step, raw)| StepSnapshot {
            step,
            grid_hex: raw
                .grid_hex
                .unwrap_or_else(|| panic!("missing G for step {step}")),
            player_positions: raw
                .positions
                .unwrap_or_else(|| panic!("missing P for step {step}")),
            walls_remaining: raw
                .walls
                .unwrap_or_else(|| panic!("missing W for step {step}")),
            current_player: raw
                .current_player
                .unwrap_or_else(|| panic!("missing C for step {step}")),
            action_mask: raw
                .mask
                .unwrap_or_else(|| panic!("missing M for step {step}")),
            tensor_hex: raw
                .tensor_hex
                .unwrap_or_else(|| panic!("missing T for step {step}")),
            rotated_mask: raw.rotated_mask,
            rotated_tensor_hex: raw.rotated_tensor_hex,
            root_value_hex: raw.root_value_hex,
            root_policy_hex: raw.root_policy_hex,
            selected_action_index: raw.selected_action_index,
        })
        .collect()
}

/// Encode a grid (Array2<i8>) as hex string of int8 bytes, row-major, matching Python.
fn grid_to_hex(grid: &ndarray::ArrayView2<i8>) -> String {
    grid.iter().map(|&v| format!("{:02x}", v as u8)).collect()
}

/// Encode an ndarray::Array4<f32> as hex string of its raw bytes (C-contiguous).
fn tensor_to_hex(tensor: &ndarray::Array4<f32>) -> String {
    tensor
        .iter()
        .flat_map(|&v| v.to_le_bytes())
        .map(|b| format!("{:02x}", b))
        .collect()
}

/// Encode f32 vector as hex string of little-endian float32 bytes.
#[cfg(feature = "binary")]
fn vec_f32_to_hex(values: &[f32]) -> String {
    values
        .iter()
        .flat_map(|&v| v.to_le_bytes())
        .map(|b| format!("{:02x}", b))
        .collect()
}

#[cfg(feature = "binary")]
fn hex_to_f32(hex: &str) -> f32 {
    assert_eq!(hex.len(), 8, "expected 4-byte float hex");
    let mut bytes = [0u8; 4];
    for i in 0..4 {
        let start = i * 2;
        let end = start + 2;
        bytes[i] = u8::from_str_radix(&hex[start..end], 16).expect("valid hex byte");
    }
    f32::from_le_bytes(bytes)
}

/// Build a rotated GameState for Player 1 (mirrors game_runner.rs logic).
fn build_rotated_state(state: &GameState) -> GameState {
    let work_grid = rotate_grid_180(&state.grid());
    let work_positions = rotate_player_positions(&state.player_positions(), state.board_size);
    let work_goals = rotate_goal_rows(&state.goal_rows());
    GameState {
        grid: work_grid,
        player_positions: work_positions,
        walls_remaining: state.walls_remaining.clone(),
        goal_rows: work_goals,
        current_player: state.current_player,
        board_size: state.board_size,
        completed_steps: state.completed_steps,
    }
}

/// Compare one snapshot field, returning a descriptive error message on mismatch.
fn assert_snapshot_matches(seq_name: &str, step: usize, py: &StepSnapshot, state: &GameState) {
    // 1. Grid
    let rust_grid_hex = grid_to_hex(&state.grid());
    assert_eq!(
        rust_grid_hex, py.grid_hex,
        "[{seq_name}] step {step}: grid mismatch"
    );

    // 2. Player positions
    let pp = state.player_positions();
    let rust_positions = [pp[[0, 0]], pp[[0, 1]], pp[[1, 0]], pp[[1, 1]]];
    assert_eq!(
        rust_positions, py.player_positions,
        "[{seq_name}] step {step}: player positions mismatch"
    );

    // 3. Walls remaining
    let wr = state.walls_remaining();
    let rust_walls = [wr[0], wr[1]];
    assert_eq!(
        rust_walls, py.walls_remaining,
        "[{seq_name}] step {step}: walls remaining mismatch"
    );

    // 4. Current player
    assert_eq!(
        state.current_player, py.current_player,
        "[{seq_name}] step {step}: current player mismatch"
    );

    // 5. Action mask
    let rust_mask = state.get_action_mask();
    assert_eq!(
        rust_mask, py.action_mask,
        "[{seq_name}] step {step}: action mask mismatch"
    );

    // 6. Tensor
    let rust_tensor = grid_game_state_to_resnet_input(state);
    let rust_tensor_hex = tensor_to_hex(&rust_tensor);
    assert_eq!(
        rust_tensor_hex, py.tensor_hex,
        "[{seq_name}] step {step}: tensor mismatch"
    );

    // 7. Rotated mask and tensor (when player 1)
    if py.current_player == 1 {
        let rotated = build_rotated_state(state);

        if let Some(ref py_rmask) = py.rotated_mask {
            let rust_rmask = rotated.get_action_mask();
            assert_eq!(
                rust_rmask, *py_rmask,
                "[{seq_name}] step {step}: rotated action mask mismatch"
            );
        }

        if let Some(ref py_rtensor_hex) = py.rotated_tensor_hex {
            let rust_rtensor = grid_game_state_to_resnet_input(&rotated);
            let rust_rtensor_hex = tensor_to_hex(&rust_rtensor);
            assert_eq!(
                rust_rtensor_hex, *py_rtensor_hex,
                "[{seq_name}] step {step}: rotated tensor mismatch"
            );
        }
    }
}

/// The four deterministic action sequences used for testing.
/// Each is: (name, action_indices).
fn test_sequences() -> Vec<(&'static str, Vec<usize>)> {
    vec![
        ("A_moves_only", vec![7, 17, 12, 18, 13, 23]),
        ("B_walls_and_moves", vec![7, 17, 30, 41, 12, 16, 13, 51]),
        ("C_straight_jump", vec![7, 17, 12, 7]),
        ("D_diagonal_jump", vec![7, 17, 42, 12, 37, 6]),
    ]
}

#[test]
fn test_step_trace_matches_python() {
    let board_size = 5;
    let max_walls = 2;

    for (name, actions) in test_sequences() {
        let output = run_step_trace_python(board_size, max_walls, &actions);
        let snapshots = parse_step_trace_output(&output);

        // We expect len(actions)+1 snapshots (initial + one after each action)
        assert_eq!(
            snapshots.len(),
            actions.len() + 1,
            "[{name}] expected {} snapshots, got {}",
            actions.len() + 1,
            snapshots.len()
        );

        let mut state = GameState::new(board_size, max_walls);

        for (i, snap) in snapshots.iter().enumerate() {
            assert_eq!(snap.step, i, "[{name}] snapshot step number mismatch");
            assert_snapshot_matches(name, i, snap, &state);

            // Apply action to advance to next state (except after the last snapshot)
            if i < actions.len() {
                let action = action_index_to_action(board_size, actions[i]);
                state.step(action);
            }
        }
    }
}

#[cfg(feature = "binary")]
#[test]
fn test_mcts_game_trace_matches_python() {
    let board_size = 5;
    let max_walls = 2;
    let max_steps = 50;
    let mcts_n = 50;

    let output = run_mcts_game_python(board_size, max_walls, max_steps, mcts_n);
    let snapshots = parse_step_trace_output(&output);
    assert!(!snapshots.is_empty(), "expected at least one MCTS snapshot");

    let mut state = GameState::new(board_size, max_walls);
    let config = MCTSConfig {
        n: Some(mcts_n),
        k: None,
        ucb_c: 1.4,
        noise_epsilon: 0.0,
        noise_alpha: Some(1.0),
        max_steps: Some(max_steps),
        penalize_visited_states: false,
    };
    let visited_states = std::collections::HashSet::new();
    let mut evaluator = UniformMockEvaluator;

    for (i, snap) in snapshots.iter().enumerate() {
        assert_eq!(snap.step, i, "MCTS snapshot step number mismatch");
        assert_snapshot_matches("MCTS", i, snap, &state);

        let should_select = !state.is_game_over() && state.completed_steps < max_steps as usize;

        if should_select {
            let work_state = if state.current_player == 1 {
                build_rotated_state(&state)
            } else {
                state.clone()
            };

            let work_mask = work_state.get_action_mask();
            let (children, root_value): (Vec<ChildInfo>, f32) =
                search(&config, work_state.clone(), &mut evaluator, &visited_states)
                    .expect("MCTS search should succeed");

            let visit_counts: Vec<u32> = children.iter().map(|c| c.visit_count).collect();
            let action_indices: Vec<usize> = children.iter().map(|c| c.action_index).collect();

            let selected_idx = apply_temperature_and_sample(&visit_counts, &action_indices, 0.0);

            let total_visits: u32 = visit_counts.iter().sum();
            let mut policy = vec![0.0f32; work_mask.len()];
            if total_visits > 0 {
                for child in &children {
                    policy[child.action_index] = child.visit_count as f32 / total_visits as f32;
                }
            }

            let py_value_hex = snap
                .root_value_hex
                .as_ref()
                .unwrap_or_else(|| panic!("missing V for step {i}"));
            let py_policy_hex = snap
                .root_policy_hex
                .as_ref()
                .unwrap_or_else(|| panic!("missing Q for step {i}"));
            let py_action_idx = snap
                .selected_action_index
                .unwrap_or_else(|| panic!("missing A for step {i}"));

            let py_value = hex_to_f32(py_value_hex);
            assert!(
                (root_value - py_value).abs() < 1e-6,
                "MCTS step {i}: root value mismatch (rust={}, py={})",
                root_value,
                py_value
            );
            assert_eq!(
                vec_f32_to_hex(&policy),
                *py_policy_hex,
                "MCTS step {i}: root policy mismatch"
            );
            assert_eq!(
                selected_idx, py_action_idx,
                "MCTS step {i}: selected action index mismatch"
            );

            let selected_action = action_index_to_action(board_size, selected_idx);
            let action = if state.current_player == 1 {
                let (rr, rc, rt) = rotate_action_coords(
                    board_size,
                    selected_action[0],
                    selected_action[1],
                    selected_action[2],
                );
                [rr, rc, rt]
            } else {
                selected_action
            };
            state.step(action);
        } else {
            assert!(
                snap.selected_action_index.is_none(),
                "MCTS step {i}: final snapshot should not include selected action"
            );
        }
    }
}
