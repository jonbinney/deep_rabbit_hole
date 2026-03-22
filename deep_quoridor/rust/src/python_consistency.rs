use std::io::ErrorKind;
use std::path::PathBuf;
use std::process::Command;
#[cfg(feature = "binary")]
use std::{fmt::Write as _, fs, panic::AssertUnwindSafe};

#[cfg(feature = "binary")]
use crate::actions::action_to_index;
use crate::actions::{action_index_to_action, policy_size};
#[cfg(feature = "binary")]
use crate::agents::alphazero::agent::apply_temperature_and_sample_with_mode;
#[cfg(feature = "binary")]
use crate::agents::alphazero::evaluator::{OnnxEvaluator, UniformMockEvaluator};
#[cfg(feature = "binary")]
use crate::agents::alphazero::mcts::{search, ChildInfo, MCTSConfig};
#[cfg(feature = "binary")]
use crate::game_runner::{GameResult, ReplayBufferItem};
use crate::game_state::GameState;
use crate::grid_helpers::grid_game_state_to_resnet_input;
#[cfg(feature = "binary")]
use crate::replay_writer::{write_game_npz, write_game_yaml, GameMetadata};
use crate::rotation::{rotate_goal_rows, rotate_grid_180, rotate_player_positions};
#[cfg(feature = "binary")]
use ndarray::{Array1, Array2, Array4, Ix2, OwnedRepr};
#[cfg(feature = "binary")]
use ndarray_npy::NpzReader;

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

#[cfg(feature = "binary")]
fn resolve_real_model_fixture_paths() -> (PathBuf, PathBuf) {
    let root_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("rust crate should live under deep_quoridor/")
        .to_path_buf();
    let fixtures_dir = root_dir.join("rust").join("fixtures");

    let pt_path = std::env::var("DEEP_QUORIDOR_PT_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|_| fixtures_dir.join("alphazero_B5W2_mv1.pt"));
    let onnx_path = std::env::var("DEEP_QUORIDOR_ONNX_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|_| fixtures_dir.join("alphazero_B5W2_mv1.onnx"));

    assert!(
        pt_path.exists(),
        "Missing PT model fixture: {}. Set DEEP_QUORIDOR_PT_MODEL or add deep_quoridor/rust/fixtures/alphazero_B5W2_mv1.pt",
        pt_path.display()
    );
    assert!(
        onnx_path.exists(),
        "Missing ONNX model fixture: {}. Set DEEP_QUORIDOR_ONNX_MODEL or add deep_quoridor/rust/fixtures/alphazero_B5W2_mv1.onnx",
        onnx_path.display()
    );

    (pt_path, onnx_path)
}

#[cfg(feature = "binary")]
fn run_real_model_selfplay_python(
    board_size: i32,
    max_walls: i32,
    max_steps: i32,
    mcts_n: u32,
    pt_model_path: &std::path::Path,
    output_dir: &std::path::Path,
    deterministic_tie_break: bool,
) -> String {
    let src_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("rust crate should live under deep_quoridor/")
        .join("src");

    let script_path = src_dir.join("selfplay_real_model_reference.py");

    let args = [
        src_dir.to_string_lossy().into_owned(),
        board_size.to_string(),
        max_walls.to_string(),
        max_steps.to_string(),
        mcts_n.to_string(),
        pt_model_path.to_string_lossy().into_owned(),
        output_dir.to_string_lossy().into_owned(),
        if deterministic_tie_break {
            "1".to_string()
        } else {
            "0".to_string()
        },
    ];

    run_python(&script_path.to_string_lossy(), &args)
}

#[cfg(feature = "binary")]
fn action_index_rotated_to_original(board_size: i32, idx: usize) -> usize {
    let rotated = action_index_to_action(board_size, idx);
    let (row, col, action_type) =
        crate::rotation::rotate_action_coords(board_size, rotated[0], rotated[1], rotated[2]);
    action_to_index(board_size, &[row, col, action_type])
}

#[cfg(feature = "binary")]
fn policy_original_to_rotated(board_size: i32, original_policy: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; original_policy.len()];
    for (rotated_idx, slot) in out.iter_mut().enumerate() {
        let original_idx = action_index_rotated_to_original(board_size, rotated_idx);
        *slot = original_policy[original_idx];
    }
    out
}

#[cfg(feature = "binary")]
fn latest_npz_in_ready_dir(output_dir: &std::path::Path) -> PathBuf {
    let ready = output_dir.join("ready");
    let mut files: Vec<PathBuf> = fs::read_dir(&ready)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", ready.display()))
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|x| x.to_str()) == Some("npz"))
        .collect();
    files.sort();
    files
        .last()
        .cloned()
        .unwrap_or_else(|| panic!("no npz file found in {}", ready.display()))
}

#[cfg(feature = "binary")]
fn parity_deterministic_ties_enabled() -> bool {
    std::env::var("DEEP_QUORIDOR_PARITY_DETERMINISTIC_TIES")
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

#[cfg(feature = "binary")]
fn explain_mcts_trace_python(trace_path: &std::path::Path) -> String {
    let src_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("rust crate should live under deep_quoridor/")
        .join("src");

    let script_path = src_dir.join("mcts_game_reference.py");
    let args = [
        "--explain-trace".to_string(),
        src_dir.to_string_lossy().into_owned(),
        trace_path.to_string_lossy().into_owned(),
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
        if line.starts_with("CFG,") {
            continue;
        }
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
fn mask_to_string(mask: &[bool]) -> String {
    mask.iter()
        .map(|&value| if value { '1' } else { '0' })
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

#[cfg(feature = "binary")]
fn hex_to_f32_vec(hex: &str) -> Vec<f32> {
    assert!(hex.len() % 8 == 0, "expected float32-packed hex string");
    let mut out = Vec::with_capacity(hex.len() / 8);
    for i in (0..hex.len()).step_by(8) {
        out.push(hex_to_f32(&hex[i..i + 8]));
    }
    out
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

#[cfg(feature = "binary")]
fn generate_rust_mcts_trace(
    board_size: i32,
    max_walls: i32,
    max_steps: i32,
    mcts_n: u32,
) -> String {
    let mut trace = String::new();
    writeln!(
        &mut trace,
        "CFG,{board_size},{max_walls},{max_steps},{mcts_n}"
    )
    .unwrap();

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
    let mut step = 0usize;

    loop {
        let pp = state.player_positions();
        let wr = state.walls_remaining();
        writeln!(&mut trace, "G,{step},{}", grid_to_hex(&state.grid())).unwrap();
        writeln!(
            &mut trace,
            "P,{step},{},{},{},{}",
            pp[[0, 0]],
            pp[[0, 1]],
            pp[[1, 0]],
            pp[[1, 1]]
        )
        .unwrap();
        writeln!(&mut trace, "W,{step},{},{}", wr[0], wr[1]).unwrap();
        writeln!(&mut trace, "C,{step},{}", state.current_player).unwrap();

        let mask = state.get_action_mask();
        writeln!(&mut trace, "M,{step},{}", mask_to_string(&mask)).unwrap();
        let tensor = grid_game_state_to_resnet_input(&state);
        writeln!(&mut trace, "T,{step},{}", tensor_to_hex(&tensor)).unwrap();

        if state.current_player == 1 {
            let rotated = build_rotated_state(&state);
            let rmask = rotated.get_action_mask();
            writeln!(&mut trace, "RM,{step},{}", mask_to_string(&rmask)).unwrap();
            let rtensor = grid_game_state_to_resnet_input(&rotated);
            writeln!(&mut trace, "RT,{step},{}", tensor_to_hex(&rtensor)).unwrap();
        }

        let should_select = !state.is_game_over() && state.completed_steps < max_steps as usize;
        if !should_select {
            break;
        }

        // Run MCTS on the original (unrotated) state, matching Python's behaviour where
        // the evaluator handles rotation internally and MCTS always operates in the
        // original action-index space for both players.
        let (children, root_value): (Vec<ChildInfo>, f32) =
            search(&config, state.clone(), &mut evaluator, &visited_states)
                .expect("MCTS search should succeed");

        let visit_counts: Vec<u32> = children.iter().map(|c| c.visit_count).collect();
        let action_indices: Vec<usize> = children.iter().map(|c| c.action_index).collect();
        // Keep this path deterministic to match `mcts_game_reference.py`, which
        // picks the first child among max-visit ties.
        let selected_idx =
            apply_temperature_and_sample_with_mode(&visit_counts, &action_indices, 0.0, true);

        let total_visits: u32 = visit_counts.iter().sum();
        let mut policy = vec![0.0f32; mask.len()];
        if total_visits > 0 {
            for child in &children {
                policy[child.action_index] = child.visit_count as f32 / total_visits as f32;
            }
        }

        writeln!(&mut trace, "V,{step},{}", vec_f32_to_hex(&[root_value])).unwrap();
        writeln!(&mut trace, "Q,{step},{}", vec_f32_to_hex(&policy)).unwrap();
        writeln!(&mut trace, "A,{step},{selected_idx}").unwrap();

        // Action is already in original frame (MCTS ran on original state), apply directly.
        let selected_action = action_index_to_action(board_size, selected_idx);
        state.step(selected_action);
        step += 1;
    }

    trace
}

#[cfg(feature = "binary")]
fn generate_rust_real_model_trace_and_write_npz(
    board_size: i32,
    max_walls: i32,
    max_steps: i32,
    mcts_n: u32,
    onnx_model_path: &std::path::Path,
    output_dir: &std::path::Path,
    deterministic_tie_break: bool,
) -> String {
    let mut trace = String::new();
    writeln!(
        &mut trace,
        "CFG,{board_size},{max_walls},{max_steps},{mcts_n}"
    )
    .unwrap();

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
    let mut evaluator = OnnxEvaluator::new(
        onnx_model_path
            .to_str()
            .expect("onnx model path should be valid utf-8"),
    )
    .expect("failed to construct onnx evaluator for real-model parity");
    let mut replay_items: Vec<ReplayBufferItem> = Vec::new();
    let mut winner: Option<i32> = None;
    let mut step = 0usize;

    loop {
        let pp = state.player_positions();
        let wr = state.walls_remaining();
        writeln!(&mut trace, "G,{step},{}", grid_to_hex(&state.grid())).unwrap();
        writeln!(
            &mut trace,
            "P,{step},{},{},{},{}",
            pp[[0, 0]],
            pp[[0, 1]],
            pp[[1, 0]],
            pp[[1, 1]]
        )
        .unwrap();
        writeln!(&mut trace, "W,{step},{},{}", wr[0], wr[1]).unwrap();
        writeln!(&mut trace, "C,{step},{}", state.current_player).unwrap();

        let mask_original = state.get_action_mask();
        writeln!(&mut trace, "M,{step},{}", mask_to_string(&mask_original)).unwrap();
        let tensor_original = grid_game_state_to_resnet_input(&state);
        writeln!(&mut trace, "T,{step},{}", tensor_to_hex(&tensor_original)).unwrap();

        if state.current_player == 1 {
            let rotated = build_rotated_state(&state);
            let rmask = rotated.get_action_mask();
            writeln!(&mut trace, "RM,{step},{}", mask_to_string(&rmask)).unwrap();
            let rtensor = grid_game_state_to_resnet_input(&rotated);
            writeln!(&mut trace, "RT,{step},{}", tensor_to_hex(&rtensor)).unwrap();
        }

        let should_select = !state.is_game_over() && state.completed_steps < max_steps as usize;
        if !should_select {
            break;
        }

        let (children, root_value): (Vec<ChildInfo>, f32) =
            search(&config, state.clone(), &mut evaluator, &visited_states)
                .expect("MCTS search should succeed with real model");

        let visit_counts: Vec<u32> = children.iter().map(|c| c.visit_count).collect();
        let action_indices: Vec<usize> = children.iter().map(|c| c.action_index).collect();
        let selected_idx = apply_temperature_and_sample_with_mode(
            &visit_counts,
            &action_indices,
            0.0,
            deterministic_tie_break,
        );

        let total_visits: u32 = visit_counts.iter().sum();
        let mut policy = vec![0.0f32; mask_original.len()];
        if total_visits > 0 {
            for child in &children {
                policy[child.action_index] = child.visit_count as f32 / total_visits as f32;
            }
        }

        writeln!(&mut trace, "V,{step},{}", vec_f32_to_hex(&[root_value])).unwrap();
        writeln!(&mut trace, "Q,{step},{}", vec_f32_to_hex(&policy)).unwrap();
        writeln!(&mut trace, "A,{step},{selected_idx}").unwrap();

        let (stored_input_3d, stored_policy, stored_mask) = if state.current_player == 1 {
            let rotated = build_rotated_state(&state);
            let input_3d = grid_game_state_to_resnet_input(&rotated)
                .index_axis(ndarray::Axis(0), 0)
                .to_owned();
            (
                input_3d,
                policy_original_to_rotated(board_size, &policy),
                rotated.get_action_mask(),
            )
        } else {
            let input_3d = grid_game_state_to_resnet_input(&state)
                .index_axis(ndarray::Axis(0), 0)
                .to_owned();
            (input_3d, policy.clone(), mask_original.clone())
        };

        replay_items.push(ReplayBufferItem {
            input_array: stored_input_3d,
            policy: stored_policy,
            action_mask: stored_mask,
            value: 0.0,
            player: state.current_player,
        });

        let acted_player = state.current_player;
        state.step(action_index_to_action(board_size, selected_idx));

        if state.check_win(acted_player) {
            winner = Some(acted_player);
            for item in replay_items.iter_mut() {
                item.value = if item.player == acted_player {
                    1.0
                } else {
                    -1.0
                };
            }
            break;
        }

        step += 1;
    }

    let result = GameResult {
        winner,
        num_turns: step as i32,
        replay_items,
    };

    let ready_dir = output_dir.join("ready");
    fs::create_dir_all(&ready_dir).expect("failed to create rust output ready dir");
    let npz_path = ready_dir.join("rust_real_model_game.npz");
    let yaml_path = ready_dir.join("rust_real_model_game.yaml");
    write_game_npz(&npz_path, &result).expect("failed to write rust real-model npz");
    write_game_yaml(
        &yaml_path,
        &GameMetadata {
            model_version: 0,
            game_length: result.replay_items.len(),
            creator: "rust-parity".to_string(),
        },
    )
    .expect("failed to write rust real-model yaml");

    trace
}

#[cfg(feature = "binary")]
fn assert_snapshot_fields_match(
    seq_name: &str,
    step: usize,
    left: &StepSnapshot,
    right: &StepSnapshot,
) {
    assert_eq!(
        left.step, right.step,
        "[{seq_name}] step {step}: step mismatch"
    );
    assert_eq!(
        left.grid_hex, right.grid_hex,
        "[{seq_name}] step {step}: grid mismatch"
    );
    assert_eq!(
        left.player_positions, right.player_positions,
        "[{seq_name}] step {step}: player positions mismatch"
    );
    assert_eq!(
        left.walls_remaining, right.walls_remaining,
        "[{seq_name}] step {step}: walls mismatch"
    );
    assert_eq!(
        left.current_player, right.current_player,
        "[{seq_name}] step {step}: current player mismatch"
    );
    assert_eq!(
        left.action_mask, right.action_mask,
        "[{seq_name}] step {step}: action mask mismatch"
    );
    assert_eq!(
        left.tensor_hex, right.tensor_hex,
        "[{seq_name}] step {step}: tensor mismatch"
    );
    assert_eq!(
        left.rotated_mask, right.rotated_mask,
        "[{seq_name}] step {step}: rotated action mask mismatch"
    );
    assert_eq!(
        left.rotated_tensor_hex, right.rotated_tensor_hex,
        "[{seq_name}] step {step}: rotated tensor mismatch"
    );

    match (&left.root_value_hex, &right.root_value_hex) {
        (Some(left_hex), Some(right_hex)) => {
            let left_value = hex_to_f32(left_hex);
            let right_value = hex_to_f32(right_hex);
            assert!(
                (left_value - right_value).abs() < 1e-6,
                "[{seq_name}] step {step}: root value mismatch (left={}, right={})",
                left_value,
                right_value
            );
        }
        (None, None) => {}
        _ => panic!("[{seq_name}] step {step}: root value presence mismatch"),
    }

    assert_eq!(
        left.root_policy_hex, right.root_policy_hex,
        "[{seq_name}] step {step}: root policy mismatch"
    );
    assert_eq!(
        left.selected_action_index, right.selected_action_index,
        "[{seq_name}] step {step}: selected action mismatch"
    );
}

#[cfg(feature = "binary")]
fn assert_snapshot_fields_match_real_model(
    seq_name: &str,
    step: usize,
    left: &StepSnapshot,
    right: &StepSnapshot,
) {
    assert_eq!(
        left.step, right.step,
        "[{seq_name}] step {step}: step mismatch"
    );
    assert_eq!(
        left.grid_hex, right.grid_hex,
        "[{seq_name}] step {step}: grid mismatch"
    );
    assert_eq!(
        left.player_positions, right.player_positions,
        "[{seq_name}] step {step}: player positions mismatch"
    );
    assert_eq!(
        left.walls_remaining, right.walls_remaining,
        "[{seq_name}] step {step}: walls mismatch"
    );
    assert_eq!(
        left.current_player, right.current_player,
        "[{seq_name}] step {step}: current player mismatch"
    );
    assert_eq!(
        left.action_mask, right.action_mask,
        "[{seq_name}] step {step}: action mask mismatch"
    );
    assert_eq!(
        left.tensor_hex, right.tensor_hex,
        "[{seq_name}] step {step}: tensor mismatch"
    );
    assert_eq!(
        left.rotated_mask, right.rotated_mask,
        "[{seq_name}] step {step}: rotated action mask mismatch"
    );
    assert_eq!(
        left.rotated_tensor_hex, right.rotated_tensor_hex,
        "[{seq_name}] step {step}: rotated tensor mismatch"
    );

    match (&left.root_value_hex, &right.root_value_hex) {
        (Some(left_hex), Some(right_hex)) => {
            let left_value = hex_to_f32(left_hex);
            let right_value = hex_to_f32(right_hex);
            assert!(
                (left_value - right_value).abs() < 2e-2,
                "[{seq_name}] step {step}: root value mismatch (left={}, right={})",
                left_value,
                right_value
            );
        }
        (None, None) => {}
        _ => panic!("[{seq_name}] step {step}: root value presence mismatch"),
    }

    match (&left.root_policy_hex, &right.root_policy_hex) {
        (Some(left_hex), Some(right_hex)) => {
            let left_values = hex_to_f32_vec(left_hex);
            let right_values = hex_to_f32_vec(right_hex);
            assert_eq!(
                left_values.len(),
                right_values.len(),
                "[{seq_name}] step {step}: root policy length mismatch"
            );
            for (i, (l, r)) in left_values.iter().zip(right_values.iter()).enumerate() {
                assert!(
                    (l - r).abs() < 1e-4,
                    "[{seq_name}] step {step}: root policy mismatch at idx {} (left={}, right={})",
                    i,
                    l,
                    r
                );
            }
        }
        (None, None) => {}
        _ => panic!("[{seq_name}] step {step}: root policy presence mismatch"),
    }

    assert_eq!(
        left.selected_action_index, right.selected_action_index,
        "[{seq_name}] step {step}: selected action mismatch"
    );
}

#[cfg(feature = "binary")]
fn temp_trace_path(prefix: &str) -> std::path::PathBuf {
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time should be after epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}_{pid}_{nanos}.trace"))
}

#[cfg(feature = "binary")]
fn panic_with_trace_explanations(message: String, py_trace: &str, rust_trace: &str) -> ! {
    let py_path = temp_trace_path("python_mcts_trace");
    let rust_path = temp_trace_path("rust_mcts_trace");
    fs::write(&py_path, py_trace).expect("should write python trace");
    fs::write(&rust_path, rust_trace).expect("should write rust trace");

    let py_explain = explain_mcts_trace_python(&py_path);
    let rust_explain = explain_mcts_trace_python(&rust_path);

    panic!(
        "{message}\n\nPython trace: {}\nRust trace: {}\n\nPython explanation:\n{}\nRust explanation:\n{}",
        py_path.display(),
        rust_path.display(),
        py_explain,
        rust_explain
    );
}

#[cfg(feature = "binary")]
struct NpzGameData {
    input_arrays: Array4<f32>,
    policies: Array2<f32>,
    action_masks: Array2<f32>,
    values: Array1<f32>,
    players: Array1<i32>,
}

#[cfg(feature = "binary")]
fn load_npz_game(path: &std::path::Path) -> NpzGameData {
    let file = std::fs::File::open(path)
        .unwrap_or_else(|e| panic!("failed to open npz {}: {e}", path.display()));
    let mut reader = NpzReader::new(file)
        .unwrap_or_else(|e| panic!("failed to parse npz {}: {e}", path.display()));

    let input_arrays: Array4<f32> = reader
        .by_name("input_arrays.npy")
        .unwrap_or_else(|e| panic!("missing/invalid input_arrays in {}: {e}", path.display()));
    let policies: Array2<f32> = reader
        .by_name("policies.npy")
        .unwrap_or_else(|e| panic!("missing/invalid policies in {}: {e}", path.display()));

    let action_masks: Array2<f32> = match reader.by_name::<OwnedRepr<f32>, Ix2>("action_masks.npy")
    {
        Ok(mask_f32) => mask_f32,
        Err(_) => {
            if let Ok(mask_bool) = reader.by_name::<OwnedRepr<bool>, Ix2>("action_masks.npy") {
                mask_bool.mapv(|x| if x { 1.0 } else { 0.0 })
            } else if let Ok(mask_i8) = reader.by_name::<OwnedRepr<i8>, Ix2>("action_masks.npy") {
                mask_i8.mapv(|x| if x != 0 { 1.0 } else { 0.0 })
            } else if let Ok(mask_u8) = reader.by_name::<OwnedRepr<u8>, Ix2>("action_masks.npy") {
                mask_u8.mapv(|x| if x != 0 { 1.0 } else { 0.0 })
            } else {
                panic!("missing/invalid action_masks in {}", path.display())
            }
        }
    };

    let values: Array1<f32> = reader
        .by_name("values.npy")
        .unwrap_or_else(|e| panic!("missing/invalid values in {}: {e}", path.display()));
    let players: Array1<i32> = reader
        .by_name("players.npy")
        .unwrap_or_else(|e| panic!("missing/invalid players in {}: {e}", path.display()));

    NpzGameData {
        input_arrays,
        policies,
        action_masks,
        values,
        players,
    }
}

#[cfg(feature = "binary")]
fn assert_f32_iter_close<'a>(
    seq_name: &str,
    field: &str,
    left: impl Iterator<Item = &'a f32>,
    right: impl Iterator<Item = &'a f32>,
    epsilon: f32,
) {
    for (idx, (l, r)) in left.zip(right).enumerate() {
        assert!(
            (*l - *r).abs() <= epsilon,
            "[{seq_name}] {field} mismatch at idx {idx}: left={l}, right={r}, eps={epsilon}"
        );
    }
}

#[cfg(feature = "binary")]
fn assert_npz_games_match(
    seq_name: &str,
    left_path: &std::path::Path,
    right_path: &std::path::Path,
) {
    let left = load_npz_game(left_path);
    let right = load_npz_game(right_path);

    assert_eq!(
        left.input_arrays.shape(),
        right.input_arrays.shape(),
        "[{seq_name}] input_arrays shape mismatch"
    );
    assert_eq!(
        left.policies.shape(),
        right.policies.shape(),
        "[{seq_name}] policies shape mismatch"
    );
    assert_eq!(
        left.action_masks.shape(),
        right.action_masks.shape(),
        "[{seq_name}] action_masks shape mismatch"
    );
    assert_eq!(
        left.values.shape(),
        right.values.shape(),
        "[{seq_name}] values shape mismatch"
    );
    assert_eq!(
        left.players.shape(),
        right.players.shape(),
        "[{seq_name}] players shape mismatch"
    );

    assert_f32_iter_close(
        seq_name,
        "input_arrays",
        left.input_arrays.iter(),
        right.input_arrays.iter(),
        1e-5,
    );
    assert_f32_iter_close(
        seq_name,
        "policies",
        left.policies.iter(),
        right.policies.iter(),
        1e-5,
    );
    assert_f32_iter_close(
        seq_name,
        "action_masks",
        left.action_masks.iter(),
        right.action_masks.iter(),
        1e-6,
    );
    assert_f32_iter_close(
        seq_name,
        "values",
        left.values.iter(),
        right.values.iter(),
        1e-6,
    );

    for (idx, (l, r)) in left.players.iter().zip(right.players.iter()).enumerate() {
        assert_eq!(l, r, "[{seq_name}] players mismatch at idx {idx}");
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

    let py_trace = run_mcts_game_python(board_size, max_walls, max_steps, mcts_n);
    let rust_trace = generate_rust_mcts_trace(board_size, max_walls, max_steps, mcts_n);

    let comparison = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let py_snapshots = parse_step_trace_output(&py_trace);
        let rust_snapshots = parse_step_trace_output(&rust_trace);
        assert!(
            !py_snapshots.is_empty(),
            "expected at least one python MCTS snapshot"
        );
        assert!(
            !rust_snapshots.is_empty(),
            "expected at least one rust MCTS snapshot"
        );

        for (i, (py_snap, rust_snap)) in py_snapshots.iter().zip(rust_snapshots.iter()).enumerate()
        {
            assert_snapshot_fields_match("MCTS", i, py_snap, rust_snap);
        }

        assert_eq!(
            py_snapshots.len(),
            rust_snapshots.len(),
            "MCTS trace length mismatch"
        );
    }));

    if let Err(payload) = comparison {
        let message = if let Some(text) = payload.downcast_ref::<String>() {
            text.clone()
        } else if let Some(text) = payload.downcast_ref::<&str>() {
            text.to_string()
        } else {
            "MCTS trace comparison failed".to_string()
        };
        panic_with_trace_explanations(message, &py_trace, &rust_trace);
    }
}

#[cfg(feature = "binary")]
#[test]
fn test_real_model_selfplay_trace_and_npz_matches_python() {
    let board_size = 5;
    let max_walls = 2;
    let max_steps = 50;
    let mcts_n = 20;
    let deterministic_tie_break = parity_deterministic_ties_enabled();

    let (pt_model_path, onnx_model_path) = resolve_real_model_fixture_paths();

    let py_output_dir = temp_trace_path("python_real_model_output");
    let rust_output_dir = temp_trace_path("rust_real_model_output");
    fs::create_dir_all(&py_output_dir).expect("failed to create python output dir");
    fs::create_dir_all(&rust_output_dir).expect("failed to create rust output dir");

    let py_trace = run_real_model_selfplay_python(
        board_size,
        max_walls,
        max_steps,
        mcts_n,
        &pt_model_path,
        &py_output_dir,
        deterministic_tie_break,
    );
    let rust_trace = generate_rust_real_model_trace_and_write_npz(
        board_size,
        max_walls,
        max_steps,
        mcts_n,
        &onnx_model_path,
        &rust_output_dir,
        deterministic_tie_break,
    );

    let comparison = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let py_snapshots = parse_step_trace_output(&py_trace);
        let rust_snapshots = parse_step_trace_output(&rust_trace);

        assert!(
            !py_snapshots.is_empty(),
            "expected at least one python real-model snapshot"
        );
        assert!(
            !rust_snapshots.is_empty(),
            "expected at least one rust real-model snapshot"
        );

        for (i, (py_snap, rust_snap)) in py_snapshots.iter().zip(rust_snapshots.iter()).enumerate()
        {
            assert_snapshot_fields_match_real_model("REAL_MODEL", i, py_snap, rust_snap);
        }

        assert_eq!(
            py_snapshots.len(),
            rust_snapshots.len(),
            "REAL_MODEL trace length mismatch"
        );

        let py_npz = latest_npz_in_ready_dir(&py_output_dir);
        let rust_npz = latest_npz_in_ready_dir(&rust_output_dir);
        assert_npz_games_match("REAL_MODEL_NPZ", &py_npz, &rust_npz);
    }));

    if let Err(payload) = comparison {
        let message = if let Some(text) = payload.downcast_ref::<String>() {
            text.clone()
        } else if let Some(text) = payload.downcast_ref::<&str>() {
            text.to_string()
        } else {
            "Real-model selfplay parity comparison failed".to_string()
        };
        panic_with_trace_explanations(message, &py_trace, &rust_trace);
    }
}
