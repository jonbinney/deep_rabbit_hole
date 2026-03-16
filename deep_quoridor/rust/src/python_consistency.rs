use std::io::ErrorKind;
use std::path::PathBuf;
use std::process::Command;

use crate::actions::{action_index_to_action, policy_size};
use crate::game_state::GameState;

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
