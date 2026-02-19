//! Writing game replay data to disk.
//!
//! Each game produces two files:
//! - A `.npz` file with numpy-compatible arrays (input_arrays, policies, action_masks, values, players).
//! - A `.yaml` companion file with game metadata.
//!
//! The `.npz` format is a zip archive of `.npy` files, readable by `numpy.load()`.
//!
//! This module is only available behind the `binary` feature flag.

use anyhow::{Context, Result};
use ndarray::{Array1, Array2, Array4};
use ndarray_npy::NpzWriter;
use serde::Serialize;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::Path;

use crate::game_runner::GameResult;

/// Metadata written as a companion YAML file for each game.
#[derive(Debug, Serialize)]
pub struct GameMetadata {
    /// Model version used for self-play.
    pub model_version: i64,
    /// Number of turns (replay items) in the game.
    pub game_length: usize,
    /// Identifier of the process that generated this game.
    pub creator: String,
}

/// Write the replay buffer for one game as a `.npz` file.
///
/// Arrays saved:
/// - `input_arrays`: shape `(N, 5, M, M)` float32
/// - `policies`: shape `(N, policy_size)` float32
/// - `action_masks`: shape `(N, policy_size)` float32 (0/1)
/// - `values`: shape `(N,)` float32
/// - `players`: shape `(N,)` int32
pub fn write_game_npz<P: AsRef<Path>>(path: P, result: &GameResult) -> Result<()> {
    let items = &result.replay_items;
    if items.is_empty() {
        anyhow::bail!("Cannot write an empty game");
    }

    let n = items.len();
    let (channels, h, w) = {
        let s = items[0].input_array.shape();
        (s[0], s[1], s[2])
    };
    let policy_len = items[0].policy.len();

    // Stack input_arrays → (N, C, H, W)
    let mut input_data = Vec::with_capacity(n * channels * h * w);
    for item in items {
        input_data.extend(item.input_array.iter());
    }
    let input_arrays =
        Array4::<f32>::from_shape_vec((n, channels, h, w), input_data).context("input_arrays")?;

    // Stack policies → (N, policy_size)
    let mut policy_data = Vec::with_capacity(n * policy_len);
    for item in items {
        policy_data.extend_from_slice(&item.policy);
    }
    let policies =
        Array2::<f32>::from_shape_vec((n, policy_len), policy_data).context("policies")?;

    // Stack action_masks → (N, policy_size) as f32
    let mut mask_data = Vec::with_capacity(n * policy_len);
    for item in items {
        mask_data.extend(item.action_mask.iter().map(|&b| if b { 1.0f32 } else { 0.0f32 }));
    }
    let action_masks =
        Array2::<f32>::from_shape_vec((n, policy_len), mask_data).context("action_masks")?;

    // Values → (N,)
    let values = Array1::<f32>::from_iter(items.iter().map(|item| item.value));

    // Players → (N,)
    let players = Array1::<i32>::from_iter(items.iter().map(|item| item.player));

    // Write npz
    if let Some(parent) = path.as_ref().parent() {
        fs::create_dir_all(parent)?;
    }
    let file = File::create(path.as_ref())
        .with_context(|| format!("Failed to create {}", path.as_ref().display()))?;
    let buf = BufWriter::new(file);
    let mut npz = NpzWriter::new(buf);
    npz.add_array("input_arrays", &input_arrays)?;
    npz.add_array("policies", &policies)?;
    npz.add_array("action_masks", &action_masks)?;
    npz.add_array("values", &values)?;
    npz.add_array("players", &players)?;
    npz.finish()?;

    Ok(())
}

/// Write companion YAML metadata for a game.
pub fn write_game_yaml<P: AsRef<Path>>(path: P, metadata: &GameMetadata) -> Result<()> {
    if let Some(parent) = path.as_ref().parent() {
        fs::create_dir_all(parent)?;
    }
    let contents = serde_yaml::to_string(metadata).context("Failed to serialize metadata")?;
    fs::write(path.as_ref(), contents)
        .with_context(|| format!("Failed to write {}", path.as_ref().display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game_runner::ReplayBufferItem;
    use ndarray::Array3;
    use tempfile::TempDir;

    fn make_dummy_game(num_turns: usize) -> GameResult {
        let items: Vec<ReplayBufferItem> = (0..num_turns)
            .map(|i| ReplayBufferItem {
                input_array: Array3::<f32>::zeros((5, 13, 13)),
                policy: vec![0.01; 57], // 25 + 16 + 16
                action_mask: vec![true; 57],
                value: if i % 2 == 0 { 1.0 } else { -1.0 },
                player: (i % 2) as i32,
            })
            .collect();
        GameResult {
            winner: Some(0),
            num_turns: num_turns as i32,
            replay_items: items,
        }
    }

    #[test]
    fn test_write_game_npz_creates_file() {
        let dir = TempDir::new().unwrap();
        let npz_path = dir.path().join("game_001.npz");
        let game = make_dummy_game(10);
        write_game_npz(&npz_path, &game).unwrap();
        assert!(npz_path.exists());
        assert!(npz_path.metadata().unwrap().len() > 0);
    }

    #[test]
    fn test_write_game_yaml_creates_file() {
        let dir = TempDir::new().unwrap();
        let yaml_path = dir.path().join("game_001.yaml");
        let meta = GameMetadata {
            model_version: 42,
            game_length: 10,
            creator: "test_pid".to_string(),
        };
        write_game_yaml(&yaml_path, &meta).unwrap();
        assert!(yaml_path.exists());

        let contents = fs::read_to_string(&yaml_path).unwrap();
        assert!(contents.contains("model_version: 42"));
        assert!(contents.contains("game_length: 10"));
        assert!(contents.contains("creator: test_pid"));
    }

    #[test]
    fn test_write_empty_game_fails() {
        let dir = TempDir::new().unwrap();
        let npz_path = dir.path().join("empty.npz");
        let game = GameResult {
            winner: None,
            num_turns: 0,
            replay_items: vec![],
        };
        assert!(write_game_npz(&npz_path, &game).is_err());
    }

    #[test]
    fn test_write_creates_parent_dirs() {
        let dir = TempDir::new().unwrap();
        let nested = dir.path().join("a").join("b").join("c").join("game.npz");
        let game = make_dummy_game(2);
        write_game_npz(&nested, &game).unwrap();
        assert!(nested.exists());
    }
}
