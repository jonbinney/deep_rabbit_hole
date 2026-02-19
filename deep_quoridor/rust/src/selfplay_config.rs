//! Configuration for self-play sessions.
//!
//! Parses the same YAML config format used by the Python v2 training pipeline,
//! extracting only the fields relevant to Rust self-play.
//!
//! This module is only available behind the `binary` feature flag.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::fs;
use std::path::Path;

/// Top-level config — mirrors the Python `UserConfig` structure.
/// Uses `deny_unknown_fields = false` (the serde default) so extra
/// sections like `alphazero`, `training`, `benchmarks`, `wandb` are
/// silently ignored.
#[derive(Debug, Deserialize)]
pub struct PipelineConfig {
    /// Game parameters.
    pub quoridor: QuoridorConfig,

    /// Self-play worker parameters (optional — not needed for Rust CLI).
    #[serde(default)]
    pub self_play: Option<SelfPlayWorkerConfig>,
}

/// Quoridor game parameters — matches Python's `QuoridorConfig`.
#[derive(Debug, Deserialize)]
pub struct QuoridorConfig {
    pub board_size: i32,
    pub max_walls: i32,
    pub max_steps: usize,
}

/// Self-play worker parameters from the YAML (subset of Python's `SelfPlayConfig`).
#[derive(Debug, Deserialize)]
pub struct SelfPlayWorkerConfig {
    #[serde(default)]
    pub num_workers: Option<usize>,
    #[serde(default)]
    pub parallel_games: Option<usize>,
}

/// Load a `PipelineConfig` from a YAML file.
pub fn load_config<P: AsRef<Path>>(path: P) -> Result<PipelineConfig> {
    let contents = fs::read_to_string(path.as_ref())
        .with_context(|| format!("Failed to read config file: {}", path.as_ref().display()))?;
    let config: PipelineConfig = serde_yaml::from_str(&contents)
        .with_context(|| format!("Failed to parse config file: {}", path.as_ref().display()))?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_load_python_compatible_config() {
        // This is a subset of an actual ci.yaml
        let yaml = r#"
run_id: test-run
quoridor:
  board_size: 5
  max_walls: 1
  max_steps: 50
alphazero:
  network:
    type: mlp
  mcts_n: 50
  mcts_c_puct: 1.2
self_play:
  num_workers: 2
  parallel_games: 2
training:
  finish_after: 2 minutes
  games_per_training_step: 8.0
  learning_rate: 0.001
  batch_size: 16
  weight_decay: 0.0001
  replay_buffer_size: 1000000
"#;
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(yaml.as_bytes()).unwrap();

        let config = load_config(f.path()).unwrap();
        assert_eq!(config.quoridor.board_size, 5);
        assert_eq!(config.quoridor.max_walls, 1);
        assert_eq!(config.quoridor.max_steps, 50);
        assert_eq!(config.self_play.unwrap().parallel_games, Some(2));
    }

    #[test]
    fn test_load_minimal_config() {
        let yaml = r#"
quoridor:
  board_size: 9
  max_walls: 10
  max_steps: 200
"#;
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(yaml.as_bytes()).unwrap();

        let config = load_config(f.path()).unwrap();
        assert_eq!(config.quoridor.board_size, 9);
        assert_eq!(config.quoridor.max_walls, 10);
        assert_eq!(config.quoridor.max_steps, 200);
        assert!(config.self_play.is_none());
    }

    #[test]
    fn test_load_config_missing_quoridor() {
        let yaml = r#"
self_play:
  num_workers: 2
"#;
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(yaml.as_bytes()).unwrap();

        let result = load_config(f.path());
        assert!(result.is_err());
    }
}
