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

use crate::agents::alphazero::agent::AlphaZeroAgentConfig;
use crate::agents::alphazero::mcts::MCTSConfig;

/// Parsed `latest.yaml` — written atomically by the Python trainer
/// whenever a new model checkpoint is produced.
#[derive(Debug, Clone, Deserialize)]
pub struct LatestModelYaml {
    /// Path to the ONNX (or .pt) model file.
    pub filename: String,
    /// Monotonically-increasing model version number.
    pub version: i64,
}

/// Network type used by the AlphaZero agent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NetworkType {
    Resnet,
    Mlp,
}

/// Minimal network config — just enough to determine the type.
#[derive(Debug, Clone, Deserialize)]
pub struct NetworkConfig {
    #[serde(rename = "type", default = "default_network_type")]
    network_type: String,
}

fn default_network_type() -> String {
    "resnet".to_string()
}

/// Top-level config — mirrors the Python `UserConfig` structure.
/// Uses `deny_unknown_fields = false` (the serde default) so extra
/// sections like `training`, `benchmarks`, `wandb` are silently ignored.
#[derive(Debug, Deserialize)]
pub struct PipelineConfig {
    /// Game parameters.
    pub quoridor: QuoridorConfig,

    /// Self-play worker parameters (optional — not needed for Rust CLI).
    #[serde(default)]
    pub self_play: Option<SelfPlayWorkerConfig>,

    /// AlphaZero parameters (optional).
    #[serde(default)]
    pub alphazero: Option<AlphaZeroConfig>,
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
    /// AlphaZero overrides specific to self-play (e.g., noise settings).
    #[serde(default)]
    pub alphazero: Option<AlphaZeroConfig>,
}

/// AlphaZero MCTS configuration — matches Python's config format.
///
/// Field names use the same keys as Python for config reusability.
#[derive(Debug, Clone, Deserialize)]
pub struct AlphaZeroConfig {
    /// Network configuration (type, num_blocks, etc.).
    #[serde(default)]
    pub network: Option<NetworkConfig>,

    /// Number of MCTS simulations.
    #[serde(default)]
    pub mcts_n: Option<u32>,

    /// Multiplier for adaptive simulations (n = k * num_valid_actions).
    #[serde(default)]
    pub mcts_k: Option<u32>,

    /// UCB exploration constant (c_puct).
    #[serde(default = "default_c_puct")]
    pub mcts_c_puct: f32,

    /// Dirichlet noise weight.
    #[serde(default = "default_noise_epsilon")]
    pub mcts_noise_epsilon: f32,

    /// Dirichlet alpha parameter. If None, auto-computed.
    #[serde(default)]
    pub mcts_noise_alpha: Option<f32>,

    /// Temperature for action selection.
    #[serde(default)]
    pub temperature: Option<f32>,

    /// Step at which to drop temperature to 0.
    #[serde(default)]
    pub drop_t_on_step: Option<usize>,

    /// Whether to penalize visited states.
    #[serde(default)]
    pub penalize_visited_states: bool,

    /// Maximum game steps for MCTS terminal check.
    #[serde(default)]
    pub max_steps: Option<i32>,
}

fn default_c_puct() -> f32 {
    1.4
}

fn default_noise_epsilon() -> f32 {
    0.25
}

impl Default for AlphaZeroConfig {
    fn default() -> Self {
        Self {
            network: None,
            mcts_n: Some(100),
            mcts_k: None,
            mcts_c_puct: 1.4,
            mcts_noise_epsilon: 0.25,
            mcts_noise_alpha: None,
            temperature: None,
            drop_t_on_step: None,
            penalize_visited_states: false,
            max_steps: None,
        }
    }
}

impl AlphaZeroConfig {
    /// Convert to an AlphaZeroAgentConfig for the agent.
    pub fn to_agent_config(&self) -> AlphaZeroAgentConfig {
        AlphaZeroAgentConfig {
            mcts: MCTSConfig {
                n: self.mcts_n,
                k: self.mcts_k,
                ucb_c: self.mcts_c_puct,
                noise_epsilon: self.mcts_noise_epsilon,
                noise_alpha: self.mcts_noise_alpha,
                max_steps: self.max_steps,
                penalize_visited_states: self.penalize_visited_states,
            },
            temperature: self.temperature.unwrap_or(1.0),
            drop_t_on_step: self.drop_t_on_step,
            penalize_visited_states: self.penalize_visited_states,
        }
    }

    /// Merge with overrides from self_play section.
    pub fn merge(&self, overrides: &AlphaZeroConfig) -> AlphaZeroConfig {
        AlphaZeroConfig {
            network: overrides
                .network
                .as_ref()
                .or(self.network.as_ref())
                .cloned(),
            mcts_n: overrides.mcts_n.or(self.mcts_n),
            mcts_k: overrides.mcts_k.or(self.mcts_k),
            mcts_c_puct: overrides.mcts_c_puct,
            mcts_noise_epsilon: overrides.mcts_noise_epsilon,
            mcts_noise_alpha: overrides.mcts_noise_alpha.or(self.mcts_noise_alpha),
            temperature: overrides.temperature.or(self.temperature),
            drop_t_on_step: overrides.drop_t_on_step.or(self.drop_t_on_step),
            penalize_visited_states: overrides.penalize_visited_states
                || self.penalize_visited_states,
            max_steps: overrides.max_steps.or(self.max_steps),
        }
    }
}

impl PipelineConfig {
    /// Return the network type from the config, defaulting to Resnet.
    pub fn network_type(&self) -> NetworkType {
        self.alphazero
            .as_ref()
            .and_then(|az| az.network.as_ref())
            .map(|n| match n.network_type.as_str() {
                "mlp" => NetworkType::Mlp,
                _ => NetworkType::Resnet,
            })
            .unwrap_or(NetworkType::Resnet)
    }
}

/// Load a `PipelineConfig` from a YAML file.
pub fn load_config<P: AsRef<Path>>(path: P) -> Result<PipelineConfig> {
    let contents = fs::read_to_string(path.as_ref())
        .with_context(|| format!("Failed to read config file: {}", path.as_ref().display()))?;
    let config: PipelineConfig = serde_yaml::from_str(&contents)
        .with_context(|| format!("Failed to parse config file: {}", path.as_ref().display()))?;
    Ok(config)
}

/// Load a `LatestModelYaml` from a YAML file.
pub fn load_latest_model<P: AsRef<Path>>(path: P) -> Result<LatestModelYaml> {
    let contents = fs::read_to_string(path.as_ref()).with_context(|| {
        format!(
            "Failed to read latest model file: {}",
            path.as_ref().display()
        )
    })?;
    let model: LatestModelYaml = serde_yaml::from_str(&contents).with_context(|| {
        format!(
            "Failed to parse latest model file: {}",
            path.as_ref().display()
        )
    })?;
    Ok(model)
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

        // Check alphazero config
        let az = config.alphazero.unwrap();
        assert_eq!(az.mcts_n, Some(50));
        assert!((az.mcts_c_puct - 1.2).abs() < 1e-6);
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
        assert!(config.alphazero.is_none());
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

    #[test]
    fn test_alphazero_config_defaults() {
        let config = AlphaZeroConfig::default();

        assert_eq!(config.mcts_n, Some(100));
        assert!((config.mcts_c_puct - 1.4).abs() < 1e-6);
        assert!((config.mcts_noise_epsilon - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_alphazero_config_to_agent_config() {
        let config = AlphaZeroConfig {
            network: None,
            mcts_n: Some(200),
            mcts_k: None,
            mcts_c_puct: 2.0,
            mcts_noise_epsilon: 0.1,
            mcts_noise_alpha: Some(0.3),
            temperature: Some(0.5),
            drop_t_on_step: Some(10),
            penalize_visited_states: true,
            max_steps: Some(100),
        };

        let agent_config = config.to_agent_config();

        assert_eq!(agent_config.mcts.n, Some(200));
        assert!((agent_config.mcts.ucb_c - 2.0).abs() < 1e-6);
        assert!((agent_config.temperature - 0.5).abs() < 1e-6);
        assert_eq!(agent_config.drop_t_on_step, Some(10));
        assert!(agent_config.penalize_visited_states);
    }

    #[test]
    fn test_alphazero_config_merge() {
        let base = AlphaZeroConfig {
            network: None,
            mcts_n: Some(100),
            mcts_k: Some(10),
            mcts_c_puct: 1.4,
            mcts_noise_epsilon: 0.25,
            mcts_noise_alpha: Some(0.3),
            temperature: Some(1.0),
            drop_t_on_step: Some(30),
            penalize_visited_states: false,
            max_steps: Some(200),
        };

        let overrides = AlphaZeroConfig {
            network: None,
            mcts_n: Some(50),              // Override
            mcts_k: None,                  // Keep base
            mcts_c_puct: 2.0,              // Override
            mcts_noise_epsilon: 0.5,       // Override
            mcts_noise_alpha: None,        // Keep base
            temperature: None,             // Keep base
            drop_t_on_step: None,          // Keep base
            penalize_visited_states: true, // Override
            max_steps: None,               // Keep base
        };

        let merged = base.merge(&overrides);

        assert_eq!(merged.mcts_n, Some(50));
        assert_eq!(merged.mcts_k, Some(10));
        assert!((merged.mcts_c_puct - 2.0).abs() < 1e-6);
        assert!((merged.mcts_noise_epsilon - 0.5).abs() < 1e-6);
        assert_eq!(merged.mcts_noise_alpha, Some(0.3));
        assert_eq!(merged.temperature, Some(1.0));
        assert_eq!(merged.drop_t_on_step, Some(30));
        assert!(merged.penalize_visited_states);
        assert_eq!(merged.max_steps, Some(200));
    }
}
