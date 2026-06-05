//! Server-side configuration: derived from `<play-dir>/config.yaml` plus the
//! list of selectable models found in `<play-dir>/models/*.onnx`.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;

/// What the server needs from the play directory.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub play_dir: PathBuf,
    pub board_size: i32,
    pub max_walls: i32,
    pub max_steps: i32,
    pub default_mcts_c_puct: f32,
    /// Filenames (not full paths) of `*.onnx` files in `<play-dir>/models/`.
    pub models: Vec<String>,
}

/// Subset of `config.yaml` we actually parse. Anything else is ignored.
#[derive(Debug, Deserialize)]
struct ConfigFile {
    quoridor: QuoridorSection,
    #[serde(default)]
    alphazero: AlphaZeroSection,
}

#[derive(Debug, Deserialize)]
struct QuoridorSection {
    board_size: i32,
    max_walls: i32,
    max_steps: i32,
}

#[derive(Debug, Deserialize, Default)]
struct AlphaZeroSection {
    #[serde(default)]
    mcts_c_puct: Option<f32>,
}

impl ServerConfig {
    pub fn load(play_dir: &Path) -> Result<Self> {
        let cfg_path = play_dir.join("config.yaml");
        let raw = std::fs::read_to_string(&cfg_path)
            .with_context(|| format!("reading {}", cfg_path.display()))?;
        let file: ConfigFile = serde_yaml::from_str(&raw)
            .with_context(|| format!("parsing {}", cfg_path.display()))?;

        let models_dir = play_dir.join("models");
        let mut models: Vec<String> = std::fs::read_dir(&models_dir)
            .with_context(|| format!("reading {}", models_dir.display()))?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("onnx"))
            .filter_map(|e| e.file_name().to_str().map(|s| s.to_string()))
            .collect();
        models.sort();

        Ok(Self {
            play_dir: play_dir.to_path_buf(),
            board_size: file.quoridor.board_size,
            max_walls: file.quoridor.max_walls,
            max_steps: file.quoridor.max_steps,
            default_mcts_c_puct: file.alphazero.mcts_c_puct.unwrap_or(1.4),
            models,
        })
    }

    /// Full path to a chosen model file. Returns `None` if the name isn't in
    /// the listed `models`.
    pub fn model_path(&self, model: &str) -> Option<PathBuf> {
        if self.models.iter().any(|m| m == model) {
            Some(self.play_dir.join("models").join(model))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn make_play_dir(yaml: &str, model_files: &[&str]) -> PathBuf {
        let dir = tempfile::Builder::new()
            .prefix("playsrv_test_")
            .tempdir()
            .expect("tempdir")
            .keep();
        fs::write(dir.join("config.yaml"), yaml).unwrap();
        fs::create_dir_all(dir.join("models")).unwrap();
        for f in model_files {
            fs::write(dir.join("models").join(f), b"not really onnx").unwrap();
        }
        dir
    }

    #[test]
    fn loads_minimal_config_and_lists_models_sorted() {
        let dir = make_play_dir(
            "quoridor:\n  board_size: 5\n  max_walls: 2\n  max_steps: 50\n",
            &["model_002.onnx", "model_000.onnx", "ignore.txt"],
        );
        let cfg = ServerConfig::load(&dir).unwrap();
        assert_eq!(cfg.board_size, 5);
        assert_eq!(cfg.max_walls, 2);
        assert_eq!(cfg.max_steps, 50);
        assert!((cfg.default_mcts_c_puct - 1.4).abs() < 1e-6);
        assert_eq!(cfg.models, vec!["model_000.onnx", "model_002.onnx"]);
    }

    #[test]
    fn picks_up_alphazero_c_puct_when_present() {
        let dir = make_play_dir(
            "quoridor:\n  board_size: 5\n  max_walls: 2\n  max_steps: 50\n\
             alphazero:\n  mcts_c_puct: 1.7\n",
            &["m.onnx"],
        );
        let cfg = ServerConfig::load(&dir).unwrap();
        assert!((cfg.default_mcts_c_puct - 1.7).abs() < 1e-6);
    }

    #[test]
    fn model_path_returns_some_for_listed_and_none_for_unlisted() {
        let dir = make_play_dir(
            "quoridor:\n  board_size: 5\n  max_walls: 2\n  max_steps: 50\n",
            &["a.onnx", "b.onnx"],
        );
        let cfg = ServerConfig::load(&dir).unwrap();
        assert!(cfg.model_path("a.onnx").is_some());
        assert!(cfg.model_path("c.onnx").is_none());
    }
}
