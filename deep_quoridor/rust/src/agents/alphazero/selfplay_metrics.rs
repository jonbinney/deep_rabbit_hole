//! Per-process self-play metric accumulator.
//!
//! Folds `GameMetrics` for the current model version, then flushes a raw-aggregate
//! JSON record per `(version, pid)` so the Python side can combine processes and
//! compute final metrics. Reset happens on each model-version change and on shutdown.

use std::collections::HashSet;

use anyhow::{Context, Result};

use crate::agents::alphazero::selfplay_game::GameMetrics;

/// Running raw aggregates for one model version within one process.
pub struct SelfPlayAccumulator {
    version: i64,
    sims: u64,
    terminal_wins: u64,
    truncations: u64,
    max_depth: u32,
    sum_depth: u64,
    moves: u64,
    sum_root_entropy: f64,
    sum_top_move_frac: f64,
    sum_nodes: u64,
    sum_internal_nodes: u64,
    games_generated: u64,
    full_hashes: HashSet<u64>,
    opening_hashes: HashSet<u64>,
}

impl SelfPlayAccumulator {
    pub fn new(version: i64) -> Self {
        Self {
            version,
            sims: 0,
            terminal_wins: 0,
            truncations: 0,
            max_depth: 0,
            sum_depth: 0,
            moves: 0,
            sum_root_entropy: 0.0,
            sum_top_move_frac: 0.0,
            sum_nodes: 0,
            sum_internal_nodes: 0,
            games_generated: 0,
            full_hashes: HashSet::new(),
            opening_hashes: HashSet::new(),
        }
    }

    pub fn set_version(&mut self, v: i64) {
        self.version = v;
    }

    pub fn fold_game(&mut self, gm: &GameMetrics) {
        self.sims += gm.sims;
        self.terminal_wins += gm.terminal_wins;
        self.truncations += gm.truncations;
        self.max_depth = self.max_depth.max(gm.max_depth);
        self.sum_depth += gm.sum_depth;
        self.moves += gm.moves;
        self.sum_root_entropy += gm.sum_root_entropy;
        self.sum_top_move_frac += gm.sum_top_move_frac;
        self.sum_nodes += gm.sum_nodes;
        self.sum_internal_nodes += gm.sum_internal_nodes;
        self.games_generated += 1;
        self.full_hashes.insert(gm.full_hash);
        self.opening_hashes.insert(gm.opening_hash);
    }

    fn clear_counts(&mut self) {
        self.sims = 0;
        self.terminal_wins = 0;
        self.truncations = 0;
        self.max_depth = 0;
        self.sum_depth = 0;
        self.moves = 0;
        self.sum_root_entropy = 0.0;
        self.sum_top_move_frac = 0.0;
        self.sum_nodes = 0;
        self.sum_internal_nodes = 0;
        self.games_generated = 0;
        self.full_hashes.clear();
        self.opening_hashes.clear();
    }

    fn to_json(&self, pid: u32) -> serde_json::Value {
        serde_json::json!({
            "model_version": self.version,
            "pid": pid,
            "sims": self.sims,
            "terminal_wins": self.terminal_wins,
            "truncations": self.truncations,
            "max_depth": self.max_depth,
            "sum_depth": self.sum_depth,
            "moves": self.moves,
            "sum_root_entropy": self.sum_root_entropy,
            "sum_top_move_frac": self.sum_top_move_frac,
            "sum_nodes": self.sum_nodes,
            "sum_internal_nodes": self.sum_internal_nodes,
            "games_generated": self.games_generated,
            "unique_full": self.full_hashes.len(),
            "unique_opening": self.opening_hashes.len(),
        })
    }

    /// Write the current version's record to `<dir>/v{version}_pid{pid}.json` (atomic
    /// tmp+rename), then clear counts. No-op (just clears) when nothing was accumulated.
    pub fn flush_and_reset(&mut self, dir: &str, pid: u32) -> Result<()> {
        if self.moves == 0 && self.games_generated == 0 {
            self.clear_counts();
            return Ok(());
        }
        let path = format!("{}/v{}_pid{}.json", dir, self.version, pid);
        let tmp = format!("{}.tmp", path);
        let bytes = serde_json::to_vec(&self.to_json(pid)).context("serialize metrics record")?;
        std::fs::write(&tmp, &bytes).with_context(|| format!("write {}", tmp))?;
        std::fs::rename(&tmp, &path).with_context(|| format!("rename to {}", path))?;
        self.clear_counts();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::alphazero::selfplay_game::GameMetrics;

    fn sample_game(full: u64, opening: u64) -> GameMetrics {
        GameMetrics {
            sims: 100,
            terminal_wins: 5,
            truncations: 2,
            max_depth: 10,
            sum_depth: 300,
            moves: 20,
            sum_root_entropy: 12.0,
            sum_top_move_frac: 8.0,
            sum_nodes: 500,
            sum_internal_nodes: 250,
            full_hash: full,
            opening_hash: opening,
        }
    }

    #[test]
    fn flush_writes_expected_aggregates() {
        let dir = std::env::temp_dir().join(format!("spm_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let dir_s = dir.to_str().unwrap().to_string();

        let mut acc = SelfPlayAccumulator::new(7);
        acc.fold_game(&sample_game(1, 100));
        acc.fold_game(&sample_game(2, 100));
        acc.fold_game(&sample_game(2, 100));
        acc.flush_and_reset(&dir_s, 4242).unwrap();

        let path = format!("{}/v7_pid4242.json", dir_s);
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(v["model_version"], 7);
        assert_eq!(v["games_generated"], 3);
        assert_eq!(v["sims"], 300);
        assert_eq!(v["unique_full"], 2);
        assert_eq!(v["unique_opening"], 1);
        assert_eq!(v["max_depth"], 10);

        std::fs::remove_file(&path).unwrap();
        acc.flush_and_reset(&dir_s, 4242).unwrap();
        assert!(
            !std::path::Path::new(&path).exists(),
            "empty flush writes nothing"
        );

        std::fs::remove_dir_all(&dir).ok();
    }
}
