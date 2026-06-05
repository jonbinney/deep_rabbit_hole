# Quoridor Play Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A small Rust binary that serves a local web app for playing Quoridor against the project's AlphaZero agent, with model selection from a directory.

**Architecture:** One new `bin/play_server.rs` plus a `play_server` module in the existing `quoridor_rs` crate. The binary uses `tiny_http` (no framework) to serve embedded HTML/CSS/JS and a small JSON API; each browser session creates a `GameSession` (owns a `QGameMechanics` + an `AlphaZeroAgent`) held in an `Arc<Mutex<HashMap>>`. Vanilla HTML+JS frontend with a `(2N-1) × (2N-1)` CSS-grid board; the server enriches legal actions with semantic shape so the client never has to know the action-index encoding.

**Tech Stack:** Rust (edition 2024), `tiny_http`, `serde`/`serde_json`, `serde_yaml`, the existing `AlphaZeroAgent`/`QGameMechanics`/`actions` modules, plain HTML/CSS/JS in the browser.

**Spec:** `docs/superpowers/specs/2026-05-29-quoridor-play-server-design.md`

---

## File structure

- `deep_quoridor/rust/Cargo.toml` — add `tiny_http` dep + `play_server` bin entry + `ureq` dev-dep. (modify)
- `deep_quoridor/rust/src/lib.rs` — register `pub mod play_server;` behind the `binary` feature. (modify)
- `deep_quoridor/rust/src/play_server/mod.rs` — re-exports. (create)
- `deep_quoridor/rust/src/play_server/state.rs` — `StateView` JSON shape + action enrichment. (create)
- `deep_quoridor/rust/src/play_server/config.rs` — `ServerConfig` loaded from the play-dir. (create)
- `deep_quoridor/rust/src/play_server/session.rs` — `GameSession` + `GameRegistry`. (create)
- `deep_quoridor/rust/src/play_server/handlers.rs` — request/response types + pure handler functions. (create)
- `deep_quoridor/rust/src/play_server/static/index.html` — frontend HTML. (create)
- `deep_quoridor/rust/src/play_server/static/app.css` — frontend CSS. (create)
- `deep_quoridor/rust/src/play_server/static/app.js` — frontend JS. (create)
- `deep_quoridor/rust/src/bin/play_server.rs` — CLI + `tiny_http` loop + route dispatch. (create)
- `deep_quoridor/rust/tests/play_server_e2e.rs` — end-to-end test using the existing B5W2 fixture. (create)

**Build/run notes for the implementer.** All work goes on the current branch (don't switch). The server is built with `--features binary` (NOT `--all-features`, which enables `gpu` and would require `ORT_DYLIB_PATH`). Run cargo commands with sandbox disabled and long timeouts (release/test builds use LTO and are slow). AGENTS.md: commit subject starts with `vibe: ` imperative ≤50 chars; do NOT run `cargo fmt` between tasks (formatting is a separate final task per the project rule).

---

## Task 1: Cargo.toml — add `tiny_http`, `ureq` dev-dep, register the `play_server` bin

**Files:**
- Modify: `deep_quoridor/rust/Cargo.toml`

- [ ] **Step 1: Add the `tiny_http` optional dep**

In `[dependencies]`, near `serde_json`:

```toml
tiny_http = { version = "0.12", optional = true }
```

- [ ] **Step 2: Add `"tiny_http"` to the `binary` feature list**

Current line:
```toml
binary = ["clap", "ort", "serde_yaml", "serde_json", "ndarray-npy", "zip", "rand_distr", "tokio", "futures"]
```
Change to:
```toml
binary = ["clap", "ort", "serde_yaml", "serde_json", "ndarray-npy", "zip", "rand_distr", "tokio", "futures", "tiny_http"]
```

- [ ] **Step 3: Register the new bin**

Below the existing `[[bin]]` entries (`create_policy_db`, `selfplay`), add:
```toml
[[bin]]
name = "play_server"
path = "src/bin/play_server.rs"
required-features = ["binary"]
```

- [ ] **Step 4: Add `ureq` as a dev-dependency for the end-to-end test**

In `[dev-dependencies]`:
```toml
ureq = { version = "2", default-features = false, features = ["json"] }
```

- [ ] **Step 5: Verify the existing CPU build still passes**

```bash
cd /home/jbinney/ws/deep_rabbit_hole/deep_quoridor/rust && cargo build --no-default-features --features binary --bin selfplay
```
Expected: builds successfully. (`play_server` will fail until later tasks create it — that's fine; we only built `selfplay` here.)

- [ ] **Step 6: Commit**

```bash
cd /home/jbinney/ws/deep_rabbit_hole
git add deep_quoridor/rust/Cargo.toml
git commit -m "vibe: add tiny_http dep + play_server bin entry"
```

---

## Task 2: `play_server` module scaffold + `StateView` + action enrichment

**Files:**
- Create: `deep_quoridor/rust/src/play_server/mod.rs`
- Create: `deep_quoridor/rust/src/play_server/state.rs`
- Modify: `deep_quoridor/rust/src/lib.rs`

- [ ] **Step 1: Add the module to `lib.rs`**

Add this `pub mod` declaration (gated by the `binary` feature) alongside the others. Find a `#[cfg(feature = "binary")]` block of `pub mod` lines in `src/lib.rs` and add:
```rust
#[cfg(feature = "binary")]
pub mod play_server;
```

- [ ] **Step 2: Create the module entry point**

Create `deep_quoridor/rust/src/play_server/mod.rs`:
```rust
//! Local web server for playing Quoridor against the AlphaZero agent.
//!
//! Architecture overview is in
//! `docs/superpowers/specs/2026-05-29-quoridor-play-server-design.md`.

pub mod config;
pub mod handlers;
pub mod session;
pub mod state;
```

(`config`, `handlers`, `session` are created in later tasks; the build will only succeed at the end of Task 5. To unblock incremental builds, comment out the not-yet-created `pub mod` lines and uncomment as each task lands. **Implementer:** at this step, only leave `pub mod state;` uncommented.)

So the file at the end of this task is:
```rust
//! Local web server for playing Quoridor against the AlphaZero agent.
//!
//! Architecture overview is in
//! `docs/superpowers/specs/2026-05-29-quoridor-play-server-design.md`.

pub mod state;
// pub mod config;    // added in Task 3
// pub mod session;   // added in Task 4
// pub mod handlers;  // added in Task 5
```

- [ ] **Step 3: Write the failing test for action enrichment + StateView**

Create `deep_quoridor/rust/src/play_server/state.rs` with the test scaffolding first:
```rust
//! Pure types and helpers for the play-server `state` JSON shape and for
//! enriching action indices with their semantic board coordinates.

use serde::Serialize;

use crate::actions::action_index_to_action;

/// Single legal action carried over the wire. The client never needs to know
/// the action-index encoding; it just looks at `kind` and the coords.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum EnrichedAction {
    Move { index: u32, to: [i32; 2] },
    Wall {
        index: u32,
        row: i32,
        col: i32,
        orientation: WallOrientation,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum WallOrientation {
    H,
    V,
}

/// Snapshot of a `GameSession` for the client to render. Built by
/// `session::GameSession::view()` from the underlying `QGameMechanics` +
/// `CompactState`.
#[derive(Debug, Clone, Serialize)]
pub struct StateView {
    pub board_size: i32,
    pub max_walls: i32,
    pub max_steps: i32,
    pub current_player: i32,
    pub p1_pos: [i32; 2],
    pub p2_pos: [i32; 2],
    pub p1_walls: i32,
    pub p2_walls: i32,
    pub walls: Vec<WallEntry>,
    pub legal_actions: Vec<EnrichedAction>,
    pub completed_steps: i32,
    pub winner: Option<i32>,
    pub human_player: i32,
    pub last_action: Option<EnrichedAction>,
    pub move_history: Vec<u32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct WallEntry {
    pub row: i32,
    pub col: i32,
    pub orientation: WallOrientation,
}

/// Map an action index to its semantic enrichment (move dest / wall coords +
/// orientation). Matches the convention in `actions::action_index_to_action`:
/// indices < N*N are moves to row/col=index/N,index%N; the next (N-1)^2 are
/// vertical walls; the remaining (N-1)^2 are horizontal walls.
pub fn enrich_action(board_size: i32, index: usize) -> EnrichedAction {
    let [row, col, action_type] = action_index_to_action(board_size, index);
    match action_type {
        0 => EnrichedAction::Move {
            index: index as u32,
            to: [row, col],
        },
        1 => EnrichedAction::Wall {
            index: index as u32,
            row,
            col,
            orientation: WallOrientation::V,
        },
        2 => EnrichedAction::Wall {
            index: index as u32,
            row,
            col,
            orientation: WallOrientation::H,
        },
        other => panic!("unexpected action type {other} for index {index}"),
    }
}

/// Apply `enrich_action` to every legal index in `mask`.
pub fn enrich_legal_actions(board_size: i32, mask: &[bool]) -> Vec<EnrichedAction> {
    mask.iter()
        .enumerate()
        .filter_map(|(i, &legal)| legal.then(|| enrich_action(board_size, i)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actions::policy_size;

    #[test]
    fn enrich_move_action_round_trips_coords() {
        // First action on a 5x5 board: move to (0, 0).
        let a = enrich_action(5, 0);
        assert_eq!(
            a,
            EnrichedAction::Move {
                index: 0,
                to: [0, 0]
            }
        );

        // Index N*N - 1 on 5x5 = last move cell = (4, 4).
        let a = enrich_action(5, 24);
        assert_eq!(
            a,
            EnrichedAction::Move {
                index: 24,
                to: [4, 4]
            }
        );
    }

    #[test]
    fn enrich_first_vertical_then_first_horizontal_wall() {
        let n: i32 = 5;
        let nn = (n * n) as usize;
        let walls = ((n - 1) * (n - 1)) as usize;

        // First vertical wall is at index N*N.
        let v = enrich_action(n, nn);
        assert_eq!(
            v,
            EnrichedAction::Wall {
                index: nn as u32,
                row: 0,
                col: 0,
                orientation: WallOrientation::V
            }
        );

        // First horizontal wall is at index N*N + (N-1)^2.
        let h = enrich_action(n, nn + walls);
        assert_eq!(
            h,
            EnrichedAction::Wall {
                index: (nn + walls) as u32,
                row: 0,
                col: 0,
                orientation: WallOrientation::H
            }
        );
    }

    #[test]
    fn enrich_legal_actions_filters_by_mask() {
        let n = 5;
        let size = policy_size(n);
        let mut mask = vec![false; size];
        mask[0] = true;
        mask[(n * n) as usize] = true; // first vertical wall

        let actions = enrich_legal_actions(n, &mask);
        assert_eq!(actions.len(), 2);
        assert!(matches!(
            actions[0],
            EnrichedAction::Move { index: 0, .. }
        ));
        assert!(matches!(
            actions[1],
            EnrichedAction::Wall {
                orientation: WallOrientation::V,
                ..
            }
        ));
    }

    #[test]
    fn enriched_action_serializes_with_kind_tag() {
        let m = EnrichedAction::Move {
            index: 3,
            to: [4, 5],
        };
        let s = serde_json::to_string(&m).unwrap();
        assert_eq!(s, r#"{"kind":"move","index":3,"to":[4,5]}"#);

        let w = EnrichedAction::Wall {
            index: 17,
            row: 3,
            col: 2,
            orientation: WallOrientation::H,
        };
        let s = serde_json::to_string(&w).unwrap();
        assert_eq!(
            s,
            r#"{"kind":"wall","index":17,"row":3,"col":2,"orientation":"h"}"#
        );
    }
}
```

- [ ] **Step 4: Run the new tests**

```bash
cd /home/jbinney/ws/deep_rabbit_hole/deep_quoridor/rust && cargo test --lib --no-default-features --features binary play_server::state -- --nocapture
```
Expected: all 4 tests pass. (If `action_index_to_action` returns a different action_type numbering than the comments above suggest, **read** `src/actions.rs` and adjust the `match action_type` arms accordingly — the goal is that vertical wall maps to `WallOrientation::V` and horizontal to `WallOrientation::H`.)

- [ ] **Step 5: Commit**

```bash
cd /home/jbinney/ws/deep_rabbit_hole
git add deep_quoridor/rust/src/lib.rs deep_quoridor/rust/src/play_server
git commit -m "vibe: add play_server state view + action enrichment"
```

---

## Task 3: `ServerConfig` — load `<play-dir>/config.yaml` + list `models/*.onnx`

**Files:**
- Create: `deep_quoridor/rust/src/play_server/config.rs`
- Modify: `deep_quoridor/rust/src/play_server/mod.rs`

- [ ] **Step 1: Uncomment the module line**

Edit `src/play_server/mod.rs` so the file contains:
```rust
//! Local web server for playing Quoridor against the AlphaZero agent.
//!
//! Architecture overview is in
//! `docs/superpowers/specs/2026-05-29-quoridor-play-server-design.md`.

pub mod config;
pub mod state;
// pub mod session;   // added in Task 4
// pub mod handlers;  // added in Task 5
```

- [ ] **Step 2: Create the config module with tests**

Create `deep_quoridor/rust/src/play_server/config.rs`:
```rust
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
            .into_path();
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
```

- [ ] **Step 3: Run the new tests**

```bash
cd /home/jbinney/ws/deep_rabbit_hole/deep_quoridor/rust && cargo test --lib --no-default-features --features binary play_server::config -- --nocapture
```
Expected: 3 tests pass.

- [ ] **Step 4: Commit**

```bash
cd /home/jbinney/ws/deep_rabbit_hole
git add deep_quoridor/rust/src/play_server
git commit -m "vibe: add play_server config loader and models scan"
```

---

## Task 4: `GameSession` + `GameRegistry`

**Files:**
- Create: `deep_quoridor/rust/src/play_server/session.rs`
- Modify: `deep_quoridor/rust/src/play_server/mod.rs`

- [ ] **Step 1: Enable the module**

Edit `src/play_server/mod.rs` so the not-yet-existing `handlers` is still commented but `session` is enabled:
```rust
pub mod config;
pub mod session;
pub mod state;
// pub mod handlers;  // added in Task 5
```

- [ ] **Step 2: Implement `GameSession` + `GameRegistry`**

Create `deep_quoridor/rust/src/play_server/session.rs`:

```rust
//! Per-game state (`GameSession`) and the shared registry that the HTTP
//! handlers look up by `game_id`.
//!
//! Each session owns its own `AlphaZeroAgent` so games run independently. The
//! registry holds an `Arc<Mutex<GameSession>>` per game; an HTTP handler takes
//! the outer `Mutex` briefly to look up the session and then holds the inner
//! `Mutex` for the duration of the move + AI response.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Context, Result};
use rand::RngCore;

use crate::actions::policy_size;
use crate::agents::ActionSelector;
use crate::agents::alphazero::agent::{AlphaZeroAgent, AlphaZeroAgentConfig};
use crate::agents::alphazero::evaluator::{Evaluator, UniformMockEvaluator};
use crate::agents::alphazero::mcts::MCTSConfig;
use crate::compact::q_bit_repr::CompactState;
use crate::compact::q_game_mechanics::QGameMechanics;
use crate::play_server::config::ServerConfig;
use crate::play_server::state::{
    enrich_action, enrich_legal_actions, EnrichedAction, StateView, WallEntry, WallOrientation,
};

pub type GameId = String;

/// One running game: owns the agent and the board state.
pub struct GameSession {
    pub mechanics: QGameMechanics,
    pub state: CompactState,
    pub agent: AlphaZeroAgent,
    pub board_size: i32,
    pub max_walls: i32,
    pub max_steps: i32,
    pub human_player: i32,
    pub last_action: Option<EnrichedAction>,
    pub move_history: Vec<u32>,
}

impl GameSession {
    /// Construct an agent config matching the server's notion of "play-mode":
    /// the requested `mcts_n`, temperature 0 (argmax visits), deterministic
    /// tie-break, no Dirichlet noise, and the server's max_steps as the MCTS
    /// search cap.
    pub fn agent_config(mcts_n: u32, c_puct: f32, max_steps: i32) -> AlphaZeroAgentConfig {
        AlphaZeroAgentConfig {
            mcts: MCTSConfig {
                n: Some(mcts_n),
                k: None,
                ucb_c: c_puct,
                noise_epsilon: 0.0,
                noise_alpha: None,
                max_steps: Some(max_steps),
                penalize_visited_states: false,
            },
            temperature: 0.0,
            drop_t_on_step: None,
            penalize_visited_states: false,
            deterministic_tie_break: true,
        }
    }

    /// Build a session that loads ONNX from disk. The session's mechanics +
    /// initial state come from `cfg`.
    pub fn new_from_onnx(
        cfg: &ServerConfig,
        model_path: &std::path::Path,
        mcts_n: u32,
        human_player: i32,
    ) -> Result<Self> {
        let mechanics = QGameMechanics::new(
            cfg.board_size as usize,
            cfg.max_walls as usize,
            cfg.max_steps as usize,
        );
        let state = mechanics.create_initial_state();
        let model_str = model_path
            .to_str()
            .ok_or_else(|| anyhow!("model path is not valid UTF-8"))?;
        let agent_config = Self::agent_config(mcts_n, cfg.default_mcts_c_puct, cfg.max_steps);
        let agent = AlphaZeroAgent::new(model_str, agent_config)
            .context("constructing AlphaZeroAgent")?;
        Ok(Self {
            mechanics,
            state,
            agent,
            board_size: cfg.board_size,
            max_walls: cfg.max_walls,
            max_steps: cfg.max_steps,
            human_player,
            last_action: None,
            move_history: Vec::new(),
        })
    }

    /// Test-only variant that injects a fake evaluator instead of loading ORT.
    #[cfg(test)]
    pub fn new_with_evaluator(
        board_size: i32,
        max_walls: i32,
        max_steps: i32,
        evaluator: Box<dyn Evaluator + Send>,
        mcts_n: u32,
        c_puct: f32,
        human_player: i32,
    ) -> Self {
        let mechanics =
            QGameMechanics::new(board_size as usize, max_walls as usize, max_steps as usize);
        let state = mechanics.create_initial_state();
        let agent_config = Self::agent_config(mcts_n, c_puct, max_steps);
        let agent = AlphaZeroAgent::with_evaluator(evaluator, agent_config);
        Self {
            mechanics,
            state,
            agent,
            board_size,
            max_walls,
            max_steps,
            human_player,
            last_action: None,
            move_history: Vec::new(),
        }
    }

    fn current_player(&self) -> i32 {
        self.mechanics.repr().get_current_player(self.state) as i32
    }

    fn is_game_over(&self) -> bool {
        self.mechanics.is_game_over(self.state)
            || self.mechanics.repr().get_completed_steps(self.state) >= self.max_steps as usize
    }

    fn legal_mask(&mut self) -> Vec<bool> {
        self.mechanics.get_action_mask_immut(self.state)
    }

    /// Apply one action (no matter whose turn). Records it in `last_action` +
    /// `move_history`. Returns an error if the action is illegal.
    pub fn apply_action(&mut self, action_index: u32) -> Result<()> {
        if self.is_game_over() {
            return Err(anyhow!("game is already over"));
        }
        let mask = self.legal_mask();
        let idx = action_index as usize;
        if idx >= mask.len() || !mask[idx] {
            return Err(anyhow!("action {action_index} is not legal"));
        }
        self.last_action = Some(enrich_action(self.board_size, idx));
        self.move_history.push(action_index);
        self.mechanics.apply_action_index(&mut self.state, idx);
        Ok(())
    }

    /// Run the AI for one move on the current state. Errors if it's actually
    /// the human's turn or the game is over.
    pub fn ai_step(&mut self) -> Result<()> {
        if self.is_game_over() {
            return Ok(());
        }
        if self.current_player() == self.human_player {
            return Err(anyhow!("not AI's turn"));
        }
        let mask = self.legal_mask();
        let (action_idx, _policy) = self
            .agent
            .select_action(self.state, &self.mechanics, &mask)
            .context("AI MCTS selection")?;
        self.last_action = Some(enrich_action(self.board_size, action_idx));
        self.move_history.push(action_idx as u32);
        self.mechanics
            .apply_action_index(&mut self.state, action_idx);
        Ok(())
    }

    /// Build the JSON-facing snapshot the client renders from.
    pub fn view(&mut self) -> StateView {
        let mask = self.legal_mask();
        let legal_actions = enrich_legal_actions(self.board_size, &mask);
        let repr = self.mechanics.repr();
        let (p1r, p1c) = repr.get_player_position(self.state, 0);
        let (p2r, p2c) = repr.get_player_position(self.state, 1);
        let p1w = repr.get_walls_remaining(self.state, 0) as i32;
        let p2w = repr.get_walls_remaining(self.state, 1) as i32;
        let completed_steps = repr.get_completed_steps(self.state) as i32;
        let winner = if self.mechanics.check_win(self.state, 0) {
            Some(0)
        } else if self.mechanics.check_win(self.state, 1) {
            Some(1)
        } else {
            None
        };

        StateView {
            board_size: self.board_size,
            max_walls: self.max_walls,
            max_steps: self.max_steps,
            current_player: self.current_player(),
            p1_pos: [p1r as i32, p1c as i32],
            p2_pos: [p2r as i32, p2c as i32],
            p1_walls: p1w,
            p2_walls: p2w,
            walls: list_walls(&self.mechanics, self.state, self.board_size),
            legal_actions,
            completed_steps,
            winner,
            human_player: self.human_player,
            last_action: self.last_action.clone(),
            move_history: self.move_history.clone(),
        }
    }
}

/// Iterate every potential wall slot (`(N-1)^2` for each orientation) and ask
/// the mechanics whether a wall is currently present at that location.
fn list_walls(mechanics: &QGameMechanics, state: CompactState, board_size: i32) -> Vec<WallEntry> {
    let mut out = Vec::new();
    let wall_size = (board_size - 1) as usize;
    for orientation_idx in 0..2 {
        let orientation = if orientation_idx == 0 {
            WallOrientation::H
        } else {
            WallOrientation::V
        };
        for row in 0..wall_size {
            for col in 0..wall_size {
                if mechanics
                    .repr()
                    .get_wall(state, row, col, orientation_idx == 0)
                {
                    out.push(WallEntry {
                        row: row as i32,
                        col: col as i32,
                        orientation,
                    });
                }
            }
        }
    }
    out
}

/// Thread-safe map `game_id -> GameSession`.
#[derive(Clone, Default)]
pub struct GameRegistry {
    inner: Arc<Mutex<HashMap<GameId, Arc<Mutex<GameSession>>>>>,
}

impl GameRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&self, session: GameSession) -> GameId {
        let id = new_game_id();
        self.inner
            .lock()
            .unwrap()
            .insert(id.clone(), Arc::new(Mutex::new(session)));
        id
    }

    pub fn get(&self, game_id: &str) -> Option<Arc<Mutex<GameSession>>> {
        self.inner.lock().unwrap().get(game_id).cloned()
    }
}

fn new_game_id() -> GameId {
    let mut bytes = [0u8; 4];
    rand::thread_rng().fill_bytes(&mut bytes);
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_session(human_player: i32) -> GameSession {
        GameSession::new_with_evaluator(
            5,
            2,
            50,
            Box::new(UniformMockEvaluator),
            8,
            1.4,
            human_player,
        )
    }

    #[test]
    fn initial_view_has_pawns_on_home_rows_and_no_walls() {
        let mut s = mock_session(0);
        let v = s.view();
        assert_eq!(v.board_size, 5);
        assert_eq!(v.max_walls, 2);
        assert_eq!(v.current_player, 0);
        assert_eq!(v.completed_steps, 0);
        assert!(v.walls.is_empty());
        assert_eq!(v.p1_walls, 2);
        assert_eq!(v.p2_walls, 2);
        assert!(v.winner.is_none());
        assert_eq!(v.human_player, 0);
        assert!(v.last_action.is_none());
        assert!(v.move_history.is_empty());
        // Some legal move actions must exist at the start.
        let has_move = v
            .legal_actions
            .iter()
            .any(|a| matches!(a, EnrichedAction::Move { .. }));
        assert!(has_move);
    }

    #[test]
    fn apply_action_records_last_action_and_advances_player() {
        let mut s = mock_session(0);
        let mask = s.legal_mask();
        let first_legal_move = mask
            .iter()
            .enumerate()
            .find(|(_, &b)| b)
            .map(|(i, _)| i as u32)
            .expect("at least one legal action");
        s.apply_action(first_legal_move).unwrap();
        assert_eq!(s.move_history, vec![first_legal_move]);
        assert!(s.last_action.is_some());
        assert_eq!(s.current_player(), 1);
    }

    #[test]
    fn apply_action_rejects_illegal_index() {
        let mut s = mock_session(0);
        let mask = s.legal_mask();
        let illegal = mask
            .iter()
            .enumerate()
            .find(|(_, &b)| !b)
            .map(|(i, _)| i as u32)
            .expect("at least one illegal action");
        let err = s.apply_action(illegal).unwrap_err();
        assert!(err.to_string().contains("not legal"));
    }

    #[test]
    fn ai_step_errors_when_its_human_turn() {
        let mut s = mock_session(0); // human is P1, AI is P2, P1 goes first
        let err = s.ai_step().unwrap_err();
        assert!(err.to_string().contains("not AI's turn"));
    }

    #[test]
    fn ai_step_runs_when_its_ai_turn() {
        let mut s = mock_session(1); // human is P2, AI is P1, AI goes first
        s.ai_step().unwrap();
        assert_eq!(s.move_history.len(), 1);
        assert!(s.last_action.is_some());
        assert_eq!(s.current_player(), 1); // now human (P2) to move
    }

    #[test]
    fn registry_insert_and_get_round_trip() {
        let reg = GameRegistry::new();
        let id = reg.insert(mock_session(0));
        assert!(reg.get(&id).is_some());
        assert!(reg.get("does-not-exist").is_none());
    }

    #[test]
    fn game_id_is_8_hex_chars() {
        let id = new_game_id();
        assert_eq!(id.len(), 8);
        assert!(id.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
```

- [ ] **Step 3: Run the session tests**

```bash
cd /home/jbinney/ws/deep_rabbit_hole/deep_quoridor/rust && cargo test --lib --no-default-features --features binary play_server::session -- --nocapture
```
Expected: all session tests pass.

If `QGameMechanics::repr().get_wall(state, row, col, is_horizontal)` doesn't have that exact signature, **read** `src/compact/q_bit_repr.rs` for the actual `get_wall` signature and adjust `list_walls` accordingly. The semantic is "iterate every potential wall slot and ask if a wall is there"; the exact API call is local to that helper.

- [ ] **Step 4: Commit**

```bash
cd /home/jbinney/ws/deep_rabbit_hole
git add deep_quoridor/rust/src/play_server
git commit -m "vibe: add GameSession + GameRegistry"
```

---

## Task 5: HTTP handlers (pure functions)

**Files:**
- Create: `deep_quoridor/rust/src/play_server/handlers.rs`
- Modify: `deep_quoridor/rust/src/play_server/mod.rs`

- [ ] **Step 1: Enable the module**

Edit `src/play_server/mod.rs` so all four `pub mod` lines are active:
```rust
pub mod config;
pub mod handlers;
pub mod session;
pub mod state;
```

- [ ] **Step 2: Implement the handlers**

Create `deep_quoridor/rust/src/play_server/handlers.rs`:
```rust
//! Pure handler functions that the HTTP layer calls. These are unit-testable
//! without an actual TCP socket — they take parsed request bodies + the
//! shared state and return response structs.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

use crate::play_server::config::ServerConfig;
use crate::play_server::session::{GameRegistry, GameSession};
use crate::play_server::state::StateView;

/// Reasonable bounds for the per-game MCTS slider. The server clamps inbound
/// values to this range.
pub const MCTS_N_MIN: u32 = 1;
pub const MCTS_N_MAX: u32 = 4000;

#[derive(Debug, Serialize)]
pub struct ConfigResponse {
    pub board_size: i32,
    pub max_walls: i32,
    pub max_steps: i32,
    pub models: Vec<String>,
    pub default_mcts_n: u32,
}

#[derive(Debug, Deserialize)]
pub struct NewGameRequest {
    pub model: String,
    pub mcts_n: u32,
    pub human_player: i32,
}

#[derive(Debug, Serialize)]
pub struct NewGameResponse {
    pub game_id: String,
    pub state: StateView,
}

#[derive(Debug, Serialize)]
pub struct StateResponse {
    pub state: StateView,
}

#[derive(Debug, Deserialize)]
pub struct MoveRequest {
    pub action_index: u32,
}

/// Tag for `handle_*` failures so the HTTP layer can pick the right status.
#[derive(Debug)]
pub enum ApiError {
    BadRequest(String),
    NotFound(String),
    Internal(String),
}

pub fn handle_config(cfg: &ServerConfig, default_mcts_n: u32) -> ConfigResponse {
    ConfigResponse {
        board_size: cfg.board_size,
        max_walls: cfg.max_walls,
        max_steps: cfg.max_steps,
        models: cfg.models.clone(),
        default_mcts_n,
    }
}

pub fn handle_new_game(
    cfg: &ServerConfig,
    registry: &GameRegistry,
    req: NewGameRequest,
) -> Result<NewGameResponse, ApiError> {
    if req.human_player != 0 && req.human_player != 1 {
        return Err(ApiError::BadRequest(format!(
            "human_player must be 0 or 1, got {}",
            req.human_player
        )));
    }
    let mcts_n = req.mcts_n.clamp(MCTS_N_MIN, MCTS_N_MAX);
    let model_path = cfg
        .model_path(&req.model)
        .ok_or_else(|| ApiError::BadRequest(format!("unknown model: {}", req.model)))?;

    let mut session = GameSession::new_from_onnx(cfg, &model_path, mcts_n, req.human_player)
        .map_err(|e| ApiError::Internal(format!("constructing session: {e:#}")))?;

    // If the AI plays first, take its move before returning the initial state.
    if session.human_player != 0
        && !session.view().winner.is_some()
        && session.view().current_player == 1 - session.human_player
    {
        session
            .ai_step()
            .map_err(|e| ApiError::Internal(format!("initial AI move: {e:#}")))?;
    }
    let game_id = registry.insert(session);
    let view = with_session(registry, &game_id, |s| Ok::<_, anyhow::Error>(s.view()))
        .map_err(api_internal)?;
    Ok(NewGameResponse {
        game_id,
        state: view,
    })
}

pub fn handle_get_state(
    registry: &GameRegistry,
    game_id: &str,
) -> Result<StateResponse, ApiError> {
    let view = with_session(registry, game_id, |s| Ok::<_, anyhow::Error>(s.view()))
        .map_err(api_internal)?;
    Ok(StateResponse { state: view })
}

pub fn handle_move(
    registry: &GameRegistry,
    game_id: &str,
    req: MoveRequest,
) -> Result<StateResponse, ApiError> {
    let view = with_session(registry, game_id, |s| {
        if s.view().winner.is_some() {
            return Err(anyhow!("game is over"));
        }
        if s.view().current_player != s.human_player {
            return Err(anyhow!("not human's turn"));
        }
        s.apply_action(req.action_index)?;
        // If after the human's move it's the AI's turn (and the game isn't
        // over), run the AI in the same round-trip.
        if s.view().winner.is_none() && s.view().current_player != s.human_player {
            s.ai_step()?;
        }
        Ok(s.view())
    })
    .map_err(|e| {
        let m = format!("{e:#}");
        if m.contains("not legal") || m.contains("not human") || m.contains("game is over") {
            ApiError::BadRequest(m)
        } else {
            ApiError::Internal(m)
        }
    })?;
    Ok(StateResponse { state: view })
}

fn with_session<F, T, E>(
    registry: &GameRegistry,
    game_id: &str,
    f: F,
) -> Result<T, anyhow::Error>
where
    F: FnOnce(&mut GameSession) -> Result<T, E>,
    E: Into<anyhow::Error>,
{
    let lock = registry
        .get(game_id)
        .ok_or_else(|| anyhow!("unknown game_id"))?;
    let mut session = lock.lock().unwrap();
    f(&mut session).map_err(Into::into)
}

fn api_internal(e: anyhow::Error) -> ApiError {
    let m = format!("{e:#}");
    if m.contains("unknown game_id") {
        ApiError::NotFound(m)
    } else {
        ApiError::Internal(m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::alphazero::evaluator::UniformMockEvaluator;

    fn mock_session(human_player: i32) -> GameSession {
        GameSession::new_with_evaluator(
            5,
            2,
            50,
            Box::new(UniformMockEvaluator),
            8,
            1.4,
            human_player,
        )
    }

    fn register(reg: &GameRegistry, s: GameSession) -> String {
        reg.insert(s)
    }

    #[test]
    fn handle_get_state_returns_view_when_game_exists() {
        let reg = GameRegistry::new();
        let id = register(&reg, mock_session(0));
        let r = handle_get_state(&reg, &id).unwrap();
        assert_eq!(r.state.board_size, 5);
    }

    #[test]
    fn handle_get_state_returns_not_found_for_bad_id() {
        let reg = GameRegistry::new();
        match handle_get_state(&reg, "ffffffff") {
            Err(ApiError::NotFound(_)) => {}
            other => panic!("expected NotFound, got {other:?}"),
        }
    }

    #[test]
    fn handle_move_rejects_illegal_action() {
        let reg = GameRegistry::new();
        let id = register(&reg, mock_session(0));
        // Pick an index that's known to be illegal (e.g. moving to own square
        // = the agent's starting cell, index = N*N - first/last cell).
        // Easier: pick a wall index that can't exist (index 0 is a move and is
        // not legal at the very start because move 0 is to (0,0)).
        match handle_move(
            &reg,
            &id,
            MoveRequest {
                action_index: u32::MAX - 1,
            },
        ) {
            Err(ApiError::BadRequest(msg)) => assert!(msg.contains("not legal") || msg.contains("range")),
            other => panic!("expected BadRequest, got {other:?}"),
        }
    }
}
```

- [ ] **Step 3: Build + run tests**

```bash
cd /home/jbinney/ws/deep_rabbit_hole/deep_quoridor/rust && cargo test --lib --no-default-features --features binary play_server -- --nocapture
```
Expected: all `play_server::*` tests pass.

- [ ] **Step 4: Commit**

```bash
cd /home/jbinney/ws/deep_rabbit_hole
git add deep_quoridor/rust/src/play_server
git commit -m "vibe: add play_server API handlers"
```

---

## Task 6: `bin/play_server.rs` — CLI, tiny_http loop, static assets

**Files:**
- Create: `deep_quoridor/rust/src/bin/play_server.rs`
- Create: `deep_quoridor/rust/src/play_server/static/index.html` (placeholder; real content in Task 8)
- Create: `deep_quoridor/rust/src/play_server/static/app.css` (placeholder)
- Create: `deep_quoridor/rust/src/play_server/static/app.js` (placeholder)

- [ ] **Step 1: Create the static asset placeholders**

The bin file uses `include_str!` to embed these at compile time, so they must exist as files even before Task 8 fills them. Minimal stand-ins:

`deep_quoridor/rust/src/play_server/static/index.html`:
```html
<!doctype html><meta charset=utf-8><title>Quoridor</title>
<p>Play UI placeholder — see Task 8.</p>
```

`deep_quoridor/rust/src/play_server/static/app.css`:
```css
/* Real styles in Task 8. */
```

`deep_quoridor/rust/src/play_server/static/app.js`:
```js
// Real frontend logic in Task 8.
```

- [ ] **Step 2: Write the binary**

Create `deep_quoridor/rust/src/bin/play_server.rs`:
```rust
//! Local web server for playing Quoridor against the project's AlphaZero
//! agent. See `docs/superpowers/specs/2026-05-29-quoridor-play-server-design.md`.

use std::io::Read;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;
use tiny_http::{Header, Method, Response, Server};

use quoridor_rs::play_server::config::ServerConfig;
use quoridor_rs::play_server::handlers::{
    self, ApiError, MoveRequest, NewGameRequest,
};
use quoridor_rs::play_server::session::GameRegistry;

const INDEX_HTML: &str = include_str!("../play_server/static/index.html");
const APP_CSS: &str = include_str!("../play_server/static/app.css");
const APP_JS: &str = include_str!("../play_server/static/app.js");

#[derive(Parser)]
#[command(about = "Quoridor play server")]
struct Cli {
    /// Directory containing `config.yaml` and `models/*.onnx`.
    #[arg(long)]
    play_dir: PathBuf,

    /// TCP port to listen on.
    #[arg(long, default_value_t = 8080)]
    port: u16,

    /// Bind address. Default is loopback; use `0.0.0.0` for LAN access.
    #[arg(long, default_value = "127.0.0.1")]
    bind: String,

    /// Default `mcts_n` the UI starts the slider at.
    #[arg(long, default_value_t = 400)]
    default_mcts_n: u32,
}

struct App {
    cfg: ServerConfig,
    default_mcts_n: u32,
    registry: GameRegistry,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let cfg =
        ServerConfig::load(&cli.play_dir).with_context(|| format!("loading {:?}", cli.play_dir))?;
    let app = Arc::new(App {
        cfg,
        default_mcts_n: cli.default_mcts_n,
        registry: GameRegistry::new(),
    });

    let addr: SocketAddr = format!("{}:{}", cli.bind, cli.port)
        .parse()
        .context("parse bind addr")?;
    let server = Server::http(addr).map_err(|e| anyhow::anyhow!("tiny_http listen: {e}"))?;
    println!("play_server listening on http://{addr}");
    println!("models: {:?}", app.cfg.models);

    for mut request in server.incoming_requests() {
        let app = Arc::clone(&app);
        // tiny_http is sync; for simplicity handle in-thread (sessions take
        // ~ms; AI moves dominate but each session has its own lock, so
        // concurrent games still progress in parallel via multiple threads
        // if we spawn one here).
        std::thread::spawn(move || {
            let response = route(&app, &mut request);
            if let Err(e) = request.respond(response) {
                eprintln!("response error: {e:#}");
            }
        });
    }
    Ok(())
}

fn route(app: &App, request: &mut tiny_http::Request) -> Response<std::io::Cursor<Vec<u8>>> {
    let url = request.url().to_string();
    let method = request.method().clone();

    // Static files.
    if method == Method::Get {
        match url.as_str() {
            "/" | "/index.html" => return html(INDEX_HTML, 200),
            "/static/app.css" => return text(APP_CSS, 200, "text/css; charset=utf-8"),
            "/static/app.js" => {
                return text(APP_JS, 200, "application/javascript; charset=utf-8")
            }
            "/api/config" => {
                let resp = handlers::handle_config(&app.cfg, app.default_mcts_n);
                return json_ok(&resp);
            }
            u if u.starts_with("/api/games/") && !u[11..].contains('/') => {
                let game_id = &u[11..];
                return match handlers::handle_get_state(&app.registry, game_id) {
                    Ok(r) => json_ok(&r),
                    Err(e) => api_err_response(e),
                };
            }
            _ => {}
        }
    } else if method == Method::Post {
        match url.as_str() {
            "/api/games" => {
                let body: NewGameRequest = match read_json(request) {
                    Ok(b) => b,
                    Err(msg) => return json_err(400, &msg),
                };
                return match handlers::handle_new_game(&app.cfg, &app.registry, body) {
                    Ok(r) => json_ok(&r),
                    Err(e) => api_err_response(e),
                };
            }
            u if u.starts_with("/api/games/") && u.ends_with("/move") => {
                let game_id = &u["/api/games/".len()..u.len() - "/move".len()];
                let body: MoveRequest = match read_json(request) {
                    Ok(b) => b,
                    Err(msg) => return json_err(400, &msg),
                };
                return match handlers::handle_move(&app.registry, game_id, body) {
                    Ok(r) => json_ok(&r),
                    Err(e) => api_err_response(e),
                };
            }
            _ => {}
        }
    }

    text("not found", 404, "text/plain; charset=utf-8")
}

fn read_json<T: serde::de::DeserializeOwned>(request: &mut tiny_http::Request) -> Result<T, String> {
    let mut buf = String::new();
    request
        .as_reader()
        .read_to_string(&mut buf)
        .map_err(|e| format!("reading body: {e}"))?;
    serde_json::from_str(&buf).map_err(|e| format!("parsing body: {e}"))
}

fn json_ok<T: serde::Serialize>(body: &T) -> Response<std::io::Cursor<Vec<u8>>> {
    let s = serde_json::to_vec(body).unwrap_or_else(|_| b"{}".to_vec());
    let mut r = Response::from_data(s).with_status_code(200);
    r.add_header(Header::from_bytes("Content-Type", "application/json; charset=utf-8").unwrap());
    r
}

fn json_err(code: u32, msg: &str) -> Response<std::io::Cursor<Vec<u8>>> {
    let body = serde_json::json!({ "error": msg }).to_string();
    let mut r = Response::from_string(body).with_status_code(code);
    r.add_header(Header::from_bytes("Content-Type", "application/json; charset=utf-8").unwrap());
    r
}

fn api_err_response(e: ApiError) -> Response<std::io::Cursor<Vec<u8>>> {
    match e {
        ApiError::BadRequest(m) => json_err(400, &m),
        ApiError::NotFound(m) => json_err(404, &m),
        ApiError::Internal(m) => json_err(500, &m),
    }
}

fn html(body: &str, code: u32) -> Response<std::io::Cursor<Vec<u8>>> {
    text(body, code, "text/html; charset=utf-8")
}

fn text(body: &str, code: u32, content_type: &str) -> Response<std::io::Cursor<Vec<u8>>> {
    let mut r = Response::from_string(body.to_string()).with_status_code(code);
    r.add_header(Header::from_bytes("Content-Type", content_type).unwrap());
    r
}
```

- [ ] **Step 3: Build the bin**

```bash
cd /home/jbinney/ws/deep_rabbit_hole/deep_quoridor/rust && cargo build --no-default-features --features binary --bin play_server
```
Expected: builds successfully. (`Response::from_string` / `from_data` return slightly different concrete types in `tiny_http` 0.12; if the `Response<std::io::Cursor<Vec<u8>>>` return signature mismatches, switch the helpers to return `Response<std::io::Empty>` for `from_string` and convert via `.boxed()` consistently. The simplest is to import `tiny_http::Response` aliased without the generic and use `Response::from_data(Vec<u8>)` throughout so all helpers share the `Cursor<Vec<u8>>` type. Adjust as the compiler guides.)

- [ ] **Step 4: Commit**

```bash
cd /home/jbinney/ws/deep_rabbit_hole
git add deep_quoridor/rust/src/bin/play_server.rs deep_quoridor/rust/src/play_server/static
git commit -m "vibe: add play_server bin with tiny_http loop"
```

---

## Task 7: End-to-end test using the existing B5W2 fixture

**Files:**
- Create: `deep_quoridor/rust/tests/play_server_e2e.rs`

- [ ] **Step 1: Write the failing test**

Create `deep_quoridor/rust/tests/play_server_e2e.rs`:
```rust
//! End-to-end test: spawn the play server bound to port 0, drive it via HTTP,
//! and verify state transitions on a tiny game using the existing B5W2
//! fixture model.

#![cfg(feature = "binary")]

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::Value;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn fixture_onnx() -> PathBuf {
    workspace_root().join("fixtures/alphazero_B5W2_mv1.onnx")
}

fn make_play_dir() -> PathBuf {
    let dir = tempfile::Builder::new()
        .prefix("play_e2e_")
        .tempdir()
        .expect("tempdir")
        .into_path();
    std::fs::write(
        dir.join("config.yaml"),
        "quoridor:\n  board_size: 5\n  max_walls: 2\n  max_steps: 50\n",
    )
    .unwrap();
    std::fs::create_dir_all(dir.join("models")).unwrap();
    std::fs::copy(
        fixture_onnx(),
        dir.join("models").join("alphazero_B5W2_mv1.onnx"),
    )
    .unwrap();
    dir
}

struct ServerProc {
    child: Child,
    port: u16,
}

impl Drop for ServerProc {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn spawn_server(play_dir: &std::path::Path) -> ServerProc {
    // Use a fixed high-ish port; if collision, the test will retry.
    let port = pick_port();
    let bin = workspace_root().join("target/debug/play_server");
    let child = Command::new(&bin)
        .args([
            "--play-dir",
            play_dir.to_str().unwrap(),
            "--port",
            &port.to_string(),
            "--bind",
            "127.0.0.1",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn play_server");
    wait_ready(port);
    ServerProc { child, port }
}

fn pick_port() -> u16 {
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    drop(l);
    port
}

fn wait_ready(port: u16) {
    let deadline = Instant::now() + Duration::from_secs(15);
    let url = format!("http://127.0.0.1:{port}/api/config");
    while Instant::now() < deadline {
        if ureq::get(&url).call().is_ok() {
            return;
        }
        std::thread::sleep(Duration::from_millis(150));
    }
    panic!("play_server never became ready on port {port}");
}

#[test]
fn e2e_new_game_and_first_move() {
    // Build the bin once; ignore exit code (`cargo test` already builds bins
    // it depends on, but we want explicit failure if it doesn't).
    assert!(
        Command::new(env!("CARGO"))
            .args([
                "build",
                "--no-default-features",
                "--features",
                "binary",
                "--bin",
                "play_server",
            ])
            .status()
            .expect("cargo build")
            .success(),
        "cargo build of play_server failed"
    );

    let dir = make_play_dir();
    let server = spawn_server(&dir);
    let base = format!("http://127.0.0.1:{}", server.port);

    // /api/config lists the fixture model + the board config.
    let cfg: Value = ureq::get(&format!("{base}/api/config"))
        .call()
        .unwrap()
        .into_json()
        .unwrap();
    assert_eq!(cfg["board_size"], 5);
    assert_eq!(cfg["max_walls"], 2);
    assert!(cfg["models"]
        .as_array()
        .unwrap()
        .iter()
        .any(|m| m == "alphazero_B5W2_mv1.onnx"));

    // POST /api/games -> initial state.
    let new_game: Value = ureq::post(&format!("{base}/api/games"))
        .send_json(serde_json::json!({
            "model": "alphazero_B5W2_mv1.onnx",
            "mcts_n": 8,
            "human_player": 0,
        }))
        .unwrap()
        .into_json()
        .unwrap();
    let game_id = new_game["game_id"].as_str().unwrap().to_string();
    assert_eq!(new_game["state"]["board_size"], 5);
    assert_eq!(new_game["state"]["current_player"], 0);
    assert_eq!(new_game["state"]["human_player"], 0);
    assert!(new_game["state"]["legal_actions"]
        .as_array()
        .unwrap()
        .iter()
        .any(|a| a["kind"] == "move"));

    // Pick any legal move action; POST /api/games/<id>/move.
    let first_move_idx = new_game["state"]["legal_actions"]
        .as_array()
        .unwrap()
        .iter()
        .find_map(|a| {
            if a["kind"] == "move" {
                a["index"].as_u64().map(|x| x as u32)
            } else {
                None
            }
        })
        .expect("at least one move action");
    let after: Value = ureq::post(&format!("{base}/api/games/{game_id}/move"))
        .send_json(serde_json::json!({ "action_index": first_move_idx }))
        .unwrap()
        .into_json()
        .unwrap();
    // After human move + AI response, it's the human's turn again, and
    // history grew by exactly 2.
    assert_eq!(after["state"]["human_player"], 0);
    assert_eq!(after["state"]["current_player"], 0);
    assert_eq!(after["state"]["move_history"].as_array().unwrap().len(), 2);
}
```

- [ ] **Step 2: Run the test**

```bash
cd /home/jbinney/ws/deep_rabbit_hole/deep_quoridor/rust && cargo test --no-default-features --features binary --test play_server_e2e -- --nocapture
```
Expected: `e2e_new_game_and_first_move` passes (≈ a few seconds — the fixture model is tiny).

- [ ] **Step 3: Commit**

```bash
cd /home/jbinney/ws/deep_rabbit_hole
git add deep_quoridor/rust/tests/play_server_e2e.rs
git commit -m "vibe: e2e test for play_server against B5W2 fixture"
```

---

## Task 8: Frontend — `index.html`, `app.css`, `app.js`

**Files:**
- Modify (replace placeholder): `deep_quoridor/rust/src/play_server/static/index.html`
- Modify: `deep_quoridor/rust/src/play_server/static/app.css`
- Modify: `deep_quoridor/rust/src/play_server/static/app.js`

- [ ] **Step 1: Replace `index.html`**

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Quoridor vs AlphaZero</title>
<link rel="stylesheet" href="/static/app.css">
</head>
<body>
<main id="app">
  <h1>Quoridor vs AlphaZero</h1>

  <section id="setup" class="panel">
    <label>Model
      <select id="model"></select>
    </label>
    <label>MCTS sims
      <input type="range" id="mcts" min="1" max="4000" step="1">
      <output id="mcts-val"></output>
    </label>
    <label>You play as
      <select id="first">
        <option value="0">Player 1 (move first)</option>
        <option value="1">Player 2 (AI moves first)</option>
      </select>
    </label>
    <button id="start">New Game</button>
  </section>

  <section id="game" hidden>
    <div id="board" aria-label="Quoridor board"></div>
    <aside id="info" class="panel">
      <p>Turn: <strong id="turn"></strong></p>
      <p>You are: <strong id="you"></strong></p>
      <p>Your walls: <strong id="hwalls"></strong></p>
      <p>AI walls: <strong id="awalls"></strong></p>
      <p>Steps: <strong id="steps"></strong></p>
      <p id="status"></p>
      <button id="newgame2">New Game</button>
    </aside>
  </section>
</main>
<script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Replace `app.css`**

```css
* { box-sizing: border-box; }
body { font-family: system-ui, sans-serif; margin: 1.5em; background: #fafafa; color: #222; }
#app { max-width: 1000px; margin: 0 auto; }
h1 { margin-top: 0; }
.panel { background: #fff; padding: 1em; border: 1px solid #ddd; border-radius: 6px; }
#setup label { display: block; margin-bottom: 0.75em; }
#setup button { padding: 0.5em 1em; }
#game { display: flex; gap: 1em; margin-top: 1em; align-items: flex-start; }

#board {
  display: grid;
  background: #d8b876;
  padding: 4px;
  border-radius: 6px;
  --cell: 44px;
  --slot: 8px;
}
#board > div {
  display: flex;
  align-items: center;
  justify-content: center;
}
#board > .cell {
  background: #f1d9a1;
  border-radius: 3px;
  font-weight: bold;
}
#board > .cell.legal { background: #c8e8a6; cursor: pointer; }
#board > .cell.legal:hover { background: #a4d878; }
#board > .cell.last { outline: 3px solid #f08020; }
#board > .pawn-p1 { color: #1a48d8; }
#board > .pawn-p2 { color: #d8281a; }
#board > .pawn-p1::before { content: '\25CF'; font-size: 1.5em; }
#board > .pawn-p2::before { content: '\25CF'; font-size: 1.5em; }
#board > .wallslot { background: transparent; }
#board > .wallslot.legal { background: #c8e8a6aa; cursor: pointer; }
#board > .wallslot.legal:hover { background: #a4d878dd; }
#board > .wallslot.wall { background: #5a3a18; border-radius: 2px; }
#board > .post { background: #b08850; border-radius: 1px; }

#info { min-width: 220px; }
#status { font-weight: bold; min-height: 1.2em; }
.thinking #status::after { content: ' (AI thinking…)'; font-weight: normal; color: #666; }
```

- [ ] **Step 3: Replace `app.js`**

```javascript
"use strict";

const $ = (id) => document.getElementById(id);
let CONFIG = null;
let GAME = null;

async function fetchJson(method, path, body) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const r = await fetch(path, opts);
  const text = await r.text();
  const data = text ? JSON.parse(text) : {};
  if (!r.ok) throw new Error(data.error || `${r.status} ${r.statusText}`);
  return data;
}

async function init() {
  CONFIG = await fetchJson("GET", "/api/config");
  const modelSel = $("model");
  modelSel.innerHTML = "";
  for (const m of CONFIG.models) {
    const opt = document.createElement("option");
    opt.value = m;
    opt.textContent = m;
    modelSel.appendChild(opt);
  }
  const slider = $("mcts");
  slider.value = CONFIG.default_mcts_n;
  $("mcts-val").value = slider.value;
  slider.addEventListener("input", () => ($("mcts-val").value = slider.value));

  $("start").addEventListener("click", startGame);
  $("newgame2").addEventListener("click", () => {
    $("setup").hidden = false;
    $("game").hidden = true;
  });
}

async function startGame() {
  const body = {
    model: $("model").value,
    mcts_n: parseInt($("mcts").value, 10),
    human_player: parseInt($("first").value, 10),
  };
  const data = await fetchJson("POST", "/api/games", body);
  GAME = { id: data.game_id, state: data.state };
  $("setup").hidden = true;
  $("game").hidden = false;
  render();
}

function render() {
  const s = GAME.state;
  const N = s.board_size;
  const board = $("board");
  board.style.gridTemplateColumns = `repeat(${2 * N - 1}, var(--col-size))`;
  board.style.gridAutoRows = "var(--row-size)";
  board.style.setProperty("--col-size", "var(--cell)");
  board.style.setProperty("--row-size", "var(--cell)");

  board.innerHTML = "";
  const mirror = s.human_player === 1;
  // (gridR, gridC) is the (2N-1)x(2N-1) grid position.
  for (let gr = 0; gr < 2 * N - 1; gr++) {
    for (let gc = 0; gc < 2 * N - 1; gc++) {
      const el = document.createElement("div");
      // Display row is mirrored so the human's home row is at the bottom.
      el.style.gridRow = `${(mirror ? gr : 2 * N - 2 - gr) + 1}`;
      el.style.gridColumn = `${gc + 1}`;

      if (gr % 2 === 0 && gc % 2 === 0) {
        // pawn cell
        const r = gr / 2;
        const c = gc / 2;
        el.className = "cell";
        el.style.width = "var(--cell)";
        el.style.height = "var(--cell)";
        if (s.p1_pos[0] === r && s.p1_pos[1] === c) el.classList.add("pawn-p1");
        if (s.p2_pos[0] === r && s.p2_pos[1] === c) el.classList.add("pawn-p2");
        el.dataset.kind = "cell";
        el.dataset.r = r;
        el.dataset.c = c;
      } else if (gr % 2 === 1 && gc % 2 === 1) {
        el.className = "post";
        el.style.width = "var(--slot)";
        el.style.height = "var(--slot)";
      } else {
        // wall slot
        el.className = "wallslot";
        const horizontal = gr % 2 === 1;
        el.dataset.kind = horizontal ? "wall-h" : "wall-v";
        el.dataset.r = horizontal ? (gr - 1) / 2 : gr / 2;
        el.dataset.c = horizontal ? gc / 2 : (gc - 1) / 2;
        if (horizontal) {
          el.style.height = "var(--slot)";
          el.style.gridColumn = `${gc + 1} / span 3`;
        } else {
          el.style.width = "var(--slot)";
          el.style.gridRow = `${(mirror ? gr : 2 * N - 2 - gr) + 1} / span 3`;
        }
      }
      board.appendChild(el);
    }
  }

  // Mark existing walls.
  for (const w of s.walls) {
    const sel =
      w.orientation === "h"
        ? `.wallslot[data-kind="wall-h"][data-r="${w.row}"][data-c="${w.col}"]`
        : `.wallslot[data-kind="wall-v"][data-r="${w.row}"][data-c="${w.col}"]`;
    document.querySelectorAll(sel).forEach((el) => el.classList.add("wall"));
  }
  // Highlight last action.
  if (s.last_action) {
    const a = s.last_action;
    if (a.kind === "move") {
      document
        .querySelectorAll(`.cell[data-r="${a.to[0]}"][data-c="${a.to[1]}"]`)
        .forEach((el) => el.classList.add("last"));
    }
  }

  // Mark legal actions only when it's the human's turn.
  if (s.winner === null && s.current_player === s.human_player) {
    for (const a of s.legal_actions) {
      let el = null;
      if (a.kind === "move") {
        el = board.querySelector(`.cell[data-r="${a.to[0]}"][data-c="${a.to[1]}"]`);
      } else {
        const kind = a.orientation === "h" ? "wall-h" : "wall-v";
        el = board.querySelector(
          `.wallslot[data-kind="${kind}"][data-r="${a.row}"][data-c="${a.col}"]`,
        );
      }
      if (el) {
        el.classList.add("legal");
        el.addEventListener("click", () => playMove(a.index), { once: true });
      }
    }
  }

  // Side panel.
  const turnName = s.winner !== null ? "Game over" : s.current_player === 0 ? "Player 1" : "Player 2";
  $("turn").textContent = turnName;
  $("you").textContent = s.human_player === 0 ? "Player 1" : "Player 2";
  const hWalls = s.human_player === 0 ? s.p1_walls : s.p2_walls;
  const aWalls = s.human_player === 0 ? s.p2_walls : s.p1_walls;
  $("hwalls").textContent = hWalls;
  $("awalls").textContent = aWalls;
  $("steps").textContent = s.completed_steps;
  const status =
    s.winner === null
      ? ""
      : s.winner === s.human_player
        ? "You won 🎉"
        : "AI won";
  $("status").textContent = status;
}

async function playMove(actionIndex) {
  document.body.classList.add("thinking");
  try {
    const data = await fetchJson("POST", `/api/games/${GAME.id}/move`, {
      action_index: actionIndex,
    });
    GAME.state = data.state;
    render();
  } catch (e) {
    alert(`move rejected: ${e.message}`);
  } finally {
    document.body.classList.remove("thinking");
  }
}

init();
```

- [ ] **Step 4: Rebuild + manual smoke check**

```bash
cd /home/jbinney/ws/deep_rabbit_hole/deep_quoridor/rust && cargo build --no-default-features --features binary --bin play_server
```
Expected: builds (`include_str!` re-embeds the new files).

Then, manually:
```bash
mkdir -p /tmp/quoridor-play/models
cp deep_quoridor/rust/fixtures/alphazero_B5W2_mv1.onnx /tmp/quoridor-play/models/
cat > /tmp/quoridor-play/config.yaml <<YAML
quoridor:
  board_size: 5
  max_walls: 2
  max_steps: 50
YAML
deep_quoridor/rust/target/debug/play_server --play-dir /tmp/quoridor-play
```
Open `http://localhost:8080` in a browser, click "New Game", verify the board renders, click a highlighted legal move, verify the AI responds and the board re-renders.

- [ ] **Step 5: Commit**

```bash
cd /home/jbinney/ws/deep_rabbit_hole
git add deep_quoridor/rust/src/play_server/static
git commit -m "vibe: add play_server frontend"
```

---

## Task 9: Formatting commit (per AGENTS.md)

- [ ] **Step 1: Format Rust**

```bash
cd /home/jbinney/ws/deep_rabbit_hole/deep_quoridor/rust && cargo fmt && cargo fmt --check && echo FMT_OK && cargo build --no-default-features --features binary --bin play_server
```
Expected: `FMT_OK` and a clean build.

- [ ] **Step 2: Commit only if formatting changed files**

```bash
cd /home/jbinney/ws/deep_rabbit_hole
git status --short deep_quoridor/rust
# If there are modified .rs files:
git add -u deep_quoridor/rust
git commit -m "vibe: cargo fmt"
```
If there are no changes, skip.

---

## Self-review (completed during plan authoring)

- **Spec coverage:** §Architecture/User flow → Tasks 4–8; §Folder layout → Task 3; §CLI → Task 6; §HTTP API → Tasks 5+6; §`State` object → Tasks 2+4; §Frontend → Task 8; §Concurrency → Task 4 (`Arc<Mutex<>>` per game); §Cargo.toml additions → Task 1; §Testing → Tasks 2–5 unit tests + Task 7 e2e; §Out of scope kept out (no auth, no persistence, no undo, etc.). All mapped.
- **Placeholder scan:** none — every step has concrete code or exact commands. Tasks 6 and 4 carry small "if API signature differs, read X and adapt" notes for two specific calls (`tiny_http::Response` generics; `repr().get_wall`) where the exact compile-time shape isn't guaranteed without reading those files; both name the file to read and what to preserve.
- **Type consistency:** `StateView` (Task 2) field names match the JSON in the spec, the e2e test assertions (Task 7), and the frontend reads (Task 8). `EnrichedAction` tag/`kind` naming consistent across server, e2e test (`"kind":"move"`), and JS. `GameRegistry::get`/`insert` signatures are the same in Tasks 4, 5, and 6. `ApiError::{BadRequest,NotFound,Internal}` defined in Task 5 and consumed in Task 6.
