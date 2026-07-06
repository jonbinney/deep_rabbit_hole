//! AlphaZero MCTS agent implementation.
//!
//! This module provides MCTS-based action selection using neural network evaluation.
//! Only available behind the `binary` feature flag.

pub mod batched_search;
pub mod evaluator;
pub mod mcts;

#[cfg(feature = "binary")]
pub mod agent;
#[cfg(feature = "binary")]
pub mod eval_pipeline;
#[cfg(feature = "binary")]
pub mod selfplay_game;
#[cfg(feature = "binary")]
pub mod selfplay_mcts;
#[cfg(feature = "binary")]
pub mod selfplay_metrics;

#[cfg(feature = "binary")]
pub use agent::{AlphaZeroAgent, AlphaZeroAgentConfig};
