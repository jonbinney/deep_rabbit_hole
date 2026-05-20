//! AlphaZero MCTS agent implementation.
//!
//! This module provides MCTS-based action selection using neural network evaluation.
//! Only available behind the `binary` feature flag.

pub mod eval_coordinator;
pub mod eval_pipeline;
pub mod evaluator;
pub mod mcts;
pub mod selfplay_mcts;

pub mod agent;
pub use agent::{AlphaZeroAgent, AlphaZeroAgentConfig};
