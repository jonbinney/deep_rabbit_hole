//! AlphaZero MCTS agent implementation.
//!
//! This module provides MCTS-based action selection using neural network evaluation.
//! Only available behind the `binary` feature flag.

pub mod evaluator;
pub mod mcts;

pub mod agent;
pub use agent::{AlphaZeroAgent, AlphaZeroAgentConfig};
