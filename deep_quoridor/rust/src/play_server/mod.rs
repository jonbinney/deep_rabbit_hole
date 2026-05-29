//! Local web server for playing Quoridor against the AlphaZero agent.
//!
//! Architecture overview is in
//! `docs/superpowers/specs/2026-05-29-quoridor-play-server-design.md`.

pub mod config;
pub mod handlers;
pub mod http;
pub mod session;
pub mod state;
