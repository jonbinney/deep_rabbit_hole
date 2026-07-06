use wasm_bindgen::prelude::*;

/// Call once from JS on startup to route Rust panics to `console.error`.
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

/// Smoke binding to prove the crate links against quoridor-rs core.
#[wasm_bindgen]
pub fn core_board_size(board_size: usize, max_walls: usize, max_steps: usize) -> usize {
    let mechanics =
        quoridor_rs::compact::q_game_mechanics::QGameMechanics::new(board_size, max_walls, max_steps);
    mechanics.repr().board_size()
}
