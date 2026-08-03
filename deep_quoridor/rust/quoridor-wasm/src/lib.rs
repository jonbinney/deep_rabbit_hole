use serde::Serialize;
use wasm_bindgen::prelude::*;

mod game;
mod search;
mod view;

use game::WasmGame;

#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

/// JS-facing handle around a game session.
#[wasm_bindgen]
pub struct Game {
    inner: WasmGame,
}

#[wasm_bindgen]
impl Game {
    #[wasm_bindgen(constructor)]
    pub fn new(board_size: i32, max_walls: i32, max_steps: i32, human_player: i32) -> Game {
        Game {
            inner: WasmGame::new(board_size, max_walls, max_steps, human_player),
        }
    }

    /// Returns the `StateView` as a JS object.
    #[wasm_bindgen(js_name = stateView)]
    pub fn state_view(&self) -> Result<JsValue, JsValue> {
        // Serialize `Option::None` as JS `null` (not `undefined`) so the fields
        // the TS types declare as `T | null` (winner, last_action) actually
        // arrive as `null` and `=== null` checks on the JS side hold.
        let ser = serde_wasm_bindgen::Serializer::new().serialize_missing_as_null(true);
        self.inner
            .view()
            .serialize(&ser)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen(js_name = applyAction)]
    pub fn apply_action(&mut self, action_index: u32) -> Result<JsValue, JsValue> {
        self.inner
            .apply_action(action_index)
            .map_err(|e| JsValue::from_str(&e))?;
        self.state_view()
    }

    pub fn undo(&mut self, count: usize) -> Result<JsValue, JsValue> {
        self.inner.undo(count);
        self.state_view()
    }

    #[wasm_bindgen(js_name = runSearch)]
    pub async fn run_search(
        &self,
        mcts_n: u32,
        c_puct: f32,
        leaf_parallelism: u32,
        virtual_loss: u32,
        eval_batch: js_sys::Function,
        progress: js_sys::Function,
    ) -> Result<JsValue, JsValue> {
        crate::search::run_search_js(
            &self.inner,
            mcts_n,
            c_puct,
            leaf_parallelism,
            virtual_loss,
            eval_batch,
            progress,
        )
        .await
    }
}
