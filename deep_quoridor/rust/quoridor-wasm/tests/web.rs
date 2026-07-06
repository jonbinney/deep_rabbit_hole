use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;

// A JS mock eval that resolves to zero values + zero logits of width 512.
#[wasm_bindgen(inline_js = "
module.exports.makeMockEval = function() {
  return function(flat, n, c, h, w) {
    const values = new Float32Array(n);
    const logits = new Float32Array(n * 512);
    return Promise.resolve({ values, logits });
  };
}
module.exports.makeProgress = function() { return function(_d, _t) {}; }
")]
extern "C" {
    fn makeMockEval() -> js_sys::Function;
    fn makeProgress() -> js_sys::Function;
}

#[wasm_bindgen_test]
async fn run_search_returns_a_legal_action() {
    quoridor_wasm::init();
    let game = quoridor_wasm::Game::new(5, 2, 50, 0);
    let result = game
        .run_search(32, 1.4, 8, 1, makeMockEval(), makeProgress())
        .await
        .unwrap();

    let action = js_sys::Reflect::get(&result, &JsValue::from_str("action")).unwrap();
    let action = action.as_f64().unwrap() as u32;

    // Cross-check against the game's own legal mask via stateView.legal_actions.
    let view = game.state_view().unwrap();
    let legal = js_sys::Reflect::get(&view, &JsValue::from_str("legal_actions")).unwrap();
    let legal: js_sys::Array = legal.dyn_into().unwrap();
    let mut found = false;
    for i in 0..legal.length() {
        let a = legal.get(i);
        let idx = js_sys::Reflect::get(&a, &JsValue::from_str("index"))
            .unwrap()
            .as_f64()
            .unwrap() as u32;
        if idx == action {
            found = true;
            break;
        }
    }
    assert!(found, "runSearch action must be one of the legal actions");
}
