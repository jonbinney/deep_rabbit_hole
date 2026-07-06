//! Wires the portable `run_batched_search` driver to JS callbacks: an async
//! batched NN eval (onnxruntime-web) and a progress reporter.

use js_sys::{Array, Float32Array, Object, Reflect};
use ndarray::Array4;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

use quoridor_rs::agents::alphazero::batched_search::{
    BatchedSearchConfig, EvalOutput, best_action, run_batched_search,
};
use quoridor_rs::agents::alphazero::mcts::MCTSConfig;

use crate::game::WasmGame;

/// Flatten a batch of [1, C, H, W] feature tensors into one contiguous
/// Float32Array of shape [N, C, H, W] plus its dims.
fn flatten_batch(batch: &[Array4<f32>]) -> (Float32Array, usize, usize, usize) {
    let n = batch.len();
    let (c, h, w) = if n == 0 {
        (0, 0, 0)
    } else {
        let s = batch[0].shape();
        (s[1], s[2], s[3])
    };
    let per = c * h * w;
    let flat = Float32Array::new_with_length((n * per) as u32);
    for (i, arr) in batch.iter().enumerate() {
        let contiguous: Vec<f32> = arr.iter().copied().collect();
        flat.subarray((i * per) as u32, ((i + 1) * per) as u32)
            .copy_from(&contiguous);
    }
    (flat, c, h, w)
}

/// Split the JS result `{ values: Float32Array[N], logits: Float32Array[N*P] }`
/// into per-leaf `EvalOutput`s.
fn split_outputs(result: &JsValue, n: usize) -> Result<Vec<EvalOutput>, JsValue> {
    let values: Float32Array = Reflect::get(result, &JsValue::from_str("values"))?.dyn_into()?;
    let logits: Float32Array = Reflect::get(result, &JsValue::from_str("logits"))?.dyn_into()?;
    let values_v = values.to_vec();
    let logits_v = logits.to_vec();
    if values_v.len() != n {
        return Err(JsValue::from_str(
            "eval result 'values' length != batch size",
        ));
    }
    if n != 0 && logits_v.len() % n != 0 {
        return Err(JsValue::from_str(
            "eval result 'logits' length is not a multiple of batch size",
        ));
    }
    let p = if n == 0 { 0 } else { logits_v.len() / n };
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(EvalOutput {
            value: values_v[i],
            policy_logits: logits_v[i * p..(i + 1) * p].to_vec(),
        });
    }
    Ok(out)
}

/// Run MCTS for the current position, calling `eval_batch` (async JS) for NN
/// forward passes and `progress` after each round. Resolves to
/// `{ action, rootValue, children: [{actionIndex, visitCount}] }`.
pub async fn run_search_js(
    game: &WasmGame,
    mcts_n: u32,
    c_puct: f32,
    leaf_parallelism: u32,
    virtual_loss: u32,
    eval_batch: js_sys::Function,
    progress: js_sys::Function,
) -> Result<JsValue, JsValue> {
    // Guard the boundary: searching a finished game leaves the root unexpanded,
    // so `run_batched_search` returns no children and `best_action` would panic
    // (a hard wasm abort under `panic = "abort"`). Surface a catchable JS error.
    if game.is_game_over() {
        return Err(JsValue::from_str("cannot run search: game is already over"));
    }

    let cfg = MCTSConfig {
        n: Some(mcts_n),
        k: None,
        ucb_c: c_puct,
        noise_epsilon: 0.0,
        noise_alpha: None,
        // Play mode terminates on a winner and the JS layer stops at game end,
        // so a hard step cap isn't needed here for M1.
        max_steps: None,
        penalize_visited_states: false,
    };
    let bs = BatchedSearchConfig {
        leaf_parallelism,
        virtual_loss,
    };

    let eval = |batch: Vec<Array4<f32>>| {
        let eval_batch = eval_batch.clone();
        async move {
            let n = batch.len();
            let (flat, c, h, w) = flatten_batch(&batch);
            let args = Array::new();
            args.push(&flat);
            args.push(&JsValue::from_f64(n as f64));
            args.push(&JsValue::from_f64(c as f64));
            args.push(&JsValue::from_f64(h as f64));
            args.push(&JsValue::from_f64(w as f64));
            let promise = eval_batch
                .apply(&JsValue::NULL, &args)
                .map_err(|e| anyhow::anyhow!("eval_batch threw: {e:?}"))?;
            let promise: js_sys::Promise = promise
                .dyn_into()
                .map_err(|_| anyhow::anyhow!("eval_batch must return a Promise"))?;
            let result = JsFuture::from(promise)
                .await
                .map_err(|e| anyhow::anyhow!("eval_batch rejected: {e:?}"))?;
            split_outputs(&result, n).map_err(|e| anyhow::anyhow!("split_outputs: {e:?}"))
        }
    };

    let report = |done: u32, total: u32| {
        // Progress is best-effort; ignore a throwing JS progress callback.
        let _ = progress.call2(
            &JsValue::NULL,
            &JsValue::from_f64(done as f64),
            &JsValue::from_f64(total as f64),
        );
    };

    let (children, root_value) =
        run_batched_search(&cfg, &bs, game.state(), game.mechanics(), eval, report)
            .await
            .map_err(|e| JsValue::from_str(&format!("{e:#}")))?;

    let action = best_action(&children);
    let out = Object::new();
    Reflect::set(
        &out,
        &JsValue::from_str("action"),
        &JsValue::from_f64(action as f64),
    )?;
    Reflect::set(
        &out,
        &JsValue::from_str("rootValue"),
        &JsValue::from_f64(root_value as f64),
    )?;
    let arr = Array::new();
    for c in &children {
        let o = Object::new();
        Reflect::set(
            &o,
            &JsValue::from_str("actionIndex"),
            &JsValue::from_f64(c.action_index as f64),
        )?;
        Reflect::set(
            &o,
            &JsValue::from_str("visitCount"),
            &JsValue::from_f64(c.visit_count as f64),
        )?;
        arr.push(&o);
    }
    Reflect::set(&out, &JsValue::from_str("children"), &arr)?;
    Ok(out.into())
}
