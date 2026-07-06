<script lang="ts">
  import { onMount } from "svelte";
  import Board from "./lib/Board.svelte";
  import ControlRail from "./lib/ControlRail.svelte";
  import ConfigDrawer from "./lib/ConfigDrawer.svelte";
  import { AiClient } from "./lib/aiClient";
  import { fetchConfig, fetchModels, type ConfigView, type ModelsView } from "./lib/api";
  import type { StateView } from "./lib/types";

  let config = $state<ConfigView | null>(null);
  let models = $state<ModelsView | null>(null);
  let view = $state<StateView | null>(null);
  let thinking = $state(false);
  let progress = $state<{ done: number; total: number } | null>(null);
  let error = $state<string | null>(null);
  let model = $state("");
  let humanPlayer = $state(0);
  let params = $state({ mctsN: 200, cPuct: 1.4, leafParallelism: 8, virtualLoss: 1 });

  // True only when the human may act: their turn, game live, AI not working.
  const awaitingHuman = $derived(
    !!view && view.winner == null && view.current_player === view.human_player && !thinking,
  );

  const ai = new AiClient();
  ai.onState = (v, t) => { view = v; thinking = t; if (!t) progress = null; };
  ai.onProgress = (done, total) => { thinking = true; progress = { done, total }; };
  ai.onError = (m) => { error = m; thinking = false; };

  onMount(async () => {
    try {
      config = await fetchConfig();
      models = await fetchModels();
      model = models.default ?? models.models[0] ?? "";
      params = { ...params, mctsN: config.defaults.mcts_n, cPuct: config.defaults.mcts_c_puct };
      newGame();
    } catch (e) {
      error = `Failed to load config/models: ${e}`;
    }
  });

  function newGame() {
    if (!config || !model) return;
    error = null; thinking = true; progress = null;
    ai.newGame({
      model, boardSize: config.board_size, maxWalls: config.max_walls,
      maxSteps: config.max_steps, humanPlayer, params,
    });
  }
  function act(index: number) { thinking = true; ai.move(index); }
</script>

<div class="layout">
  <div>
    {#if error}<p class="err">Error: {error}</p>{/if}
    {#if view}
      <Board {view} disabled={!awaitingHuman} onaction={act} />
    {:else}
      <p>Loading…</p>
    {/if}
  </div>
  <ControlRail {view} {thinking} {progress} onundo={() => ai.undo(2)} onnewgame={newGame} />
  <ConfigDrawer {config} {models} {model} {params}
    onchange={(o) => { model = o.model; params = o.params; ai.setParams(o.params); }} />
</div>

<style>
  .layout { display: flex; gap: 20px; align-items: flex-start; flex-wrap: wrap; }
  .err { color: #c0392b; }
</style>
