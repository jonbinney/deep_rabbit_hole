<script lang="ts">
  import type { ConfigView, ModelsView } from "./api";
  let { config, models, model, params, humanPlayer, onchange, onhumanplayer }: {
    config: ConfigView | null;
    models: ModelsView | null;
    model: string;
    params: { mctsN: number; cPuct: number; leafParallelism: number; virtualLoss: number };
    humanPlayer: number;
    onchange: (o: { model: string; params: typeof params }) => void;
    onhumanplayer: (p: number) => void;
  } = $props();
</script>

<div class="drawer">
  <h3>Setup</h3>
  <label>Model
    <select value={model} onchange={(e) => onchange({ model: e.currentTarget.value, params })}>
      {#each models?.models ?? [] as m}<option value={m}>{m}</option>{/each}
    </select>
  </label>

  <div class="who">
    <span class="who-label">You play</span>
    <div class="segmented">
      <label class:sel={humanPlayer === 0}>
        <input type="radio" name="human-player" checked={humanPlayer === 0} onchange={() => onhumanplayer(0)} />
        First (P1)
      </label>
      <label class:sel={humanPlayer === 1}>
        <input type="radio" name="human-player" checked={humanPlayer === 1} onchange={() => onhumanplayer(1)} />
        Second (P2)
      </label>
    </div>
    <small class="hint">
      {humanPlayer === 0 ? "You move first." : "The AI moves first."} Applies when you click New game.
    </small>
  </div>

  <label>MCTS sims: {params.mctsN}
    <input type="range" min="16" max="2000" step="16" value={params.mctsN}
      oninput={(e) => onchange({ model, params: { ...params, mctsN: +e.currentTarget.value } })} />
  </label>
  <label>c_puct: {params.cPuct}
    <input type="range" min="0.5" max="3" step="0.1" value={params.cPuct}
      oninput={(e) => onchange({ model, params: { ...params, cPuct: +e.currentTarget.value } })} />
  </label>
  <label>leaf parallelism: {params.leafParallelism}
    <input type="range" min="1" max="32" step="1" value={params.leafParallelism}
      oninput={(e) => onchange({ model, params: { ...params, leafParallelism: +e.currentTarget.value } })} />
  </label>
</div>

<style>
  .drawer { display: flex; flex-direction: column; gap: 10px; width: 240px; }
  label { display: flex; flex-direction: column; font-size: 0.85rem; gap: 2px; }
  .who { display: flex; flex-direction: column; gap: 4px; font-size: 0.85rem; }
  .who-label { font-weight: 600; }
  .segmented { display: flex; gap: 6px; }
  .segmented label {
    flex: 1;
    flex-direction: row;
    align-items: center;
    justify-content: center;
    gap: 4px;
    padding: 4px 6px;
    border: 1px solid #c9b48a;
    border-radius: 6px;
    cursor: pointer;
    background: #fffaf1;
  }
  .segmented label.sel { background: #1e3a8a; color: #fff; border-color: #1e3a8a; }
  .segmented input { margin: 0; }
  .hint { color: #6b5a3f; }
</style>
