<script lang="ts">
  import type { ConfigView, ModelsView } from "./api";
  let { config, models, model, params, onchange }: {
    config: ConfigView | null;
    models: ModelsView | null;
    model: string;
    params: { mctsN: number; cPuct: number; leafParallelism: number; virtualLoss: number };
    onchange: (o: { model: string; params: typeof params }) => void;
  } = $props();
</script>

<div class="drawer">
  <h3>Setup</h3>
  <label>Model
    <select value={model} onchange={(e) => onchange({ model: e.currentTarget.value, params })}>
      {#each models?.models ?? [] as m}<option value={m}>{m}</option>{/each}
    </select>
  </label>
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
</style>
