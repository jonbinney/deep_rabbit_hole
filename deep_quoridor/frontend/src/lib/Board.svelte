<script lang="ts">
  import type { StateView } from "./types";
  import { deriveBoard } from "./board";
  let { view, disabled, onaction }: {
    view: StateView; disabled: boolean; onaction: (index: number) => void;
  } = $props();

  const board = $derived(deriveBoard(view));
  const cells = $derived(
    Array.from({ length: board.size }, (_, r) =>
      Array.from({ length: board.size }, (_, c) => ({ r, c })),
    ),
  );
  function pawnAt(r: number, c: number): number | null {
    if (board.pawns[0][0] === r && board.pawns[0][1] === c) return 0;
    if (board.pawns[1][0] === r && board.pawns[1][1] === c) return 1;
    return null;
  }
</script>

<div class="board" style="--n:{board.size}">
  {#each cells as row}
    {#each row as { r, c }}
      {@const move = board.moveActionAt(r, c)}
      {@const pawn = pawnAt(r, c)}
      <button
        class="cell"
        class:legal={move !== undefined}
        class:p1={pawn === 0}
        class:p2={pawn === 1}
        disabled={disabled || move === undefined}
        onclick={() => move !== undefined && onaction(move)}
      >{pawn === 0 ? "●" : pawn === 1 ? "○" : ""}</button>
    {/each}
  {/each}
</div>
<!-- Wall placement: list legal wall actions as buttons (functional M1 UI). -->
<div class="walls">
  {#each view.legal_actions.filter((a) => a.kind === "wall") as a}
    <button disabled={disabled} onclick={() => onaction(a.index)}>
      wall {a.kind === "wall" ? `${a.orientation} @${a.row},${a.col}` : ""}
    </button>
  {/each}
</div>

<style>
  .board { display: grid; grid-template-columns: repeat(var(--n), 44px); gap: 3px; }
  .cell { width: 44px; height: 44px; font-size: 1.4rem; }
  .cell.legal { outline: 2px dashed #2a7; }
  .cell.p1 { background: #c0392b; color: #fff; }
  .cell.p2 { background: #2980b9; color: #fff; }
  .walls { margin-top: 8px; display: flex; flex-wrap: wrap; gap: 4px; max-width: 360px; }
  .walls button { font-size: 0.75rem; }
</style>
