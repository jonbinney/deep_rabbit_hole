<script lang="ts">
  import type { StateView } from "./types";
  import { buildBoardGrid, cellKey, type GridCell } from "./boardGrid";

  let { view, disabled, onaction }: {
    view: StateView; disabled: boolean; onaction: (index: number) => void;
  } = $props();

  const grid = $derived(buildBoardGrid(view, !disabled));
  const tracks = $derived(
    Array.from({ length: grid.size }, (_, i) =>
      i % 2 === 0 ? "var(--pawn-size)" : "var(--post-size)",
    ).join(" "),
  );

  // Cells to tint while previewing the wall under the cursor.
  let hovered = $state(new Set<string>());
  function enter(group: string[]) { hovered = new Set(group); }
  function clearHover() { if (hovered.size) hovered = new Set(); }

  function click(cell: GridCell) {
    if (cell.legalMoveIndex !== null) onaction(cell.legalMoveIndex);
    else if (cell.wall) onaction(cell.wall.index);
  }
</script>

<div class="board" style="grid-template-columns:{tracks}; grid-template-rows:{tracks}">
  {#each grid.cells as cell (cellKey(cell.gr, cell.gc))}
    {@const clickable = cell.legalMoveIndex !== null || cell.wall !== null}
    <!-- svelte-ignore a11y_click_events_have_key_events -->
    <!-- svelte-ignore a11y_no_static_element_interactions -->
    <div
      class="cell {cell.base}"
      class:placed={cell.placed !== null}
      class:legal-move={cell.legalMoveIndex !== null}
      class:legal-wall={cell.wall !== null}
      class:preview={hovered.has(cellKey(cell.gr, cell.gc))}
      class:last-move={cell.lastMove}
      class:last-wall={cell.lastWall}
      class:clickable
      role={clickable ? "button" : null}
      tabindex={clickable ? 0 : null}
      onclick={() => click(cell)}
      onmouseenter={() => cell.wall && enter(cell.wall.group)}
      onmouseleave={clearHover}
    >
      {#if cell.pawn !== null}
        <div class="pawn" class:you={cell.pawn === view.human_player} class:ai={cell.pawn !== view.human_player}></div>
      {/if}
    </div>
  {/each}
</div>

<style>
  .board {
    --pawn-size: 54px;
    --post-size: 14px;
    display: inline-grid;
    gap: 0;
    padding: 8px;
    background: #2b1a0c;
    border-radius: 6px;
  }
  .cell { position: relative; }
  .pawn-cell { background: #ead7ad; }
  .wall-h-half, .wall-v-half { background: #c9b07a; }
  .post { background: #b59b65; }

  /* Placed walls: darkest walnut, wins over slot color. */
  .cell.placed { background: #3a2412; }

  /* Wall preview: only shown while hovering a placeable wall (no persistent
     tint on the 32 empty legal-wall slots). */
  .cell.preview { background: rgba(58, 36, 18, 0.55); }

  /* Legal wall anchors are clickable but otherwise invisible until hovered. */
  .cell.legal-wall { cursor: pointer; }

  .cell.clickable { cursor: pointer; }

  /* Legal move destination: green dot in the pawn cell. */
  .cell.legal-move::before {
    content: "";
    position: absolute;
    inset: 30%;
    border-radius: 50%;
    background: rgba(34, 113, 50, 0.55);
    transition: inset 0.08s;
  }
  .cell.legal-move:hover::before { inset: 18%; }

  .cell.last-move::after {
    content: "";
    position: absolute;
    inset: 8%;
    border: 2px dashed #d97706;
    border-radius: 6px;
  }
  .cell.last-wall { outline: 2px solid #d97706; outline-offset: -2px; }

  .pawn {
    position: absolute;
    inset: 12%;
    border-radius: 50%;
    box-shadow: inset 0 -3px 6px rgba(0, 0, 0, 0.35);
  }
  .pawn.ai { background: #b91c1c; }                 /* AI (top) */
  .pawn.you { background: #1e3a8a; outline: 3px solid #93c5fd; outline-offset: 1px; }  /* you (bottom) */
</style>
