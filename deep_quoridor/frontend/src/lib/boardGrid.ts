// Pure geometry for rendering a Quoridor board as a (2N-1)x(2N-1) grid of
// pawn cells, wall-half slots, and posts. The human (player 0) is always shown
// with their home row at the visual BOTTOM, moving UP toward the top — so we
// vertically flip server coordinates when the human is player 0.
//
// Adapted from the original Rust play-server's app.js rendering, which used the
// same mirror + wall-group scheme.

import type { Orientation, StateView } from "./types";

export interface GridCell {
  gr: number;
  gc: number;
  base: "pawn-cell" | "wall-h-half" | "wall-v-half" | "post";
  pawn: 0 | 1 | null;
  placed: Orientation | null;
  /** Set on a legal move's destination pawn-cell. */
  legalMoveIndex: number | null;
  /** Set ONLY on a legal wall's anchor (display top-left) cell. */
  wall: { index: number; orientation: Orientation; group: string[] } | null;
  lastMove: boolean;
  lastWall: boolean;
}

export interface BoardGrid {
  n: number;
  size: number; // 2n - 1
  cells: GridCell[];
}

export function cellKey(gr: number, gc: number): string {
  return `${gr},${gc}`;
}

/** The 3 grid cells (two halves + the post between) covered by a wall whose
 *  display top-left corner is (dr, dc). For "h" it extends right, for "v" down. */
function wallGroup(dr: number, dc: number, o: Orientation): [number, number][] {
  if (o === "h") {
    return [
      [2 * dr + 1, 2 * dc],
      [2 * dr + 1, 2 * dc + 1],
      [2 * dr + 1, 2 * dc + 2],
    ];
  }
  return [
    [2 * dr, 2 * dc + 1],
    [2 * dr + 1, 2 * dc + 1],
    [2 * dr + 2, 2 * dc + 1],
  ];
}

/** Build the renderable grid. `interactive` gates legal-action markers (off
 *  when it's the AI's turn / the board is disabled). */
export function buildBoardGrid(view: StateView, interactive: boolean): BoardGrid {
  const n = view.board_size;
  const size = 2 * n - 1;
  const flip = view.human_player === 0; // put the human's (P0) home row at the bottom
  const mp = (r: number, c: number): [number, number] => (flip ? [n - 1 - r, c] : [r, c]);
  // A wall's (top,left) corner spans two pawn rows, so its flipped display row
  // is one less than a pawn's: (n-1-r) - 1 = n-2-r.
  const mw = (r: number, c: number): [number, number] => (flip ? [n - 2 - r, c] : [r, c]);

  const cells: GridCell[] = [];
  const idx = new Map<string, GridCell>();
  for (let gr = 0; gr < size; gr++) {
    for (let gc = 0; gc < size; gc++) {
      const rowEven = gr % 2 === 0;
      const colEven = gc % 2 === 0;
      const base =
        rowEven && colEven
          ? "pawn-cell"
          : !rowEven && colEven
            ? "wall-h-half"
            : rowEven && !colEven
              ? "wall-v-half"
              : "post";
      const cell: GridCell = {
        gr, gc, base, pawn: null, placed: null,
        legalMoveIndex: null, wall: null, lastMove: false, lastWall: false,
      };
      cells.push(cell);
      idx.set(cellKey(gr, gc), cell);
    }
  }
  const at = (gr: number, gc: number): GridCell => idx.get(cellKey(gr, gc))!;

  const [p1r, p1c] = mp(view.p1_pos[0], view.p1_pos[1]);
  const [p2r, p2c] = mp(view.p2_pos[0], view.p2_pos[1]);
  at(2 * p1r, 2 * p1c).pawn = 0;
  at(2 * p2r, 2 * p2c).pawn = 1;

  for (const w of view.walls) {
    const [dr, dc] = mw(w.row, w.col);
    for (const [gr, gc] of wallGroup(dr, dc, w.orientation)) at(gr, gc).placed = w.orientation;
  }

  if (view.last_action) {
    const la = view.last_action;
    if (la.kind === "move") {
      const [dr, dc] = mp(la.to[0], la.to[1]);
      at(2 * dr, 2 * dc).lastMove = true;
    } else {
      const [dr, dc] = mw(la.row, la.col);
      for (const [gr, gc] of wallGroup(dr, dc, la.orientation)) at(gr, gc).lastWall = true;
    }
  }

  // `== null` catches both null and undefined (older serializers emit undefined).
  const humanTurn = view.winner == null && view.current_player === view.human_player;
  if (interactive && humanTurn) {
    for (const a of view.legal_actions) {
      if (a.kind === "move") {
        const [dr, dc] = mp(a.to[0], a.to[1]);
        at(2 * dr, 2 * dc).legalMoveIndex = a.index;
      } else {
        const [dr, dc] = mw(a.row, a.col);
        const group = wallGroup(dr, dc, a.orientation);
        at(group[0][0], group[0][1]).wall = {
          index: a.index,
          orientation: a.orientation,
          group: group.map(([r, c]) => cellKey(r, c)),
        };
      }
    }
  }

  return { n, size, cells };
}
