import type { Orientation, StateView, WallEntry } from "./types";

export interface BoardModel {
  size: number;
  pawns: [[number, number], [number, number]];
  walls: WallEntry[];
  /** Action index for the legal move landing on (r,c), or undefined. */
  moveActionAt(r: number, c: number): number | undefined;
  /** Action index for the legal wall at (r,c,orientation), or undefined. */
  wallActionAt(r: number, c: number, o: Orientation): number | undefined;
}

export function deriveBoard(view: StateView): BoardModel {
  const moves = new Map<string, number>();
  const walls = new Map<string, number>();
  for (const a of view.legal_actions) {
    if (a.kind === "move") moves.set(`${a.to[0]},${a.to[1]}`, a.index);
    else walls.set(`${a.row},${a.col},${a.orientation}`, a.index);
  }
  return {
    size: view.board_size,
    pawns: [view.p1_pos, view.p2_pos],
    walls: view.walls,
    moveActionAt: (r, c) => moves.get(`${r},${c}`),
    wallActionAt: (r, c, o) => walls.get(`${r},${c},${o}`),
  };
}
