import { describe, expect, it } from "vitest";
import { deriveBoard } from "./board";
import type { StateView } from "./types";

function baseView(over: Partial<StateView> = {}): StateView {
  return {
    board_size: 5, max_walls: 2, max_steps: 50, current_player: 0,
    p1_pos: [4, 2], p2_pos: [0, 2], p1_walls: 2, p2_walls: 2,
    walls: [], legal_actions: [], completed_steps: 0, winner: null,
    human_player: 0, last_action: null, move_history: [],
    ...over,
  };
}

describe("deriveBoard", () => {
  it("maps legal move destinations to their action index", () => {
    const view = baseView({
      legal_actions: [
        { kind: "move", index: 7, to: [3, 2] },
        { kind: "wall", index: 30, row: 1, col: 1, orientation: "v" },
      ],
    });
    const b = deriveBoard(view);
    expect(b.moveActionAt(3, 2)).toBe(7);
    expect(b.moveActionAt(0, 0)).toBeUndefined();
    expect(b.wallActionAt(1, 1, "v")).toBe(30);
  });

  it("exposes pawn positions and placed walls", () => {
    const view = baseView({ walls: [{ row: 2, col: 2, orientation: "h" }] });
    const b = deriveBoard(view);
    expect(b.pawns[0]).toEqual([4, 2]);
    expect(b.pawns[1]).toEqual([0, 2]);
    expect(b.walls).toHaveLength(1);
    expect(b.size).toBe(5);
  });
});
