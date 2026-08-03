import { describe, expect, it } from "vitest";
import { buildBoardGrid, cellKey } from "./boardGrid";
import type { StateView } from "./types";

// Server coords: player 0 starts at row 0 (top), player 1 at row N-1 (bottom).
// Human is player 0, so the grid flips vertically -> P0 shown at the bottom.
function view(over: Partial<StateView> = {}): StateView {
  return {
    board_size: 5, max_walls: 2, max_steps: 50, current_player: 0,
    p1_pos: [0, 2], p2_pos: [4, 2], p1_walls: 2, p2_walls: 2,
    walls: [], legal_actions: [], completed_steps: 0, winner: null,
    human_player: 0, last_action: null, move_history: [],
    ...over,
  };
}

function cell(g: ReturnType<typeof buildBoardGrid>, gr: number, gc: number) {
  return g.cells.find((c) => c.gr === gr && c.gc === gc)!;
}

describe("buildBoardGrid", () => {
  it("produces a (2N-1)^2 grid with the right cell kinds", () => {
    const g = buildBoardGrid(view(), true);
    expect(g.size).toBe(9);
    expect(g.cells).toHaveLength(81);
    expect(cell(g, 0, 0).base).toBe("pawn-cell");
    expect(cell(g, 1, 0).base).toBe("wall-h-half");
    expect(cell(g, 0, 1).base).toBe("wall-v-half");
    expect(cell(g, 1, 1).base).toBe("post");
  });

  it("flips so the human (P0) sits at the visual bottom and the AI (P1) at the top", () => {
    const g = buildBoardGrid(view(), true);
    // P0 server row 0 -> display row 4 -> grid row 8 (bottom).
    expect(cell(g, 8, 4).pawn).toBe(0);
    // P1 server row 4 -> display row 0 -> grid row 0 (top).
    expect(cell(g, 0, 4).pawn).toBe(1);
  });

  it("maps a legal forward move to the mirrored cell (above the pawn = up)", () => {
    const g = buildBoardGrid(
      view({ legal_actions: [{ kind: "move", index: 7, to: [1, 2] }] }),
      true,
    );
    // to (1,2) -> display (3,2) -> grid (6,4), which is above the pawn at (8,4).
    expect(cell(g, 6, 4).legalMoveIndex).toBe(7);
  });

  it("puts a legal wall on its anchor cell with the 3-cell hover group", () => {
    const g = buildBoardGrid(
      view({ legal_actions: [{ kind: "wall", index: 30, row: 1, col: 1, orientation: "v" }] }),
      true,
    );
    // wall (1,1) -> display (2,1); vertical group anchored at (4,3).
    const anchor = cell(g, 4, 3);
    expect(anchor.wall?.index).toBe(30);
    expect(anchor.wall?.group).toEqual([cellKey(4, 3), cellKey(5, 3), cellKey(6, 3)]);
    // Non-anchor cells of the group carry no wall marker.
    expect(cell(g, 5, 3).wall).toBeNull();
  });

  it("marks placed walls across their whole group", () => {
    const g = buildBoardGrid(view({ walls: [{ row: 1, col: 1, orientation: "h" }] }), true);
    // wall (1,1) h -> display (2,1); horizontal group [(5,2),(5,3),(5,4)].
    expect(cell(g, 5, 2).placed).toBe("h");
    expect(cell(g, 5, 3).placed).toBe("h");
    expect(cell(g, 5, 4).placed).toBe("h");
  });

  it("omits legal markers when not interactive or not the human's turn", () => {
    const acts: StateView["legal_actions"] = [{ kind: "move", index: 7, to: [1, 2] }];
    expect(buildBoardGrid(view({ legal_actions: acts }), false).cells.every((c) => c.legalMoveIndex === null)).toBe(true);
    expect(
      buildBoardGrid(view({ legal_actions: acts, current_player: 1 }), true).cells.every((c) => c.legalMoveIndex === null),
    ).toBe(true);
  });
});
