// Quoridor play server -- vanilla JS frontend.
//
// Coordinates: the server speaks absolute Quoridor coordinates with (0,0)
// at the top-left and player 0 starting on row 0. We always render the
// human's home row at the bottom, so when `human_player == 1` we mirror
// coordinates 180 deg before placing anything on the board grid.
//
// The board is a (2N-1) x (2N-1) CSS grid alternating pawn cells, wall
// slots, and wall posts. Server `legal_actions` already carry the
// kind/coords so the client never has to mirror Python's action-encoding
// logic.
//
// Move flow uses an optimistic update: when the human clicks an action,
// we immediately apply a local approximation of the new state (move the
// pawn, place the wall, flip the turn) so the UI reflects the click
// without waiting for the AI. When the server responds with the
// authoritative post-AI state, we replace the local state and re-render.

const STATE = {
  cfg: null,    // /api/config response
  gameId: null,
  view: null,   // last StateView (server- or optimistically-derived)
  pending: false,
};

const $ = (sel) => document.querySelector(sel);

function make(tag, attrs = {}, children = []) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") el.className = v;
    else if (k === "text") el.textContent = v;
    else if (k.startsWith("on") && typeof v === "function") {
      el.addEventListener(k.slice(2), v);
    } else {
      el.setAttribute(k, v);
    }
  }
  for (const c of children) el.appendChild(c);
  return el;
}

async function fetchJson(url, options) {
  const resp = await fetch(url, options);
  if (!resp.ok) {
    let detail = resp.statusText;
    try {
      const body = await resp.json();
      if (body && body.error) detail = body.error;
    } catch (_) { /* non-JSON error body */ }
    throw new Error(detail);
  }
  return resp.json();
}

// ---- setup ----

async function init() {
  try {
    STATE.cfg = await fetchJson("/api/config");
    renderSetup();
  } catch (e) {
    showError("Failed to load /api/config: " + e.message);
  }
}

function renderSetup() {
  const sel = $("#model-select");
  sel.innerHTML = "";
  for (const name of STATE.cfg.models) {
    sel.appendChild(make("option", { value: name, text: name }));
  }
  if (STATE.cfg.models.length === 0) {
    sel.appendChild(make("option", { value: "", text: "(no models found)" }));
    $("#new-game-button").disabled = true;
  }

  const slider = $("#mcts-n");
  slider.value = STATE.cfg.default_mcts_n;
  $("#mcts-n-display").textContent = slider.value;

  $("#board-size-display").textContent =
    `${STATE.cfg.board_size}x${STATE.cfg.board_size}, ${STATE.cfg.max_walls} walls each`;
}

async function startGame() {
  const body = {
    model: $("#model-select").value,
    mcts_n: parseInt($("#mcts-n").value, 10),
    human_player: parseInt(
      document.querySelector('input[name="human-player"]:checked').value,
      10,
    ),
  };
  clearError();
  setPending(true);
  try {
    const data = await fetchJson("/api/games", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    STATE.gameId = data.game_id;
    STATE.view = data.state;
    render();
  } catch (e) {
    showError("New game failed: " + e.message);
  } finally {
    setPending(false);
  }
}

// ---- optimistic update ----

// Apply a local approximation of `action` to `view` so the UI reflects
// the human's move immediately. The server's response will overwrite
// this with the authoritative state, which also includes the AI's reply.
//
// We do not attempt to recompute the post-action legal_actions; clicks
// are disabled while we wait on the server.
function applyOptimistic(view, action) {
  const o = JSON.parse(JSON.stringify(view));
  const mover = o.current_player;
  if (action.kind === "move") {
    if (mover === 0) o.p1_pos = action.to;
    else o.p2_pos = action.to;
    // Detect immediate win: reaching the opposite home row ends the game.
    const N = o.board_size;
    const goalRow = mover === 0 ? N - 1 : 0;
    if (action.to[0] === goalRow) o.winner = mover;
  } else {
    o.walls.push({
      row: action.row,
      col: action.col,
      orientation: action.orientation,
    });
    if (mover === 0) o.p1_walls -= 1;
    else o.p2_walls -= 1;
  }
  o.last_action = { ...action };
  o.move_history = [...o.move_history, action.index];
  o.current_player = 1 - mover;
  o.completed_steps += 1;
  o.legal_actions = [];
  return o;
}

async function sendMove(action) {
  if (STATE.pending || !STATE.gameId) return;
  clearError();

  // Optimistic render: show the human's move now.
  STATE.view = applyOptimistic(STATE.view, action);
  render();
  setPending(true);

  try {
    const data = await fetchJson(`/api/games/${STATE.gameId}/move`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action_index: action.index }),
    });
    STATE.view = data.state;
    render();
  } catch (e) {
    showError("Move rejected: " + e.message);
    // Roll back to the truth from the server.
    try {
      const fresh = await fetchJson(`/api/games/${STATE.gameId}`);
      STATE.view = fresh.state;
      render();
    } catch (_) { /* leave optimistic view; user will retry */ }
  } finally {
    setPending(false);
  }
}

// ---- ui-state helpers ----

function setPending(p) {
  STATE.pending = p;
  $("#spinner").hidden = !p;
}

function showError(msg) {
  const el = $("#error-display");
  el.textContent = msg;
  el.hidden = false;
}
function clearError() {
  const el = $("#error-display");
  el.textContent = "";
  el.hidden = true;
}

// ---- coordinate transforms ----
//
// We always render the human's home row at the bottom. Server P0
// starts at server row 0; P1 starts at server row N-1. CSS-grid row 0
// renders at the top of the screen, so:
//
//   - human == P0 -> vertical flip (row N-1 - r), cols unchanged.
//   - human == P1 -> no flip (P1 already at server row N-1 = bottom).
//
// Cols stay put so a wall's "left" half in server space stays the
// left half in display space; the anchor/extends-right rule the
// hover logic below relies on then needs no further inversion.

function mirrorPawn(r, c) {
  const N = STATE.view.board_size;
  return STATE.view.human_player === 0 ? [N - 1 - r, c] : [r, c];
}

// Server walls are indexed by their (top, left) corner. After a
// vertical flip, what was server-top becomes display-bottom -- so the
// display (top, left) corner of the wall is at row N-2-r (one less
// than N-1-r because the wall spans two pawn rows).
function mirrorWall(r, c) {
  const N = STATE.view.board_size;
  return STATE.view.human_player === 0 ? [N - 2 - r, c] : [r, c];
}

// Return the (gr, gc) grid coordinates of the 3 cells that make up a
// wall at display (dr, dc): the two halves and the post between them.
function wallGroupCells(dr, dc, orientation) {
  if (orientation === "h") {
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

// ---- render ----

function render() {
  const v = STATE.view;
  const N = v.board_size;
  const size = 2 * N - 1;
  const board = $("#board");
  board.innerHTML = "";

  // Alternating column/row sizes: pawn cell, post, pawn cell, post, ...
  const tracks = Array.from({ length: size }, (_, i) =>
    i % 2 === 0 ? "var(--pawn-size)" : "var(--post-size)",
  ).join(" ");
  board.style.gridTemplateColumns = tracks;
  board.style.gridTemplateRows = tracks;

  // Build the grid and remember each cell so we can decorate.
  const cells = [];
  for (let gr = 0; gr < size; gr++) {
    cells.push([]);
    for (let gc = 0; gc < size; gc++) {
      const isRowEven = gr % 2 === 0;
      const isColEven = gc % 2 === 0;
      let cls = "post";
      if (isRowEven && isColEven) cls = "pawn-cell";
      else if (!isRowEven && isColEven) cls = "wall-h-half";
      else if (isRowEven && !isColEven) cls = "wall-v-half";
      const el = make("div", { class: `cell ${cls}` });
      cells[gr].push(el);
      board.appendChild(el);
    }
  }

  // Pawns -- colors stay tied to the server player index so the
  // walls-left counters in the side panel always match the pawn colors
  // regardless of orientation.
  const [p1r, p1c] = mirrorPawn(v.p1_pos[0], v.p1_pos[1]);
  const [p2r, p2c] = mirrorPawn(v.p2_pos[0], v.p2_pos[1]);
  cells[2 * p1r][2 * p1c].appendChild(make("div", { class: "pawn p1" }));
  cells[2 * p2r][2 * p2c].appendChild(make("div", { class: "pawn p2" }));

  // Placed walls
  for (const w of v.walls) {
    const [dr, dc] = mirrorWall(w.row, w.col);
    const placedCls = `wall-placed-${w.orientation}`;
    for (const [gr, gc] of wallGroupCells(dr, dc, w.orientation)) {
      cells[gr][gc].classList.add(placedCls);
    }
  }

  // Last-action highlight
  if (v.last_action) {
    const la = v.last_action;
    if (la.kind === "move") {
      const [dr, dc] = mirrorPawn(la.to[0], la.to[1]);
      cells[2 * dr][2 * dc].classList.add("last-move");
    } else {
      const [dr, dc] = mirrorWall(la.row, la.col);
      for (const [gr, gc] of wallGroupCells(dr, dc, la.orientation)) {
        cells[gr][gc].classList.add("last-wall");
      }
    }
  }

  // Click handlers on legal actions -- only when it's the human's turn.
  // sendMove() ignores clicks while STATE.pending is true, so we don't
  // need to also gate handler attachment on pending.
  //
  // Anchor-only attachment: each wall is interactive *only* on its
  // display-top-left cell -- the left half for horizontal walls, the
  // top half for vertical walls. This means:
  //   - Each grid cell triggers at most one wall, so hovering doesn't
  //     light up two overlapping walls at once.
  //   - Intersection posts (the small squares between four pawn cells)
  //     never trigger a wall, since they are never any wall's anchor.
  //   - The wall the user sees on hover is the one that "starts here
  //     and extends right (H) or down (V)" in display coordinates.
  const humanTurn = v.winner === null && v.current_player === v.human_player;
  if (humanTurn) {
    for (const a of v.legal_actions) {
      if (a.kind === "move") {
        const [dr, dc] = mirrorPawn(a.to[0], a.to[1]);
        const cell = cells[2 * dr][2 * dc];
        cell.classList.add("legal-move");
        cell.addEventListener("click", () => sendMove(a));
      } else {
        const [dr, dc] = mirrorWall(a.row, a.col);
        const group = wallGroupCells(dr, dc, a.orientation).map(
          ([gr, gc]) => cells[gr][gc],
        );
        const anchor = group[0]; // first cell is the display top-left
        anchor.classList.add(`legal-wall-${a.orientation}`);
        anchor.addEventListener("click", () => sendMove(a));
        anchor.addEventListener("mouseenter", () => {
          for (const c of group) c.classList.add("wall-hover");
        });
        anchor.addEventListener("mouseleave", () => {
          for (const c of group) c.classList.remove("wall-hover");
        });
      }
    }
  }

  // Status panel
  const turnEl = $("#turn-display");
  if (v.winner !== null) {
    turnEl.textContent = "Game over";
  } else if (v.current_player === v.human_player) {
    turnEl.textContent = "Your move";
  } else {
    turnEl.textContent = "AI thinking";
  }
  // Map walls-left by role, not server index: when the human is P1 we
  // want "You" to show p2_walls, not p1_walls.
  const youWalls = v.human_player === 0 ? v.p1_walls : v.p2_walls;
  const aiWalls  = v.human_player === 0 ? v.p2_walls : v.p1_walls;
  $("#walls-you").textContent = youWalls;
  $("#walls-ai").textContent = aiWalls;
  $("#completed-steps").textContent = `${v.completed_steps} / ${v.max_steps}`;

  // Game-over banner
  const banner = $("#game-over-banner");
  if (v.winner === null) {
    if (v.completed_steps >= v.max_steps) {
      banner.style.display = "block";
      banner.textContent = "Draw (move limit reached)";
    } else {
      banner.style.display = "none";
    }
  } else {
    banner.style.display = "block";
    banner.textContent = v.winner === v.human_player ? "You won!" : "AI won";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  init();
  $("#mcts-n").addEventListener("input", (e) => {
    $("#mcts-n-display").textContent = e.target.value;
  });
  $("#new-game-button").addEventListener("click", startGame);
});
