// Quoridor play server — vanilla JS frontend.
//
// Coordinates: the server speaks absolute Quoridor coordinates with (0,0) at
// the top-left and player 0 starting on row 0. We always render the human's
// home row at the bottom, so when `human_player == 1` we mirror coordinates
// 180° before placing anything on the board grid.
//
// The board is a (2N-1) x (2N-1) CSS grid alternating pawn cells, wall slots,
// and wall posts. Server `legal_actions` already carry the kind/coords so the
// client never has to mirror Python's action-encoding logic.

const STATE = {
  cfg: null,    // /api/config response
  gameId: null,
  view: null,   // last StateView from server
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

async function sendMove(actionIndex) {
  if (STATE.pending || !STATE.gameId) return;
  clearError();
  setPending(true);
  try {
    const data = await fetchJson(`/api/games/${STATE.gameId}/move`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action_index: actionIndex }),
    });
    STATE.view = data.state;
    render();
  } catch (e) {
    showError("Move rejected: " + e.message);
  } finally {
    setPending(false);
  }
}

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

function mirrorPawn(r, c) {
  const N = STATE.view.board_size;
  return STATE.view.human_player === 1 ? [N - 1 - r, N - 1 - c] : [r, c];
}

// A wall at server (r, c) sits between rows r and r+1, spanning cols c and
// c+1. Under 180 deg rotation it lives between rows (N-2-r) and (N-1-r),
// spanning cols (N-2-c) and (N-1-c) -- same orientation.
function mirrorWall(r, c) {
  const N = STATE.view.board_size;
  return STATE.view.human_player === 1 ? [N - 2 - r, N - 2 - c] : [r, c];
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

  // Pawns -- keep colors tied to the server player index so walls-left
  // counters line up with the pawn colors regardless of orientation.
  const [p1r, p1c] = mirrorPawn(v.p1_pos[0], v.p1_pos[1]);
  const [p2r, p2c] = mirrorPawn(v.p2_pos[0], v.p2_pos[1]);
  cells[2 * p1r][2 * p1c].appendChild(make("div", { class: "pawn p1" }));
  cells[2 * p2r][2 * p2c].appendChild(make("div", { class: "pawn p2" }));

  // Placed walls
  for (const w of v.walls) {
    const [dr, dc] = mirrorWall(w.row, w.col);
    if (w.orientation === "h") {
      cells[2 * dr + 1][2 * dc].classList.add("wall-placed-h");
      cells[2 * dr + 1][2 * dc + 1].classList.add("wall-placed-h");
      cells[2 * dr + 1][2 * dc + 2].classList.add("wall-placed-h");
    } else {
      cells[2 * dr][2 * dc + 1].classList.add("wall-placed-v");
      cells[2 * dr + 1][2 * dc + 1].classList.add("wall-placed-v");
      cells[2 * dr + 2][2 * dc + 1].classList.add("wall-placed-v");
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
      if (la.orientation === "h") {
        cells[2 * dr + 1][2 * dc].classList.add("last-wall");
        cells[2 * dr + 1][2 * dc + 1].classList.add("last-wall");
        cells[2 * dr + 1][2 * dc + 2].classList.add("last-wall");
      } else {
        cells[2 * dr][2 * dc + 1].classList.add("last-wall");
        cells[2 * dr + 1][2 * dc + 1].classList.add("last-wall");
        cells[2 * dr + 2][2 * dc + 1].classList.add("last-wall");
      }
    }
  }

  // Click handlers on legal actions -- only when it's the human's turn.
  const humanTurn = v.winner === null && v.current_player === v.human_player;
  if (humanTurn) {
    for (const a of v.legal_actions) {
      if (a.kind === "move") {
        const [dr, dc] = mirrorPawn(a.to[0], a.to[1]);
        const cell = cells[2 * dr][2 * dc];
        cell.classList.add("legal-move");
        cell.addEventListener("click", () => sendMove(a.index));
      } else {
        const [dr, dc] = mirrorWall(a.row, a.col);
        const tag = (gr, gc) => {
          const c = cells[gr][gc];
          c.classList.add(`legal-wall-${a.orientation}`);
          c.addEventListener("click", () => sendMove(a.index));
        };
        if (a.orientation === "h") {
          tag(2 * dr + 1, 2 * dc);
          tag(2 * dr + 1, 2 * dc + 1);
          tag(2 * dr + 1, 2 * dc + 2);
        } else {
          tag(2 * dr, 2 * dc + 1);
          tag(2 * dr + 1, 2 * dc + 1);
          tag(2 * dr + 2, 2 * dc + 1);
        }
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
  $("#walls-p1").textContent = v.p1_walls;
  $("#walls-p2").textContent = v.p2_walls;
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
