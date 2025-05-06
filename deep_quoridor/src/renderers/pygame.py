import time
from dataclasses import dataclass, field
from enum import Enum
from threading import Event
from typing import List, Optional, Tuple

import numpy as np
import pygame
from agents import ActionLog, Agent
from agents.human import HumanAgent
from arena import GameResult
from quoridor import Action, MoveAction, Player, WallAction, WallOrientation

from renderers import Renderer

WALL_TO_CELL_RATIO = 5


def rgb(hexs):
    return [tuple(int(hex[i : i + 2], 16) for i in (0, 2, 4)) for hex in hexs]


# Borrowed from https://yeun.github.io/open-color/
PALETTE_GRAY = rgb(["212529", "343a40", "495057", "868e96", "adb5bd", "ced4da", "dee2e6", "e9ecef", "f1f3f5", "f8f9fa"])
PALETTE_TEAL = rgb(["087f5b", "099268", "0ca678", "12b886", "20c997", "38d9a9", "63e6be", "96f2d7", "c3fae8", "e6fcf5"])
PALETTE_RED = rgb(["c92a2a", "e03131", "f03e3e", "fa5252", "ff6b6b", "ff8787", "ffa8a8", "ffc9c9", "ffe3e3", "fff5f5"])
PALETTE_ORANGE = rgb(
    ["d9480f", "e8590c", "f76707", "fd7e14", "ff922b", "ffa94d", "ffc078", "ffd8a8", "ffe8cc", "fff4e6"]
)
PALETTE_BLUE = rgb(["1864ab", "1971c2", "1c7ed6", "228be6", "339af0", "4dabf7", "74c0fc", "a5d8ff", "d0ebff", "e7f5ff"])


PALETTES = [PALETTE_TEAL, PALETTE_RED, PALETTE_ORANGE, PALETTE_BLUE]

COLOR_BOARD = (222, 184, 135)
COLOR_GRID = (160, 120, 95)
COLOR_WALL = (70, 40, 20)
COLOR_PREVIEW = PALETTE_TEAL[1]


COLOR_BUTTON = PALETTE_GRAY[5]
COLOR_SCREEN = (255, 255, 255)
COLOR_PLAYER1 = PALETTE_GRAY[1]
COLOR_PLAYER2 = PALETTE_GRAY[6]
COLOR_ACTIVE_PLAYER = PALETTE_RED[0]

COLOR_BLACK = (0, 0, 0)
COLOR_GREY = (128, 128, 128)


class GameState(Enum):
    READY = 0
    RUNNING = 1
    PAUSED = 2
    STEP = 3
    FINISHED = 4


@dataclass
class BoardState:
    p1_position: Tuple[int, int]
    p1_walls_remaining: int
    p2_position: Tuple[int, int]
    p2_walls_remaining: int
    is_p1_turn: bool
    walls: np.ndarray
    result: Optional[GameResult] = None
    log: Optional[ActionLog] = None


@dataclass
class HumanInput:
    enabled: bool = False
    valid_moves: set = field(default_factory=set)
    hover_action: Action | None = None
    click_action: Action | None = None


class PygameQuoridor:
    _instance = None

    def __init__(self):
        self.board_pixels = 530
        self.board_position = (50, 50)
        self.log_actions = True
        self.buttons = {
            "start_pause": {"text": "Start", "method": self._handle_start_pause},
            "next": {"text": "Step", "method": self._handle_next},
            "log": {"text": "Log off", "method": self._handle_log},
        }
        self.state: Optional[BoardState] = None
        self.human_input = HumanInput
        self.running = True
        PygameQuoridor._instance = self

    @classmethod
    def instance(cls) -> "PygameQuoridor":
        assert cls._instance
        return cls._instance

    @classmethod
    def active(cls) -> bool:
        return cls._instance is not None

    def start(self, game):
        self.board_size = game.board_size
        self.compute_sizes()
        self.game_state = GameState.READY

    def start_game(self, player1: str, player2: str, state: BoardState):
        self.player1 = player1
        self.player2 = player2
        self.state = state

    def end_game(self, game, result: GameResult):
        self.game_state = GameState.FINISHED
        self.update_result(result)

    def compute_sizes(self):
        self.wall_size = self.board_pixels // ((self.board_size - 1) + self.board_size * WALL_TO_CELL_RATIO)
        self.cell_size = WALL_TO_CELL_RATIO * self.wall_size
        self.board_pixels = (self.board_size - 1) * self.wall_size + self.board_size * self.cell_size

    def update_players_state(
        self,
        p1_position: Tuple[int, int],
        p1_walls_remaining: int,
        p2_position: Tuple[int, int],
        p2_walls_remaining: int,
        walls: np.ndarray,
    ):
        assert self.state
        self.state.p1_position = p1_position
        self.state.p1_walls_remaining = p1_walls_remaining
        self.state.p2_position = p2_position
        self.state.p2_walls_remaining = p2_walls_remaining
        self.state.walls = walls

    def update_turn(self, is_p1_turn: bool):
        assert self.state
        self.state.is_p1_turn = is_p1_turn

    def update_log(self, log: ActionLog | None):
        assert self.state
        self.state.log = log

    def update_result(self, result: GameResult):
        assert self.state
        self.state.result = result

    def _cell_pos(self, row, col):
        x = col * (self.cell_size + self.wall_size) + self.board_position[0]
        y = row * (self.cell_size + self.wall_size) + self.board_position[1]
        return [x, y, x + self.cell_size, y + self.cell_size]

    def _cell_center(self, row, col):
        x0, y0, x1, y1 = self._cell_pos(row, col)
        return [(x0 + x1) // 2, (y0 + y1) // 2]

    def _draw_player(self, color, row, col, active):
        cx, cy = self._cell_center(row, col)
        pygame.draw.circle(self.screen, color, (cx, cy), self.cell_size * 0.4)
        if active:
            pygame.draw.circle(self.screen, COLOR_ACTIVE_PLAYER, (cx, cy), self.cell_size * 0.4 + 3, width=2)

    def _draw_board(self):
        mx, my = self.board_position  # margins

        # Background of the board
        rect = pygame.Rect(mx, my, self.board_pixels, self.board_pixels)
        pygame.draw.rect(self.screen, COLOR_BOARD, rect)

        # A rectangle around the board
        rect = pygame.Rect(mx - 2, my - 2, self.board_pixels + 4, self.board_pixels + 4)
        pygame.draw.rect(self.screen, COLOR_GRID, rect, width=4, border_radius=4)

        # Vertical lines
        for i in range(self.board_size - 1):
            _, _, x1, _ = self._cell_pos(0, i)
            xc = x1 + self.wall_size // 2
            pygame.draw.line(self.screen, COLOR_GRID, (xc, my), (xc, my + self.board_pixels), self.wall_size)

        # Horizontal lines
        for i in range(self.board_size - 1):
            _, _, _, y1 = self._cell_pos(i, 0)
            yc = y1 + self.wall_size // 2
            pygame.draw.line(self.screen, COLOR_GRID, (mx, yc), (mx + self.board_pixels, yc), self.wall_size)

    def _draw_wall(self, row, col, is_horizontal: bool, is_preview: bool = False):
        offset1 = offset2 = 0
        if is_horizontal:
            offset1 = 1
        else:
            offset2 = 1
        _, y0, x1, _ = self._cell_pos(row + offset1, col + offset1)
        x0, _, _, y1 = self._cell_pos(row + offset2, col + offset2)

        color = COLOR_PREVIEW if is_preview else COLOR_WALL

        pygame.draw.rect(self.screen, color, (x0, y0, x1 - x0, y1 - y0), border_radius=4)

    def _draw_walls(self):
        assert self.state
        m = self.board_size - 1

        for row in range(m):
            for col in range(m):
                if self.state.walls[row, col, 0] == 1 and (row == 0 or self.state.walls[row - 1, col, 0] == 0):
                    self._draw_wall(row, col, is_horizontal=False)

                if self.state.walls[row, col, 1] == 1 and (col == 0 or self.state.walls[row, col - 1, 1] == 0):
                    self._draw_wall(row, col, is_horizontal=True)

    def _draw_players_and_data(self):
        assert self.state is not None
        ys = [self.board_position[1] - 38, self.board_position[1] + self.board_pixels + 6]
        texts = [
            f"P0 {self.player1} ({self.state.p1_walls_remaining})",
            f"P1 {self.player2} ({self.state.p2_walls_remaining})",
        ]
        actives = [self.state.is_p1_turn, not self.state.is_p1_turn]
        colors = [COLOR_PLAYER1, COLOR_PLAYER2]
        positions = [self.state.p1_position, self.state.p2_position]

        for y, text, active, color, position in zip(ys, texts, actives, colors, positions):
            # Draw the player on the board
            self._draw_player(color, position[0], position[1], active)

            # Write the name of the player and the walls remaining
            text_surface = self.font24.render(text, True, COLOR_BLACK)
            x = self.board_position[0] + (self.board_pixels - text_surface.get_width()) // 2
            self.screen.blit(text_surface, (x, y))

            # Draw the player color and an active circle next to the name
            pygame.draw.circle(self.screen, color, (x - 15, y + 15), 5)
            if active:
                pygame.draw.circle(self.screen, COLOR_ACTIVE_PLAYER, (x - 15, y + 15), radius=9, width=2)

    def _draw_buttons(self):
        text = {
            GameState.READY: "Start",
            GameState.RUNNING: "Pause",
            GameState.PAUSED: "Resume",
            GameState.FINISHED: "Done",
        }
        if self.game_state in text:
            self.buttons["start_pause"]["text"] = text[self.game_state]

        self.buttons["log"]["text"] = "Log off" if self.log_actions else "Log on"

        btn_width = 100
        btn_spacing = 20
        total_width = btn_width * len(self.buttons) + btn_spacing * (len(self.buttons) - 1)
        x = self.board_position[0] + (self.board_pixels - total_width) // 2
        y = self.board_position[1] + self.board_pixels + 50

        for _, button in self.buttons.items():
            button["rect"] = pygame.Rect(x, y, btn_width, 24)
            pygame.draw.rect(self.screen, COLOR_BUTTON, button["rect"], border_radius=4)
            text = self.font16.render(button["text"], True, COLOR_BLACK)
            self.screen.blit(text, dest=text.get_rect(center=button["rect"].center))
            x += btn_width + btn_spacing

    def _draw_result(self):
        assert self.state
        result = self.state.result
        if result is None:
            return

        text = f"Game Over! {result.winner} won in {result.steps} moves."
        text_surface = self.font24.render(text, True, COLOR_BLACK)
        self.screen.blit(text_surface, (self.board_position[0] + 15, self.board_position[1] + self.board_pixels + 80))

    def _draw_human_hover(self):
        if self.human_input.hover_action is None or not self.human_input.enabled:
            return

        if isinstance(self.human_input.hover_action, MoveAction):
            row, col = self.human_input.hover_action.destination
            self._draw_player(COLOR_PREVIEW, row, col, False)
        elif isinstance(self.human_input.hover_action, WallAction):
            row, col = self.human_input.hover_action.position
            is_horizontal = self.human_input.hover_action.orientation == WallOrientation.HORIZONTAL
            self._draw_wall(row, col, is_horizontal, is_preview=True)

    def draw_screen(self):
        if self.state is None:
            return

        self.screen.fill(COLOR_SCREEN)
        self._draw_board()
        self._draw_players_and_data()
        self._draw_walls()
        self._draw_buttons()
        self._draw_result()
        self._draw_log()
        self._draw_human_hover()

    def _draw_log_action(self, action: Action, text: str, color):
        wall_len = 2 * self.cell_size + self.wall_size

        if isinstance(action, MoveAction):
            x0, y0, x1, y1 = self._cell_pos(*action.destination)
            x = x0 + self.cell_size // 2
            y = y0 + self.cell_size // 2
            pygame.draw.circle(self.screen, color, (x, y), self.cell_size * 0.3)

        elif isinstance(action, WallAction):
            x0, y0, x1, y1 = self._cell_pos(*action.position)
            if action.orientation == WallOrientation.VERTICAL:
                x = x1 + self.wall_size // 2
                y = y0 + self.cell_size // 2
                pygame.draw.rect(
                    self.screen, color, (x1 + 1, y0 + 1, self.wall_size - 1, wall_len - 1), border_radius=4
                )
            else:
                x = x0 + self.cell_size // 2
                y = y1 + self.wall_size // 2
                pygame.draw.rect(
                    self.screen, color, (x0 + 1, y1 + 1, wall_len - 1, self.wall_size - 1), border_radius=4
                )

        text_surface = self.font12.render(text, True, COLOR_BLACK)
        self.screen.blit(text_surface, dest=text_surface.get_rect(center=(x, y)))

    def _draw_log_action_score_ranking(self, entry: ActionLog.ActionScoreRanking, palette_id: int):
        palette_size = len(PALETTES[palette_id])

        coeff = palette_size / max([r for r, _, _ in entry.ranking])
        for ranking, action, score in entry.ranking:
            text = f"{score:0.2f}" if score < 10 else f"{int(score)}"
            color = PALETTES[palette_id][int((ranking - 1) * coeff)]
            self._draw_log_action(action, text, color)

        return palette_id + 1

    def _draw_path(self, path: list[tuple[int, int]], path_id):
        for (r1, c1), (r2, c2) in zip(path, path[1:]):
            p1 = self._cell_center(r1, c1)
            p2 = self._cell_center(r2, c2)
            color = PALETTE_GRAY[4]
            pygame.draw.line(self.screen, color, p1, p2, width=3)
            pygame.draw.circle(self.screen, color, p2, radius=7)

        return path_id + 1  # todo

    def _draw_log(self):
        assert self.state
        log = self.state.log
        if not self.log_actions or log is None:
            return

        palette_id_asr = 0
        path_id = 0

        for entry in log.records:
            if isinstance(entry, ActionLog.ActionScoreRanking):
                palette_id_asr = self._draw_log_action_score_ranking(entry, palette_id_asr)
            if isinstance(entry, ActionLog.ActionText):
                self._draw_log_action(entry.action, entry.text, PALETTE_GRAY[4])
            if isinstance(entry, ActionLog.Path):
                path_id = self._draw_path(entry.path, path_id)

    def _handle_click(self, pos):
        x, y = pos
        for id, button in self.buttons.items():
            if button["rect"].collidepoint(x, y):
                button["method"]()

        if self.human_input.enabled and self.human_input.hover_action:
            # Call this so that we make sure that the click is happening
            # in the right place and not behind
            self.handle_hover(pos)
            self.human_input.click_action = self.human_input.hover_action
            self.human_input.hover_action = None

    def _handle_start_pause(self):
        if self.game_state in [GameState.READY, GameState.PAUSED]:
            self.game_state = GameState.RUNNING
        elif self.game_state == GameState.RUNNING:
            self.game_state = GameState.PAUSED
        elif self.game_state == GameState.FINISHED:
            self.game_state = GameState.READY
        self._draw_buttons()

    def _handle_next(self):
        self.game_state = GameState.STEP

    def _handle_log(self):
        self.log_actions = not self.log_actions
        self._draw_buttons()

    def handle_hover(self, pos):
        self.human_input.hover_action = None
        if not self.human_input.enabled:
            return

        x = pos[0] - self.board_position[0]
        y = pos[1] - self.board_position[1]
        if x < 0 or x > self.board_pixels or y < 0 or y > self.board_pixels:
            return

        col, x_offset = divmod(x, self.cell_size + self.wall_size)
        row, y_offset = divmod(y, self.cell_size + self.wall_size)
        h_wall, v_wall = False, False
        if x_offset > self.cell_size:
            v_wall = True
        if y_offset > self.cell_size:
            h_wall = True

        if h_wall and v_wall:
            # It's just in the corner of a cell, so we ignore it because we don't know
            # if the user wants a verticar or horizontal wall
            return

        type_action = 0
        if v_wall:
            action = WallAction((row, col), WallOrientation.VERTICAL)
        elif h_wall:
            action = WallAction((row, col), WallOrientation.HORIZONTAL)
        else:
            action = MoveAction((row, col))

        if action in self.human_input.valid_moves:
            self.human_input.hover_action = action

    def get_human_input(self, valid_moves: set) -> Action | None:
        self.human_input.valid_moves = valid_moves
        self.human_input.click_action = None
        self.human_input.enabled = True

        while self.human_input.click_action is None and self.running:
            pygame.time.delay(50)

        self.human_input.enabled = False
        return self.human_input.click_action

    def run(self):
        pygame.init()
        self.font24 = pygame.font.SysFont("Sans", 24)
        self.font16 = pygame.font.SysFont("Sans", 16)
        self.font12 = pygame.font.SysFont("sfnsmono", 12)

        self.screen = pygame.display.set_mode((630, 700))
        pygame.display.set_caption("Quoridor")

        clock = pygame.time.Clock()

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self._handle_click(event.pos)
                elif event.type == pygame.MOUSEMOTION:
                    self.handle_hover(event.pos)

            self.draw_screen()

            pygame.display.flip()
            clock.tick(30)

        pygame.quit()


class PygameRenderer(Renderer):
    def __init__(self):
        super().__init__()
        # To sync between threads and make sure gui is instantiated before calling it
        self.gui_created = Event()

        # When set it indicates that the gui wishes to terminate, so the Arena thread needs to exit as well
        self.terminated = Event()

        # A move shouldn't take less than this time (could take longer if the agent is slow)
        # so that it doesn't go too fast
        self.min_time_ms_per_move = 1000

        self.last_action_time = 0

    def start_arena(self, game, total_games: int):
        self.gui_created.wait()

        PygameQuoridor.instance().start(game)

    def _wait_not_in_states(self, states: List[GameState] | GameState):
        """Wait until the game reaches one of the specified states.
        If self.terminated is set while waiting it will exit the program.
        """

        wait_states = [states] if isinstance(states, GameState) else states
        while True:
            if self.terminated.is_set():
                exit()
            if PygameQuoridor.instance().game_state not in wait_states:
                break

            time.sleep(0.01)

    def start_game(self, env, agent1: Agent, agent2: Agent):
        gui = PygameQuoridor.instance()

        # We always enable the log from the agents and decide later if we want to show it, since it's
        # easier and the performance impact of logging is negligible when compared to the rendering.
        agent1.action_log.set_enabled()
        agent2.action_log.set_enabled()
        self.has_human_player = isinstance(agent1, HumanAgent) or isinstance(agent2, HumanAgent)

        initial_state = BoardState(
            p1_position=env.game.board.get_player_position(Player.ONE),
            p1_walls_remaining=env.game.board.get_walls_remaining(Player.ONE),
            p2_position=env.game.board.get_player_position(Player.TWO),
            p2_walls_remaining=env.game.board.get_walls_remaining(Player.TWO),
            is_p1_turn=True,
            walls=env.game.board.get_old_style_walls(),
        )
        gui.start_game(agent1.name(), agent2.name(), initial_state)

        if self.has_human_player:
            gui.game_state = GameState.RUNNING

        self._wait_not_in_states(GameState.READY)

        if gui.game_state == GameState.STEP:
            gui.game_state = GameState.PAUSED

    def end_game(self, game, result: GameResult):
        PygameQuoridor.instance().end_game(game, result)
        self._wait_not_in_states(GameState.FINISHED)

    def before_action(self, game, agent):
        gui = PygameQuoridor.instance()
        gui.update_log(agent.action_log)
        gui.update_turn(game.agent_selection == "player_0")
        if isinstance(agent, HumanAgent):
            # The human can play as soon as their turn is ready, no need to wait.
            # Set the last action time so the wait is just for the other agent.
            self.last_action_time = time.time()
            return

        self._wait_not_in_states([GameState.READY, GameState.PAUSED])

        if gui.game_state == GameState.STEP:
            gui.game_state = GameState.PAUSED
        else:
            # When we have a human and we're not logging there's no need to wait
            if not self.has_human_player or PygameQuoridor.instance().log_actions:
                # Wait for the minimum time to pass before the next move, but don't limit if the game is
                # in step mode, so that the user can go as fast as they want
                while (time.time() - self.last_action_time) < (self.min_time_ms_per_move / 1000.0):
                    time.sleep(0.01)
                self.last_action_time = time.time()

    def after_action(self, env, step, agent_id, action):
        gui = PygameQuoridor.instance()
        gui.update_log(None)
        gui.update_turn(env.agent_selection == "player_0")

        gui.update_players_state(
            p1_position=env.game.board.get_player_position(Player.ONE),
            p1_walls_remaining=env.game.board.get_walls_remaining(Player.ONE),
            p2_position=env.game.board.get_player_position(Player.TWO),
            p2_walls_remaining=env.game.board.get_walls_remaining(Player.TWO),
            walls=env.game.board.get_old_style_walls(),
        )

    def main_thread(self):
        PygameQuoridor()
        self.gui_created.set()
        PygameQuoridor.instance().run()
        self.terminated.set()
