import time
from enum import Enum
from queue import Queue
from threading import Event
from typing import List, Optional

import pygame

from agents import ActionLog, Agent
from arena import GameResult
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


class PygameQuoridor:
    def __init__(self):
        self._msg_to_gui = Queue()
        self.board_pixels = 530
        self.board_position = (50, 50)
        self.log_actions = True
        self.buttons = {
            "start_pause": {"text": "Start", "method": self._handle_start_pause},
            "next": {"text": "Step", "method": self._handle_next},
            "log": {"text": "Log off", "method": self._handle_log},
        }

    def start(self, game):
        self.board_size = game.board_size
        self.compute_sizes()
        self.game_state = GameState.READY

    def start_game(self, player1: str, player2: str):
        self.player1 = player1
        self.player2 = player2

    def end_game(self, game, result: GameResult):
        self.game_state = GameState.FINISHED
        self.update_board(game, result.winner, result)

    def compute_sizes(self):
        self.wall_size = self.board_pixels // ((self.board_size - 1) + self.board_size * WALL_TO_CELL_RATIO)
        self.cell_size = WALL_TO_CELL_RATIO * self.wall_size
        self.board_pixels = (self.board_size - 1) * self.wall_size + self.board_size * self.cell_size

    def update_board(self, game, current_player: str, result=None):
        self._msg_to_gui.put(
            {
                "action": "update_board",
                "positions": game.positions,
                "walls": game.walls,
                "walls_remaining": game.walls_remaining,
                "current_player": current_player,
                "result": result,
            }
        )

    def show_log(self, game, log):
        self._msg_to_gui.put(
            {
                "action": "show_log",
                "game": game,
                "log": log,
            }
        )

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

    def _draw_walls(self, positions):
        m = self.board_size - 1

        for row in range(m):
            for col in range(m):
                if positions[row, col, 0] == 1 and (row == 0 or positions[row - 1, col, 0] == 0):
                    # Vertical wall
                    _, y0, x1, _ = self._cell_pos(row, col)
                    x0, _, _, y1 = self._cell_pos(row + 1, col + 1)
                    pygame.draw.rect(self.screen, COLOR_WALL, (x0, y0, x1 - x0, y1 - y0), border_radius=4)

                if positions[row, col, 1] == 1 and (col == 0 or positions[row, col - 1, 1] == 0):
                    # Horizontal wall
                    x0, _, _, y1 = self._cell_pos(row, col)
                    _, y0, x1, _ = self._cell_pos(row + 1, col + 1)
                    pygame.draw.rect(self.screen, COLOR_WALL, (x0, y0, x1 - x0, y1 - y0), border_radius=4)

    def _draw_players_and_data(self, message):
        positions = message["positions"]
        ys = [self.board_position[1] - 38, self.board_position[1] + self.board_pixels + 6]
        texts = [
            f"P0 {self.player1} ({message['walls_remaining']['player_0']})",
            f"P1 {self.player2} ({message['walls_remaining']['player_1']})",
        ]
        actives = [message["current_player"] == "player_1", message["current_player"] == "player_0"]
        colors = [COLOR_PLAYER1, COLOR_PLAYER2]
        positions = [positions["player_0"], positions["player_1"]]

        for y, text, active, color, position in zip(ys, texts, actives, colors, positions):
            # Draw the player on the board
            self._draw_player(color, position[0], position[1], active)

            # Write the name of the player and the walls remaining
            text_surface = self.font24.render(text, True, COLOR_BLACK)
            x = self.board_position[0] + (self.board_pixels - text_surface.get_width()) // 2
            self.screen.blit(text_surface, (x, y))

            # Draw the player color and an active circle next to the name
            pygame.draw.circle(self.screen, color, (x - 10, y + 16), 5)
            if active:
                pygame.draw.circle(self.screen, COLOR_ACTIVE_PLAYER, (x - 10, y + 16), radius=9, width=2)

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

    def _draw_result(self, result: Optional[GameResult]):
        if result is None:
            return

        text = f"Game Over! {result.winner} won in {result.steps} moves."
        text_surface = self.font24.render(text, True, COLOR_BLACK)
        self.screen.blit(text_surface, (self.board_position[0] + 15, self.board_position[1] + self.board_pixels + 80))

    def draw_screen(self, message):
        self.screen.fill(COLOR_SCREEN)
        self._draw_board()
        self._draw_players_and_data(message)
        self._draw_walls(message["walls"])
        self._draw_buttons()
        self._draw_result(message["result"])

    def _draw_log_action(self, game, action: int, text: str, color):
        r, c, type = game.action_index_to_params(action)
        x0, y0, x1, y1 = self._cell_pos(r, c)
        wall_len = 2 * self.cell_size + self.wall_size
        if type == 0:
            x = x0 + self.cell_size // 2
            y = y0 + self.cell_size // 2
            pygame.draw.circle(self.screen, color, (x, y), self.cell_size * 0.3)

        elif type == 1:  # vertical wall
            x = x1 + self.wall_size // 2
            y = y0 + self.cell_size // 2
            pygame.draw.rect(self.screen, color, (x1 + 1, y0 + 1, self.wall_size - 1, wall_len - 1), border_radius=4)

        elif type == 2:
            x = x0 + self.cell_size // 2
            y = y1 + self.wall_size // 2
            pygame.draw.rect(self.screen, color, (x0 + 1, y1 + 1, wall_len - 1, self.wall_size - 1), border_radius=4)

        text_surface = self.font12.render(text, True, COLOR_BLACK)
        self.screen.blit(text_surface, dest=text_surface.get_rect(center=(x, y)))

    def _draw_log_action_score_ranking(self, game, entry: ActionLog.ActionScoreRanking, palette_id: int):
        palette_size = len(PALETTES[palette_id])

        coeff = palette_size / max([r for r, _, _ in entry.ranking])
        for ranking, action, score in entry.ranking:
            text = f"{score:0.2f}" if score < 10 else f"{int(score)}"
            color = PALETTES[palette_id][int((ranking - 1) * coeff)]
            self._draw_log_action(game, action, text, color)

        return palette_id + 1

    def _draw_path(self, game, path: list[tuple[int, int]], path_id):
        for (r1, c1), (r2, c2) in zip(path, path[1:]):
            p1 = self._cell_center(r1, c1)
            p2 = self._cell_center(r2, c2)
            color = PALETTE_GRAY[4]
            pygame.draw.line(self.screen, color, p1, p2, width=3)
            pygame.draw.circle(self.screen, color, p2, radius=7)

        return path_id + 1  # todo

    def _draw_log(self, message):
        if not self.log_actions:
            return
        palette_id_asr = 0
        path_id = 0

        log = message["log"]
        game = message["game"]
        for entry in log.records:
            if isinstance(entry, ActionLog.ActionScoreRanking):
                palette_id_asr = self._draw_log_action_score_ranking(game, entry, palette_id_asr)
            if isinstance(entry, ActionLog.ActionText):
                self._draw_log_action(game, entry.action, entry.text, PALETTE_GRAY[4])
            if isinstance(entry, ActionLog.Path):
                path_id = self._draw_path(game, entry.path, path_id)

    def _handle_click(self, pos):
        x, y = pos
        for id, button in self.buttons.items():
            if button["rect"].collidepoint(x, y):
                button["method"]()

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

    def run(self):
        pygame.init()
        self.font24 = pygame.font.SysFont("Sans", 24)
        self.font16 = pygame.font.SysFont("Sans", 16)
        self.font12 = pygame.font.SysFont("sfnsmono", 12)

        self.screen = pygame.display.set_mode((630, 700))
        pygame.display.set_caption("Quoridor")

        clock = pygame.time.Clock()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self._handle_click(event.pos)

            if not self._msg_to_gui.empty():
                message = self._msg_to_gui.get()
                if message["action"] == "update_board":
                    self.draw_screen(message)

                if message["action"] == "show_log":
                    self._draw_log(message)

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
        self.min_time_ms_per_move = 500

        self.last_action_time = 0

    def start_arena(self, game, total_games: int):
        self.gui_created.wait()

        self.gui.start(game)

    def _wait_not_in_states(self, states: List[GameState] | GameState):
        """Wait until the game reaches one of the specified states.
        If self.terminated is set while waiting it will exit the program.
        """

        wait_states = [states] if isinstance(states, GameState) else states
        while True:
            if self.terminated.is_set():
                exit()
            if self.gui.game_state not in wait_states:
                break

            time.sleep(0.01)

    def start_game(self, game, agent1: Agent, agent2: Agent):
        # We always enable the log from the agents and decide later if we want to show it, since it's
        # easier and the performance impact of logging is negligible when compared to the rendering.
        agent1.action_log.set_enabled()
        agent2.action_log.set_enabled()

        self.gui.start_game(agent1.name(), agent2.name())
        self.gui.update_board(game, "player_1")

        self._wait_not_in_states(GameState.READY)

        if self.gui.game_state == GameState.STEP:
            self.gui.game_state = GameState.PAUSED

    def end_game(self, game, result: GameResult):
        self.gui.end_game(game, result)
        self._wait_not_in_states(GameState.FINISHED)

    def before_action(self, game, agent):
        self.gui.show_log(game, agent.action_log)
        self._wait_not_in_states([GameState.READY, GameState.PAUSED])

        if self.gui.game_state == GameState.STEP:
            self.gui.game_state = GameState.PAUSED
        else:
            # Wait for the minimum time to pass before the next move, but don't limit if the game is
            # in step mode, so that the user can go as fast as they want
            while (time.time() - self.last_action_time) < (self.min_time_ms_per_move / 1000.0):
                time.sleep(0.01)
            self.last_action_time = time.time()

    def after_action(self, game, step, agent, action):
        self.gui.update_board(game, agent)

    def main_thread(self):
        self.gui = PygameQuoridor()
        self.gui_created.set()
        self.gui.run()
        self.terminated.set()
