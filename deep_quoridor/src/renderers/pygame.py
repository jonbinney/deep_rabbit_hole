import pygame.freetype
from renderers import Renderer
from arena import GameResult
from agents import Agent
import pygame
from queue import Queue
import time
from enum import Enum
from typing import Optional, List

WALL_TO_CELL_RATIO = 5

COLOR_BOARD = (222, 184, 135)
COLOR_GRID = (160, 120, 95)
COLOR_WALL = (224, 32, 32)  # (100, 50, 20)

COLOR_BLACK = (0, 0, 0)


class GameState(Enum):
    READY = 0
    RUNNING = 1
    PAUSED = 2
    STEP = 3
    FINISHED = 4


class PQ:
    def __init__(self):
        self._msg_to_gui = Queue()
        self.board_pixels = 500
        self.board_position = (40, 50)

    def start(self, game):
        self.board_size = game.board_size
        self.resize()
        self.game_state = GameState.READY

    def start_game(self, player1: str, player2: str):
        self.player1 = player1
        self.player2 = player2

    def end_game(self, game, result: GameResult):
        self.game_state = GameState.FINISHED
        self.update_board(game, result.winner, result)  # TODO

    def resize(self):  # TO DO, maybe pass size
        self.wall_size = self.board_pixels // ((self.board_size - 1) + self.board_size * WALL_TO_CELL_RATIO)
        self.cell_size = WALL_TO_CELL_RATIO * self.wall_size
        self.board_pixels = (self.board_size - 1) * self.wall_size + self.board_size * self.cell_size

        # where to put this?
        self.buttons = {
            "start_pause": {"text": "Start", "method": self._handle_start_pause, "rect": pygame.Rect(10, 580, 100, 24)},
            "next": {"text": "Next", "method": self._handle_next, "rect": pygame.Rect(120, 580, 100, 24)},
        }

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

    def _cell_pos(self, row, col):
        x = col * (self.cell_size + self.wall_size) + self.board_position[0]
        y = row * (self.cell_size + self.wall_size) + self.board_position[1]
        return [x, y, x + self.cell_size, y + self.cell_size]

    def _draw_player(self, color, row, col):
        x0, y0, x1, y1 = self._cell_pos(row, col)
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        pygame.draw.circle(self.screen, color, (cx, cy), self.cell_size * 0.4)

    def _draw_board(self):
        mx, my = self.board_position  # margins

        # Background of the board
        pygame.draw.rect(
            self.screen,
            COLOR_BOARD,
            (
                mx,
                my,
                self.board_pixels,
                self.board_pixels,
            ),
        )

        # A rectangle around the board
        pygame.draw.rect(
            self.screen,
            COLOR_GRID,
            (
                mx - 2,
                my - 2,
                self.board_pixels + 4,
                self.board_pixels + 4,
            ),
            width=4,
            border_radius=4,
        )

        # Vertical lines
        for i in range(self.board_size - 1):
            _, _, x1, _ = self._cell_pos(0, i)
            xc = x1 + self.wall_size // 2
            pygame.draw.line(
                self.screen,
                COLOR_GRID,
                (xc, my),
                (xc, my + self.board_pixels),
                self.wall_size,
            )

        # Horizontal lines
        for i in range(self.board_size - 1):
            _, _, _, y1 = self._cell_pos(i, 0)
            yc = y1 + self.wall_size // 2
            pygame.draw.line(
                self.screen,
                COLOR_GRID,
                (mx, yc),
                (mx + self.board_pixels, yc),
                self.wall_size,
            )

    def _draw_walls(self, positions):
        m = self.board_size - 1

        # Vertical walls
        for col in range(m):
            skip = False
            for row in range(m):
                if skip:
                    skip = False
                    continue
                if positions[row, col, 0] == 1:
                    skip = True
                    _, y0, x1, _ = self._cell_pos(row, col)
                    _, _, _, y1 = self._cell_pos(row + 1, col)
                    xc = x1 + self.wall_size // 2

                    pygame.draw.line(self.screen, COLOR_WALL, (xc, y0), (xc, y1), width=self.wall_size)

        # Horizontal walls
        for row in range(m):
            skip = False
            for col in range(m):
                if skip:
                    skip = False
                    continue
                if positions[row, col, 1] == 1:
                    skip = True
                    x0, _, _, y1 = self._cell_pos(row, col)
                    _, _, x1, _ = self._cell_pos(row, col + 1)
                    yc = y1 + self.wall_size // 2

                    pygame.draw.line(self.screen, COLOR_WALL, (x0, yc), (x1, yc), width=self.wall_size)

    def _draw_players_and_data(self, message):
        # Draw the pawns
        positions = message["positions"]

        row, col = positions["player_0"]
        self._draw_player((32, 32, 32), row, col)

        row, col = positions["player_1"]
        self._draw_player((224, 224, 224), row, col)

        # Write the player names and number of walls left
        p1 = f"{self.player1} ({message['walls_remaining']['player_0']})"
        text_surface = self.font24.render(p1, True, COLOR_BLACK)
        self.screen.blit(text_surface, (self.board_position[0] + 15, self.board_position[1] - 32))

        p2 = f"{self.player2} ({message['walls_remaining']['player_1']})"
        text_surface = self.font24.render(p2, True, COLOR_BLACK)
        self.screen.blit(text_surface, (self.board_position[0] + 15, self.board_position[1] + self.board_pixels + 6))

        # Draw a green dot for the player that has the turn
        if message["current_player"] == "player_0":
            y = self.board_position[1] - 17
        else:
            y = self.board_position[1] + self.board_pixels + 18

        pygame.draw.circle(self.screen, (0, 192, 0), (self.board_position[0] + 5, y), 5)

    def _draw_buttons(self):
        if self.game_state == GameState.READY:
            self.buttons["start_pause"]["text"] = "Start"
        if self.game_state == GameState.RUNNING:
            self.buttons["start_pause"]["text"] = "Pause"
        if self.game_state == GameState.PAUSED:
            self.buttons["start_pause"]["text"] = "Resume"
        if self.game_state == GameState.FINISHED:
            self.buttons["start_pause"]["text"] = "Done"

        for _, button in self.buttons.items():
            pygame.draw.rect(self.screen, (200, 200, 200), button["rect"])
            text = self.font16.render(button["text"], True, COLOR_BLACK)
            self.screen.blit(text, dest=text.get_rect(center=button["rect"].center))

    def _draw_result(self, result: Optional[GameResult]):
        if result is None:
            return

        text = f"Game Over! {result.winner} won in {result.steps} moves."
        text_surface = self.font24.render(text, True, COLOR_BLACK)
        self.screen.blit(text_surface, (self.board_position[0] + 15, self.board_position[1] + self.board_pixels + 80))

    def draw_screen(self, message):
        self.screen.fill((255, 255, 255))
        self._draw_board()
        self._draw_players_and_data(message)
        self._draw_walls(message["walls"])
        self._draw_buttons()
        self._draw_result(message["result"])

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

    def _handle_next(self):
        self.game_state = GameState.STEP

    def run(self):
        pygame.init()
        self.font24 = pygame.font.SysFont("Sans", 24)
        self.font16 = pygame.font.SysFont("Sans", 16)

        self.screen = pygame.display.set_mode((600, 660))
        pygame.display.set_caption("Quoridor")

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self._handle_click(event.pos)

            if self._msg_to_gui.empty():
                pygame.time.delay(1)
                continue

            message = self._msg_to_gui.get()
            if message["action"] == "update_board":
                self.draw_screen(message)

            pygame.display.flip()

        pygame.quit()


class PygameRenderer(Renderer):
    def __init__(self):
        super().__init__()
        # To sync between threads and make sure gui is instantiated before calling it
        self.gui_created = False

        # When set to true it indicates that the gui wishes to terminate, so the Arena
        # thread needs to exit as well
        self.terminate = False

    def start_arena(self, game, total_games: int):
        while not self.gui_created:
            time.sleep(0.1)

        self.gui.start(game)

    def end_arena(self, game, results: list[GameResult]):
        pass

    def _wait_for_states(self, states: List[GameState] | GameState):
        """Wait until the game reaches one of the specified states.
        If self.terminate is set to true while waiting it will exit the program.
        """
        wait_states = [states] if isinstance(states, GameState) else states
        while self.gui.game_state in wait_states:
            if self.terminate:
                exit()
            time.sleep(0.01)

    def start_game(self, game, agent1: Agent, agent2: Agent):
        self.gui.start_game(agent1.name(), agent2.name())
        self.gui.update_board(game, "player_0")

        self._wait_for_states(GameState.READY)

        if self.gui.game_state == GameState.STEP:
            self.gui.game_state = GameState.PAUSED

    def end_game(self, game, result: GameResult):
        self.gui.end_game(game, result)
        self._wait_for_states(GameState.FINISHED)

    def action(self, game, step, agent, action):
        self.gui.update_board(game, agent)
        self._wait_for_states([GameState.READY, GameState.PAUSED])

        if self.gui.game_state == GameState.STEP:
            self.gui.game_state = GameState.PAUSED

        time.sleep(0.02)

    def main_thread(self):
        self.gui = PQ()
        self.gui_created = True
        self.gui.run()
        self.terminate = True
