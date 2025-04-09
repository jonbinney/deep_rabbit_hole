from arena import GameResult
from renderers import Renderer


class ProgressBarRenderer(Renderer):
    def update_progress_bar(self, bar_width: int = 50):
        progress = self.games_played / self.total_games * 100
        filled = int(bar_width * progress / 100)
        bar = "=" * filled + "-" * (bar_width - filled)
        print(f"\rProgress: [{bar}] {progress:.1f}% ({self.games_played}/{self.total_games})", end="", flush=True)

    def start_arena(self, game, total_games):
        self.total_games = total_games
        self.games_played = 0
        self.update_progress_bar()

    def end_game(self, game, result):
        self.games_played += 1
        self.update_progress_bar()

    def end_arena(self, game, results: list[GameResult]):
        print()
