from typing import Optional
from quoridor_env import env
from agents import Agent
from agents import AgentRegistry
from agents.replay import ReplayAgent
from dataclasses import dataclass
import time


@dataclass
class GameResult:
    player1: str
    player2: str
    winner: str
    steps: int
    time_ms: int
    game_id: str


class ArenaPlugin:
    """
    Base class for all arena plugins.
    The plug in can override any combinantion of the methods below in order to provide additional functionality.
    """

    def start_game(self, game, agent1: Agent, agent2: Agent):
        pass

    def end_game(self, game, result: GameResult):
        pass

    def start_arena(self, game):
        pass

    def end_arena(self, game, results: list[GameResult]):
        pass

    def action(self, game, step, agent, action):
        pass


class CompositeArenaPlugin:
    """
    Allows to combine multiple plugins into a single one, calling them sequentially for each method.

    """

    def __init__(self, plugins: list[ArenaPlugin]):
        """
        For the sake of convenience, the plugin list is allowed to be empty, in which case the plugin will be a no-op.
        """
        self.plugins = plugins

    def start_game(self, game, agent1: Agent, agent2: Agent):
        [plugin.start_game(game, agent1, agent2) for plugin in self.plugins]

    def end_game(self, game, result: GameResult):
        [plugin.end_game(game, result) for plugin in self.plugins]

    def start_arena(self, game):
        [plugin.start_arena(game) for plugin in self.plugins]

    def end_arena(self, game, results: list[GameResult]):
        [plugin.end_arena(game, results) for plugin in self.plugins]

    def action(self, game, step, agent, action):
        [plugin.action(game, step, agent, action) for plugin in self.plugins]


class Arena:
    def __init__(
        self,
        board_size: int = 9,
        max_walls: int = 10,
        step_rewards: bool = False,
        renderer: Optional[ArenaPlugin] = None,
        saver: Optional[ArenaPlugin] = None,
        plugins: list[ArenaPlugin] = [],
    ):
        self.board_size = board_size
        self.max_walls = max_walls
        self.step_rewards = step_rewards
        self.game = env(
            board_size=board_size,
            max_walls=max_walls,
            step_rewards=step_rewards,
        )

        self.plugins = CompositeArenaPlugin([p for p in plugins + [renderer, saver] if p is not None])

    def _play_game(self, agent1: Agent, agent2: Agent, game_id: str) -> GameResult:
        self.game.reset()
        agents = {
            "player_0": agent1,
            "player_1": agent2,
        }

        self.plugins.start_game(self.game, agent1, agent2)

        start_time = time.time()
        step = 0
        for agent in self.game.agent_iter():
            _, _, termination, truncation, _ = self.game.last()
            if termination or truncation:
                action = None
                break

            action = int(agents[agent].get_action(self.game))
            self.game.step(action)
            self.plugins.action(self.game, step, agent, action)
            step += 1

        end_time = time.time()

        result = GameResult(
            player1=agent1.name(),
            player2=agent2.name(),
            winner=[agent1, agent2][self.game.winner()].name(),
            steps=step,
            time_ms=int((end_time - start_time) * 1000),
            game_id=game_id,
        )
        self.plugins.end_game(self.game, result)

        self.game.close()
        return result

    def play_games(self, players: list[str | Agent], times: int):
        self.plugins.start_arena(self.game)

        match_id = 1
        results = []
        agents = []
        for p in players:
            if isinstance(p, Agent):
                agents.append(p)
            else:
                agents.append(AgentRegistry.create(p))

        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                for t in range(times):
                    agent_1, agent_2 = (agents[i], agents[j]) if t % 2 == 0 else (agents[j], agents[i])
                    agent_1.reset()
                    agent_2.reset()

                    result = self._play_game(agent_1, agent_2, f"game_{match_id:04d}")
                    results.append(result)
                    match_id += 1

        self.plugins.end_arena(self.game, results)

    def replay_games(self, arena_data: dict, game_ids_to_replay: list[str]):
        """Replays a series of games from previously recorded arena data.

        This method simulates games using recorded moves from previous matches, allowing for
        replay and analysis of historical games.
        """
        self.plugins.start_arena(self.game)

        results = []

        if len(game_ids_to_replay) == 0:
            game_ids_to_replay = arena_data["games"].keys()

        for game_id in game_ids_to_replay:
            game_data = arena_data["games"][game_id]
            steps_player1 = game_data["actions"][::2]
            steps_player2 = game_data["actions"][1::2]

            agent_1 = ReplayAgent(game_data["player1"], steps_player1)
            agent_2 = ReplayAgent(game_data["player2"], steps_player2)

            result = self._play_game(agent_1, agent_2, game_id)
            results.append(result)

        self.plugins.end_arena(self.game, results)
