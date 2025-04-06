import time
from threading import Thread
from typing import Optional

import numpy as np
from agents import Agent, AgentRegistry, ReplayAgent
from agents.core import AbstractTrainableAgent
from arena_utils import ArenaPlugin, CompositeArenaPlugin, GameResult
from quoridor_env import env
from renderers import PygameRenderer


class Arena:
    def __init__(
        self,
        board_size: int = 9,
        max_walls: int = 10,
        step_rewards: bool = False,
        renderers: list[ArenaPlugin] = [],
        saver: Optional[ArenaPlugin] = None,
        plugins: list[ArenaPlugin] = [],
        swap_players: bool = True,
    ):
        self.board_size = board_size
        self.max_walls = max_walls
        self.step_rewards = step_rewards
        self.swap_players = swap_players
        self.game = env(board_size=board_size, max_walls=max_walls, step_rewards=step_rewards)

        self.renderers = renderers
        self.plugins = CompositeArenaPlugin([p for p in np.concatenate([plugins, renderers, [saver]]) if p is not None])

    def _play_game(self, agent1: Agent, agent2: Agent, game_id: str) -> GameResult:
        self.game.reset()
        agents = {
            "player_0": agent1,
            "player_1": agent2,
        }
        for p, a in agents.items():
            a.start_game(self.game, p)
        self.plugins.start_game(self.game, agent1, agent2)
        start_time = time.time()
        step = 0
        for player_id in self.game.agent_iter():
            observation, _, termination, truncation, _ = self.game.last()
            agent = agents[player_id]
            if termination or truncation:
                if agent.is_trainable() and isinstance(agent, AbstractTrainableAgent):
                    # Handle end of game (in case winner was not this agent)
                    agent.handle_step_outcome(observation, None, self.game)
                break

            action = int(agent.get_action(self.game))

            self.plugins.before_action(self.game, agent)
            self.game.step(action)

            if agent.is_trainable() and isinstance(agent, AbstractTrainableAgent):
                agent.handle_step_outcome(observation, action, self.game)

            opponent_agent = agents[self.game.agent_selection]
            if opponent_agent.is_trainable() and isinstance(opponent_agent, AbstractTrainableAgent):
                opponent_agent.handle_opponent_step_outcome(observation, action, self.game)

            self.plugins.after_action(self.game, step, player_id, action)
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
        for p, a in agents.items():
            a.end_game(self.game)
        self.plugins.end_game(self.game, result)

        self.game.close()
        return result

    def _play_games(self, players: list[str | Agent], times: int):
        self.plugins.start_arena(self.game, total_games=len(players) * (len(players) - 1) * times // 2)

        match_id = 1
        results = []
        agents = []
        for p in players:
            if isinstance(p, Agent):
                agents.append(p)
            else:
                agents.append(
                    AgentRegistry.create_from_encoded_name(p, board_size=self.board_size, max_walls=self.max_walls)
                )

        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                for t in range(times):
                    agent_1, agent_2 = (
                        (agents[i], agents[j]) if not self.swap_players or t % 2 == 0 else (agents[j], agents[i])
                    )
                    result = self._play_game(agent_1, agent_2, f"game_{match_id:04d}")
                    results.append(result)
                    match_id += 1

        self.plugins.end_arena(self.game, results)

    def _replay_games(self, arena_data: dict, game_ids_to_replay: list[str]):
        """Replays a series of games from previously recorded arena data.

        This method simulates games using recorded moves from previous matches, allowing for
        replay and analysis of historical games.
        """
        results = []

        if len(game_ids_to_replay) == 0:
            game_ids_to_replay = arena_data["games"].keys()

        self.plugins.start_arena(self.game, len(game_ids_to_replay))

        for game_id in game_ids_to_replay:
            game_data = arena_data["games"][game_id]
            steps_player1 = game_data["actions"][::2]
            steps_player2 = game_data["actions"][1::2]

            agent_1 = ReplayAgent(game_data["player1"], steps_player1)
            agent_2 = ReplayAgent(game_data["player2"], steps_player2)

            result = self._play_game(agent_1, agent_2, game_id)
            results.append(result)

        self.plugins.end_arena(self.game, results)

    def play_games(self, players: list[str | Agent], times: int):
        pygame_renderer = next((r for r in self.renderers if isinstance(r, PygameRenderer)), None)

        if pygame_renderer is None:
            self._play_games(players, times)
        else:
            # When using PygameRenderer, pygame needs to run in the main thread (at least on MacOS),
            # so we need to start a new thread for the game loop.
            thread = Thread(target=self._play_games, args=(players, times))
            thread.start()

            pygame_renderer.main_thread()
            thread.join()

    def replay_games(self, arena_data: dict, game_ids_to_replay: list[str]):
        pygame_renderer = next((r for r in self.renderers if isinstance(r, PygameRenderer)), None)

        if pygame_renderer is None:
            self._replay_games(arena_data, game_ids_to_replay)
        else:
            # When using PygameRenderer, pygame needs to run in the main thread (at least on MacOS),
            # so we need to start a new thread for the game loop.
            thread = Thread(target=self._replay_games, args=(arena_data, game_ids_to_replay))
            thread.start()

            pygame_renderer.main_thread()
            thread.join()
