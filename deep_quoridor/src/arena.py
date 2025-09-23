import random
import time
from enum import Enum
from threading import Thread
from typing import Optional

from agents import Agent, AgentRegistry, ReplayAgent
from agents.core.trainable_agent import TrainableAgent
from arena_utils import ArenaPlugin, CompositeArenaPlugin, GameResult, MoveInfo
from quoridor_env import env
from renderers import PygameRenderer
from utils.misc import get_opponent_player_id


# Add after imports
class PlayMode(Enum):
    ALL_VS_ALL = "all_vs_all"  # Legacy mode where all players play against each other
    FIRST_VS_RANDOM = "first_vs_random"  # First player plays against randomly selected opponents
    FIRST_VS_ALL = "first_vs_all"  # First player plays against all the opponents


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
        max_steps: int = 1000,
    ):
        self.board_size = board_size
        self.max_walls = max_walls
        self.step_rewards = step_rewards
        self.max_steps = max_steps
        self.swap_players = swap_players
        self.game = env(board_size=board_size, max_walls=max_walls, max_steps=max_steps, step_rewards=step_rewards)

        self.renderers = renderers
        self.plugins = CompositeArenaPlugin([p for p in plugins + renderers + [saver] if p is not None])

    def _play_game(self, agent1: Agent, agent2: Agent, game_id: str) -> GameResult:
        self.game.reset()
        agents = {
            "player_0": agent1,
            "player_1": agent2,
        }
        # If it's self play, we only have one agent, and we use this to make sure that some calls like
        # start_game and end_game are called only once.
        if agent1 == agent2:
            unique_agents = {"both_players": agent1}
        else:
            unique_agents = agents

        for p, a in unique_agents.items():
            a.start_game(self.game, p)
        self.plugins.start_game(self.game, agent1, agent2)
        start_time = time.time()
        step = 0
        moves = []
        for agent_id in self.game.agent_iter():
            observation_before_action, _, termination, truncation, _ = self.game.last()
            agent = agents[agent_id]
            opponent_agent_id = get_opponent_player_id(agent_id)
            opponent_agent = agents[opponent_agent_id]

            if termination or truncation:
                if agent.is_trainable() and isinstance(agent, TrainableAgent):
                    # Handle end of game (in case winner was not this agent)
                    agent.handle_step_outcome(
                        observation_before_action=observation_before_action,
                        opponent_observation_after_action=self.game.observe(opponent_agent_id),
                        observation_after_action=None,
                        reward=self.game.rewards[agent_id],
                        action=None,
                        done=True,
                    )

                if truncation:
                    # Print the game state to help debug.
                    print(f"\nP1: {agent1.name()} P2: {agent2.name()}")
                    print(self.game.render())

                break

            # opponent_agent = agents["player_0" if player_id == "player_1" else "player_1"]
            # if opponent_agent.is_trainable() and isinstance(opponent_agent, AbstractTrainableAgent):
            #    opponent_agent.inspect_opponent_possible_actions(self.game, observation, agent.action_log)
            assert (observation_before_action["action_mask"] == self.game.last_action_mask[agent_id]).all()

            get_action_start_time = time.time()
            action = int(agent.get_action(observation_before_action))
            get_action_finish_time = time.time()
            moves.append(MoveInfo(agent.name(), action, get_action_finish_time - get_action_start_time))

            self.plugins.before_action(self.game, agent)
            self.game.step(action)

            if agent.is_trainable() and isinstance(agent, TrainableAgent):
                agent.handle_step_outcome(
                    observation_before_action=observation_before_action,
                    opponent_observation_after_action=self.game.observe(opponent_agent_id),
                    observation_after_action=self.game.observe(agent_id),
                    reward=self.game.rewards[agent_id],
                    action=action,
                    done=self.game.is_done(),
                )

            if opponent_agent.is_trainable() and isinstance(opponent_agent, TrainableAgent):
                opponent_agent.handle_opponent_step_outcome(
                    opponent_observation_before_action=observation_before_action,
                    my_observation_after_opponent_action=self.game.observe(opponent_agent_id),
                    opponent_observation_after_action=self.game.observe(agent_id),
                    opponent_reward=self.game.rewards[agent_id],
                    opponent_action=action,
                    done=self.game.is_done(),
                )

            self.plugins.after_action(self.game, step, agent_id, action)
            step += 1

        end_time = time.time()
        # print(self.game.render())
        winner = self.game.winner()

        if agent1.name() == agent2.name():
            agent1_name = f"{agent1.name()}-P1"
            agent2_name = f"{agent2.name()}-P2"
        else:
            agent1_name = agent1.name()
            agent2_name = agent2.name()

        result = GameResult(
            player1=agent1_name,
            player2=agent2_name,
            winner=[agent1_name, agent2_name][winner] if winner is not None else "None",
            steps=step,
            time_ms=int((end_time - start_time) * 1000),
            game_id=game_id,
            moves=moves,
        )
        for p, a in unique_agents.items():
            a.end_game(self.game)
        self.plugins.end_game(self.game, result)

        self.game.close()
        return result

    # Replace the existing _play_games method
    def _play_games(self, players: list[str | Agent], times: int, mode: PlayMode) -> list[GameResult]:
        agents = []
        for p in players:
            if isinstance(p, Agent):
                agents.append(p)
            else:
                agents.append(AgentRegistry.create_from_encoded_name(p, self.game))

        if mode == PlayMode.ALL_VS_ALL:
            total_games = len(agents) * (len(agents) - 1) * times // 2
        else:  # FIRST_VS_RANDOM or FIRST_VS_ALL mode
            total_games = (len(agents) - 1) * times

        self.plugins.start_arena(self.game, total_games=total_games)

        match_id = 1
        results = []

        try:
            if mode == PlayMode.ALL_VS_ALL:
                for i in range(len(agents)):
                    for j in range(i + 1, len(agents)):
                        for t in range(times):
                            agent_1, agent_2 = (
                                (agents[i], agents[j])
                                if not self.swap_players or t % 2 == 0
                                else (agents[j], agents[i])
                            )
                            result = self._play_game(agent_1, agent_2, f"game_{match_id:04d}")
                            results.append(result)
                            match_id += 1
            else:  # FIRST_VS_RANDOM or FIRST_VS_ALL mode
                first_agent = agents[0]
                remaining_agents = agents[1:]

                for opp in remaining_agents:
                    for _ in range(times):
                        opponent = random.choice(remaining_agents) if mode == PlayMode.FIRST_VS_RANDOM else opp
                        agent_1, agent_2 = (
                            (first_agent, opponent)
                            if not self.swap_players or match_id % 2 == 0
                            else (opponent, first_agent)
                        )
                        result = self._play_game(agent_1, agent_2, f"game_{match_id:04d}")
                        results.append(result)
                        match_id += 1

        except KeyboardInterrupt:
            print("!!! Interrupted by user. Closing the arena before exit.")
        except Exception as e:
            print(f"!!! Exception occurred: {e}. Closing the arena before exit.")
            raise
        finally:
            self.plugins.end_arena(self.game, results)

        return results

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

    def play_games(self, players: list[str | Agent], times: int, mode: PlayMode = PlayMode.ALL_VS_ALL):
        pygame_renderer = next((r for r in self.renderers if isinstance(r, PygameRenderer)), None)

        if pygame_renderer is None:
            self._play_games(players, times, mode)
        else:
            # When using PygameRenderer, pygame needs to run in the main thread (at least on MacOS),
            # so we need to start a new thread for the game loop.
            thread = Thread(target=self._play_games, args=(players, times, mode))
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
