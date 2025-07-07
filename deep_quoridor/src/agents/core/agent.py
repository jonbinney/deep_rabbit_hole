import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Type

from quoridor import Action
from utils import parse_subargs


class ActionLog:
    @dataclass
    class ActionText:
        """
        Log an action with an associated text.
        """

        action: Action
        text: str

    @dataclass
    class ActionScoreRanking:
        """
        Log a ranking of actions with their scores.
        """

        ranking: list[tuple[int, Action, float]]  # rank, action, score

    @dataclass
    class Path:
        """
        Log a path of coordinates.
        """

        path: list[tuple[int, int]]  # list of coordinates

    def __init__(self):
        self.records = []
        self.set_enabled(False)

    def clear(self):
        self.records = []

    def set_enabled(self, enabled=True):
        self._enabled = enabled

    def is_enabled(self) -> bool:
        return self._enabled

    def action_text(self, action: Action, text: str):
        if not self.is_enabled():
            return
        self.records.append(self.ActionText(action, text))

    def action_score_ranking(self, action_scores: dict[Action, float]):
        """
        Log a ranking of actions with their scores.
        action_scores is a dictionary where the key is the action and the value is the score.
        The ranking is created automatically and sorted in descending order.
        """
        if not self.is_enabled():
            return
        sorted_actions = sorted(action_scores.items(), key=lambda x: x[1], reverse=True)
        ranking = []
        curr_rank = 0
        prev_score = None
        for action, score in sorted_actions:
            if prev_score is None or score != prev_score:
                curr_rank += 1

            ranking.append((curr_rank, action, score))
            prev_score = score

        self.records.append(self.ActionScoreRanking(ranking))

    def path(self, path: list[tuple[int, int]]):
        if not self.is_enabled():
            return
        self.records.append(self.Path(path))


class Agent(ABC):
    """
    Base class for all agents.
    Given a game state, the agent should return an action.
    """

    def __init__(self, **kwargs):
        self.action_log = ActionLog()

    @staticmethod
    def _friendly_name(class_name: str):
        return class_name.replace("Agent", "").lower()

    def name(self) -> str:
        return Agent._friendly_name(self.__class__.__name__)

    def is_trainable(self) -> bool:
        """Returns True if the agent is a learning agent, False otherwise."""
        return False

    def start_game(self, game, player_id):
        """This method is called when a new game starts.
        It allows the agent to reset its state if needed.
        """
        pass

    def end_game(self, game):
        """This method is called when a game ends.
        It allows the agent to clean up its state if needed.
        """
        pass

    @abstractmethod
    def get_action(self, observation) -> int:
        pass


@dataclass
class AgentRegistryEntry:
    class_name: str
    module_name: str
    agent_class: Type[Agent] = None


class AgentRegistry:
    agents = {
        "alphazero": AgentRegistryEntry("AlphaZeroAgent", "agents.alphazero"),
        "alphazero_os": AgentRegistryEntry("AlphaZeroOSAgent", "agents.alphazero_os"),
        "cnn": AgentRegistryEntry("CnnAgent", "agents.adapter_based_agents"),
        "cnn3c": AgentRegistryEntry("Cnn3CAgent", "agents.adapter_based_agents"),
        "daz": AgentRegistryEntry("DAZAgent", "agents.alphazero_dexp"),
        "daz_mimic": AgentRegistryEntry("DAZAgent.create_from_trained_instance", ""),
        "dexp": AgentRegistryEntry("DExpAgent", "agents.dexp"),
        "dexp_mimic": AgentRegistryEntry("DExpAgent.create_from_trained_instance", ""),
        "greedy": AgentRegistryEntry("GreedyAgent", "agents.greedy"),
        "human": AgentRegistryEntry("HumanAgent", "agents.human"),
        "mcts": AgentRegistryEntry("MCTSAgent", "agents.mcts"),
        "ndexp": AgentRegistryEntry("NDexpAgent", "agents.adapter_based_agents"),
        "random": AgentRegistryEntry("RandomAgent", "agents.random"),
        "simple": AgentRegistryEntry("SimpleAgent", "agents.simple"),
        "sb3ppo": AgentRegistryEntry("SB3PPOAgent", "agents.sb3_ppo"),
    }

    @staticmethod
    def get_agent_class(agent_type: str) -> Type[Agent]:
        registry_entry = AgentRegistry.agents[agent_type]

        if registry_entry.agent_class is None:
            agent_module = importlib.import_module(registry_entry.module_name)
            registry_entry.agent_class = getattr(agent_module, registry_entry.class_name)

        return registry_entry.agent_class

    @staticmethod
    def create(friendly_name: str, **kwargs) -> Agent:
        return AgentRegistry.agents[friendly_name](**kwargs)

    @staticmethod
    def create_from_encoded_name(
        encoded_name: str, env, remove_training_args=False, keep_args: set[str] = set(), **kwargs
    ) -> Agent:
        parts = encoded_name.split(":")
        agent_type = parts[0]
        agent_class = AgentRegistry.get_agent_class(agent_type)
        if len(parts) == 2:
            subargs_class = agent_class.params_class()
            if subargs_class is None:
                raise ValueError(f"The agent {agent_type} doesn't support subarguments, but '{parts[1]}' was passed")

            if remove_training_args:
                args_to_remove = subargs_class.training_only_params().difference(keep_args)
                subargs = parse_subargs(parts[1], subargs_class, ignore_fields=args_to_remove)
            else:
                subargs = parse_subargs(parts[1], subargs_class)
            kwargs["params"] = subargs

        return agent_class(
            board_size=env.board_size,
            max_walls=env.max_walls,
            observation_space=env.observation_space(None),
            action_space=env.action_space(None),
            **kwargs,
        )

    @staticmethod
    def is_valid_encoded_name(encoded_name: str):
        parts = encoded_name.split(":")
        agent_type = parts[0]
        return agent_type in AgentRegistry.names()

    @staticmethod
    def names():
        return list(AgentRegistry.agents.keys())
