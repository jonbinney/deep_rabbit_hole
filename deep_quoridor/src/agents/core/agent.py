import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Type

from quoridor import Action
from utils import SubargsBase, parse_subargs


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
    agent_class: Optional[Type[Agent]] = None
    params_class: Optional[Type[SubargsBase]] = None


class AgentRegistry:
    agents = {}

    @staticmethod
    def get_registry_entry(agent_type: str) -> AgentRegistryEntry:
        registry_entry = AgentRegistry.agents[agent_type]

        if registry_entry.agent_class is None:
            agent_module = importlib.import_module(registry_entry.module_name)
            element_names = registry_entry.class_name.split(".")

            # If the class_name is heirarchical, e.g. "foo.bar.Baz", then we need to gettatr
            # the first element, then the second, etc. until we get to Baz.
            registry_entry.agent_class = getattr(agent_module, element_names[0])
            for element_name in element_names[1:]:
                registry_entry.agent_class = getattr(registry_entry.agent_class, element_name)

            if hasattr(registry_entry.agent_class, "params_class"):
                registry_entry.params_class = registry_entry.agent_class.params_class()

        return registry_entry

    @staticmethod
    def create(friendly_name: str, **kwargs) -> Agent:
        return AgentRegistry.agents[friendly_name](**kwargs)

    @staticmethod
    def create_from_encoded_name(
        encoded_name: str, env, remove_training_args=False, keep_args: set[str] = set(), **kwargs
    ) -> Agent:
        parts = encoded_name.split(":")
        agent_type = parts[0]
        registry_entry = AgentRegistry.get_registry_entry(agent_type)
        if len(parts) == 2:
            if registry_entry.params_class is None:
                raise ValueError(f"The agent {agent_type} doesn't support subarguments, but '{parts[1]}' was passed")

            if remove_training_args:
                args_to_remove = registry_entry.params_class.training_only_params().difference(keep_args)
                subargs = parse_subargs(parts[1], registry_entry.params_class, ignore_fields=args_to_remove)
            else:
                subargs = parse_subargs(parts[1], registry_entry.params_class)

            kwargs["params"] = subargs

        assert registry_entry.agent_class is not None
        return registry_entry.agent_class(
            board_size=env.board_size,
            max_walls=env.max_walls,
            max_steps=env.max_steps,
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

    @staticmethod
    def register(name: str, class_name: str, module_name: str):
        AgentRegistry.agents[name] = AgentRegistryEntry(class_name, module_name)
