__all__ = [
    "AbstractTrainableAgent",
    "ActionLog",
    "Agent",
    "AgentRegistry",
    "DExpAgent",
    "FlatDQNAgent",
    "RandomAgent",
    "ReplayAgent",
    "ReplayBuffer",
    "SimpleAgent",
    "TrainableAgentParams",
]


from agents.core import (  # noqa: E402, F401  # noqa: E402, F401
    AbstractTrainableAgent,
    ActionLog,
    Agent,
    AgentRegistry,
    ReplayBuffer,
    TrainableAgentParams,
)
from agents.dexp import DExpAgent  # noqa: E402
from agents.flat_dqn import FlatDQNAgent  # noqa: E402
from agents.greedy import GreedyAgent  # noqa: E402, F401
from agents.random import RandomAgent  # noqa: E402, F401
from agents.replay import ReplayAgent  # noqa: E402, F401
from agents.simple import SimpleAgent  # noqa: E402, F401

AgentRegistry.register("dexp", DExpAgent)
AgentRegistry.register("flatdqn", FlatDQNAgent)
AgentRegistry.register("greedy", GreedyAgent)
AgentRegistry.register("random", RandomAgent)
AgentRegistry.register("simple", SimpleAgent)
