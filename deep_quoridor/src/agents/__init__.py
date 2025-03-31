__all__ = [
    "AbstractTrainableAgent",
    "Agent",
    "AgentRegistry",
    "DExpAgent",
    "FlatDQNAgent",
    "Pretrained01FlatDQNAgent",
    "RandomAgent",
    "ReplayAgent",
    "ReplayBuffer",
    "SelfRegisteringAgent",
    "SimpleAgent",
]


from agents.core import (  # noqa: E402, F401  # noqa: E402, F401
    AbstractTrainableAgent,
    Agent,
    AgentRegistry,
    Log,
    ReplayBuffer,
)
from agents.dexp import DExpAgent, DExpPretrainedAgent  # noqa: E402
from agents.flat_dqn import FlatDQNAgent, Pretrained01FlatDQNAgent  # noqa: E402
from agents.greedy import GreedyAgent  # noqa: E402, F401
from agents.random import RandomAgent  # noqa: E402, F401
from agents.replay import ReplayAgent  # noqa: E402, F401
from agents.simple import SimpleAgent  # noqa: E402, F401

AgentRegistry.register("dexppretrained", DExpPretrainedAgent)
AgentRegistry.register("pretrained01flatdqn", Pretrained01FlatDQNAgent)
AgentRegistry.register("greedy", GreedyAgent)
AgentRegistry.register("random", RandomAgent)
AgentRegistry.register("simple", SimpleAgent)
