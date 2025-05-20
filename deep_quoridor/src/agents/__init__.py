__all__ = [
    "AbstractTrainableAgent",
    "ActionLog",
    "Agent",
    "AgentRegistry",
    "DExpAgent",
    "HumanAgent",
    "NDexpAgent",
    "RandomAgent",
    "ReplayAgent",
    "ReplayBuffer",
    "SimpleAgent",
    "TrainableAgentParams",
]


from agents.adapter_based_agents import CnnAgent, NDexpAgent
from agents.alphazero import AlphaZeroAgent  # noqa: E402, F401
from agents.core import (  # noqa: E402, F401  # noqa: E402, F401
    AbstractTrainableAgent,
    ActionLog,
    Agent,
    AgentRegistry,
    ReplayBuffer,
    TrainableAgentParams,
)
from agents.dexp import DExpAgent  # noqa: E402
from agents.greedy import GreedyAgent  # noqa: E402, F401
from agents.human import HumanAgent  # noqa: E402, F401
from agents.random import RandomAgent  # noqa: E402, F401
from agents.replay import ReplayAgent  # noqa: E402, F401
from agents.sb3_ppo import SB3PPOAgent  # noqa: E402, F401
from agents.simple import SimpleAgent  # noqa: E402, F401

AgentRegistry.register("alphazero", AlphaZeroAgent)
AgentRegistry.register("cnn", CnnAgent)
AgentRegistry.register("dexp", DExpAgent)
AgentRegistry.register("dexp_mimic", DExpAgent.create_from_trained_instance)
AgentRegistry.register("greedy", GreedyAgent)
AgentRegistry.register("human", HumanAgent)
AgentRegistry.register("ndexp", NDexpAgent)
AgentRegistry.register("random", RandomAgent)
AgentRegistry.register("simple", SimpleAgent)
AgentRegistry.register("sb3ppo", SB3PPOAgent)
