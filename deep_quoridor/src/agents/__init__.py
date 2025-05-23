__all__ = [
    "AbstractTrainableAgent",
    "ActionLog",
    "Agent",
    "AgentRegistry",
    "DExpAgent",
    "HumanAgent",
    "MCTSAgent",
    "NDexpAgent",
    "RandomAgent",
    "ReplayAgent",
    "ReplayBuffer",
    "SimpleAgent",
    "TrainableAgentParams",
]


from agents.adapter_based_agents import Cnn3CAgent, CnnAgent, NDexpAgent
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
from agents.mcts import MCTSAgent  # noqa: E402, F401
from agents.random import RandomAgent  # noqa: E402, F401
from agents.replay import ReplayAgent  # noqa: E402, F401
from agents.sb3_ppo import SB3PPOAgent  # noqa: E402, F401
from agents.simple import SimpleAgent  # noqa: E402, F401

AgentRegistry.register("cnn", CnnAgent)
AgentRegistry.register("cnn3c", Cnn3CAgent)
AgentRegistry.register("dexp", DExpAgent)
AgentRegistry.register("dexp_mimic", DExpAgent.create_from_trained_instance)
AgentRegistry.register("greedy", GreedyAgent)
AgentRegistry.register("human", HumanAgent)
AgentRegistry.register("mcts", MCTSAgent)
AgentRegistry.register("ndexp", NDexpAgent)
AgentRegistry.register("random", RandomAgent)
AgentRegistry.register("simple", SimpleAgent)
AgentRegistry.register("sb3ppo", SB3PPOAgent)
