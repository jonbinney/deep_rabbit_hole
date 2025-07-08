__all__ = [
    "AbstractTrainableAgent",
    "ActionLog",
    "Agent",
    "AgentRegistry",
    "ReplayAgent",
    "ReplayBuffer",
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
from agents.replay import ReplayAgent

AgentRegistry.register("alphazero", "AlphaZeroAgent", "agents.alphazero")
AgentRegistry.register("alphazero_os", "AlphaZeroOSAgent", "agents.alphazero_os")
AgentRegistry.register("cnn", "CnnAgent", "agents.adapter_based_agents")
AgentRegistry.register("cnn3c", "Cnn3CAgent", "agents.adapter_based_agents")
AgentRegistry.register("daz", "DAZAgent", "agents.alphazero_dexp")
AgentRegistry.register("daz_mimic", "DAZAgent.create_from_trained_instance", "agents.alphazero_dexp")
AgentRegistry.register("dexp", "DExpAgent", "agents.dexp")
AgentRegistry.register("dexp_mimic", "DExpAgent.create_from_trained_instance", "agents.dexp")
AgentRegistry.register("greedy", "GreedyAgent", "agents.greedy")
AgentRegistry.register("human", "HumanAgent", "agents.human")
AgentRegistry.register("mcts", "MCTSAgent", "agents.mcts")
AgentRegistry.register("ndexp", "NDexpAgent", "agents.adapter_based_agents")
AgentRegistry.register("random", "RandomAgent", "agents.random")
AgentRegistry.register("simple", "SimpleAgent", "agents.simple")
AgentRegistry.register("sb3ppo", "SB3PPOAgent", "agents.sb3_ppo")
