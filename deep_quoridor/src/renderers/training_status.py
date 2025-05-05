from agents.core.trainable_agent import AbstractTrainableAgent

from renderers import Renderer


class TrainingStatusRenderer(Renderer):
    def __init__(self, total_episodes: int):
        self.update_every = 1
        self.total_episodes = total_episodes
        self.episode_count = 0

    def start_game(self, game, agent1, agent2):
        self.step = 0
        self.agents = [agent1, agent2]
        self.player1 = agent1.name()
        self.player2 = agent2.name()

    def end_game(self, game, result):
        for agent in self.agents:
            if not isinstance(agent, AbstractTrainableAgent) or not agent.training_mode:
                continue
            agent_name = agent.name()
            avg_loss, avg_reward = agent.compute_loss_and_reward(self.update_every)
            won = result.winner == agent.name()
            print(
                f"{agent_name} Episode {self.episode_count + 1:5d}/{self.total_episodes} [{'*' if won else ' '}], Steps: {self.step:3d}, Avg Reward: {avg_reward:6.2f}, "
                f"Avg Loss: {avg_loss:7.4f}, Epsilon: {agent.epsilon:.4f} opponent: {self.player1 if agent_name == self.player2 else self.player2}"
            )
        self.episode_count += 1
        return

    def after_action(self, game, step, agent, action):
        self.step += 1
