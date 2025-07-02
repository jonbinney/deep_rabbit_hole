import time

from agents.core.trainable_agent import TrainableAgent

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
        self.init_time = time.time()

    def end_game(self, game, result):
        self.episode_count += 1
        total_time = round((time.time() - self.init_time) * 1000)
        for agent in self.agents:
            if not isinstance(agent, TrainableAgent) or not agent.is_training():
                continue
            agent_name = agent.name()
            avg_loss, avg_reward = agent.compute_loss_and_reward(self.update_every)
            won = result.winner == agent.name()
            p = "1" if result.player1 == agent.name() else "2"
            print(
                f"{agent_name} P{p} Episode {self.episode_count:5d}/{self.total_episodes} [{'*' if won else ' '}], Steps: {self.step:3d}, Time: {total_time:5d}ms,  Avg Reward: {avg_reward:6.2f}, "
                f"Avg Loss: {avg_loss:7.4f}, Epsilon: {agent.epsilon:.4f} opponent: {self.player1 if agent_name == self.player2 else self.player2}"
            )
        return

    def after_action(self, game, step, agent, action):
        self.step += 1
