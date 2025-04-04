from agents.core import AbstractTrainableAgent

from renderers import Renderer


class TrainingStatusRenderer(Renderer):
    def __init__(self, update_every: int, total_episodes: int, agents: list[AbstractTrainableAgent]):
        self.agents = agents
        self.update_every = update_every
        self.total_episodes = total_episodes
        self.episode_count = 0

    def start_game(self, game, agent1, agent2):
        self.step = 0
        self.player1 = agent1.name()
        self.player2 = agent2.name()

    def end_game(self, game, result):
        if self.episode_count % self.update_every == 0:
            for agent in self.agents:
                agent_name = agent.name()
                avg_reward, avg_loss = agent.compute_loss_and_reward(self.update_every)
                print(
                    f"{agent_name} Episode {self.episode_count + 1}/{self.total_episodes}, Steps: {self.step}, Avg Reward: {avg_reward:.2f}, "
                    f"Avg Loss: {avg_loss:.4f}, Epsilon: {agent.epsilon:.4f}"
                )
        self.episode_count += 1
        return

    def action(self, game, step, agent, action):
        self.step += 1
        if self.step == 1000:
            print("board")
            print(game.render())
            print(f"player0: {self.player1}")
            print(game.observe("player_0"))
            print(f"player1:  {self.player2}")
            print(game.observe("player_1"))
