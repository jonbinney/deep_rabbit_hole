from agents.agent import Agent


def sample_random_action_sequence(game, max_path_length):
    """
    Sample a random sequence of actions for a given game. Stops early if the game terminates."""
    agent_name = game.agent_selection

    action_sequence = []
    total_reward = 0.0
    while len(action_sequence) < max_path_length:
        observation, reward, termination, truncation, _ = game.last()
        mask = observation["action_mask"]

        # For now, assume the other agent takes random actions.
        if game.agent_selection != agent_name:
            action = game.action_space(game.agent_selection).sample(mask)
            game.step(action)
            continue

        total_reward += reward
        if termination or truncation:
            break

        action = game.action_space(game.agent_selection).sample(mask)
        action_sequence.append(action)
        game.step(action)

    return action_sequence, total_reward


class SimpleAgent(Agent):
    def __init__(self, sequence_length=3, num_sequences=10):
        super().__init__()
        self.sequence_length = sequence_length
        self.num_sequences = num_sequences

    def get_action(self, game):
        _, _, termination, truncation, _ = game.last()
        if termination or truncation:
            return None

        possible_action_sequences = []
        for _ in range(self.num_sequences):
            action_sequence, total_reward = sample_random_action_sequence(game.copy(), self.sequence_length)
            possible_action_sequences.append((action_sequence, total_reward))

        # Choose the action sequence with the highest reward.
        best_sequence, _ = max(possible_action_sequences, key=lambda x: x[1])
        return best_sequence[0]


class BetterSimple(SimpleAgent):
    def __init__(self, sequence_length=3, num_sequences=10):
        super().__init__(20, 20)
