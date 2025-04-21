import numpy as np
from quoridor import ActionEncoder, Quoridor, construct_game_from_observation

from agents.core import Agent


def sample_random_action_sequence(game: Quoridor, max_path_length: int):
    """
    Sample a random sequence of actions for a given game. Stops early if the game terminates."""
    player = game.get_current_player()

    action_sequence = []
    total_reward = 0.0
    while len(action_sequence) < max_path_length:
        # For now, assume the other agent takes random actions.
        valid_actions = game.get_valid_actions()
        assert len(valid_actions) > 0, "No valid actions available... this shouldn't be possible."

        action = np.random.choice(valid_actions)
        if game.get_current_player() == player:
            action_sequence.append(action)

        game.step(action)

        # TODO: Implement a reward function
        total_reward += 0

        if game.is_game_over():
            break

    return action_sequence, total_reward


class SimpleAgent(Agent):
    def __init__(self, sequence_length=3, num_sequences=10, **kwargs):
        super().__init__()
        self.sequence_length = sequence_length
        self.num_sequences = num_sequences
        self.board_size = kwargs["board_size"]
        self.action_encoder = ActionEncoder(self.board_size)

    def get_action(self, observation, action_mask):
        possible_action_sequences = []
        for _ in range(self.num_sequences):
            game = construct_game_from_observation(observation)
            action_sequence, total_reward = sample_random_action_sequence(game, self.sequence_length)
            possible_action_sequences.append((action_sequence, total_reward))

        # Choose the action sequence with the highest reward.
        best_sequence, _ = max(possible_action_sequences, key=lambda x: x[1])
        return self.action_encoder.action_to_index(best_sequence[0])
