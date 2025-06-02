# Test script to check the actual observation shape from Quoridor

import numpy as np
import pyspiel


def main():
    # Create the Quoridor game with specified parameters
    board_size = 9
    players = 2
    wall_count = 10

    game = pyspiel.load_game("quoridor", {"board_size": board_size, "players": players, "wall_count": wall_count})

    # Create an initial state
    state = game.new_initial_state()

    # Get the observation tensor
    observation_tensor = state.observation_tensor()
    print(f"Observation tensor shape: {np.array(observation_tensor).shape}")

    # Get the observation tensor for other players
    for player in range(players):
        obs = state.observation_tensor(player)
        print(f"Player {player} observation shape: {np.array(obs).shape}")
        obs = state.observation_string(player)
        print(f"Player {player} observation string: {obs}")

    # Get information state tensor
    # info_state_tensor = state.information_state_tensor()
    # print(f"Information state tensor shape: {np.array(info_state_tensor).shape}")

    # Print number of actions
    print(f"Number of distinct actions: {game.num_distinct_actions()}")

    # Trace a simple game to understand state and action space


def trace_simple_game():
    """
    Trace a simple Quoridor game by applying actions and observing how the state changes.
    This helps understand the action and observation spaces in OpenSpiel's Quoridor implementation.
    """
    print("\n--- TRACING A SIMPLE QUORIDOR GAME ---")

    # Create the Quoridor game with specified parameters
    board_size = 9
    players = 2
    wall_count = 10

    game = pyspiel.load_game("quoridor", {"board_size": board_size, "players": players, "wall_count": wall_count})
    state = game.new_initial_state()

    # Print game parameters
    print(f"Game type: {game.get_type()}")
    print(f"Number of players: {game.num_players()}")
    print(f"Max game length: {game.max_game_length()}")
    print(f"Number of distinct actions: {game.num_distinct_actions()}")

    # Calculate the action space breakdown
    wall_actions = (board_size - 1) * (board_size - 1) * 2
    move_actions = 4  # Up, Right, Down, Left
    print(f"Wall placement actions: 0 to {wall_actions - 1}")
    print(f"  - Horizontal walls: 0 to {(board_size - 1) * (board_size - 1) - 1}")
    print(f"  - Vertical walls: {(board_size - 1) * (board_size - 1)} to {wall_actions - 1}")
    print(f"Pawn movement actions: {wall_actions} to {wall_actions + move_actions - 1}")

    # Print initial state
    print(f"\nInitial state: {state.observation_string(0)}")
    print(f"Legal actions for player 0: {state.legal_actions()}")

    # Trace a few moves
    moves_to_trace = 100
    move_counter = 0

    print(f"\nTracing {moves_to_trace} moves:")
    while not state.is_terminal() and move_counter < moves_to_trace:
        current_player = state.current_player()
        legal_actions = state.legal_actions()

        if not legal_actions:
            print("No legal actions available. Something went wrong.")
            break

        # Choose the first legal action (for simplicity)
        action = int(input(f"Enter an action for player {current_player}: "))

        # Decode and print the action
        action_str = decode_action(action, board_size)

        print(f"\nMove {move_counter + 1}:")
        print(f"Player {current_player} taking action {action} ({action_str})")

        # Apply the action
        state.apply_action(action)

        # Print the new state
        print(f"New state after action: {state.observation_string(0)}")
        print(f"Legal actions for next player: {state.legal_actions()}")

        move_counter += 1

    if state.is_terminal():
        print("\nGame ended in a terminal state")
        returns = state.returns()
        print(f"Returns: {returns}")
        if returns[0] > returns[1]:
            print("Player 0 won")
        elif returns[1] > returns[0]:
            print("Player 1 won")
        else:
            print("Game ended in a draw")


def decode_action(action, board_size):
    """Decode an action number into a human-readable format for Quoridor."""
    wall_actions = (board_size - 1) * (board_size - 1) * 2

    if action < wall_actions:
        # Wall placement
        is_horizontal = action < (board_size - 1) * (board_size - 1)
        wall_type = "horizontal" if is_horizontal else "vertical"

        if is_horizontal:
            wall_index = action
        else:
            wall_index = action - (board_size - 1) * (board_size - 1)

        row = wall_index // (board_size - 1)
        col = wall_index % (board_size - 1)
        return f"{wall_type} wall at position ({row}, {col})"
    else:
        # Pawn movement
        move_index = action - wall_actions
        directions = ["Up", "Right", "Down", "Left"]
        direction = directions[move_index % 4]
        return f"Move pawn {direction}"


if __name__ == "__main__":
    trace_simple_game()
    # main()
