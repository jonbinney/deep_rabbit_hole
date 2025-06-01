# Test script to check the actual observation shape from Quoridor

import numpy as np
import pyspiel

def main():
    # Create the Quoridor game with specified parameters
    board_size = 5
    players = 2
    wall_count = 3
    
    game = pyspiel.load_game(
        "quoridor", 
        {"board_size": board_size, "players": players, "wall_count": wall_count}
    )
    
    # Create an initial state
    state = game.new_initial_state()
    
    # Get the observation tensor
    observation_tensor = state.observation_tensor()
    print(f"Observation tensor shape: {np.array(observation_tensor).shape}")
    
    # Get the observation tensor for other players
    for player in range(players):
        obs = state.observation_tensor(player)
        print(f"Player {player} observation shape: {np.array(obs).shape}")
    
    # Get information state tensor
    info_state_tensor = state.information_state_tensor()
    print(f"Information state tensor shape: {np.array(info_state_tensor).shape}")
    
    # Print number of actions
    print(f"Number of distinct actions: {game.num_distinct_actions()}")
    
if __name__ == "__main__":
    main()
