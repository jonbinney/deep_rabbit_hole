from quoridor_env import env  # Import the environment from your file

# Initialize environment
game = env()
game.reset()

# Print initial board
print("Initial Board State:")
board = game.render()
print(board)

# Manually select moves (modify these to test different scenarios)
actions = [
    game.action_params_to_index(1, 4, 0),
    game.action_params_to_index(7, 4, 0),
    game.action_params_to_index(6, 4, 2),
    game.action_params_to_index(0, 3, 1),
]  # Example moves for testing

step = 0
for agent in game.agent_iter():
    observation, reward, termination, truncation, info = game.last()

    mask = observation["action_mask"]
    print(f"Valid moves for agent {agent}:")
    for i in range(game.board_size**2):  # For now only showing moves
        if mask[i] == 1:
            print(f"{i}: {game.action_index_to_params(i)}")

    # End of hardcoded test actions
    if len(actions) == 0:
        break

    if termination or truncation:
        action = None
        print(f"\nGame Over! {agent} wins.")
    else:
        # Hardcoded actions for now
        action = actions.pop(0)
        print(f"\nStep {step + 1}: {agent} takes action {action}")

    game.step(action)  # Apply action

    board = game.render()  # Print updated board
    print(board)

    step += 1

game.close()
