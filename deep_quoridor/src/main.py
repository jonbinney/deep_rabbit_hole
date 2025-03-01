from quoridor_env import env  # Import the environment from your file

# Initialize environment
game = env()
game.reset()

# Print initial board
print("Initial Board State:")
board = game.render()
print(board)

# Manually select moves (modify these to test different scenarios)
actions = [game.rowcol_to_idx(4, 1), game.rowcol_to_idx(4, 7)]  # Example moves for testing

for step, action in enumerate(actions):
    agent = game.agent_selection
    print(f"\nStep {step + 1}: {agent} takes action {action}")

    game.step(action)  # Apply action
    board = game.render()  # Print updated board
    print(board)

    if game.terminations[agent]:  # Check if someone won
        print(f"\nGame Over! {agent} wins.")
        break
