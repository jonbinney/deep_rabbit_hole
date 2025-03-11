from quoridor_env import env  # Import the environment from your file

# Initialize environment
game = env()
game.reset()

# Print initial board
print("Initial Board State:")
board = game.render()
print(board)

step = 0
for agent in game.agent_iter():
    observation, reward, termination, truncation, info = game.last()

    mask = observation["action_mask"]

    if termination or truncation:
        action = None
        print(f"\nGame Over! {game.get_opponent(agent)} wins.")
        break
    else:
        # Hardcoded actions for now
        action = game.action_space(agent).sample(mask)
        print(f"\nStep {step + 1}: {agent} takes action {action}")

    game.step(action)  # Apply action

    board = game.render()  # Print updated board
    print(board)

    step += 1

game.close()
