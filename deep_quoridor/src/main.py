import curses

from quoridor_env import env  # Import the environment from your file

# Initialize environment
game = env()
game.reset()

# Print initial board
print("Initial Board State:")
board = game.render()
print(board)

step = 0
stdscr = curses.initscr()
curses.noecho()
curses.cbreak()
curses.start_color()

for agent in game.agent_iter():
    observation, reward, termination, truncation, info = game.last()

    mask = observation["action_mask"]

    if termination or truncation:
        action = None
        winner = max(game.rewards, key=game.rewards.get)
        print(f"\nGame Over! {winner} wins.")
        break
    else:
        # Hardcoded actions for now
        action = game.action_space(agent).sample(mask)
        print(f"\nStep {step + 1}: {agent} takes action {action}")

    game.step(action)  # Apply action

    board = game.render()
    stdscr.clear()
    stdscr.addstr(0, 0, board)
    stdscr.refresh()
    curses.napms(500)
    print(board)

    step += 1

game.close()
curses.nocbreak()
curses.echo()
curses.endwin()
