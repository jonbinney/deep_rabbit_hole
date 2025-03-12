import curses
import argparse

from quoridor_env import env  # Import the environment from your file


def play(board_size: int | None, max_walls: int | None, render: str):
    # Don't pass the None arguments to env so it uses the defaults
    args = {"board_size": board_size, "max_walls": max_walls}
    args = {k: v for k, v in args.items() if v is not None}
    game = env(**args)

    game.reset()

    if render == "print":
        print("Initial Board State:")
        board = game.render()
        print(board)
    elif render == "curses":
        stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        curses.start_color()

    step = 0

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

        if render == "curses":
            stdscr.clear()
            stdscr.addstr(2, 2, board)
            stdscr.refresh()
            curses.napms(500)

        print(board)

        step += 1

    if render == "curses":
        curses.nocbreak()
        curses.echo()
        curses.endwin()

    game.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Quoridor")
    parser.add_argument("-N", "--board_size", type=int, default=None, help="Board Size")
    parser.add_argument("-W", "--max_walls", type=int, default=None, help="Max walls per player")
    parser.add_argument("-r", "--render", choices=["print", "curses"], default="curses", help="Render mode")

    args = parser.parse_args()
    play(args.board_size, args.max_walls, args.render)
