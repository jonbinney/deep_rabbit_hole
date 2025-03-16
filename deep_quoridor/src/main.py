import argparse
import curses

from quoridor_env import env  # Import the environment from your file
from simple_agent import SimpleAgent


class RandomAgent:
    def __init__(self):
        pass

    def get_action(self, game):
        observation, _, termination, truncation, _ = game.last()
        mask = observation["action_mask"]
        if termination or truncation:
            return None
        return game.action_space(game.agent_selection).sample(mask)


def play(board_size: int | None, max_walls: int | None, render: str, step_rewards: bool):
    # Don't pass the None arguments to env so it uses the defaults
    args = {"board_size": board_size, "max_walls": max_walls, "step_rewards": step_rewards}
    args = {k: v for k, v in args.items() if v is not None}
    game = env(**args)

    game.reset()

    agents = {
        "player_0": RandomAgent(),
        "player_1": SimpleAgent(),
    }

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
        _, _, termination, truncation, _ = game.last()
        if termination or truncation:
            action = None
            winner = max(game.rewards, key=game.rewards.get)
            print(f"\nGame Over! {winner} wins after {step} steps.")
            break

        action = agents[agent].get_action(game)
        print(f"\nStep {step + 1}: {agent} takes action {action}")
        game.step(action)  # Apply action
        print(f"Rewards: {game.rewards}")

        board = game.render()

        if render == "curses":
            stdscr.clear()
            stdscr.addstr(2, 2, board)
            stdscr.refresh()
            curses.napms(100)

        if render == "print":
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
    parser.add_argument("-r", "--render", choices=["print", "curses"], default="print", help="Render mode")
    parser.add_argument("--step_rewards", action="store_true", default=False, help="Enable step rewards")

    args = parser.parse_args()
    play(**vars(args))
