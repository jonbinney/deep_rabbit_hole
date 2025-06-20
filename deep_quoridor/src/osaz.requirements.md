# META: Used prompts:

 1. Delete the osaz directory. Then look at the requirements in file osaz.requirements.md and generate the code incrementally.
 2. Look at the requirements in file osaz.requirements.md and generete the thierd request, the previous two are ready. Use the contents of this file as context for every new generated code. When you're finished, create a file called 02_generation_summary.md in the osaz/logs directory and write a summary of the generation process in it.

# OpenSpiel AlphaZero Agent

This file enumarates all the requirements for the generation of an AlphaZero agent using the OpenSpiel framework.

# Requirements

 - Generate everything in an osaz directory
 - Generate the different pieces of code incrementally, first in one file, then the next improvement in a subsequent file, and so on
 - If any new python pip requirements show up, add them to the requirements.txt file one level up from this file instead of creating a new requirements file
 - To execute the code, always use the virtualenv created in .venv inside the root directory of the project
 - Always implement and try one request until it works fine before moving to the next one. Don't start with the next one until the previous one is working.

## Details to take into account in all files

 - All files should have a main function which can be called from the command line
 - After you've run and confirmed that the new code works and only then, clean-up the code to adhere to linter requirements


## First request

 - First of all, generate a file tictactoe.py which uses OpenSpiel to play an MCTS agent against a random agent in the game of Tic Tac Toe
 - Make sure to add the evaluator parameter to the MCTSBot constructor in both files.  Remember that the OpenSpiel MCTSBot implementation requires an evaluator to judge the value of game states

### Things learned from when this was generated

 - Don't include `simple_plotlib` import, it's not used
 - Avoid duplicating evaluator parameters in MCTSBot constructor
 - Use OpenSpiel's built-in RandomRolloutEvaluator

## Second request
 - Then, in a different file, modify what you did in the first one and change the game to Quoridor

### Things learned from when this was generated

 - Make sure parameters in quoridor.py are converted to integers
 - The correct parameter name is "wall_count" not "walls_per_player"

## Third request
 - Then, in a next different file, create a program that trains an AlphaZero agent to play Quoridor
 - Make it so that the model is saved to a file after a configurable number of training iterations
 - Include the ability to run a number of games alternating as P0 and P1 against a random agent, starting from the loaded model file
