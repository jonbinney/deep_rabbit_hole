# META: Prompt to use:

Delete the osaz directory. Then look at the requirements in file osaz.requirements.md and generate the code incrementally.

# OpenSpiel AlphaZero Agent

This file enumarates all the requirements for the generation of an AlphaZero agent using the OpenSpiel framework.

# Requirements

 - Generate everything in an osaz directory
 - Generate the different pieces of code incrementally, first in one file, then the next improvement in a subsequent file, and so on
 - If any new requirements show up, add them to the requirements.txt file one level up from this file instead of creating a new requirements file

## Details to take into account in all files

 - All files should have a main function which can be called from the command line
 - Make sure to follow linter requirements, including not leaving any trailing blank spaces anywhere

## First request

 - First of all, generate a file tictactoe.py which uses OpenSpiel to play an MCTS agent against a random agent in the game of Tic Tac Toe

## Second request
 - Then, in a different file, modify what you did in the first one and change the game to Quoridor

