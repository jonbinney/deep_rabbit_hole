# Conversation Transcript - Part 3

## Assistant (continued)
It looks like we're still having an issue with the MCTS implementation. Let's try a different approach by using OpenSpiel's built-in evaluator instead of our custom one. I'll update both files:

[Function: replace_file_content to use OpenSpiel's built-in RandomRolloutEvaluator in tictactoe.py]

Let me update the quoridor.py file in the same way as tictactoe.py, using OpenSpiel's built-in RandomRolloutEvaluator:

[Function: replace_file_content to use OpenSpiel's built-in RandomRolloutEvaluator in quoridor.py]

Now let me run the tictactoe.py file to see if our fix worked:

[Function: run_command to execute tictactoe.py with fixed evaluator]

```
Exit code: 0
Output:
Game 1/1
Initial state: ...
...
...
Player 0 chooses action: 4
State after action: ...
.x.
...
Player 1 chooses action: 7
State after action: ...
.x.
.o.
Player 0 chooses action: 6
State after action: ...
.x.
xo.
Player 1 chooses action: 3
State after action: ...
ox.
xo.
Player 0 chooses action: 2
State after action: ..x
ox.
xo.
Game 1 complete. Winner: MCTS Bot (Player 0)

Overall results:
MCTS Bot (Player 0) wins: 1
Random Bot (Player 1) wins: 0
Draws: 0
Win percentages: MCTS: 100.0%, Random: 0.0%, Draw: 0.0%
```

Great! The tictactoe.py file is now working successfully. Let's run the quoridor.py file to ensure it works as well:

[Function: run_command to execute quoridor.py]

```
Exit code: 1
Output:
Warning! This game has known issues. Please see the games list on github or the code for details.
OpenSpiel exception: Wrong type for parameter board_size. Expected type: kInt, got kString with 5
Traceback (most recent call last):
  File "/home/julian/aaae/deep-rabbit-hole/code/deep_rabbit_hole/deep_quoridor/src/osaz/quoridor.py", line 144, in <module>
    app.run(main)
  File "/home/julian/aaae/deep-rabbit-hole/code/deep_rabbit_hole/.venv/lib/python3.12/site-packages/absl/app.py", line 308, in run
    _run_main(main, args)
  File "/home/julian/aaae/deep-rabbit-hole/code/deep_rabbit_hole/.venv/lib/python3.12/site-packages/absl/app.py", line 254, in _run_main
    sys.exit(main(argv))
             ^^^^^^^^^^
  File "/home/julian/aaae/deep-rabbit-hole/code/deep_rabbit_hole/deep_quoridor/src/osaz/quoridor.py", line 42, in main
    game = pyspiel.load_game("quoridor", game_parameters)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
pyspiel.SpielError: Wrong type for parameter board_size. Expected type: kInt, got kString with 5
```

## User
Continue

## Assistant
I see the issue in the quoridor.py file. The game parameters need to be integers rather than strings. Let me fix that:

[Function: replace_file_content to fix parameter types in quoridor.py]

Let me run the quoridor.py file to see if our fix for the parameter types has resolved the issue:

[Function: run_command to execute quoridor.py with fixed parameter types]

```
Exit code: 1
Output:
Warning! This game has known issues. Please see the games list on github or the code for details.
OpenSpiel exception: Unknown parameter 'walls_per_player'. Available parameters are: ansi_color_output, board_size, players, wall_count
Traceback (most recent call last):
  File "/home/julian/aaae/deep-rabbit-hole/code/deep_rabbit_hole/deep_quoridor/src/osaz/quoridor.py", line 144, in <module>
    app.run(main)
  File "/home/julian/aaae/deep-rabbit-hole/code/deep_rabbit_hole/.venv/lib/python3.12/site-packages/absl/app.py", line 308, in run
    _run_main(main, args)
  File "/home/julian/aaae/deep-rabbit-hole/code/deep_rabbit_hole/.venv/lib/python3.12/site-packages/absl/app.py", line 254, in _run_main
    sys.exit(main(argv))
             ^^^^^^^^^^
  File "/home/julian/aaae/deep-rabbit-hole/code/deep_rabbit_hole/deep_quoridor/src/osaz/quoridor.py", line 42, in main
    game = pyspiel.load_game("quoridor", game_parameters)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
pyspiel.SpielError: Unknown parameter 'walls_per_player'. Available parameters are: ansi_color_output, board_size, players, wall_count
```
