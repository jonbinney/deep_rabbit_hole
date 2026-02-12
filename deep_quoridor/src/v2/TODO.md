# V1 Parity

- Replay buffer length: righ now we're not rolling out old games to respect the length
- Overrides from the command line
- Continuation

# Other improvements and new features

- Use a schedule for the learning rate
- Use a logger class rather than just printing out.
- Instead of drop_t_on_step, do near-tie-randomness, e.g. if the second best choice is 95% or more of the first, pick between them.

# Performance

- In train, sample from all the training iterations together
- Allow num_workers in benchmarks if we want to run them faster, or at least run jobs in parallel
- For the replay buffer files, either:
  - Move them in directories based on their game number (e.g. games_1000)
  - Join multiple games in one file (a bit more tricky with sampling and race conditions)

# Ideas

- Self-healing processes
- Allow to dynamically change the number of workers and parallel games, to experiment with performance
- Mount the run directory and make other processes play from another computer
- The processes could write status files and we could have a script to watch the status (e.g. elapsed time.)
- Tuning: a script that would take a list of num-workers and a list of parallel games, as well as time to run, and will execute the different combinations for that time and output how many games were played in each combination, to find the fastest one.
