Whenever you commit to git, create the commit message starting with "vibe: " and then a one line summary of the changes.

Whenever you change rust files, before commit, make sure to run cargo fmt to format all files and then check formatting, build and run before committing.

Rules specific to the rust implementation (deep_quoridor/rust folder):
 - Whenever possible, keep compatibility with the python version implemented in deep_quoridor/src
 - keep clear separation of responsibilities. e.g.: game state functions in game_state.rs