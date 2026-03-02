Whenever you commit to git, create the commit message starting with "vibe: " and then a one line summary of the changes.

Whenever you change rust files, before commit, make sure to run cargo fmt to format all files and then check formatting, build and run before committing.

Rules specific to the rust implementation (deep_quoridor/rust folder):
 - Whenever possible, keep compatibility with the python version implemented in deep_quoridor/src
 - keep clear separation of responsibilities. e.g.: game state functions in game_state.rs

If the work is relatively large, write one commit per change, so that the updates are easier to understand and review.

Always write the plan to file as a markdown file before starting implementation.

When you finish, write the results of what you did on a markdown file so that its contents can be written in a PR request.