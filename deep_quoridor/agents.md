Always separate commits between actual function change and formatting / linting. Create one commit with the actual functional changes, and then another separate
commit with only the formatting and linting changes. This will make it easier for reviewers to review the functionality first and the formatting separately.

whenever you need to use python, remember to activate the virtualenv which is in the .venv directory in the root of the repo.

Whenever you commit to git, follow these commit message guidelines:
- Start the subject line with "vibe: " followed by a summary in imperative mood
  (e.g. "vibe: add feature" not "vibe: added feature").
- Keep the subject line to 50 characters max (not counting the "vibe: " prefix).
- For non-trivial changes, add a body separated by a blank line from the subject.
  Wrap body lines at 72 characters. Focus on **why** the change was made, not
  what files were changed (the diff shows that).
- Do not enumerate modified files in the commit message.

Whenever you change rust files, before commit, make sure to run cargo fmt to format all files and then check formatting, build and run before committing.

Rules specific to the rust implementation (deep_quoridor/rust folder):
 - Whenever possible, keep compatibility with the python version implemented in deep_quoridor/src
 - keep clear separation of responsibilities. e.g.: game state functions in game_state.rs

If the work is relatively large, write one commit per change, so that the updates are easier to understand and review.

Always write the plan to file as a markdown file before starting implementation.

When you finish, write the results of what you did on a markdown file so that its contents can be written in a PR request.