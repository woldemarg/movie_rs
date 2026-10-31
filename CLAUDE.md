# CLAUDE.md

@AGENTS.md

The rules are in AGENTS.md (imported above). This file only adds what is specific to Claude Code.

- Temporary scripts and dumps go to the session scratchpad, never into the repository.
- On Windows, heredocs with backslashes or `\n` get mangled by the shell layer: put edit scripts in a file and run them with the venv interpreter.
