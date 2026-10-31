"""Quality gate (AGENTS.md Rule 0): lint, format check, then the whole test suite."""

import subprocess
import sys

STEPS = (
    ("ruff", "check", "."),
    ("ruff", "format", "--check", "."),
    ("pytest",),
)

for step in STEPS:
    print("$", *step, flush=True)
    if subprocess.call([sys.executable, "-m", *step]) != 0:
        sys.exit(f"gate failed at: {' '.join(step)}")
print("gate passed")
