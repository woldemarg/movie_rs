"""Quality gate (AGENTS.md Rule 0): lint, format check, the catalogue, then the whole test suite.

The API, e2e, catalogue and benchmark tests run against the real catalogue in data/, which is
generated, not versioned: the gate builds it first when it is missing (once, ~3 min; with
TMDB_API_KEY in .env also the posters, through the TMDB API) and otherwise does nothing.
"""

import subprocess
import sys

STEPS = (
    ("ruff", "check", "."),
    ("ruff", "format", "--check", "."),
    ("scripts.prepare_data",),
    ("pytest",),
)

for step in STEPS:
    print("$", *step, flush=True)
    if subprocess.call([sys.executable, "-m", *step]) != 0:
        sys.exit(f"gate failed at: {' '.join(step)}")
print("gate passed")
