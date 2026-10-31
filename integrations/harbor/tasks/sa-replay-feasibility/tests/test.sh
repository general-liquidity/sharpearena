#!/bin/bash
# Run the grader with nothing the agent could have planted on its path:
# - an empty environment with a fixed PATH, so no PYTHONPATH, PYTHONSTARTUP,
#   LD_PRELOAD or PATH entry survives;
# - the image's interpreter by absolute path, in isolated mode (-I: no user site,
#   no PYTHON* variables, no script or working directory on sys.path) and without
#   the site module (-S), so a sitecustomize.py anywhere is never imported;
# - a working directory the agent never writes to;
# - a wall-clock bound below the verifier's 300 s timeout. A killed grader writes
#   no reward, which metric.py counts as crashed, never as a score.
set -uo pipefail
cd /
exec /usr/bin/env -i PATH=/usr/local/bin:/usr/bin:/bin \
  /usr/bin/timeout --kill-after=5 240 \
  /usr/local/bin/python3 -I -S /tests/grade.py
