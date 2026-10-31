#!/bin/bash
# Run the grader with nothing the agent could have planted on its path:
# - an empty environment with a fixed PATH, so no PYTHONPATH, PYTHONSTARTUP,
#   LD_PRELOAD or PATH entry survives;
# - the image's interpreter by absolute path, in isolated mode (-I: no user site,
#   no PYTHON* variables, no script or working directory on sys.path) and without
#   the site module (-S), so a sitecustomize.py anywhere is never imported;
# - a working directory the agent never writes to;
# - a wall-clock bound below the verifier's 300 s timeout.
#
# If the grader leaves no reward.json (killed by the timeout, out of memory, or
# never started because no process could be created), write an infrastructure
# verdict with shell builtins only: printf and redirection need no new process.
# metric.py counts it as unavailable, never as a score and never dropped.
set -uo pipefail
cd /
/usr/bin/env -i PATH=/usr/local/bin:/usr/bin:/bin \
  /usr/bin/timeout --kill-after=5 240 \
  /usr/local/bin/python3 -I -S /tests/grade.py
rc=$?
out=/logs/verifier
if [[ ! -e "$out/reward.json" ]]; then
  printf '{"accepted": false, "detail": "grader exited %d without a verdict", "fault": "infrastructure", "grader": "sa-replay-feasibility", "refusal": "verifier_unavailable", "reward": 0.0, "version": 3}' "$rc" > "$out/report.json"
  printf '{"accepted": 0, "agent_fault": 0, "evaluator_fault": 0, "infrastructure_fault": 1, "reward": 0.0}' > "$out/reward.json"
  printf '0.0' > "$out/reward.txt"
fi
exit 0
