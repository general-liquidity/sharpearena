#!/bin/bash
set -euo pipefail
mkdir -p /logs/artifacts
cat > /logs/artifacts/actions.jsonl <<'JSON'
{"step": 0, "action": "buy", "size": 1.0}
{"step": 1, "action": "sell", "size": 1.0}
{"step": 2, "action": "buy", "size": 1.0}
{"step": 3, "action": "buy", "size": 1.0}
{"step": 4, "action": "sell", "size": 1.0}
{"step": 5, "action": "buy", "size": 1.0}
{"step": 6, "action": "buy", "size": 1.0}
{"step": 7, "action": "close", "size": 0}
JSON
