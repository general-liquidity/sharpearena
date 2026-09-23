Write one decision per line to `/logs/artifacts/actions.jsonl`.

Each line is a JSON object with:

- `step`: integer, starting at 0 and increasing by 1
- `action`: one of `"hold"`, `"buy"`, `"sell"`
- `size`: number in [0, 1]

The final line must be `{"step": <n>, "action": "close", "size": 0}`.

Write exactly 8 lines: steps 0 through 6 are decisions, step 7 is the close.
Nothing else you write is graded.
