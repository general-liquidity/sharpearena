# Prospective local-model forecast field

This phased runner creates public evidence that local model forecasts were
sealed before their resolving market candles existed. It uses Binance's
read-only public market-data host and loads model snapshots with
`local_files_only=True`. It never downloads a model and never trades.

The field has four phases:

1. `prepare` freezes the observation, contracts, models, stopping rule, and
   analysis plan.
2. `forecast` runs each preregistered local model and writes pending evidence.
   Each probability is the normalized next-token preference for `1` versus
   `0`, with both logits recorded. This fixed scaffold produces exact support
   without repairing or selecting free-form model text.
3. `seal` binds every pending ledger and raw inference record before the frozen
   deadline. Commit and push that directory at this point.
4. `resolve` refuses to run until Binance server time is later than every target
   candle, then writes the raw candles and resolved evidence.

Model arguments use the form
`AGENT_ID,MODEL_ID,IMMUTABLE_REVISION,SNAPSHOT_PATH`. The revision is a
40-character lowercase content revision and must equal the snapshot directory
name. The snapshot is hashed by relative path, file length, and every file byte
during both preparation and inference. The plan also freezes the four Python
modules that define inference, contracts, ledgers, and settlement.

```bash
python -m sharpearena.prospective_field prepare \
  --field-dir paper/evidence/prospective-forecast-field \
  --deadline-delay-minutes 30 \
  --model 'agent-small,owner/model-small,REVISION,/absolute/local/snapshot' \
  --model 'agent-large,owner/model-large,REVISION,/absolute/local/snapshot'

python -m sharpearena.prospective_field forecast \
  --field-dir paper/evidence/prospective-forecast-field \
  --model 'agent-small,owner/model-small,REVISION,/absolute/local/snapshot'

python -m sharpearena.prospective_field seal \
  --field-dir paper/evidence/prospective-forecast-field

# Commit and push the pending evidence before this command is allowed to run.
python -m sharpearena.prospective_field resolve \
  --field-dir paper/evidence/prospective-forecast-field
```

The default design contains 24 binary contracts: four crypto pairs at six
future one-minute resolution clocks. Its primary output is descriptive Brier
loss and calibration. Pairwise bootstrap results are exploratory because six
time blocks are below the preregistered 30-block threshold for comparative
claims.
