# Counterfactual execution evidence

The Python paper-session ledger uses schema version 2. Its consumers are local
evidence readers, not a broker API. Version 1 inferred execution from risk
preflight approval and cannot establish actual fills. Do not relabel V1 records
as V2. A V1 file must be retained separately; opening it as a V2 ledger fails.

## Quantities and availability

- `executed_quantity` is the cumulative quantity confirmed by a broker receipt,
  not the requested, submitted or acknowledged quantity. It is a lower bound
  while `execution_complete` is false.
- `client_order_id` and `receipt_sha256` bind a reported fill to the order and
  the broker response. These hashes are local integrity links, not independent
  signatures or proof that a broker was truthful.
- An atomic preflight refusal leaves every order unsubmitted. The refused order
  retains its risk reason; an approved prefix is not executed.
- Accepted, pending, partially filled, unknown and malformed responses do not
  establish a final execution total. Confirmed partial quantities remain visible;
  final executed/foregone notional and P&L are `null` until execution is complete.
- Missing settlement prices produce `null`, never a fabricated zero return.
  A known zero quantity contributes zero without requiring a settlement price.
- P&L fields use the observation reference price and supplied settlement price.
  They are hypothetical reference-price marks, not realized cash P&L, fill-price
  attribution, transaction-cost estimates, or evidence of a causal risk-gate effect.
  Records label this basis `valuation: reference_price_mark`. The caller supplies
  the marks and their horizon; this ledger alone does not certify settlement timing.

## Repeated attempts

Each invocation appends a snapshot. A paper session supplies a stable decision
key binding agent, model, broker, window, decision and market observation. Later
snapshots of that key retain the original intended orders and cumulative receipts.
The aggregate uses the latest snapshot per key, so a retry cannot count a fill or
intended notional twice. The append-only history retains earlier unknown states.
Unkeyed standalone records remain independent decisions.

Call `PaperTradingSession.refresh_execution()` to query acknowledged or partially
filled orders for newer cumulative fills. This method never submits an order. A
missing or failed query leaves the outcome unresolved; absence of an already
accepted order does not authorize replacement. `reconcile_all()` handles unknown
submissions, including a persisted `submitted` state with no recorded verdict.

V2 client order IDs include broker, window and symbol-axis identity. If the same
decision finds an old client ID in a supplied lifecycle store, execution refuses
instead of assigning a new ID and risking duplicate submission. Reconcile those
legacy orders explicitly and retain their evidence before starting a new decision
window. The old lifecycle's quantity alone is not upgraded into a V2 receipt.

The forward journal's additive `preflight_order_indices` field identifies which
proposed orders were checked in this invocation. Already submitted orders are not
reserved a second time in the risk guard's projected account.

The persistent ledger reloads and validates V2 snapshots before appending. It is
a single-writer local log, not a concurrent database. Attempt-cost accounting is
separate: a snapshot count is not a model-call or token-cost measure. A process
crash may leave a submitted intent without a receipt. Such state is unresolved,
not a confirmed rejection; broker reconciliation is required before replacement.
Use a separate lifecycle store and ledger for each broker account. A broker type
name is not an account identity, and credentials are not used as public identifiers.
After a failed write, the writer refuses further appends until the log is reopened
and validated. A truncated last record is a detectable failure, not silently dropped.

## Compatibility

V2 adds execution availability, receipt identity, decision keys and snapshot
counts. Final notional/P&L and `acted` can be `null`; `confirmed_notional` remains
the observed quantity valued at the reference price. Decision counts partition
into confirmed-acted, known-unacted and unknown. Consumers must inspect the schema
version and availability rather than applying V1's all-numeric assumptions.

Direct callers constructing `GhostOrder` records attest to their own supplied
facts. Only the paper-session path binds those facts to broker responses. No
unobserved fill, experiment result or independent broker attestation is generated.
