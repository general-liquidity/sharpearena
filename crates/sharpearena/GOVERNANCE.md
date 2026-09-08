# SharpeArena Agent Interface: Governance

The contract is the product. Agents are **external** programs in any language that
read a `MarketObservation` (JSON) and reply with a `Decision` (JSON). The whole
adoption story is that this surface stays tiny and stable, so any vendor can build
against it once and not have it move under them. This document is the promise that
backs that.

The authoritative artifacts are:

- `contract/observation.schema.json` and `contract/decision.schema.json`: JSON
  Schema (draft 2020-12) that non-Rust implementers validate against.
- `src/contract.rs`: `CONTRACT_VERSION`, the single source of truth for the wire
  version.
- `contract/conformance/*.json` + `tests/conformance.rs`: the conformance kit.

## 1. Additive-only evolution

The contract evolves **additively**. The only backwards-compatible change is adding
a **new field that is optional with a default**: every field a producer may omit
must deserialize on the consumer to a sensible default.

The compatibility that buys is **one-directional, and the direction matters**. An
optional default lets a reader accept a message written against an *older* version,
because the absent field takes its default. It does not let a reader accept a message
written against a *newer* version. The published schemas set
`additionalProperties: false`, so a consumer validating against the schema it was
written for **rejects** a newer message carrying the new field.

Apply that to the two message directions:

- **Decisions** travel agent to harness. The harness is always the newer reader, so an
  added optional decision field is genuinely safe without coordination.
- **Observations** travel harness to agent. The agent is the older reader, so an added
  optional observation field is **not** transparent to an agent validating against the
  published schema it already has. It needs the same announcement and deprecation
  treatment as any other change an agent must act on.

Before adding an observation field under the additive promise, pick one and say which:
version negotiation at the handshake, a parallel namespace as for a breaking change, or
a deliberately open extension envelope whose contents are declared unvalidated. Do not
rely on the field being optional alone.

Concretely, in Rust this means the field carries `#[serde(default)]` (or
`#[serde(default = "…")]`). Today's optional-with-default fields are `fundamentals`
and `news` on `SymbolSnapshot`, `confidence` and `rationale` on `Order`, and
`reasoning` on `Decision`. They are **not** in the schemas' `required` lists, by
construction.

The following are **breaking** and are forbidden on the v1 surface:

- removing a field,
- renaming a field,
- retyping a field (e.g. `number` → `string`, widening an enum's existing variant),
- making a previously optional field required, or
- removing or renaming an `action` enum value.

A breaking change does not mutate `MarketObservation` / `Decision` in place. It ships
a **parallel namespace**: `ObservationV2` / `DecisionV2` with their own schemas and
their own `CONTRACT_VERSION` major, and the two run side by side through the
deprecation window below. Adding a *new* `action` variant is itself breaking for
consumers that exhaustively match, so it also goes through V2, not an additive bump.

## 2. `CONTRACT_VERSION` semantics

`CONTRACT_VERSION` (currently `"1.0"`) versions the **wire shape only**. It is
deliberately decoupled from the `sharpearena` crate version and the npm / PyPI package
versions, which move on their own release cadence.

- **Major** (`1.0` → `2.0`): a breaking change shipped as a parallel `…V2` namespace.
- **Minor** (`1.0` → `1.1`): reserved for a *batch* of additive fields significant
  enough to advertise. A single additive **decision** field needs no bump: the harness
  is the newer reader and existing agents are unaffected. A single additive
  **observation** field does need the bump, because an agent validating against the
  older schema rejects it; the conformance badge stays valid only for the schema
  version it was issued against.

A package release never, on its own, bumps `CONTRACT_VERSION`. Bug fixes, new baseline
agents, docs, and extra re-exports change the package version and leave the contract
version frozen.

## 3. Deprecation window

When a major (`V2`) lands, the previous major is **supported in parallel for at least
two minor package releases (no less than 90 days)**, whichever is longer. During the
window:

1. both namespaces deserialize and run; the harness accepts either,
2. the superseded version is marked `#[deprecated]` in Rust and flagged as deprecated
   in the schema `description`, and
3. the changelog states the removal release up front.

Only after the window closes may the old namespace be removed, and that removal is
itself a major package release.

## 4. Conformance badge

An implementation that passes the conformance kit
(`contract/conformance/*.json` exercised by `tests/conformance.rs`, or the equivalent
validation against the published JSON Schemas) may state:

> **conforms to the SharpeArena Agent Interface v1.0**

To earn it, an agent must, for every observation in the kit:

1. parse the `MarketObservation` against `observation.schema.json`,
2. emit a `Decision` that validates against `decision.schema.json`: every `action`
   in the enum, every `target_weight` finite, and every order `symbol` a subset of
   the observed symbols, and
3. round-trip the legacy decision shape (no `confidence` / `rationale` / `reasoning`)
   without error, proving the additive-only discipline holds.

The badge names the **contract** version it was earned against (`v1.0`), not the
package version. It remains valid across additive (non-bumping) changes and must be
re-earned against a new major.
