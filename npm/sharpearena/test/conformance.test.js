/**
 * The npm leg of the cross-language conformance kit.
 *
 * The kit lives at `crates/sharpearena/contract/` and is indexed by the versioned
 * `conformance-kit.v1.json`. Rust checks it in `crates/sharpearena/tests/`, Python in
 * `crates/sharpearena-py/tests/test_wire_conformance.py`, and this file checks it from
 * Node. All three read the same files by repository path, exactly as `golden.test.js` and
 * `spec-hash.test.js` already read the attestation records, so no package here depends on
 * either of the other two and the kit gains no dependency cycle.
 *
 * The validator below is a deliberate subset of draft 2020-12: object/array/scalar types,
 * `required`, `properties`, `additionalProperties` (boolean or schema), `items`, `enum`,
 * `minimum` and local `$ref`. That is every keyword the two contract schemas actually use,
 * and it keeps this package free of a validation dependency it would otherwise ship or
 * pin. A schema that starts using a keyword outside the subset fails loudly rather than
 * being silently ignored.
 */
const test = require("node:test");
const assert = require("node:assert");
const fs = require("node:fs");
const path = require("node:path");

const REPO = path.resolve(__dirname, "../../..");
const CONTRACT = path.join(REPO, "crates/sharpearena/contract");

const readJson = (...parts) => JSON.parse(fs.readFileSync(path.join(...parts), "utf8"));

const KIT = readJson(CONTRACT, "conformance-kit.v1.json");
const schema = (kind) => readJson(CONTRACT, KIT.schemas[kind]);
const fixture = (name) => readJson(CONTRACT, "conformance", name);

const SUPPORTED = new Set([
  "$schema",
  "$id",
  "$ref",
  "$defs",
  "title",
  "description",
  "type",
  "required",
  "properties",
  "additionalProperties",
  "items",
  "enum",
  "minimum",
]);

function typeMatches(type, value) {
  switch (type) {
    case "object":
      return typeof value === "object" && value !== null && !Array.isArray(value);
    case "array":
      return Array.isArray(value);
    case "string":
      return typeof value === "string";
    case "number":
      return typeof value === "number" && Number.isFinite(value);
    case "integer":
      return Number.isInteger(value);
    case "boolean":
      return typeof value === "boolean";
    case "null":
      return value === null;
    default:
      throw new Error(`unsupported schema type: ${type}`);
  }
}

function validate(node, value, root, where, errors) {
  for (const keyword of Object.keys(node)) {
    if (!SUPPORTED.has(keyword)) {
      throw new Error(`${where}: schema keyword outside this validator's subset: ${keyword}`);
    }
  }
  if (node.$ref) {
    const match = /^#\/\$defs\/(.+)$/.exec(node.$ref);
    if (!match) throw new Error(`${where}: only local #/$defs refs are supported: ${node.$ref}`);
    const target = root.$defs?.[match[1]];
    if (!target) throw new Error(`${where}: unresolved $ref ${node.$ref}`);
    validate(target, value, root, where, errors);
    return errors;
  }
  if (node.type && !typeMatches(node.type, value)) {
    errors.push(`${where}: expected ${node.type}, got ${JSON.stringify(value)}`);
    return errors;
  }
  if (node.enum && !node.enum.includes(value)) {
    errors.push(`${where}: ${JSON.stringify(value)} is not one of ${node.enum.join(", ")}`);
  }
  if (node.minimum !== undefined && typeof value === "number" && value < node.minimum) {
    errors.push(`${where}: ${value} is below the minimum ${node.minimum}`);
  }
  if (Array.isArray(value) && node.items) {
    value.forEach((item, i) => validate(node.items, item, root, `${where}[${i}]`, errors));
  }
  if (typeMatches("object", value)) {
    for (const key of node.required ?? []) {
      if (!Object.hasOwn(value, key)) errors.push(`${where}: missing required property ${key}`);
    }
    for (const [key, child] of Object.entries(value)) {
      const property = node.properties?.[key];
      if (property) {
        validate(property, child, root, `${where}.${key}`, errors);
      } else if (node.additionalProperties === false) {
        errors.push(`${where}: unknown property ${key}`);
      } else if (typeMatches("object", node.additionalProperties)) {
        validate(node.additionalProperties, child, root, `${where}.${key}`, errors);
      }
    }
  }
  return errors;
}

function assertValid(kind, value, where) {
  const root = schema(kind);
  const errors = validate(root, value, root, where, []);
  assert.deepEqual(errors, [], `${where} does not conform to ${KIT.schemas[kind]}`);
}

test("the kit names exactly the fixtures on disk", () => {
  const onDisk = fs
    .readdirSync(path.join(CONTRACT, "conformance"))
    .filter((name) => name.endsWith(".json"))
    .sort();
  assert.ok(KIT.fixtures.length > 0, "the kit must list at least one fixture");
  assert.deepEqual([...KIT.fixtures].sort(), onDisk);
});

test("the kit version and both schemas carry the same contract major", () => {
  const major = KIT.contract_version.split(".")[0];
  for (const kind of ["observation", "decision"]) {
    assert.ok(
      schema(kind).$id.includes(`/contract/v${major}/`),
      `${KIT.schemas[kind]} does not carry the v${major} contract major`,
    );
  }
  assert.ok(Number.isInteger(KIT.kit_version), "kit_version must be an integer");
});

test("every fixture observation validates against the published schema", () => {
  for (const name of KIT.fixtures) {
    assertValid("observation", fixture(name).observation, `${name}.observation`);
  }
});

test("every recorded legacy decision validates against the published schema", () => {
  for (const name of KIT.legacy_decision_fixtures) {
    const recorded = fixture(name).legacy_decision;
    assert.ok(recorded, `${name} is listed as carrying a legacy decision but does not`);
    assertValid("decision", recorded, `${name}.legacy_decision`);
  }
});

test("the kit exercises every additive optional field as omitted", () => {
  // The additive-only rule in GOVERNANCE.md is only proved by a fixture that actually
  // leaves the field out. Asserted over the kit as a whole: a fixture may also carry them.
  const decisions = KIT.legacy_decision_fixtures.map((name) => fixture(name).legacy_decision);
  const orders = decisions.flatMap((decision) => decision.orders);
  assert.ok(
    decisions.some((decision) => !Object.hasOwn(decision, "reasoning")),
    "no fixture omits `reasoning`, so its default is never exercised",
  );
  for (const field of ["confidence", "rationale"]) {
    assert.ok(
      orders.some((order) => !Object.hasOwn(order, field)),
      `no fixture order omits \`${field}\`, so its default is never exercised`,
    );
  }
});

// --- The TypeScript restatement of the wire contract -------------------------------
//
// A15: `types.ts` is a hand copy of the published schemas, and TypeScript types are
// erased before any validator sees them, so the fixture validation above cannot catch
// drift in them. It had already drifted: the schema's optional `cost` object
// (`DecisionCost`) was absent from `Decision` entirely, so a consumer typing a
// cost-reporting agent against this package got a type error for a field the contract
// defines and the kernel accumulates. These read the TS source as text, which is the
// only way to assert on a declaration that does not exist at runtime.

const TYPES_SRC = fs.readFileSync(path.join(__dirname, "../src/types.ts"), "utf8");

/** The member names declared in `export interface <name> { ... }`. */
function interfaceFields(name) {
  const start = TYPES_SRC.indexOf(`export interface ${name} {`);
  assert.notEqual(start, -1, `src/types.ts declares no interface ${name}`);
  const body = TYPES_SRC.slice(start, TYPES_SRC.indexOf("\n}", start));
  return new Set([...body.matchAll(/^\s{2}(\w+)\??:/gm)].map((m) => m[1]));
}

/** The members of `export type <name> = "a" | "b";`. */
function unionMembers(name) {
  const match = new RegExp(`export type ${name} =([^;]+);`).exec(TYPES_SRC);
  assert.ok(match, `src/types.ts declares no type ${name}`);
  return [...match[1].matchAll(/"([^"]+)"/g)].map((m) => m[1]).sort();
}

test("every decision-schema property is declared on the TypeScript types", () => {
  const decision = schema("decision");
  const pairs = [
    ["Decision", decision.properties],
    ["Order", decision.$defs.Order.properties],
    ["DecisionCost", decision.$defs.DecisionCost.properties],
  ];
  for (const [name, properties] of pairs) {
    const declared = interfaceFields(name);
    for (const property of Object.keys(properties)) {
      assert.ok(declared.has(property), `${name} does not declare the schema's \`${property}\``);
    }
    for (const field of declared) {
      assert.ok(
        Object.hasOwn(properties, field),
        `${name} declares \`${field}\`, which the published schema does not define`,
      );
    }
  }
});

test("every observation-schema property is declared on the TypeScript types", () => {
  const observation = schema("observation");
  const pairs = [
    ["MarketObservation", observation.properties],
    ["SymbolSnapshot", observation.$defs.SymbolSnapshot.properties],
    ["PositionState", observation.$defs.PositionState.properties],
  ];
  for (const [name, properties] of pairs) {
    const declared = interfaceFields(name);
    for (const property of Object.keys(properties)) {
      assert.ok(declared.has(property), `${name} does not declare the schema's \`${property}\``);
    }
  }
});

test("the Action union is the schema's enum, not a hand copy that drifted", () => {
  assert.deepEqual(unionMembers("Action"), [...schema("decision").$defs.Order.properties.action.enum].sort());
});

// --- The TypeScript restatement of the engine-output enums ------------------------
//
// The checks above cover the two wire-contract types, which have published schemas. The
// engine-output enums have none: they restate Rust declarations, which is the part of A15
// left open ("that would need a generated contract file rather than a test"). That file now
// exists at contract/engine-enums.v1.json, emitted by
// crates/sharpearena/tests/engine_enum_contract.rs through wildcard-free exhaustive matches,
// so a variant added to the engine fails that crate's compile and a union that drifts from
// it fails here. `BaselineAgent` is still not covered: it has no Rust enum to generate from,
// only a string dispatch in the wasm export layer.

const ENGINE_ENUMS = readJson(CONTRACT, KIT.engine_enums);

test("the engine-output unions are the generated contract, not hand copies", () => {
  for (const [name, labels] of Object.entries(ENGINE_ENUMS.enums)) {
    assert.deepEqual(
      unionMembers(name),
      [...labels].sort(),
      `the ${name} union in src/types.ts has drifted from the engine`,
    );
  }
});

test("the scenario interfaces declare exactly the engine's serde fields", () => {
  for (const [name, fields] of Object.entries(ENGINE_ENUMS.structs)) {
    const declared = interfaceFields(name);
    for (const field of fields) {
      assert.ok(declared.has(field), `${name} does not declare the engine's \`${field}\``);
    }
    for (const field of declared) {
      assert.ok(
        fields.includes(field),
        `${name} declares \`${field}\`, which the engine does not serialize`,
      );
    }
  }
});

test("the generated enum contract is not vacuous", () => {
  // A contract that named nothing would make both cross-checks above pass for free.
  assert.equal(ENGINE_ENUMS.schema_version, 1);
  assert.ok(Object.keys(ENGINE_ENUMS.enums).length >= 2);
  for (const labels of Object.values(ENGINE_ENUMS.enums)) {
    assert.ok(labels.length > 0);
  }
  for (const fields of Object.values(ENGINE_ENUMS.structs)) {
    assert.ok(fields.length > 0);
  }
});

test("a malformed decision is rejected, so the validator is not vacuous", () => {
  const root = schema("decision");
  const bad = { orders: [{ symbol: "SPY", action: "liquidate", target_weight: "0.5" }] };
  assert.ok(validate(root, bad, root, "probe", []).length >= 2, "probe should fail twice over");
});
