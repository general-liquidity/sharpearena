# Third-party logos used in this repository

Five logos are vendored here and shown beside the interfaces they name, in the
root README's "Choose an interface" table and in
[`docs/integrations/inventory.md`](../../integrations/inventory.md). Each file was
downloaded from the project's own official source on 2026-09-23 and committed
byte for byte: none has been recoloured, cropped, rescaled or otherwise modified.

| File | Project | Downloaded from | Governing terms |
|---|---|---|---|
| `gymnasium.svg` | Gymnasium (Farama Foundation) | `Farama-Foundation/Gymnasium`, `main`, `docs/_static/img/gymnasium_black.svg` | Repository LICENSE is the MIT License (Copyright 2016 OpenAI, 2022 Farama Foundation). |
| `pettingzoo.svg` | PettingZoo (Farama Foundation) | `Farama-Foundation/PettingZoo`, `main`, `docs/_static/img/PettingZoo.svg` | Repository LICENSE is the MIT License. |
| `minari.svg` | Minari (Farama Foundation) | `Farama-Foundation/Minari`, `main`, `docs/_static/img/Minari.svg` | Repository LICENSE states that all assets in that repository are the copyright of the Farama Foundation and that the Foundation "releases the elements of this repository they copyright to under the MIT license". It is the only one of the three that names assets rather than code. |
| `webassembly.svg` | WebAssembly | `carlosbaraza/web-assembly-logo`, `master`, `dist/icon/web-assembly-icon.svg` | CC0 1.0 Universal, per that repository's LICENSE file. This is the logo the WebAssembly design process selected. |
| `typescript.svg` | TypeScript | `ts-logo-128.svg`, from the official design-assets pack linked at <https://www.typescriptlang.org/branding/> | The branding page permits use of the logo to refer to TypeScript, and asks that the shape not be modified, that the logo not be used as a product's own logo or folded into one, and that nothing imply TypeScript's endorsement. This use satisfies all four. |

## Why the other interfaces have no logo here

The absence is deliberate in every case, and it is not an oversight to be filled
in later without checking the same things again.

**MCP.** The Model Context Protocol was donated to the Agentic AI Foundation, a
Linux Foundation directed fund, on 2025-12-09, and the mark is held by LF
Projects, LLC. The LF Projects trademark policy asks for written permission
before a logo appears on a third party's website or materials. Its
compatibility allowance is written for the word mark, in the form "your product
compatible with the mark", and does not extend to the logo. The protocol's own
specification repository carries no logo asset; the only file the project
publishes sits behind a content-hashed documentation CDN URL, which is not a
stable asset to depend on. Naming MCP in text is what the policy allows, so that
is what the tables do.

**HUD.** The terms are clean: `hud-evals/hud-python` is MIT and publishes
`docs/logo/hud_logo.svg`. The row is the problem, not the licence. HUD is a
local feasibility fixture here, not a supported integration, and a logo next to
it would read as a supported route to anyone scanning the table, which is the
impression [`support-status.md`](../../integrations/support-status.md) exists to
prevent.

**Harbor.** No logo is published to use. `harbor-framework/harbor` is
Apache-2.0, and neither its README nor <https://www.harborframework.com/> carries
a logo image, only badges. The same not-a-supported-integration point applies
regardless.

**`verifiers` and Prime-RL.** The image in both READMEs is Prime Intellect's
company logo rather than a project mark, it is served from a GitHub
user-attachment upload rather than a stable asset path, and the organisation
publishes no brand policy. A company mark carries a stronger endorsement
reading than a project mark, so text is the safer and more accurate choice.

**Everything not supported.** CleanRL, Ray and RLlib, Stable-Baselines3,
TorchRL, PufferLib and EnvPool have no integration in this tree. A logo beside
any of them would assert something false.

## The point that governs all of this

No open-source licence grants trademark rights. The Apache 2.0 licence excludes
them in terms, at section 6, and the MIT licence is silent on them. So "the
repository is MIT" answers whether the file may be copied, not whether the mark
may be used. What makes the five uses above defensible is that each names an
interface this package genuinely implements, next to that interface, without
claiming sponsorship or endorsement by the project named. If a logo ever moves
to a row whose interface is not real, that reasoning stops holding.

## Checking these files against upstream

Each file is byte-identical to the source named above. To re-verify:

```bash
curl -sL https://raw.githubusercontent.com/Farama-Foundation/Gymnasium/main/docs/_static/img/gymnasium_black.svg | sha256sum
sha256sum docs/assets/logos/gymnasium.svg
```

The hashes recorded at the time of vendoring:

```
2c02ebee9015c8129e98e1f897a998f3a62ed6e15016aa0d2c4c18b430b5a63c  gymnasium.svg
dc6e121d2361174d6eb06e6ec47806d35a2d07ff12738d32c77f9e7846ff6f4b  minari.svg
b98827308858b79d5eddf4dad68ba5e74f994ea58439ac0f642365dc001af031  pettingzoo.svg
25e995fa85ae9ff6b5749f4ffeb21b2fe71d74e9b8d8db8eac4b9732cd9ef287  typescript.svg
59adfc16172ad52c99436b44ef60e2aa3120a45cd82fbf57e4cc184f1f55511c  webassembly.svg
```

An upstream hash that no longer matches means the project has changed its logo,
not that this copy is wrong. Replace the file from the same path and update the
hash, rather than editing the file here.
