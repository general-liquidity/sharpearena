# Third-party logos used in this repository

Nine logos are vendored here and shown beside the interfaces they name, in the
root README's "Choose an interface" table and in
[`docs/integrations/inventory.md`](../../integrations/inventory.md). Each file
was downloaded from the project's own official source and committed byte for
byte: none has been recoloured, cropped, rescaled or otherwise modified.

| File | Project | Downloaded from | Governing terms |
|---|---|---|---|
| `gymnasium.svg` | Gymnasium (Farama Foundation) | `Farama-Foundation/Gymnasium`, `main`, `docs/_static/img/gymnasium_black.svg` | Repository LICENSE is the MIT License (Copyright 2016 OpenAI, 2022 Farama Foundation). |
| `pettingzoo.svg` | PettingZoo (Farama Foundation) | `Farama-Foundation/PettingZoo`, `main`, `docs/_static/img/PettingZoo.svg` | Repository LICENSE is the MIT License. |
| `minari.svg` | Minari (Farama Foundation) | `Farama-Foundation/Minari`, `main`, `docs/_static/img/Minari.svg` | Repository LICENSE states that all assets in that repository are the copyright of the Farama Foundation and that the Foundation "releases the elements of this repository they copyright to under the MIT license". It is the only one of the three that names assets rather than code. |
| `webassembly.svg` | WebAssembly | `carlosbaraza/web-assembly-logo`, `master`, `dist/icon/web-assembly-icon.svg` | CC0 1.0 Universal, per that repository's LICENSE file. This is the logo the WebAssembly design process selected. |
| `typescript.svg` | TypeScript | `ts-logo-128.svg`, from the official design-assets pack linked at <https://www.typescriptlang.org/branding/> | The branding page permits use of the logo to refer to TypeScript, and asks that the shape not be modified, that the logo not be used as a product's own logo or folded into one, and that nothing imply TypeScript's endorsement. This use satisfies all four. |
| `mcp.svg` | Model Context Protocol | `modelcontextprotocol/docs`, `main`, `favicon.svg` | Repository LICENSE is the MIT License (Copyright 2024-2025 Anthropic, PBC and contributors). The mark itself is held by LF Projects, LLC; see the note below on why this one is wrapped in a link. |
| `hud.svg` | HUD | `hud-evals/hud-python`, `main`, `docs/logo/hud_logo.svg` | Repository LICENSE is the MIT License. |
| `harbor.png` | Harbor | `harbor-framework/harbor`, `main`, `docs-mintlify/harbor-logo.png` | Repository LICENSE is Apache 2.0. |
| `prime-intellect.png` | Prime Intellect, for `verifiers` and Prime-RL | `https://www.primeintellect.ai/icons/logo-icon.png` | No published brand policy. Both project READMEs use this company mark as their own header, so it is the mark upstream itself puts on these projects. The use here is nominative: it names the library this package integrates with. |

## The MCP logo is a link, deliberately

The Model Context Protocol was donated to the Agentic AI Foundation, a Linux
Foundation directed fund, on 2025-12-09, and the mark is held by LF Projects,
LLC. The LF Projects trademark policy restricts putting a project logo on a
third party's materials, but it grants this in terms:

> You are also allowed to use a trademark or logo of LF Projects as a link to
> the home page of the applicable project or to a web page on LF Projects web
> site that is relevant to the reference so long as the link is in a manner that
> is consistent with the preservation of the goodwill and value of the mark.

So the MCP logo in both tables is wrapped in an anchor to
<https://modelcontextprotocol.io>, using the official file unmodified. That is
the form the policy names. If that anchor is ever removed, the permission goes
with it, and the logo should come out rather than stay as a bare image.

The same policy has a separate allowance for communicating compatibility, but it
is written for the word mark rather than the logo, so it is not what this rests
on.

## What a logo here does and does not say

A logo credits the project this package talks to. It is not a support claim.
HUD and Harbor are local feasibility fixtures rather than supported
integrations, and they carry their logos next to row text that says exactly
that, with [`support-status.md`](../../integrations/support-status.md) and the
INT-08 and INT-09 reports behind it. Support status is stated in words, in the
column beside the logo, because that is where a reader can check it.

No logo appears on a row this tree does not implement. CleanRL, Ray and RLlib,
Stable-Baselines3, TorchRL, PufferLib and EnvPool have no integration here, so
crediting them would assert something false.

Three interfaces carry no logo because no third party stands behind them: the
JSON contract, the SharpeBench bridge and the functional view are this
project's own surfaces.

## The point that governs all of this

No open-source licence grants trademark rights. The Apache 2.0 licence excludes
them in terms, at section 6, and the MIT licence is silent on them. So "the
repository is MIT" answers whether the file may be copied, not whether the mark
may be used. What makes these uses defensible is that each names an interface
this package genuinely implements, beside that interface, without claiming
sponsorship or endorsement by the project named. If a logo ever moves to a row
whose interface is not real, that reasoning stops holding.

## Checking these files against upstream

Each file is byte-identical to the source named above. To re-verify:

```bash
curl -sL https://raw.githubusercontent.com/Farama-Foundation/Gymnasium/main/docs/_static/img/gymnasium_black.svg | sha256sum
sha256sum docs/assets/logos/gymnasium.svg
```

The hashes recorded at the time of vendoring:

```
2c02ebee9015c8129e98e1f897a998f3a62ed6e15016aa0d2c4c18b430b5a63c  gymnasium.svg
b98827308858b79d5eddf4dad68ba5e74f994ea58439ac0f642365dc001af031  pettingzoo.svg
dc6e121d2361174d6eb06e6ec47806d35a2d07ff12738d32c77f9e7846ff6f4b  minari.svg
59adfc16172ad52c99436b44ef60e2aa3120a45cd82fbf57e4cc184f1f55511c  webassembly.svg
25e995fa85ae9ff6b5749f4ffeb21b2fe71d74e9b8d8db8eac4b9732cd9ef287  typescript.svg
05f47fb3ffb7323bdbf6b397330229a7f32a3a1f2d17365e748f8777a82dd6c0  mcp.svg
ec16d2d86e91da3e2c598d8609115562c04e47b576b905b131ffbc4a621224dc  hud.svg
b5004bb031b04aa6d55d93df8547ac39cf17f5341e1056c21e05581612b972fb  harbor.png
3d7b0ddac38e785bace1523d2be274591b47dad82a0a5f3572230a40ff3e9328  prime-intellect.png
```

An upstream hash that no longer matches means the project has changed its logo,
not that this copy is wrong. Replace the file from the same path and update the
hash, rather than editing the file here.

## A correction worth keeping

The first version of this page, and the pull request that added it, said Harbor
published no logo and that MCP published none outside a content-hashed
documentation CDN. Both were wrong, and both came from searching too narrowly:
Harbor's README and landing page carry only badges, but the logo is at
`docs-mintlify/harbor-logo.png` in the same repository, and the MCP
specification repository indeed has no logo while `modelcontextprotocol/docs`
publishes `logo/light.svg`, `logo/dark.svg` and `favicon.svg` under the MIT
licence. The LF Projects policy was also read as more restrictive than it is:
its logo-as-link clause, quoted above, covers precisely this use. Checking one
repository and one page is not checking whether an asset exists.
