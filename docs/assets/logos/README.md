# Third-party logos used in this repository

Thirteen logos are vendored here and shown beside the ecosystems they name, in
the root README's "Choose an interface" table, in
[`docs/integrations/inventory.md`](../../integrations/inventory.md) and in
[`docs/integrations/support-status.md`](../../integrations/support-status.md). Each file
was downloaded from the project's own official source and committed byte for
byte: none has been recoloured, cropped, rescaled or otherwise modified.

Eight of them have a `-dark` companion, and the tables select between them with a
`<picture>` element keyed on `prefers-color-scheme`. That matters because most of
these marks are near-black ink on transparency, which disappears against GitHub's
dark theme. Every `-dark` file is the project's own published light-on-dark
variant, downloaded from the path recorded below, so no recolouring was needed to
make them legible.

Five have no companion, for two different reasons. TypeScript, Ray and EnvPool
need none: each is a coloured mark that holds up on either background.
Stable-Baselines3 and TorchRL publish exactly one image and no dark variant, so
that is what is committed. They are legible on a dark background but not tuned
for it. Recolouring either one to fix that is not an option, since it would break
the rule that every file here is byte-identical to what its project publishes.

| File | Project | Downloaded from | Governing terms |
|---|---|---|---|
| `gymnasium.svg`, `gymnasium-dark.svg` | Gymnasium (Farama Foundation) | `Farama-Foundation/Gymnasium`, `main`, `docs/_static/img/gymnasium_black.svg` and `gymnasium_white.svg` | Repository LICENSE is the MIT License (Copyright 2016 OpenAI, 2022 Farama Foundation). |
| `pettingzoo.svg`, `pettingzoo-dark.svg` | PettingZoo (Farama Foundation) | `Farama-Foundation/PettingZoo`, `main`, `docs/_static/img/PettingZoo.svg` and `PettingZoo_White.svg` | Repository LICENSE is the MIT License. |
| `minari.svg`, `minari-dark.svg` | Minari (Farama Foundation) | `Farama-Foundation/Minari`, `main`, `docs/_static/img/Minari.svg` and `Minari_White.svg` | Repository LICENSE states that all assets in that repository are the copyright of the Farama Foundation and that the Foundation "releases the elements of this repository they copyright to under the MIT license". It is the only one of the three that names assets rather than code. |
| `webassembly.svg`, `webassembly-dark.svg` | WebAssembly | `carlosbaraza/web-assembly-logo`, `master`, `dist/icon/web-assembly-icon.svg` and `web-assembly-icon-white.svg` | CC0 1.0 Universal, per that repository's LICENSE file. This is the logo the WebAssembly design process selected. |
| `typescript.svg` | TypeScript | `ts-logo-128.svg`, from the official design-assets pack linked at <https://www.typescriptlang.org/branding/> | The branding page permits use of the logo to refer to TypeScript, and asks that the shape not be modified, that the logo not be used as a product's own logo or folded into one, and that nothing imply TypeScript's endorsement. This use satisfies all four. |
| `mcp.svg`, `mcp-dark.svg` | Model Context Protocol | `modelcontextprotocol/docs`, `main`, `logo/light.svg` and `logo/dark.svg` | Repository LICENSE is the MIT License (Copyright 2024-2025 Anthropic, PBC and contributors). The mark itself is held by LF Projects, LLC; see the note below on why this one is wrapped in a link. |
| `hud.svg`, `hud-dark.svg` | HUD | `hud-evals/hud-python`, `main`, `docs/logo/hud_logo.svg` and `docs/logo/hud_logo_dark.svg` | Repository LICENSE is the MIT License. |
| `harbor.png`, `harbor-dark.png` | Harbor | `harbor-framework/harbor`, `main`, `docs-mintlify/harbor-wordmark-light.png` and `harbor-wordmark-dark.png`, the pair its own `docs.json` selects between | Repository LICENSE is Apache 2.0. |
| `ray.svg` | Ray and RLlib | `ray-project/ray`, `master`, `doc/source/_static/img/ray_logo.svg` | Repository LICENSE is Apache 2.0. Ray was transferred to the PyTorch Foundation, a Linux Foundation project, on 2025-10-22, and the announcement names neutral trademark management as a reason for the move, so this one is wrapped in a link to <https://www.ray.io> on the same LF clause as MCP. |
| `torchrl.png` | TorchRL | `pytorch/rl`, `main`, `docs/source/_static/img/logo.png` | Repository LICENSE is the MIT License. The PyTorch mark is held by the Linux Foundation, so this one is wrapped in a link to <https://pytorch.org/rl>, again on the LF logo-as-link clause. |
| `stable-baselines3.png` | Stable-Baselines3 | `DLR-RM/stable-baselines3`, `master`, `docs/_static/img/logo.png` | Repository LICENSE is the MIT License. |
| `envpool.svg` | EnvPool | `sail-sg/envpool`, `main`, `docs/_static/images/envpool-logo.svg` | Repository LICENSE is Apache 2.0. |
| `prime-intellect.png`, `prime-intellect-dark.png` | Prime Intellect, for `verifiers` and Prime-RL | the light-mode and dark-mode pair the Prime-RL README itself serves, at `github.com/user-attachments/assets/40c36e38-c5bd-4c5a-9cb3-f7b902cd155d` and `.../6414bc9b-126b-41ca-9307-9e982430cde8` | No published brand policy. Both project READMEs use this company mark as their own header, so it is the mark upstream itself puts on these projects. The use here is nominative: it names the library this package integrates with. |

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

The ecosystems this tree does not integrate carry their logos too, in the "Not
present in this tree" table of `support-status.md` and in the "No" rows of
`inventory.md`. That is identification, not a claim, and it is safe precisely
because of where it sits: the column immediately to the right of each of those
logos reads "Not supported" or "Ruled out". A logo cannot be read as a support
claim in a table whose own status column denies support on the same line. The
rule that matters is therefore not "only supported things get a logo" but "the
status column must always be visible beside the logo". If a logo is ever moved
somewhere that column does not travel with it, it has to come out.

Three interfaces carry no logo because no third party stands behind them: the
JSON contract, the SharpeBench bridge and the functional view are this
project's own surfaces.

Two projects carry no logo because neither publishes one. CleanRL's README shows
only badges, and the favicon its documentation site serves is the stock MkDocs
Material book glyph rather than a mark of its own, so vendoring it would have
credited CleanRL with someone else's icon. PufferLib's README header is a
composite banner of game art rather than a logo, and no separate mark exists in
its repository or on its site. Both are named in text, and both should get a
logo the day they publish one.

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
744bc2255252daf7f8236dc85f2a4b8ca9c026d3350b71528785b2fdb4cfe7b9  gymnasium-dark.svg
2c02ebee9015c8129e98e1f897a998f3a62ed6e15016aa0d2c4c18b430b5a63c  gymnasium.svg
5230634ab602c06c2e7a906767f29f7a4aa610e0b35e3f1b7dfd4c1da1d828e1  pettingzoo-dark.svg
b98827308858b79d5eddf4dad68ba5e74f994ea58439ac0f642365dc001af031  pettingzoo.svg
2e35e6f50e7f4c525bbcb084485354ea51114521a6084d737f0cec3884d4b658  minari-dark.svg
dc6e121d2361174d6eb06e6ec47806d35a2d07ff12738d32c77f9e7846ff6f4b  minari.svg
285a1e4032d3964c013f073293efce2e6bac9994885cac530da0e6619cc6e988  webassembly-dark.svg
59adfc16172ad52c99436b44ef60e2aa3120a45cd82fbf57e4cc184f1f55511c  webassembly.svg
25e995fa85ae9ff6b5749f4ffeb21b2fe71d74e9b8d8db8eac4b9732cd9ef287  typescript.svg
ae239dea9d037a331575487390bf900d6baacf896991185b3dd861ac4b9941d5  mcp-dark.svg
3163a85f9db4b98c3b5af846ea284b3296295dbd51a138b2e77ebd438342e902  mcp.svg
7a7876c8b682dd3a8f1f93de793ab336c6090f3e173bb7e491bae953b0e8dcb4  hud-dark.svg
ec16d2d86e91da3e2c598d8609115562c04e47b576b905b131ffbc4a621224dc  hud.svg
fcd011e2bd2c689ef2fcca792bddd8008431f8a31a73449a5957334432512908  harbor-dark.png
b83402339c81f73b83c7e6409d3fa9e8fbab3a333060a06ffe0d3692a80bc4de  harbor.png
3e0a4485bc180e36ffd18aa63b0850b64b5234182cf625fcb7247851ab8b1e44  ray.svg
486e50f468a323aa5530a958f5f19cc36853c0b8150aae1024fcd99d9c933df5  torchrl.png
e50155b4afd58f3bbbb2f052f837bc221223fde9e0094047390c69ede3782f51  stable-baselines3.png
2e776fa7b2db5b54d4e53a4364446342443155b4fd97a70d13f7153b7ba46b5a  envpool.svg
e70261c6840a7f0e590c492f560c40692f958e3ddfa324452a2e4e4d615c2cf0  prime-intellect-dark.png
70a2a6f0cc4d86355485adc5ea9aca0c7171a3c6c5a426ba6fb8496d95490335  prime-intellect.png
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
