#!/usr/bin/env python3
"""Fail when a tracked Markdown file asserts that something is absent from a tree that
in fact contains it.

Three times in one day this repository shipped a document asserting absence where the
thing existed: `docs/integrations/inventory.md` carried a "Nothing in the tree" row for
HUD and Harbor after both had merged, and `docs/assets/logos/README.md` said Harbor
published no logo and that MCP published no logo asset, when both publish one. Each was
written by someone who had looked once, in one place. The repository record already says
to grep before claiming a gap, and repeating that is training rather than a device.
Training degrades. This is the device.

Three rules, all decided from the repository tree alone, so the check runs offline.

RULE A, the path rule. A path named as the OBJECT of an absence claim, in a sentence or
a table row, which nonetheless exists in the tree. "There is no `examples/hud/`" when
`examples/hud/` is tracked is a contradiction on its face.

That the path must be the claim's object is the whole design of rule A, not a detail.
The first draft fired whenever an absence word shared a sentence with any existing path,
and on this repository it produced forty findings before it produced a real one: an
audit trail is full of true sentences like "`types.ts:23-28` has no `cost` field", where
the absence is a property of something the path names rather than the path itself. A
gate at that signal-to-noise ratio gets switched off, which is worse than no gate.

RULE B, the subject rule. A Markdown table row whose status cell says its subject is
wholly absent from this tree, while a directory or file in the tree is named after that
subject. This is the exact shape the HUD and Harbor row had: a "Nothing in the tree"
cell beside a name that `examples/hud/` and `integrations/harbor/` both answer. Rule A
cannot catch it, because that row names no path at all; the missing path IS the defect.

RULE C, the vendored-asset rule. A sentence saying a project publishes, carries or has
no logo, mark, icon or brand asset, while naming a project whose asset is vendored in
`docs/assets/logos/` under a filename that names it. This is the other two defects
exactly: the page said Harbor published no logo, and that MCP published no logo asset,
while `harbor.png` and `mcp.svg` sat beside the sentence. The gate cannot read upstream,
but it can read the contradiction between one paragraph of a file and the directory the
same file indexes.

What the rules deliberately do NOT cover:

  * Upstream absence in general. "Vendor X publishes no logo", for an X this tree has
    vendored nothing from, is a claim about another repository. This gate is offline by
    construction, so it cannot check it, and pretending otherwise would be the same
    overclaim it exists to stop. Rule C reaches the cases where the contradicting
    evidence is already in this tree, and no further. What covers the rest is
    `scripts/check-logo-assets.py`, which pins every vendored asset to a recorded hash,
    plus a human re-reading upstream.
  * Paths inside markup attributes (`src=`, `srcset=`, `href=`) and Markdown link
    targets. A row's own inline logo is how the row is drawn, not a statement about
    what the tree contains, and counting it would fire on every true "Nothing in the
    tree" row that carries a mark.
  * Anything under `docs/assets/logos/`. Those files are named after third-party
    projects by construction, so their existence says nothing about whether this tree
    integrates the project. Reading `torchrl.png` as evidence of a TorchRL integration
    is precisely the false positive that gets a gate disabled.
  * Bare basenames. `build.rs` in "`crates/sharpearena-py` has no `build.rs`" is not
    treated as a repository path: a basename matches in too many places to be evidence.
    Rule A wants a token with a slash in it, or a code span that resolves exactly.
  * Absence of a behaviour, symbol, guarantee or API rather than a file. "No SB3 import
    or adapter exists" is about code, and no tree listing can adjudicate it.

Exemptions live in `scripts/absence-claim-exemptions.json`, one verbatim claim each.
`load_exemptions` explains why that shape and no other.

Usage:

    python scripts/check-absence-claims.py [--root PATH]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

EXEMPTIONS_RELATIVE = "scripts/absence-claim-exemptions.json"

# Directories whose contents are named after third parties rather than being evidence
# about this tree. Kept to one entry on purpose: every addition is a place the gate
# stops looking, so each one has to earn itself.
EVIDENCE_EXCLUDED_PREFIXES = ("docs/assets/logos/",)

# A path-like token: a backticked span, or a bare run containing a slash. Always
# contributes exactly two groups, so a match reads `m.group(1) or m.group(2)`.
_PATH = r"(?:`([^`\n]+)`|((?:\.{1,2}/)*[\w][\w.-]*(?:/[\w.-]+)+/?))"

# Absence phrasings in which the path is the object of the claim.
_OBJECT_AFTER = (
    r"\bthere\s+(?:is|are|was|were)\s+(?:currently\s+)?no\s+" + _PATH,
    r"\bno\s+such\s+(?:file|directory|path|module)\s+(?:as\s+)?" + _PATH,
    r"\b(?:has|have|had|contains?|contained|carries|carry|ships?|publishes?"
    r"|provides?|includes?|tracks?)\s+no\s+" + _PATH,
    r"\bno\s+" + _PATH + r"\s+exists?\b",
    # There is deliberately no "nothing under <path>" pattern. In this repository every
    # occurrence of that phrasing reports an absence of CHANGE ("nothing under
    # `paper/evidence/` was regenerated"), which is a true statement about a path that
    # exists, and no cheap rule separates the two readings. The existence sense is
    # already covered by "there is no <path>" and by rule B.
)

# The mirror image: the path comes first and the absence claim follows it.
_OBJECT_BEFORE = (
    _PATH + r"\s+(?:does|do|did)\s+not\s+exist\b",
    _PATH + r"\s+is\s+(?:not\s+present|absent|missing)\b",
    _PATH + r"\s+(?:was|were)\s+never\s+(?:added|created|committed)\b",
)

ABSENCE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE) for pattern in _OBJECT_AFTER + _OBJECT_BEFORE
)

# Rule B's vocabulary: a cell claiming the row's subject is wholly missing from this
# tree. Narrower than rule A's on purpose. "publishes no logo" says nothing about
# whether this tree holds a directory for the project.
WHOLE_SUBJECT_ABSENCE = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bnothing\s+(?:at\s+all\s+)?(?:in|under)\s+(?:the|this)\s+(?:tree|repo(?:sitory)?)\b",
        r"\bnot\s+present\s+(?:in|under)\s+(?:this|the)\s+(?:tree|repo(?:sitory)?)\b",
        r"\bno\s+(?:integration|implementation|adapter|binding|fixture|example)"
        r"\s+(?:in|under)\s+(?:this|the)\s+(?:tree|repo(?:sitory)?)\b",
    )
)

# Rule C's trigger: a sentence saying some project publishes no brand asset. Two of the
# three shipped defects were exactly this sentence, about a project whose asset is
# vendored in `docs/assets/logos/` under a filename that names it.
PUBLISHES_NO_ASSET = re.compile(
    r"\b(?:publish(?:es|ed)?|carr(?:ies|y|ied)|ship(?:s|ped)?)\s+"
    r"(?:no\s+(?:logo|mark|icon|brand|asset|image)\w*|none)\b"
    r"|\b(?:has|have|had|provides?)\s+no\s+(?:logo|mark|icon|brand|asset|image)\w*\b",
    re.IGNORECASE,
)

MARKUP_TAG = re.compile(r"<[^<>]*>")
INLINE_CODE = re.compile(r"`([^`\n]+)`")
LINK_TARGET = re.compile(r"\]\([^)\s]+(?:\s+\"[^\"]*\")?\)")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z`\"'(\[])")
TABLE_SEPARATOR = re.compile(r"^\s*\|?[\s:|-]+\|[\s:|-]*$")
FENCE = re.compile(r"^\s*(?:```|~~~)")
LINE_SUFFIX = re.compile(r":\d+(?:[-:]\d+)?$")

# Words a first table cell may carry that name nothing.
SUBJECT_STOPWORDS = frozenset(
    {"and", "the", "for", "with", "its", "not", "yes", "all", "any", "via", "own"}
)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def tracked_files(root: Path) -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [entry for entry in out.split("\0") if entry]


class Tree:
    """Everything the two rules are allowed to know about the repository."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.files = tracked_files(root)
        existing: set[str] = set()
        slugs: set[str] = set()
        for path in self.files:
            existing.add(path)
            parts = path.split("/")
            for index in range(1, len(parts)):
                existing.add("/".join(parts[:index]))
            if path.startswith(EVIDENCE_EXCLUDED_PREFIXES):
                continue
            for part in parts:
                stem = part.rsplit(".", 1)[0] if "." in part else part
                if len(slug(stem)) >= 3:
                    slugs.add(slug(stem))
        self.existing = frozenset(existing)
        self.component_slugs = frozenset(slugs)
        self.vendored_assets: dict[str, str] = {}
        for path in self.files:
            if not path.startswith("docs/assets/logos/") or path.endswith(".md"):
                continue
            stem = re.sub(r"-dark$", "", path.rsplit("/", 1)[-1].rsplit(".", 1)[0])
            self.vendored_assets.setdefault(slug(stem), path)

    def resolve(self, token: str, relative_to: str) -> str | None:
        """Return the tracked path a claim's object names, or None.

        Resolution is exact: either the token is a tracked path (or a directory prefix
        of one) read from the repository root, or it resolves that way relative to the
        directory of the document making the claim. Nothing fuzzier. A gate that guesses
        which of nine `vector.py` files a sentence meant is a gate that cries wolf.
        """
        token = token.strip().strip("\"'()[],;:").rstrip("/")
        token = LINE_SUFFIX.sub("", token)
        if not token or token in {".", ".."} or "/" not in token:
            return None
        if token.startswith(EVIDENCE_EXCLUDED_PREFIXES):
            return None

        if not token.startswith((".", "/")) and token in self.existing:
            return token

        base = relative_to.rsplit("/", 1)[0] if "/" in relative_to else ""
        parts = (base.split("/") if base else []) + token.split("/")
        stack: list[str] = []
        for part in parts:
            if part in {"", "."}:
                continue
            if part == "..":
                if not stack:
                    return None
                stack.pop()
                continue
            stack.append(part)
        joined = "/".join(stack)
        if joined and joined in self.existing and not joined.startswith(EVIDENCE_EXCLUDED_PREFIXES):
            return joined
        return None


def strip_markup(text: str) -> str:
    """Drop HTML tags and Markdown link targets.

    Both carry paths that are rendering references rather than assertions about the
    tree: a row drawing `<img src="../assets/logos/torchrl.png">` beside the words
    "Nothing in the tree" is not contradicting itself.
    """
    return LINK_TARGET.sub("]()", MARKUP_TAG.sub(" ", text))


def iter_units(text: str):
    """Yield (line number, unit). Table rows stay whole; prose splits into sentences.

    Fenced blocks are skipped: a shell transcript is not a claim about the tree.
    """
    in_fence = False
    paragraph: list[tuple[int, str]] = []

    def flush():
        if not paragraph:
            return []
        start = paragraph[0][0]
        joined = " ".join(line for _, line in paragraph)
        return [(start, piece) for piece in SENTENCE_SPLIT.split(joined) if piece.strip()]

    for number, line in enumerate(text.splitlines(), start=1):
        if FENCE.match(line):
            yield from flush()
            paragraph = []
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        stripped = line.strip()
        if stripped.startswith("|"):
            yield from flush()
            paragraph = []
            if not TABLE_SEPARATOR.match(line):
                yield number, line
            continue
        if not stripped:
            yield from flush()
            paragraph = []
            continue
        paragraph.append((number, stripped))

    yield from flush()


def row_subjects(row: str) -> list[str]:
    cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
    if not cells:
        return []
    subject = re.sub(r"\[([^\]]*)\]\(\)", r"\1", strip_markup(cells[0]))
    subject = INLINE_CODE.sub(r"\1", subject)
    out: list[str] = []
    for name in re.split(r"[,/]|\band\b", subject):
        words = [word for word in name.split() if word.lower() not in SUBJECT_STOPWORDS]
        if words:
            out.append(" ".join(words))
    return out


def named_vendored_project(prose: str, assets: dict[str, str]) -> tuple[str, str] | None:
    """Return (name as written, vendored file) if the prose names a vendored project.

    Single words and adjacent pairs are both tried, so "Prime Intellect" matches
    `prime-intellect.png` as readily as "Harbor" matches `harbor.png`.
    """
    words = re.findall(r"[A-Za-z0-9][\w.+-]*", prose)
    for size in (2, 1):
        for index in range(len(words) - size + 1):
            phrase = " ".join(words[index : index + size])
            key = slug(phrase)
            if len(key) >= 3 and key in assets:
                return phrase, assets[key]
    return None


def load_exemptions(root: Path) -> list[dict]:
    """Read the exemption list.

    The shape is the argument. An exemption names one file, quotes the excused claim
    verbatim, and gives a written reason. There is no file-level, directory-level or
    pattern-level opt-out, and no inline marker in the prose, so an exemption cannot be
    added sideways or made to cover text nobody has read. Because it is bound to exact
    wording, editing the prose retires it automatically and the gate speaks again; and
    an exemption whose quote no longer appears anywhere is itself a failure, so stale
    ones cannot accumulate unnoticed.

    What this does not do is stop someone who sets out to silence a true finding.
    Nothing short of review can, and a check that claimed otherwise would be making the
    same kind of unverified absence claim this file exists to catch. The design goal is
    narrower and achievable: silencing a finding cannot be done quietly or in bulk. It
    costs one legible line in one small central file, naming the exact sentence and the
    reason, which is a diff a reviewer sees.
    """
    path = root / EXEMPTIONS_RELATIVE
    if not path.is_file():
        return []
    entries = json.loads(path.read_text(encoding="utf-8"))["exemptions"]
    for entry in entries:
        missing = sorted({"file", "claim", "reason"} - set(entry))
        if missing:
            raise ValueError(f"exemption is missing {missing}: {entry}")
        if len(entry["reason"].split()) < 5:
            raise ValueError(f"exemption reason is too thin to review: {entry['reason']!r}")
    return entries


def check(root: Path) -> tuple[list[str], list[str]]:
    tree = Tree(root)
    exemptions = load_exemptions(root)
    used = [False] * len(exemptions)
    findings: list[str] = []

    for relative in tree.files:
        if not relative.endswith(".md"):
            continue
        text = (root / relative).read_text(encoding="utf-8", errors="replace")

        for number, unit in iter_units(text):
            excused = False
            for index, entry in enumerate(exemptions):
                if entry["file"] == relative and entry["claim"] in unit:
                    used[index] = True
                    excused = True
            if excused:
                continue

            prose = strip_markup(unit)

            reported = False
            for pattern in ABSENCE_PATTERNS:
                for match in pattern.finditer(prose):
                    token = match.group(1) or match.group(2) or ""
                    resolved = tree.resolve(token, relative)
                    if resolved is None:
                        continue
                    findings.append(
                        f"{relative}:{number}: claims {match.group(0).strip()!r}, "
                        f"but {resolved} is tracked in this tree\n"
                        f"    {unit.strip()[:300]}"
                    )
                    reported = True
                    break
                if reported:
                    break
            if reported:
                continue

            asset_claim = PUBLISHES_NO_ASSET.search(prose)
            if asset_claim is not None:
                named = named_vendored_project(prose, tree.vendored_assets)
                if named is not None:
                    stem, path = named
                    findings.append(
                        f"{relative}:{number}: claims {asset_claim.group(0).strip()!r} of "
                        f"{stem}, but {path} is vendored from that project\n"
                        f"    {unit.strip()[:300]}"
                    )
                    continue

            if not unit.strip().startswith("|"):
                continue
            phrase = next((p.search(prose) for p in WHOLE_SUBJECT_ABSENCE if p.search(prose)), None)
            if phrase is None:
                continue
            for name in row_subjects(unit):
                if len(slug(name)) >= 3 and slug(name) in tree.component_slugs:
                    findings.append(
                        f"{relative}:{number}: row says {phrase.group(0)!r} of {name!r}, "
                        f"but a path in this tree is named after it\n"
                        f"    {unit.strip()[:300]}"
                    )
                    break

    stale = [
        f"{EXEMPTIONS_RELATIVE}: exemption matched nothing, so it is stale: "
        f"{entry['file']} / {entry['claim'][:90]!r}"
        for entry, was_used in zip(exemptions, used)
        if not was_used
    ]
    return findings, stale


def main() -> int:
    parser = argparse.ArgumentParser(description="Absence-claim gate for tracked Markdown.")
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()

    root = args.root
    if root is None:
        root = Path(
            subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )

    findings, stale = check(root.resolve())
    for line in findings + stale:
        print(line, file=sys.stderr)
    if findings or stale:
        print(
            f"\n{len(findings)} absence claim(s) contradicted by the tree; "
            f"{len(stale)} stale exemption(s).",
            file=sys.stderr,
        )
        return 1
    print("absence claims: nothing asserted missing is present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
