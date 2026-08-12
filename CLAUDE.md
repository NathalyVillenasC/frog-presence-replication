# Working rules for this repository

This is a replication study of *Presence-absence estimation in audio recordings of tropical
frog communities* (arXiv:1901.02495). The material delivered for the replication is not
identical to the material the article describes, so a large part of the work is establishing
what is actually on disk and where it differs from the paper. These rules exist to keep that
distinction visible.

They apply to every contributor, human or agent.

## Evidence discipline

**Any figure stated in prose must be printed by the cell above it.** No exceptions. A number
computed on the side, remembered from an earlier run, or estimated from a truncated output
does not go into the text. If a claim is worth making, print the number that supports it; if
printing it is not worth a line of code, drop the claim.

This is the rule the repository is built on. A reader must be able to check every assertion
against the output immediately above it, without running anything.

Claims carry a tag saying what kind of evidence is behind them:

| Tag | Meaning |
|---|---|
| `[PAPER]` | Stated in the article |
| `[SOURCE]` | Stated by the authors in direct communication |
| `[OBSERVED]` | Visible in the output of the cell above |
| `[INFERENCE]` | Plausible reading, not provable from the data |

An `[INFERENCE]` that is really a guess should say so. Downgrading a claim is always better
than defending it.

## Notebooks

Every section follows the same four-part shape:

```
[Markdown]  what this section asks, and why it matters
[Code]      one leading comment stating the purpose, then plain code
[Output]    the table, figure or printed result
[Markdown]  the tagged findings that follow from that output
```

- **No forward references.** A section may rely on what came before it, never on what comes
  after. The notebook has to read correctly top to bottom on a first pass.
- **The exploration notebook is read-only.** It reads headers, sidecars and annotations. It
  never modifies, moves or rewrites a file in the collections.
- **Commit executed notebooks.** After changing any cell, re-run the whole notebook
  (`jupyter nbconvert --to notebook --execute --inplace`) so the stored outputs match the
  stored code. Outputs that disagree with the code that produced them are worse than no
  outputs at all.

## Prose editing boundaries

Markdown written or edited by the repository owner is not rewritten by an agent. An agent may
change code, re-run cells and update the stored outputs. If a change makes the surrounding
prose wrong, the agent **says so and stops** rather than fixing the wording itself.

## Configuration

The audio is not in the repository — the three collections weigh about 35 GB — and lives in a
different place on every machine. Paths are read from a local `.env`, never written into the
code, and `.env.example` is the committed template.

The three collections are referred to by these names everywhere: in code, in prose, in
comments and in output.

| Constant | Collection |
|---|---|
| `MENDELEY_CALL_DIR` | Dataset 1 — labelled frog calls, published on Mendeley Data |
| `PONDS_CALL_DIR` | Dataset 2 — pond recordings digitised from cassettes |
| `SPECIES_CALL_DIR` | Dataset 3 — clean species calls |

Printed output shows the folder name only, never the full path: the path is personal to
whoever is running the notebook.

## Code

Python 3.12, PEP 8, type hints, Google-style docstrings. Prefer an existing, well-known
package over hand-rolled logic, and add it to `requirements.txt` with a comment saying why
it is there.

## Commits

One concern per commit. The subject line says what changed; the body says why it was worth
changing. A commit that only moves code without changing behaviour says so explicitly
(`No output changes.`), so a reviewer facing a large notebook diff knows what to expect.
