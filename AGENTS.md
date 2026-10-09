# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## What this repository is

An Obsidian vault of deep-learning / ML knowledge notes (robotics, generative modeling, LLMs, optimization, math foundations). There is no build, lint, or test tooling — the "work" here is writing and maintaining interlinked Markdown notes.

**Git**: the vault auto-commits with `vault backup: <timestamp>` messages. Leave changes uncommitted unless explicitly asked to commit; never rewrite history.

**The user hand-edits notes between sessions.** Always re-read a note (or at least the section you're touching) before editing — exact-match edits against stale content will fail or, worse, clobber their changes. Preserve user-added material verbatim when restructuring: `==highlights==`, colored `<span>`s, CJK annotations (e.g. 微分同胚), inline comments.

## Layout

- `_00_maps_of_contents/dl_moc.md` — the single Map of Contents. Six sections (Mathematics & Statistics · Learning Methods & Theories · Sequence Modeling & LLMs · Vision & Self-Supervised Learning · Generative Modeling · RL & Embodied AI), each with a `Key notes:` line of wikilinks plus a dataview query.
- `_01_notes/` — all knowledge notes, **flat** (no subfolders). One concept per note.
- `_03_templates/` — `knowledge-integration-note.md` (the note skeleton) and a Zotero literature-import template.
- `_04_images/` — images, embedded as `![[name.png|500]]`.
- Root-level loose notes (cheat sheets etc.) are outside the knowledge system — leave them alone unless asked.
- Never edit `.obsidian/`.

## Knowledge note construction rules

### Skeleton (from `_03_templates/knowledge-integration-note.md`)

```markdown
### [[dl_moc]]  
# **Note Title**

**Tags**:: #tag1 #tag2
**Links**:: [[Related Note]] · [[Another Note]]


One-paragraph intro: bolded term, what it is, why it matters, landmark citations
as [Author et al., YYYY](https://arxiv.org/abs/XXXX.XXXXX). Then a line naming the
primary source(s) the note follows.

**Notation** paragraph if the note is math-heavy (see Notation below).

## Section Title
---
Content…

## Relationship to Other Notes
---
- [[Other Note]] — one line on how the two connect (not just "related")
```

- Every `##` section heading is immediately followed by a `---` rule.
- File names: ASCII only (e.g. `muTransfer.md` for a note titled μP); the `# **Title**` may use Unicode.
- Tag vocabulary (don't invent new tags casually): `#math` `#architecture` `#learning` `#llm` `#llama` `#vision` `#dl_theo` `#rl` `#generative` `#diffusion`.

### Writing style

- **Bold concept lead-ins**, not bullet walls: paragraphs open with the term being defined in bold. Key defined terms may use the vault's yellow span, `<span style="color:rgb(255, 255, 0)"><b>Term.</b></span>`; blue spans mark intuition/commentary.
- Theorems as compact italic statements with the source pinpointed (e.g. "*6.S184 Thm 22*"); include proofs or proof sketches when short, counterexamples where they earn their keep.
- Display math in `$$ … $$`; inline math in `$…$`. Loss pairs annotated as `\text{(… loss — intractable)}` / `\text{(conditional …)}`.
- Short, runnable-looking `python` blocks for training/sampling loops; no boilerplate.
- Comparison tables for method families; a table needs a blank line before it, and an empty header cell should get a label or the table won't render.
- Cite primary sources with arXiv links and **verify claims/numbers/links against the actual papers** before writing them down — this vault has a history of hunting down wrong equations, misattributions, and arXiv links pointing at the wrong paper. Prefer exact numbers over vague claims.
- For reference-style notes (e.g. GPU Architecture): categorized glossary at the top with block anchors (bullet ends with `^slug`, forward links `→ [[#Section]]`), body links back via `[[#^slug|term]]` at first mention.

### Notation conventions

Generative-modeling notes (Flow Matching, Score Matching, CTMC, DDPM-adjacent, VAE…) use **MIT 6.S184 notation throughout**: source $p_\text{init}$, data $p_\text{data}$; data point $z$, noise $\epsilon$; schedulers $(\alpha_t, \beta_t)$; target fields $u_t^\text{target}(x)$, $u_t^\text{target}(x|z)$; network $u_t^\theta$; rate matrix $Q_t(y|x)$ with columns summing to zero; **FM time runs noise → data** ($t{=}0$ noise, $t{=}1$ data — the reverse of DDPM time). When a note follows a source with different symbols (the FM guide's $p,q$/$x_1$/$(\alpha_t,\sigma_t)$), add a translation sentence rather than switching notation. Flag clashes explicitly (e.g. $z$ = latent in VAE vs. $z$ = data point in FM).

### Cross-linking (required for every new note)

1. Add the note to the `Key notes:` line of the right section(s) in `dl_moc.md`.
2. Add back-links: put the new note on the `**Links**::` line of closely related notes, and add inline mentions where a related note discusses the same concept.
3. Give the new note a `## Relationship to Other Notes` section — each bullet says *how* the notes connect.
4. `**Links**::` entries are separated by ` · `.

### Splitting and moving content

When a note outgrows a section (or the user asks to split), move the section into its own note and leave behind a **compact digest + pointer**, not a bare link: keep the notation and the few formulas/results that downstream sections of the source note still use, and state where the full treatment lives. Update all heading-level links (`[[Note#Section]]`) that pointed at the moved content — convert to internal `[[#Section]]` links if source and target now share a file. Precedents: DDIM out of DDPM; the Background of Flow Matching and the Probability Paths/KFE sections of CTMC into "ODEs, SDEs, and Probability Paths".
