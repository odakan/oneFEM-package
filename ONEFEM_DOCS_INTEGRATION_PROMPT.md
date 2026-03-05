# ONEFEM_DOCS_INTEGRATION_PROMPT.md
#
# PASTE THIS VERBATIM AS YOUR FIRST MESSAGE TO CLAUDE CODE
# after placing the docs/ directory in the repo root.
# ─────────────────────────────────────────────────────────────────────────────

I have added three reference documents to docs/ in this repository:

  docs/oneFEM_blueprint.md   — full architecture reference (read this first)
  docs/ADR_LOG.md            — closed architecture decisions (do not re-argue)
  docs/CLAUDE_additions.md   — status tables, stop/ask rules, task cookbook
                               (merge this content into CLAUDE.md now)

Please do the following, in this exact order:

---

STEP 1 — Read all three documents completely.

Read docs/oneFEM_blueprint.md, docs/ADR_LOG.md, and docs/CLAUDE_additions.md
in full before doing anything else. Do not skim. The blueprint Section 0
(Research Vision) is the most important — it tells you what this project
actually is and why every design decision exists.

---

STEP 2 — Merge docs/CLAUDE_additions.md into CLAUDE.md.

Open the existing CLAUDE.md and insert the following sections from
docs/CLAUDE_additions.md:

  a) "Architecture Reference" section — insert immediately after
     "## Project Overview". This is the mandatory blueprint pointer.

  b) "Current Implementation Status" section — insert after "## Testing".
     This is the ground truth for what exists vs. what is planned.

  c) "Stop and Ask Rules" — insert after the status table.

  d) "Task Cookbook" — insert at the end of CLAUDE.md.

Do not delete any existing content from CLAUDE.md. Do not paraphrase —
copy the sections verbatim from docs/CLAUDE_additions.md.

---

STEP 3 — Confirm your understanding by answering these five questions.

Answer all five before proceeding to any coding work:

  1. What Python library implements the GPU sparse solver (WP13)?
     What is the exact function call for a sparse direct solve?

  2. What does "zero-copy on SoC" mean architecturally?
     Why is cupy.asarray() on a Jetson AGX Orin different from on an RTX 3090?

  3. What is the Tier-1 acceptance gate for WP23-24?
     State it as a binary pass/fail condition.

  4. A contributor suggests wrapping the element assembly loop in Cython
     for performance. What is the correct response?

  5. The Material interface has _setTrialStrain, _commitState,
     _revertToLastCommit, getStress, getTangent, getInitialTangent,
     getStrain, getCopy, revertToStart.
     Which two of these are critical for (a) FiberSection and (b) FE²?
     Why?

---

STEP 4 — Update the Current Implementation Status table.

Scan the actual source tree under src/oneFEM/ and verify the ✅/⚠️/🔲
entries in the status table you just merged into CLAUDE.md. Correct any
entries that do not match what you find in the source. Add a comment
"# verified YYYY-MM-DD" at the top of the status table.

---

Only after completing all four steps are you ready to receive coding tasks.
