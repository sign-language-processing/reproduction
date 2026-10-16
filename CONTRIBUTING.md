# Contributing

This repository holds Team R's reproduction attempts for the REPRO-SIGN study. Most pull
requests add or update a single `papers/<paper-slug>/` directory.

[`AGENTS.md`](AGENTS.md) is the authority on policy: scope, the preference ladder, patch rules,
data and compute gates, status vocabulary, and the evidence contract. This file does not restate
it. It covers the part `AGENTS.md` does not: **how to get a reproduction PR ready, and how to
review one.**

## Before you open a PR

One branch per paper. Then, from the repository root:

```bash
python3 .agents/skills/reproduce-paper/scripts/validate_reproduction.py papers/<paper-slug>
```

Fix every reported issue before pushing. Then read your own diff end to end — the validator
checks structure, not sense.

In the PR description, say which paper, which preference level, and what a reviewer should look
at hardest. If the attempt stopped at a gate, say which gate and what evidence was gathered;
a blocked attempt is a legitimate, reportable outcome and should not be dressed up as progress.

## Reviewing a reproduction PR

### 1. Run the validator first

```bash
python3 .agents/skills/reproduce-paper/scripts/validate_reproduction.py papers/<paper-slug>
```

Do this before reading anything. It already enforces a large share of what used to be reviewed
by hand, including:

- `status.pipeline` and `status.numerical_agreement` are drawn from the allowed vocabularies, so
  invented values like `not_fully_reproduced` are rejected;
- the pipeline status is **derived** correctly — all in-scope targets produced ⇒ `complete`,
  some ⇒ `partial`, none ⇒ a blocker status, and nothing produced ⇒ `not_assessed`;
- `status.numerical_agreement_basis` is non-empty, and a `complete` pipeline retains no open
  gate;
- `README.md` surfaces the statuses and preference level in a fixed format, matching
  `reproduction.json` exactly:

  ```markdown
  **Pipeline status:** `complete`
  **Numerical agreement:** `does_not_agree` — optional explanation after an em dash
  **Preference level:** 1
  ```

- every produced target carries a numeric `original_value`, `reproduced_value`, and a
  `difference` that actually equals reproduced − original;
- every result points to a real run and artifact by internal ID;
- `README.md` contains no `<…>` placeholders (note this forbids literal angle brackets anywhere
  in the report, inline HTML included);
- `schema_version` is gone, since there is one current contract.

Some records committed before the current contract still fail. Hold a PR to its own paper
directory; do not ask a contributor to repair unrelated legacy records, but do flag them in a
separate issue.

### 2. Read the report as a stranger would

The validator cannot tell whether the science is right. These are the checks that need a human:

**Targets match the paper.** Open the actual paper table, caption, and surrounding text. Confirm
each target's system, dataset version and split, metric implementation and direction, and
published value. Vague queue text and an opaque metric ID are not a target specification. This
is the most common place a reproduction goes quietly wrong, and no amount of green validation
catches it.

**Score pairs are legible.** When the pipeline is `complete` or `partial`, the published and
reproduced values must appear side by side in `README.md` as a table a reader can scan — one row
per target, with the difference. The pairs are enforced inside `reproduction.json`; whether a
human can actually *find* them in the report is the reviewer's call.

**Preference level is honest, and matches the shape of the diff.** The level claimed must match
what the diff actually does:

| Level | Situation | What the diff should look like |
| --- | --- | --- |
| 1 | Published code runs | Container and entry points only. No upstream code. |
| 2 | Published code does not execute | A pinned upstream commit plus `upstream.patch` (one patch) or `patches/*.patch` (several, ordered). Still no vendored upstream code. |
| 3 | No usable code exists after a real search | An original implementation written from the paper, with every inferred detail documented. |

So patches are the signature of **level 2**, not level 3 — at level 3 there is no upstream to
patch. At levels 1 and 2, upstream source copied into the repo instead of expressed as a patch
against a pinned commit is a defect, however convenient. At level 3, demand the evidence that
the source search really happened before accepting that no code exists; a missing link in the
queue is not proof.

**Patches are corrective only.** Read each patch, not just its header. Each should carry one
concern and a header saying why it is necessary: dependency or API compatibility, hardcoded
paths, containerization, or a genuine execution bug. Reject style, structure, speed, and
preference rewrites of the authors' code. Watch hardest for any change that moves a number
toward the published score — hyperparameters, split handling, checkpoint selection, metric
variants. That is the one failure that invalidates the study rather than just the PR.

**Required files are present, and nothing extra is.** Always `README.md` and
`reproduction.json`. A `Dockerfile` only when the shared base image is not enough — its absence
is fine and often correct. An entry point (`modal_app.py` and friends) for anything that ran;
a paper with no run, because it stopped at a gate, legitimately has neither. Per the layout
rules, a lone script or patch stays at the paper root with a descriptive name — `scripts/`,
`patches/`, and `artifacts/` exist only once there are several related files. No empty
directories.

**No junk, no bulk, no secrets.** Datasets, checkpoints, large logs, and licensed artifacts
belong in Modal or Hugging Face, referenced by immutable URI and checksum. Check the diff for
`__pycache__/`, `*.pyc`, editor state, and stray outputs — `.gitignore` covers these, but it
does not untrack files that were committed before the rule, and some already in this repo slipped
through exactly that way.

**No personal data.** Reports, records, run commands, patch headers, and log excerpts must not
carry email addresses, usernames or account handles, operator real names, machine-local absolute
paths (`/Users/…`, `/home/…`, `~`), environment identity, or tokens. Queue exports must have
reviewer and assignee addresses stripped while keeping the source SHA-256. Published author
names and contact details that the paper itself states are fine, and are the point of the
author-contact section.

**The paper is identifiable.** `README.md` must carry the report's identity line, and it must
match `reproduction.json.paper_id`:

```markdown
**Paper ID:** `8526aecd1407305d815883725a864405e31a54c1`
```

For a queue-backed assignment this is the reviewing platform's own unique record ID — the value
the record was selected by — so that a report can be traced back to the exact annotated
candidate and matched up with Team S's pool during analysis. Do not substitute the directory
slug there; the slug is the path, not the identity. A `direct_user_request` assignment has no
platform record, and uses the paper slug instead. The validator does not cross-check this line
against the JSON the way it does the statuses, so it is on the reviewer.

**Pins are immutable.** Every Git source pinned to a full commit SHA, never a branch or tag;
non-Git artifacts carrying a checksum; Hugging Face revisions pinned. A reviewer should be able
to fetch exactly what ran.

**Provenance and attribution hold up.** `assignment` preserves one final/confirmed record (or
direct-assignment provenance) and its source hash, with no invented queue-review fields.
`agents` records model and harness separately, and each run's `agent_ids` points at whoever ran
it; unknown identity is `null` with an explanation, never a plausible guess. Modal runs record
the `repro-sign` profile, app and run IDs, GPU, and timings — and never token IDs or credentials.

**Data is gated, not assumed.** Each dataset entry should establish version, subset, split,
preprocessing, access path, licence or permission basis, expected counts, and a checksum.
`available: yes` in a queue record proves nothing. A substituted or similarly named dataset is a
recorded deviation, never a silent one.

**The two status axes stay independent.** A clean run that disagrees with the paper is a complete
attempt and a finding. A number that lands close is not a result if the experiment behind it was
under-specified or involved an invented split, aggregation, or feature reduction — that is
conditional evidence, kept with its run but not marked `produced`. Push back on any PR that
collapses the axes into a single verdict of success or failure.

**A stranger could repeat it.** Commands idempotent and actually present, environment pinned,
every reported score traceable to a run and a raw metric artifact, and `README.md` agreeing with
`reproduction.json` with no placeholders.

### 3. Review tone

The study assigns no blame to the papers it reproduces and singles out no researcher, and review
comments should read the same way. A reproduction that failed is data, not an accusation, and a
contributor who documented a dead end honestly did the job right.

Do not review a reproduction of your own work, and do not reproduce your own work — flag the
conflict of interest and hand it to someone else.

Interpretation of outcomes is Team C's job. Reviewers here confirm that the numbers and evidence
are sound, not whether a paper "passed".
