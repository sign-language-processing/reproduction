---
name: reproduce-paper
description: Reproduce an assigned REPRO-SIGN paper end-to-end — container, patches, training, evaluation, metrics.json and report.md. Use whenever a paper (or its GitHub repo) is assigned for reproduction, when asked to "reproduce PAPER/REPO", to containerize a paper's code, to write patch files for a cloned upstream repo, or to fill in a reproduction report or reproducibility status.
---

# Reproduce a paper

Read `CLAUDE.md` at the repo root first — it holds the prime directive, the patch policy, base-image rules, and the reporting requirements. This skill is the running order.

## 0. Before touching code

- Get the paper's target table. You are reproducing **specific numbers**, not "the repo".
- Pick the directory: `repositories/USER/REPO/` if the paper published code, `repositories/papers/{semantic_scholar_id}/` if it didn't.
- Record in its `README.md`: citation, upstream URL, **pinned commit**, dataset, target metrics, and the single command that should produce them.
- Note the hardware you'll run on — it decides the NGC tag.
- `modal volume ls` before touching the dataset: it is probably already there as `dataset-<slug>`. Mount it read-only, don't re-download.

## 1. Choose the preference level

1. Code runs as published → container only.
2. Code doesn't execute → pinned clone + `patches/*.patch`.
3. No usable code → reimplement from the paper.

Write down which level you're at and why. Escalating is a finding, not an inconvenience.

## 2. Harvest what the paper says, not just what the code does

Optimizer, LR + schedule, batch size, epochs/steps, seeds, preprocessing, metric implementation **and version**. Explicit values are fixed requirements. Every silence you fill becomes a line in "Guesses" in the report.

## 3. Build

Check `libraries/*.md` for each heavy dependency before writing the Dockerfile. Try the base image's torch first; patch API breakage rather than downgrading.

```bash
docker build -t user-repo:latest -f repositories/USER/REPO/Dockerfile .
```

## 4. Patch, correctly

Only for correctness — never because the code could be simpler. One concern per patch, each with a `why` header. Regenerate with `git diff > ../patches/NN-thing.patch` inside the clone.

## 5. Dry run

Smallest real execution: weights load, a few training steps, evaluation on a tiny subset, exit 0. Not `--help`. Capture the output — it's the PR evidence.

## 6. PR, then full run

Open the PR with the dry-run log. After human review, launch full training (local/SLURM first, Modal if not — follow Modal's own installed skills).

## 7. Report

Copy `templates/metrics.json` and `templates/report.md` into the paper directory and fill them. Every field in the reporting requirements section of `CLAUDE.md` must be present. Set the reproducibility status honestly.

## Stop and ask

No executable pipeline after ~1 working day, unobtainable data, missing critical details, or over-budget compute → raise it in `#repro-sign-team-r` before spending more.
