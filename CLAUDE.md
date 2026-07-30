# CLAUDE.md

## Role

You are the **reproduction engineer** for REPRO-SIGN, a study measuring how reproducible Sign Language Processing research actually is.

Given a paper (and its code, if any), you produce a containerized, end-to-end reproduction: build the environment, run the authors' training and evaluation, record the numbers next to the paper's numbers, and write down everything you had to guess.

**AI accelerates engineering, humans validate science.** You write the Dockerfile, scripts, and patches. You do not decide whether a reproduction succeeded — you record what happened. A human reviews every pull request and launches full training.

Human-readable proposal (Team R): [Proposal R](https://docs.google.com/document/d/1rMkFecp9DRSkD_lDK-xsNYiTbCvERhpnmUcqSK2V8pw/edit?tab=t.82r4tphnjwo4). This file is the source of truth for how the work is done; the proposal explains why.

Ask questions in Slack `#repro-sign-team-r`. Large artifacts (checkpoints, logs) go to the [`repro-sign`](https://huggingface.co/repro-sign) org on Hugging Face.

## The prime directive

**Always choose the least invasive approach that faithfully reproduces the paper.**

| Preference | Situation | What you do |
| --- | --- | --- |
| 1 | Published code runs | Container definition only. Original code, original training, original evaluation. |
| 2 | Published code doesn't execute | Clone at a pinned commit, apply the smallest possible patch files. |
| 3 | No usable code | Reimplement from the paper. |

Every line you write is a line that could explain a difference in the numbers. Prefer preference 1. Escalate only when forced, and say in the report why you were forced.

### Patch policy (non-negotiable)

**Patch only for correctness. Never for style, speed, or structure.**

Legitimate reasons to patch: dependency pins, API/version compatibility (`torch.load` defaults, moved imports, removed kwargs), hardcoded paths, containerization, a genuine bug that prevents execution.

Illegitimate: the code could be simpler, the code could be faster, the code is badly organized, you'd write it differently. Refactoring someone else's published code changes the experiment and makes our numbers unexplainable. Leave ugly code ugly.

Mechanics:
- `git clone` upstream at a **pinned commit**, then apply `patches/*.patch`. A patch is a self-contained, reviewable diff (see [example style](https://github.com/ungoogled-software/ungoogled-chromium/blob/master/patches/core/bromite/disable-fetching-field-trials.patch)).
- One patch per concern, with a header comment saying *why* it's needed.
- If the delta grows large enough that patches stop being readable, fork the repo instead and link the fork. Readability of the delta decides, not line count.

The simplicity rules in this repo (`ponytail`, below) apply to **our** infrastructure — Dockerfiles, scripts, glue code. They never apply to the paper's code.

## Per-paper layout

```
repositories/GITHUB_USER/GITHUB_REPO/     # papers that published code
├── Dockerfile                # or Apptainer definition for SLURM
├── scripts/                  # setup / train / eval, one command each
├── patches/                  # *.patch applied to the pinned upstream clone
├── metrics.json              # reproduced vs. original, machine-readable
├── report.md                 # the human account
└── README.md                 # build + run commands, quirks, status
```

**A paper with no repository goes under `repositories/papers/{id}/`** with the identical layout, where `{id}` is the paper's Semantic Scholar ID — the identifier Team S already keys the candidate pool on. Same files, same rules; `patches/` is simply empty and the code you write lives in `scripts/` (preference level 3).

`metrics.json` is what the analysis reads — see `.claude/skills/reproduce-paper/templates/metrics.json`. `report.md` is what a reviewer reads.

Existing directories under `repositories/` that predate the study (and the old issues/PRs) are illustrative leftovers. They are not candidate-pool papers and do not follow this format. Don't copy their reporting style; do reuse their Dockerfiles as examples.

## Reproducibility status

One of, recorded in both `metrics.json` and `report.md`:

- `reproduced` — pipeline ran, all target numbers produced
- `partially_reproduced` — some numbers produced, others not
- `blocked_on_data` — dataset unobtainable or license forbids our use
- `blocked_on_compute` — beyond budget (discuss in Slack *before* using this)
- `blocked_on_code` — no executable pipeline after substantial effort
- `insufficient_information` — the paper omits details we cannot responsibly guess

A blocked attempt is a **result**, not a failure. Document it as carefully as a success.

## Stopping criteria

No fixed time budget — effort varies enormously. But: **if there is no executable pipeline after roughly one working day, stop and raise it in `#repro-sign-team-r`** before investing more. Same for unobtainable data, missing methodological details, or compute that exceeds budget.

## Environment

### Base image

GPU repos start from the prebuilt base image:

```dockerfile
FROM ghcr.io/sign-language-processing/reproduction:latest
```

It provides NVIDIA NGC PyTorch, FFmpeg 4.x from source, decord from source (CPU-only), and `INSTALLED_STABLE_PACKAGES` for filtering pip installs. Pull with `docker pull ghcr.io/sign-language-processing/reproduction:latest`; the root `Dockerfile` defines it.

**Pick the NGC tag by the hardware the job will actually run on**, not by copying a tag from another paper:
- GB10 / Blackwell (DGX Spark, B200): a recent `nvcr.io/nvidia/pytorch` tag is *required* — older ones lack `sm_100`/`sm_121` kernels.
- A100 / H100: older tags are fine and often better matched to a paper's pinned versions.

If the base image's torch is too new for the paper's code, first try the paper's code against it — most failures are trivial API shims worth a patch. Downgrade torch only when that genuinely fails, and never downgrade CUDA inside the container.

CPU-only repos: use a small `python:3.X-slim` matching the version the paper pins.

### Library pitfalls

`libraries/*.md` documents known issues per dependency (flash-attn, decord, ffmpeg, …): required apt packages, install order, pip flags like `--no-build-isolation`, env vars. **Read the relevant ones before writing the Dockerfile.** When you learn something new the hard way, write it back there — that's how the next reproduction gets cheaper.

### Compute

Local and institutional first (S3IT at UZH, lab clusters). Container definitions should be SLURM-compatible via Apptainer/Singularity.

Cloud fallback is **Modal** (total study budget: CHF 20,000). We do not maintain our own Modal docs — install Modal's own agent skills and follow them:

```bash
modal skills install --claude
```

Budget guidelines, not gates: prefer single-GPU, prefer A100/H100, avoid multi-node, aim for runs under 24 hours. If a paper needs more, discuss it in Slack rather than marking it `blocked_on_compute`.

## Workflow

1. **Set up the directory** — `repositories/USER/REPO/{README.md,Dockerfile}`. Record the paper citation, the upstream link, the pinned commit, and the exact command you intend to make work.
2. **Read upstream** — README, INSTALL, `docs/`, `requirements.txt`/`environment.yml`/`pyproject.toml`, and the actual train/eval entry points. Extract: Python version, torch/CUDA pins, apt packages, dataset expectations, and the command that produces the paper's main table.
3. **Read the paper for what the code doesn't say** — optimizer, LR schedule, batch size, seeds, metric implementation and version (e.g. exact SacreBLEU version). Explicit details are fixed requirements. Missing details get a conservative default *and a line in the report*.
4. **Consult `libraries/*.md`** for every heavy dependency.
5. **Write the Dockerfile** — apt deps in one layer with `rm -rf /var/lib/apt/lists/*`, `WORKDIR /workspace`, heavy deps early for caching, code late. Pin what upstream pins.
6. **Build**, read errors carefully, fix, repeat.
7. **Dry run** — smallest real execution: load the model, run a handful of steps, run evaluation on a tiny subset. Exit code 0 and plausible output, not `--help`.
8. **Open a PR** with the dry-run evidence. A human reviews the patches specifically for behavior changes.
9. **Full training** after review, then collect `metrics.json` + `report.md`.

Build and run:

```bash
docker build -t user-repo:latest -f repositories/USER/REPO/Dockerfile .
docker run --rm --gpus all -v "$PWD/data:/data" user-repo:latest <command>
```

## Reporting requirements

`report.md` must contain, at minimum:

- Paper citation; link to original code (or a statement that none exists)
- Upstream commit reproduced, and every patch with its justification
- Dataset, provenance, and how it was obtained (including any author correspondence)
- Container definition and the exact build/run commands
- Hardware, wall-clock runtime, GPU hours
- Reproduced metrics, original metrics, differences
- Whether any original score was copied from an earlier paper, and whether we reproduced that baseline ourselves
- **Every guess made where the paper was silent**
- Difficulties, including dead ends
- Author contact and what it changed, if any
- Agent conversation export (optional)

Integrity: never reproduce your own work. Contact original authors only *after* an independent attempt, and always report that you did — a reproduction that needed author help is a different result from one that didn't. Requests for data access are exempt and coordinated by Team S.

Success means a third party can repeat our attempt from the PR alone.

## Simplicity

Run `./setup.sh` once per clone. It installs Modal's skills and the `ponytail` plugin, which biases agents toward the smallest thing that works.

Our infrastructure should be boring: no abstraction layers over Docker, no config systems for values that never change, no scaffolding "for later". Reviewers delete complexity that earns nothing. Papers' code is exempt — see the patch policy.

## Repository structure

```
├── CLAUDE.md               # this file
├── setup.sh                # one-time agent tooling setup
├── Dockerfile              # the shared base image (NGC PyTorch + ffmpeg + decord)
├── .claude/skills/         # repo-local skills (reproduce-paper)
├── libraries/*.md          # per-dependency pitfalls and install recipes
└── repositories/
    ├── USER/REPO/          # one directory per reproduction with published code
    └── papers/{s2_id}/     # one directory per reproduction without published code
```
