# REPRO-SIGN — Reproduction

This repository holds the **reproduction attempts** of REPRO-SIGN, a quantitative meta-analysis
of how reproducible research in Sign Language Processing (SLP) is.

Reproducibility in SLP is widely believed to be low, but the evidence has been anecdotal. The
study's goal is to replace that with a number and a citable reference: take a sample of
published SLP papers, genuinely try to reproduce their main quantitative results, and report
what fraction can be reproduced and why the rest cannot.

For each paper we try to get from *published number* to *number we produced ourselves* —
recovering the code, environment, and data; running training and evaluation end-to-end in a
container; and recording every pin, command, guess, and deviation along the way. **The output is
evidence, not a verdict.**

If you are here from a paper, a model card, or a Hugging Face repo: the directory under
[`papers/`](papers) matching that paper is the full report.

> **This is not an audit of anyone's work.** The study assigns no blame and singles out no
> researcher. A failed reproduction is usually a finding about documentation, tooling, and data
> availability in the field — the goal is to improve reproducibility in SLP, and the final paper
> is meant to end in actionable recommendations.

## Where this repository sits in the study

REPRO-SIGN is a multi-institution collaboration split into three teams, so no one has to hold
the whole project in their head at once:

| Team | Owns | In this repo? |
| --- | --- | --- |
| **S — Survey** | A structured, bias-free literature search; the pool of candidate papers and their metadata; random sampling of the next paper to reproduce. | No — supplies the candidate records. |
| **R — Reproduction** | Reading an assigned paper and actually reproducing its experiments; documenting obstacles, compute, and outcomes. | **Yes. This repository is Team R's output.** |
| **C — Coordination & analysis** | Scope and success criteria, per-metric tolerance thresholds, statistical analysis of outcomes, paper writing. | No — consumes the results here. |

That division explains the most surprising thing about this repository: **Team R deliberately
does not decide whether a reproduction succeeded.** Whether a produced number counts as
reproduced — exactly, approximately, or not at all — depends on tolerance thresholds and
analysis that belong to Team C. We produce numbers and evidence; interpretation happens
downstream.

It also explains the paper selection. Papers are **randomly sampled** from Team S's pool
specifically to eliminate selection bias, so the list below is not a reading list of influential
work. It is a representative sample, and that is the point.

Reproductions are carried out by AI agents working autonomously through source discovery,
containerization, bounded experiments, and reporting, pausing only at defined human gates
(data access and licensing, ethics review, ambiguous targets, behavior-changing deviations,
compute beyond budget). The agent policy that governs all of it is [`AGENTS.md`](AGENTS.md) —
worth reading if you want to know what the attempts were and were not allowed to do.

A guiding rule throughout is *least invasive wins*: prefer running the authors' published code
untouched, then the smallest correctness patches to make it execute, and only reimplement from
the paper when a genuine search finds no usable code. Published code is never refactored, and
implementation choices are never tuned toward the published score.

## Reading a reproduction

Every paper is one directory, `papers/<first-author>-<year>-<short-title>/`, containing:

| File | What it is |
| --- | --- |
| `README.md` | The complete human report: targets, pins, data provenance, commands, results, guesses, deviations, dead ends. Doubles as the model card when weights are published. |
| `reproduction.json` | The machine-readable source of truth: assignment provenance, agent attribution, source pins, datasets, targets, runs, artifacts, gates, status. |
| `Dockerfile` | The environment, when the shared base image is not enough. |
| `scripts/`, `patches/` | Entry points and retained upstream patches, when more than one exists. |

Datasets, checkpoints, and large logs are **not** in Git — they live on Modal or Hugging Face
and are referenced by immutable URI and checksum.

### Two axes of status, never collapsed into one verdict

- **`status.pipeline`** — did the pipeline produce the requested numbers?
  `complete` · `partial` · `blocked_on_data` · `blocked_on_compute` · `blocked_on_code` · `insufficient_information`
- **`status.numerical_agreement`** — do the produced values agree with the published ones?
  `agrees` · `does_not_agree` · `not_assessed`

These are independent on purpose. A reproduction can run perfectly and disagree with the paper;
that is a complete, successful *attempt* and a finding in its own right. Likewise, a number that
happens to land close is not a result if the experiment it came from was under-specified.

## Current attempts

As recorded in each `reproduction.json` (last surveyed 2026-10-12):

| Paper | Datasets | Pipeline | Numerical agreement |
| --- | --- | --- | --- |
| [Camgöz et al. — Neural Sign Language Translation](papers/camgoz-2018-nslt) | PHOENIX14T | `complete` | `agrees` |
| [Camgöz et al. — Sign Language Transformers](papers/camgoz-2020-slt) | PHOENIX14T | `complete` | `does_not_agree` |
| [Mohamed et al. — DenseNet121 with Harris Hawks Optimization](papers/mohamed-2024-densenet121-hho) | ArASL2018 | `complete` | `not_fully_reproduced` |
| [Silanon et al. — Classification Layer Technique for ASL](papers/silanon-2023-asl-cnn) | ASL Alphabet | `complete` | `not_fully_reproduced` |
| [Chen et al. — Sign Language-Gloss Translations With Pretrained Models](papers/chen-2026-gloss-pretrained) | PHOENIX14T, ASLG-PC12, CSL-Daily | `partial` | `not_fully_reproduced` |
| [Partaourides et al. — Variational Bayesian Seq2Seq for SLT](papers/partaourides-2021-variational-bayesian-slt) | PHOENIX14T | `partial` | `does_not_agree` |
| [Ahmad et al. — Intelligent Signs Language Understanding](papers/ahmad-2022-intelligent-landmarks) | ASL Alphabet, ISL-HS | `insufficient_information` | `not_assessed` |
| [Ben Abderrahmane et al. — Temporal-to-Spatial Transmutation](papers/ben-abderrahmane-2025-temporal-to-spatial) | ArabSign | `insufficient_information` | `not_assessed` |
| [Huang & Chouvatut — Video-Based SLR via ResNet and LSTM](papers/huang-chouvatut-2024-resnet-lstm) | LSA64 | `insufficient_information` | `not_assessed` |
| [Sivamohan et al. — Hand Gesture Recognition and Translation](papers/sivamohan-2024-isl-cnn) | Author-supplied ISL landmarks | `insufficient_information` | `not_assessed` |

`insufficient_information` is itself a frequent outcome, and one of the study's more interesting
signals: the attempt stopped because the paper and its artifacts did not pin down which
experiment, split, or metric the published number came from. Each of those reports documents
exactly what was searched and what was missing.

## Repeating an attempt

Each paper's `README.md` carries the exact, idempotent commands for that reproduction. In
general:

```bash
./setup.sh                      # one-time per clone: agent tooling, Modal skill, base image
cd papers/<paper-slug>          # then follow that README's setup / data / train / eval steps
```

Runs execute on [Modal](https://modal.com) in the `repro-sign` workspace, with shared volumes
for datasets and the Hugging Face cache, funded by a seed grant with a fixed total GPU budget.
That ceiling is a real constraint on the study: it is why runs are bounded, single-GPU by
default, and gated before anything expensive. Published weights, where licensing allows, go to
the [`repro-sign`](https://huggingface.co/repro-sign) organization on Hugging Face.

## Also in this repository

- **[`Dockerfile`](Dockerfile)** — the shared GPU base image
  (`ghcr.io/sign-language-processing/reproduction:latest`): NVIDIA NGC PyTorch, FFmpeg, decord,
  `simple-video-utils`. Most reproductions start `FROM` it.
- **[`libraries/`](libraries)** — hard-won build notes for dependencies that routinely break
  (decord, FFmpeg, flash-attn). Read these before changing an environment.
- **[`.agents/skills/reproduce-paper/`](.agents/skills/reproduce-paper)** — the agent workflow:
  candidate ingestion, the Modal wrapper, stopping criteria, and record templates.
- **`repositories/`** — pre-study Dockerfiles for third-party repos, kept for reference. Not a
  template for new work.

## Caveats

Numbers here describe **our attempt**, not a judgment of the paper. A disagreement may come from
an under-documented protocol, an unavailable dataset version, hardware nondeterminism, our own
error, or a problem in the original work — the reports record the difference and the evidence,
and leave the conclusion to human reviewers. Statuses also change as attempts are revisited;
`reproduction.json` is always authoritative over this summary.

Licensed under [MIT](LICENSE). Papers, datasets, and upstream code keep their own licenses.
