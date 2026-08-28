# <paper short name> reproduction

**Paper ID:** `<paper_id>`

**Citation:** <full citation>

**Paper:** <canonical URL> · **Code/artifacts:** <URLs or none found after search>

**Preference level:** <1, 2, or 3>

**Status:** `<reproducibility_status>`

**Attempt date:** <date>

## Scope and target contract

Exact assignment text and how it was resolved into `reproduction.json.targets`. Cite the table/figure/section, rows, systems, datasets/splits, metric definitions and versions, aggregation, seeds, checkpoint rules, and published values. Explain any ambiguity and its resolution. Keep this README as the complete human-readable report; do not create a second report or model-card file.

## Source provenance

| Artifact | Canonical source | Pinned revision / SHA-256 | Role |
| --- | --- | --- | --- |
| Paper PDF |  |  | Target and protocol |
| Published code |  |  |  |
| Weights/configs/supplements |  |  |  |

List all sources considered, including searches performed when the candidate said no code was available, and explain why the selected artifacts are authoritative.

## Results

| Target ID | Paper location | System | Dataset/split | Metric + version | Original | Reproduced | Difference | Terminal reason / evidence |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
|  |  |  |  |  |  |  |  |  |

Account for every target, including numbers not produced. State which original scores were copied from earlier work and whether this attempt reproduced those baselines.

When no target was produced, state the selected `status.blocker` reason code and why it is the earliest blocker that independently prevents the requested pipeline. Report other blockers separately rather than hiding them behind the selected one.

Pipeline completeness and numerical agreement are separate: report the observed differences without declaring scientific success or failure.

## How to repeat this

```bash
# Exact setup, data, build, train, and evaluation commands from a fresh checkout.
```

State expected inputs, mounts, outputs, checkpoint/resume procedure, and the one command that reaches each target metric.

## Data provenance and permissions

| Dataset | Version/subset/splits | Source and access date | License/permission and cloud-use basis | Path in Volume `datasets` | Counts / manifest / checksum | Deviations |
| --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |

Document unavailable, mismatched, private, or substituted data explicitly. Never include restricted data or credentials.

## Environment and patches

Container base and digest, resolved dependency manifest, upstream revisions, patch hashes, and relevant host/GPU/driver details. For Modal, record the `huggingface-cache` mount and Hugging Face cache environment; list pinned Hub repositories/revisions rather than treating cached files as provenance.

| Patch | Demonstrated failure | Hypothesis | Why necessary | Behavioral effect | Evidence |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |

If no patches were needed, say so.

## Execution evidence

| Run ID | Attempt / max | Kind / targets | Platform / hardware | Seed/config | Start/end | Exit / terminal state / reason | Failure class | Stop ceilings | Logs/artifacts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |  |  |

For Modal runs, record profile `repro-sign`, environment, app/function-call IDs, and dashboard links. State the predefined terminal reason code and its specific detail, not an improvised synonym. Link raw metric outputs for every score.

## Guesses and deviations

| Detail | Paper/evidence says | This attempt used | Rationale | Effect on interpretation |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

Include every missing detail filled in and every departure from the published protocol, even if it seems harmless.

## Attempts, failures, and dead ends

For each meaningful attempt: hypothesis, cheapest test, observed result, evidence, and whether the change was kept or reverted. Include transient retries, OOMs, build failures, invalid runs, interrupted full runs, and unresolved blockers.

## Candidate flags, ethics, and human evaluation

Record how queue comments, copied-score, ethics, and human-evaluation flags were investigated. State any privacy/ethics gate and whether the reproduction used participants or sensitive data.

## Author and team contact

State none, or who was contacted, when, why, the response, and what it changed. Distinguish Team S data-access coordination from post-attempt author help.
