# Modal, data, and compute

Read this before any Modal, dataset-volume, or cloud-compute action.

## Required workspace preflight

The Modal profile associated with the REPRO-SIGN workspace/team is exactly `repro-sign`. The active/default profile is not sufficient because it can change between commands.

If `modal` is unavailable, the wrapper runs:

```bash
python3 -m pip install modal
```

It then stops. The user must authenticate interactively with:

```bash
modal setup
```

Ask the user to select or confirm the `repro-sign` workspace, then resume only after wrapper preflight succeeds.

Use the wrapper for preflight and every subsequent command:

```bash
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh volume list --json
```

The wrapper exports `MODAL_PROFILE=repro-sign`, checks `modal profile current`, privately verifies both credentials and the returned workspace identity with `modal token info`, and then forwards non-credential-sensitive arguments. It does not print token information, activate another profile, or alter credentials.

If profile, workspace, or credential preflight fails, do not inspect volumes, billing, apps, secrets, or start compute elsewhere. Ask the user to run:

```bash
modal setup
```

Resume only after wrapper preflight succeeds. Do not inspect or print credential files, token IDs, or secrets.

Official references: [Modal setup](https://modal.com/docs/cli/latest/setup), [profiles](https://modal.com/docs/cli/latest/profile), and [workspaces](https://modal.com/docs/guide/workspaces).

## Modal skill and CLI

Prefer Modal's installed agent skill for current SDK mechanics rather than copying a stale internal manual. Install/update it in the generic `.agents/` location through the wrapper with `modal_repro_sign.sh skills install -y` or `modal_repro_sign.sh skills update -y`; do not pass `--claude`.

Even when following Modal's own skill, retain this repository's stricter wrapper, data-license, cost, and evidence rules.

## Canonical shared Volumes

The `repro-sign` workspace has two canonical VolumeFS v2 Volumes:

- `datasets`: authoritative approved dataset trees, one `<slug>/` directory per dataset;
- `huggingface-cache`: shared cache for Hugging Face model, tokenizer, and Hub artifacts.

Check both before a run:

```bash
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh volume list --json
```

If either is missing, create it once with the wrapper and `volume create --version 2 NAME`. Never create per-paper Hugging Face caches or per-dataset Volumes.

## Dataset gate

The canonical shared Volume is named `datasets` and uses VolumeFS v2. Each dataset lives at root path `<slug>/`; experiments mount the Volume read-only at `/datasets`, making that dataset available as `/datasets/<slug>`.

Start with a non-mutating inventory and exact-path check:

```bash
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh volume list --json
.agents/skills/reproduce-paper/scripts/check_modal_dataset.sh <slug> [manifest-relative-path]
```

The dataset check also fails if the required `huggingface-cache` Volume is absent.

If `datasets` is missing, create it once with:

```bash
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh volume create --version 2 datasets
```

Do not create per-dataset Volumes. For the requested dataset path, verify:

- `<slug>/` exists and maps unambiguously to the paper's exact dataset/version/subset;
- expected split manifests and sample counts exist;
- files or manifests have stable checksums;
- the license/permission allows project-cloud processing;
- preprocessing provenance is known;
- it can be mounted read-only by training/evaluation.

Do not equate the shared Volume's existence with dataset availability. Record the checked path and validation in `reproduction.json.datasets`, `README.md`, and the run entry.

If the requested dataset path is absent:

1. Confirm authoritative source, version, license/terms, cloud-processing rights, and whether redistribution is permitted.
2. Prefer a committed, idempotent `data.sh` (or `scripts/data.sh` when a real script suite exists) that downloads/verifies into `/datasets/<slug>` on the shared Volume.
3. Store source URLs, access dates, checksums, expected counts, and transformations.
4. Keep download credentials outside code and logs.
5. If a click-through, account approval, private transfer, author request, or legal/identity judgment is needed, stop at the data gate and append a gate with the exact missing action and evidence to `reproduction.json.gates`. Route dataset acquisition/identity work to Team S.

An ethics flag alone is not a stop condition. Open an ethics/privacy gate when the planned work adds participants or human raters, or when existing data is identifiable/sensitive and its consent, terms, access, cloud processing, storage, or reporting basis is not clear.

Never silently use a public surrogate for unavailable experiment data. It may be run only as an explicitly labeled deviation after the protocol gate, and its result cannot stand in for the requested target.

## Hugging Face cache

Mount `huggingface-cache` read-write at `/cache/huggingface` in every Modal reproduction function and set:

```text
HF_HOME=/cache/huggingface
HF_HUB_CACHE=/cache/huggingface/hub
```

Use the shared cache for Hugging Face Hub downloads, model weights, tokenizers, and reusable Hub assets. Pin repository revisions in code/config and record them in the run entry; a cache hit does not establish identity or provenance.

Do not use this Volume as the authoritative home for datasets, checkpoints, predictions, metrics, logs, or secrets. Material needed for the exact dataset goes under `/datasets/<slug>` subject to the data gate. Outputs go to paper-specific storage.

## Compute plan

Use a representative real-data preflight to measure peak GPU memory, examples/steps per second, checkpoint size, evaluation throughput, and fixed startup overhead. Estimate:

- total optimizer/evaluation steps;
- wall time and GPU-hours for every required seed;
- GPU type/count and memory headroom;
- persistent volume and artifact storage;
- likely cost and the cost of permitted retries.

Inspect current workspace billing through the wrapper when needed and available. Record the estimate, assumptions, allowed GPU/concurrency, checkpoint cadence, and the schema-version-2 attempt and stop policy from [stopping-criteria.md](stopping-criteria.md) in the full-run entry before launch.

Proceed without asking only inside the root compute gate. If the estimate is near a boundary, use the conservative side. Never reduce required seeds, data, steps, precision, model size, or evaluation scope solely to fit the gate; request a protocol decision.

## Full-run reliability

- Use stable resource names containing the paper ID or a short collision-resistant derivative.
- Mount `huggingface-cache` and set the required Hugging Face cache environment on every Modal reproduction function.
- Make training checkpointable and test resume before the full run.
- Keep the `datasets` Volume read-only during experiments; write checkpoints/results to a separate paper-specific Volume or other permitted destination.
- Record the Modal app ID, function-call/run ID, environment, GPU, image ID/digest, start/end timestamps, terminal state, and dashboard/log links.
- Monitor to completion. Fetch final logs and output manifests even after a remote failure.
- Cancel runaway or clearly invalid work within the recorded stop ceiling; preserve evidence before retrying.

Do not deploy persistent endpoints or services when an ephemeral run suffices.
