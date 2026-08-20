# Evidence, retries, and terminal decisions

## Evidence bundle

Every meaningful retained attempt that changes the reproduction's state or informs a terminal decision needs a small entry in `reproduction.json.runs`. Use stable unique IDs such as `compatibility-check-001`, `full-seed-42`, and `eval-seed-42`. A disposable preflight superseded by a successful full run does not need a permanent entry. Keep bounded small evidence in the entry and put large logs/results on permitted external storage with immutable URIs and hashes.

Capture:

- purpose/hypothesis and run kind;
- exact command, working directory, UTC timestamps, duration, and exit code;
- repository commit plus dirty-diff hash, upstream revision, patch hashes;
- container image reference/digest and dependency lock/freeze artifact;
- config, seed, precision, effective batch, checkpoint input/output;
- dataset volume/version/splits/counts/checksums;
- hardware, GPU count/type, peak memory, wall time, GPU-hours, and cost when available;
- Modal profile `repro-sign`, environment, app and function-call/run IDs, and links when applicable;
- shared cache mount `huggingface-cache`, its mount path, and Hugging Face cache environment when the run uses Modal;
- bounded stdout/stderr and immutable references/hashes for large logs or outputs;
- target IDs exercised and raw metric artifacts produced;
- observed result and whether the hypothesis was supported.

Never place credentials, private URLs with embedded tokens, dataset samples, or restricted content in evidence.

## Attempt ledger

Put human-readable attempt summaries in `README.md` and link their run IDs. Before changing anything, write the current hypothesis and cheapest discriminating test. After the test, record actual behavior. Keep or revert the change; do not stack untested fixes.

### Retry classes

| Class | Response |
| --- | --- |
| Auth/workspace | No retry or fallback. Ask the user to run `modal setup` and select `repro-sign`. |
| License/access | No workaround. Use the data gate. |
| Budget/protocol | Do not shrink the experiment silently. Use the compute/protocol gate. |
| Transient network/infrastructure | Retry at most three times with bounded backoff; count cost and record each attempt. |
| Deterministic build/code | One scoped fix per hypothesis; rerun the smallest failing path. |
| OOM/resource | Measure cause; preserve effective optimization semantics or request a protocol decision. |
| Numerical instability | Reproduce declared seeds/precision first; inspect loss, gradients, inputs, and implementation. Do not tune toward the paper. |
| Metric mismatch | Audit split, preprocessing, checkpoint, decoding, aggregation, metric implementation/version, and copied-score provenance. |
| Interrupted full run | Resume only from a checkpoint already proven reloadable; otherwise diagnose before paying for restart. |

A retry ceiling is a maximum, not a target. Stop early when a retry cannot test a new hypothesis.

## Terminal target states

Each target embeds one terminal `result`:

- `produced`: the requested pipeline emitted a traceable value; or
- `not_produced`: no value was emitted, with a specific blocker and evidence.

Do not use numerical closeness to assign these states. Store the reproduced value, mechanically computed delta, run IDs, and artifact IDs beside the target contract without repeating the target in another file.

Choose the overall reproducibility status from target coverage and the actual blocker:

- all targets produced → `complete`;
- only some produced → `partial`;
- none produced because exact data is unavailable/impermissible → `blocked_on_data`;
- none produced after reviewed out-of-budget requirements → `blocked_on_compute`;
- none produced because no executable path survives documented attempts → `blocked_on_code`;
- none produced because essential target/protocol details remain unknowable → `insufficient_information`.

If multiple blockers exist, report all and choose the earliest blocker that independently prevents the requested target pipeline. Explain the choice.

## Final consistency checks

Before handoff:

- every target ID is unique and has one terminal result;
- every result points to an existing run and raw artifact by internal ID;
- paper/candidate IDs agree across JSON files;
- `reproduction.json` and `README.md` use the same overall status and preference level;
- commands, hashes, pins, data identity, guesses, deviations, patches, dead ends, contacts, runtime, and compute identifiers are complete;
- templates contain no examples, alternatives, or placeholder markers;
- the retained full run reaches metric parsing, or a representative preflight does when the full run is gated.

Run the validator, fix all errors, and inspect the final diff. The validator checks structure and cross-file invariants; it does not replace scientific review.
