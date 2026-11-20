# Stopping criteria and terminal records

Read this before recording the first retained run and again before closing a
target. All reports use the same controlled fields and values below. When a
historical fact was not recorded, represent it explicitly as unknown; never
invent a value.

## Retained run contract

Every retained run has one attempt identity, one stop-policy record, and one
terminal classification:

```json
{
  "run_id": "full-seed-42-attempt-1",
  "agent_ids": ["reproduction-agent"],
  "command": "exact command",
  "exit_code": 0,
  "started_at_utc": "2026-08-28T10:00:00Z",
  "finished_at_utc": "2026-08-28T10:30:00Z",
  "recording": {
    "mode": "contemporaneous",
    "source_record_sha256": null,
    "unknown_fields": []
  },
  "attempt": {
    "group_id": "full-seed-42",
    "number": 1,
    "max_attempts": 3
  },
  "stop_policy": {
    "declared_at_utc": "2026-08-28T09:55:00Z",
    "max_wall_time_seconds": 86400,
    "max_gpu_hours": 24.0,
    "max_cost_usd": 100.0
  },
  "terminal": {
    "state": "succeeded",
    "reason_code": "completed",
    "detail": "Evaluation completed and emitted every requested metric.",
    "failure_class": null,
    "retry_decision": "not_needed",
    "gate_ids": []
  }
}
```

`agent_ids` references the executing entries in `reproduction.json.agents` (see
`AGENTS.md`). An unattributable historical run uses an empty array and a specific
`agent_attribution_note`; this does not change its stop-policy recording mode.

`command` is the exact command as a repo-relative or container path. Remove local home paths before recording it.

All fields shown in `recording`, `attempt`, `stop_policy`, and `terminal` are
required; use explicit `null` only where this contract permits it.

For new work, use `recording.mode: contemporaneous` with an empty
`unknown_fields` array and a null `source_record_sha256`. The start, finish, and
declaration are UTC timestamps, the declaration precedes the start, and
`max_wall_time_seconds` is positive. `max_gpu_hours` and `max_cost_usd` are
positive when they apply and `null` otherwise.

The cost ceiling is in US dollars, the currency Modal bills in and
`compute.usage.cost_usd` records, so the two compare without conversion. Runs
whose ceiling was declared before 2026-11-20, or whose declaration time is a
retrospective unknown, keep the `max_cost_chf` they were declared with: never
convert or relabel a declared ceiling. Such a run may also carry
`max_cost_usd`. A run declared from 2026-11-20 uses `max_cost_usd` only.

For a historical run whose original stop policy was not recorded, keep the same
shape and use explicit unknowns:

```json
{
  "started_at_utc": null,
  "finished_at_utc": null,
  "recording": {
    "mode": "retrospective",
    "source_record_sha256": "<SHA-256 of the record before migration>",
    "unknown_fields": [
      "started_at_utc",
      "finished_at_utc",
      "attempt.max_attempts",
      "stop_policy.declared_at_utc",
      "stop_policy.max_wall_time_seconds"
    ],
    "detail": "Backfilled from an already-committed report."
  },
  "attempt": {
    "group_id": "historical-run",
    "number": 1,
    "max_attempts": null
  },
  "stop_policy": {
    "declared_at_utc": null,
    "max_wall_time_seconds": null,
    "max_gpu_hours": null,
    "max_cost_chf": null
  }
}
```

`retrospective` exists only to migrate already-committed evidence. Preserve any
historical value that evidence supports. Every unknown field must be selected
from the validator's narrow allowlist, present as `null`, and named in
`unknown_fields`; the source hash and explanation make the backfill auditable.
Terminal state, reason, failure class, and retry decision can never be unknown.
The validator accepts retrospective mode only for paper IDs and pre-migration
record hashes in its reviewed migration registry; an arbitrary hash cannot opt a
new run out of contemporaneous requirements. Never select retrospective mode
for a newly launched run. A ceiling is a maximum, not a retry target. Stop
earlier when another attempt cannot test a new hypothesis.

Attempt numbers are unique and contiguous from 1 within a `group_id`, and may
not exceed a known `max_attempts`. All attempts in one group use the same known
maximum. A retrospective record may use null only when the historical ceiling
was not recorded.
Authentication, license/access, and budget/protocol failures have
`max_attempts: 1` and reference the matching gate. A transient infrastructure
failure permits the initial attempt plus at most three retries, so
`max_attempts` may not exceed 4.

For Modal runs, record `compute.gpu_count` and `compute.modal_app_ids`: every
Modal app ID (`ap-` plus 22 letters or digits) the run used, in a list, because
a resumed or relaunched run can span several apps. Any app ID mentioned
elsewhere in the run entry must also be in that list, and a run that names a
Modal app declares `compute.platform: "modal"`. The ID fields `app_id`
and `modal_app_id`, on the run, its `compute`, or its `modal` object, are
rejected. A run whose app ID was not recorded keeps an empty list, and the
lapse is recorded as a `noncompliance` entry whose `run_ids` include it.

A contemporaneous GPU run has non-null GPU-hour and cost ceilings. If it was
launched without declaring them, keep them `null` rather than backfilling, and
record the lapse as a `noncompliance` entry whose `run_ids` include the run; its
`description` says what limited the run instead, such as a Modal function
timeout. A retrospective GPU run may use null only when each missing ceiling is
explicitly listed as unknown. A CPU or non-compute run may use `null` for ceilings that do
not apply.

### Actual compute usage

Ceilings say what a run may use; `compute.usage` records what it did use. Every
GPU Modal run records it, whatever its terminal state: an interrupted or failed
run still consumed GPU time. CPU runs may omit it.

```json
"compute": {
  "platform": "modal",
  "modal_profile": "repro-sign",
  "modal_app_ids": ["ap-..."],
  "gpu_count": 1,
  "usage": {
    "gpu_hours": 3.42,
    "gpu_hours_basis": "measured",
    "cost_usd": 12.3,
    "cost_basis": "billed",
    "source": "modal_repro_sign.sh billing report --start 2026-11-20 --end 2026-11-21 -r h --show-resources --json, read 2026-11-21: rows with object_id ap-...; GPU cost / L40S rate 1.95 USD/h"
  }
}
```

All five fields are required, and no other fields are allowed. `source` says
how both numbers were obtained: the command, when it was run, the app IDs, and
any rate or apportioning used. `usage` is the only place a GPU run records
GPU-hours and cost: other `compute` fields whose names contain `gpu_hours` or
`cost` are rejected. Raw durations, such as `wall_time_hours` or a process's
measured seconds, may stay beside it.

- `gpu_hours` is the GPU time Modal bills for the run: the wall time of its
  GPU-attached containers, multiplied by `gpu_count`. Choose
  `gpu_hours_basis`:
  - `measured`: taken from Modal's records. From billing: run the report below
    with `--show-resources`, sum the GPU rows (for example `L40S`) for the run's
    app, and divide by that GPU's hourly rate from
    `modal_repro_sign.sh billing rates`;
  - `upper_bound`: bounded by a longer interval that contains it, such as the
    run's `started_at_utc` to `finished_at_utc` times `gpu_count`, which also
    covers image build and queueing;
  - `estimated`: anything weaker.
- `cost_usd` is in US dollars, the currency Modal bills in. Do not convert it to
  CHF in the record; the study converts totals once, at one rate, when it
  compares them with the CHF budget.
- `cost_basis: billed` reads Modal's billing data: run
  `modal_repro_sign.sh billing report` over the run's interval and sum the
  `cost` of every row whose `object_id` is listed in `compute.modal_app_ids`.
  It therefore needs a non-empty list; a run whose app ID was not recorded can
  only be estimated. If an app also served other runs, apportion its cost and
  say how in `source`. Billing data can lag by a few hours; record the cost
  once the report covers the whole run.
- `cost_basis: rate_estimate` multiplies `gpu_hours` by the GPU's rate from
  `modal_repro_sign.sh billing rates`. It covers the GPU only, not CPU or
  memory; say so in `source`.

Billing history is retained, so the same recipe recovers usage for past runs
from their recorded app IDs. A daily report (`-r d`) can span at most 31 days;
query longer ranges in pieces. `billing rates` returns current rates only, so
GPU-hours derived from older billing assume the rate has not changed; note the
rate in `source`.

#### Who records billed usage

Reading billing requires the Owner or Manager role in the `repro-sign`
workspace; Members cannot. A contributor without that role records an estimate
when the run ends:

- `gpu_hours`: `upper_bound`, `finished_at_utc` minus `started_at_utc`, times
  `gpu_count`;
- `cost_usd`: `rate_estimate`, `gpu_hours` times the GPU's rate from
  `modal_repro_sign.sh billing rates`, with the rate and the date it was read
  in `source`. This estimate covers the GPU only, so unlike the hours it is not
  an upper bound: CPU and memory are billed on top.

The reviewer who merges the pull request, holding one of those roles, replaces
every estimate whose run has an app ID with `measured` and `billed` values from
the recipe above, and updates `source`. After merge, only runs without a
recorded app ID keep estimates.

### Terminal states and reason codes

| State | Allowed reason codes | Meaning |
| --- | --- | --- |
| `succeeded` | `completed` | The intended scope completed and the command exited zero. |
| `failed` | `command_failed` | The command terminated without completing the intended scope. |
| `stopped` | `human_gate_opened`, `retry_ceiling_reached`, `wall_time_ceiling_reached`, `gpu_hour_ceiling_reached`, `cost_ceiling_reached`, `invalid_run`, `no_new_hypothesis` | The agent deliberately ended the run or retry sequence. |
| `interrupted` | `external_interruption` | Infrastructure, preemption, or another external event interrupted the run. |

For every non-success state, choose exactly one `failure_class`:

- `auth_workspace`
- `license_access`
- `budget_protocol`
- `transient_infrastructure`
- `deterministic_code`
- `oom_resource`
- `numerical_instability`
- `interrupted_full_run`
- `invalid_run`

The class selects the response; the reason code records why execution stopped.
Keep the human-readable `detail` specific and evidence-based. Do not invent a
new code for nuance that belongs in `detail`.

Choose `retry_decision` from `not_needed`, `retry`, `resume`, `await_gate`, or
`stop`. Successful runs use `not_needed`. `retry` and `resume` name an existing
`next_run_id` in the same attempt group with the next attempt number. They are
invalid on the final allowed attempt, and every attempt after the first must be
linked from its immediately preceding attempt this way.
`human_gate_opened`/`await_gate` also list at least one open `gate_id`.
`retry_ceiling_reached` is valid only on the final allowed attempt.

A numerical or metric mismatch is not a run failure. If evaluation completes
and emits the requested traceable value, the run is `succeeded`, the target is
`produced`, and the disagreement is recorded through the target difference and
overall numerical-agreement status.

## Target terminal contract

A produced target retains its value, mechanical difference, and run/artifact
references. A target that was not produced uses a controlled reason:

```json
{
  "status": "not_produced",
  "reason": {
    "code": "target_ambiguous",
    "detail": "The paper does not identify the held-out signers."
  },
  "run_ids": [],
  "artifact_ids": [],
  "gate_ids": ["paper-signer-split"],
  "evidence": ["Paper section 4.2 specifies only an 8/2 learner count."]
}
```

Choose one reason code:

| Reason code | Overall status when no target is produced |
| --- | --- |
| `data_unavailable` | `blocked_on_data` |
| `data_permission_blocked` | `blocked_on_data` |
| `compute_budget_blocked` | `blocked_on_compute` |
| `compute_infrastructure_blocked` | `blocked_on_compute` |
| `code_unavailable` | `blocked_on_code` |
| `code_not_executable` | `blocked_on_code` |
| `target_ambiguous` | `insufficient_information` |
| `protocol_ambiguous` | `insufficient_information` |
| `metric_ambiguous` | `insufficient_information` |
| `human_evaluation_required` | `insufficient_information` |
| `copied_baseline` | No blocker mapping; use only when `copied_baseline` is true. |

Every `not_produced` result has a non-empty `evidence` list. Data, compute,
ambiguity, and human-evaluation reasons also reference a matching gate.
`code_not_executable` references at least one run classified as
`deterministic_code` or `invalid_run`; a transient interruption cannot establish
that no executable path survives. A copied baseline also requires the target's
verified `copied_baseline: true` determination.

When no target is produced, `status.blocker` makes the selected overall reason
explicit and traceable:

```json
{
  "pipeline": "insufficient_information",
  "numerical_agreement": "not_assessed",
  "preference_level": 3,
  "blocker": {
    "reason_code": "target_ambiguous",
    "detail": "The unpublished signer split independently prevents every target.",
    "target_ids": ["table4-accuracy", "table3-macro-f1"]
  }
}
```

The blocker reason must map to the pipeline status, and each referenced target
must carry the same reason code. If multiple blockers exist, report all of them
in target results and gates, then choose the earliest one that independently
prevents the requested pipeline. `complete` and `partial` records set
`status.blocker` to `null`.
