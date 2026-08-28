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
    "max_cost_chf": 100.0
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

All fields shown in `recording`, `attempt`, `stop_policy`, and `terminal` are
required; use explicit `null` only where this contract permits it.

For new work, use `recording.mode: contemporaneous` with an empty
`unknown_fields` array and a null `source_record_sha256`. The start, finish, and
declaration are UTC timestamps, the declaration precedes the start, and
`max_wall_time_seconds` is positive. `max_gpu_hours` and `max_cost_chf` are
positive when they apply and `null` otherwise.

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

For Modal runs, record `compute.gpu_count`. A contemporaneous GPU run has
non-null GPU-hour and cost ceilings. A retrospective GPU run may use null only
when each missing ceiling is explicitly listed as unknown. A CPU or non-compute
run may use `null` for ceilings that do not apply.

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
