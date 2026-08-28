# Stopping criteria and terminal records

Read this before recording the first retained run and again before closing a
target. Schema version 2 replaces ad hoc status/reason strings with the
controlled values below. Existing schema-version-1 records remain valid, but
new reproductions use version 2.

## Retained run contract

Every retained run has one attempt identity, a stop policy declared before the
run, and one terminal classification:

```json
{
  "run_id": "full-seed-42-attempt-1",
  "command": "exact command",
  "exit_code": 0,
  "started_at_utc": "2026-08-28T10:00:00Z",
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

`max_wall_time_seconds` is always a positive number. `max_gpu_hours` and
`max_cost_chf` are positive numbers when they apply and `null` otherwise. The
ceiling is a maximum, not a retry target. Stop earlier when another attempt
cannot test a new hypothesis.

The declaration timestamp must be UTC and no later than `started_at_utc`.
Attempt numbers are unique and contiguous from 1 within a `group_id`, and may
not exceed `max_attempts`. All attempts in one group use the same maximum.
Authentication, license/access, and budget/protocol failures have
`max_attempts: 1` and reference the matching gate. A transient infrastructure
failure permits the initial attempt plus at most three retries, so
`max_attempts` may not exceed 4.

For schema-version-2 Modal runs, record `compute.gpu_count`. A GPU run has
non-null GPU-hour and cost ceilings; a CPU or non-compute run may use `null`.

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
prevents the requested pipeline. `complete` and `partial` records do not use a
top-level blocker.
