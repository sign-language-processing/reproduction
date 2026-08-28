#!/usr/bin/env python3
"""Validate one normalized REPRO-SIGN paper reproduction."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PIPELINE_STATUSES = {
    "complete",
    "partial",
    "blocked_on_data",
    "blocked_on_compute",
    "blocked_on_code",
    "insufficient_information",
}
NUMERICAL_AGREEMENTS = {
    "fully_reproduced",
    "not_fully_reproduced",
    "not_assessed",
}
TARGET_STATUSES = {"produced", "not_produced"}
RUN_STATES = {"succeeded", "failed", "stopped", "interrupted"}
RUN_REASON_CODES = {
    "succeeded": {"completed"},
    "failed": {"command_failed"},
    "stopped": {
        "human_gate_opened",
        "retry_ceiling_reached",
        "wall_time_ceiling_reached",
        "gpu_hour_ceiling_reached",
        "cost_ceiling_reached",
        "invalid_run",
        "no_new_hypothesis",
    },
    "interrupted": {"external_interruption"},
}
FAILURE_CLASSES = {
    "auth_workspace",
    "license_access",
    "budget_protocol",
    "transient_infrastructure",
    "deterministic_code",
    "oom_resource",
    "numerical_instability",
    "interrupted_full_run",
    "invalid_run",
}
RETRY_DECISIONS = {"not_needed", "retry", "resume", "await_gate", "stop"}
TARGET_REASON_PIPELINES = {
    "copied_baseline": None,
    "data_unavailable": "blocked_on_data",
    "data_permission_blocked": "blocked_on_data",
    "compute_budget_blocked": "blocked_on_compute",
    "compute_infrastructure_blocked": "blocked_on_compute",
    "code_unavailable": "blocked_on_code",
    "code_not_executable": "blocked_on_code",
    "target_ambiguous": "insufficient_information",
    "protocol_ambiguous": "insufficient_information",
    "metric_ambiguous": "insufficient_information",
    "human_evaluation_required": "insufficient_information",
}
GATE_TYPES = {
    "modal_auth",
    "data",
    "target",
    "ethics",
    "protocol",
    "compute",
    "secrets",
    "destructive",
    "authority",
}
EXTERNAL_PREFIXES = ("https://", "http://", "hf://", "modal://", "s3://")
SHA256 = re.compile(r"[0-9a-f]{64}")
RETROSPECTIVE_UNKNOWN_FIELDS = {
    "started_at_utc",
    "finished_at_utc",
    "attempt.max_attempts",
    "stop_policy.declared_at_utc",
    "stop_policy.max_wall_time_seconds",
    "stop_policy.max_gpu_hours",
    "stop_policy.max_cost_chf",
}
HISTORICAL_RECORD_HASHES = {
    "8526aecd1407305d815883725a864405e31a54c1": (
        "6c215d86e281c7cb7c4accde67efae0f5d9da44b33852bd3310dd0eb51aad1b6"
    ),
    "camgoz-2018-nslt": (
        "ef710734dc1a00069f62974491f6d9cbfd8e6787051e915e7ac84f7ebc1f1f13"
    ),
    "camgoz-2020-slt": (
        "9fe456f04eae036309ce538cd0e650299002390772d2d4daddcc815bb29bc15f"
    ),
    "990030f8dfefb06e99c05218741e11ccf7b08fdb": (
        "8313e5a5d4c6471b5d50f8d7d040105d002dffe564881fb938a246cf7f918383"
    ),
}


def nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def valid_choice(value: Any, choices: set[str] | dict[str, Any]) -> bool:
    return isinstance(value, str) and value in choices


def finite_number(
    value: Any, *, positive: bool = False, nonnegative: bool = False
) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if not math.isfinite(value):
        return False
    if positive:
        return value > 0
    if nonnegative:
        return value >= 0
    return True


def parse_utc_timestamp(value: Any, label: str, issues: list[str]) -> datetime | None:
    if not nonempty(value):
        issues.append(f"{label} must be an ISO-8601 UTC timestamp")
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        issues.append(f"{label} must be an ISO-8601 UTC timestamp")
        return None
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        issues.append(f"{label} must use UTC")
        return None
    return parsed


def check_sha(value: Any, label: str, issues: list[str]) -> None:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        issues.append(f"{label} must be a lowercase SHA-256")


def check_reference(root: Path, value: Any, label: str, issues: list[str]) -> None:
    if not nonempty(value):
        issues.append(f"{label} must be a non-empty path or immutable URI")
    elif not value.startswith(EXTERNAL_PREFIXES) and not (root / value).exists():
        issues.append(f"{label} does not exist: {value}")


def keyed(
    values: Any, key: str, label: str, issues: list[str]
) -> dict[str, dict[str, Any]]:
    if not isinstance(values, list):
        issues.append(f"{label} must be an array")
        return {}
    result: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(values):
        if not isinstance(value, dict) or not nonempty(value.get(key)):
            issues.append(f"{label}[{index}] must be an object with {key}")
            continue
        identity = value[key]
        if identity in result:
            issues.append(f"duplicate {key} {identity!r}")
        else:
            result[identity] = value
    return result


def checked_references(
    values: Any,
    known: dict[str, Any],
    label: str,
    kind: str,
    issues: list[str],
    *,
    required: bool = False,
) -> list[str]:
    if not isinstance(values, list):
        issues.append(f"{label} must be an array")
        return []
    if required and not values:
        issues.append(f"{label} must not be empty")
    valid: list[str] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        if not nonempty(value):
            issues.append(f"{label}[{index}] must be a non-empty {kind} ID")
        elif value not in known:
            issues.append(f"{label} references unknown {kind} {value!r}")
        elif value in seen:
            issues.append(f"{label} repeats {kind} {value!r}")
        else:
            valid.append(value)
            seen.add(value)
    return valid


def validate_assignment(document: dict[str, Any], issues: list[str]) -> None:
    assignment = document.get("assignment")
    if not isinstance(assignment, dict):
        issues.append("assignment must be an object")
        return
    kind = assignment.get("kind")
    if not valid_choice(kind, {"queue_record", "direct_user_request"}):
        issues.append("assignment.kind must be queue_record or direct_user_request")
    source = assignment.get("source")
    if not isinstance(source, dict):
        issues.append("assignment.source must be an object")
    else:
        check_sha(source.get("sha256"), "assignment.source.sha256", issues)
        path = source.get("path")
        if nonempty(path) and Path(path).exists():
            actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            if actual != source.get("sha256"):
                issues.append("assignment source file no longer matches its SHA-256")
    if kind == "queue_record":
        record = assignment.get("record")
        if not isinstance(record, dict):
            issues.append("queue assignment must preserve record")
        elif (
            record.get("paper_id") != document.get("paper_id")
            or record.get("id") not in (None, document.get("paper_id"))
            or record.get("confirmation") != "confirmed"
            or record.get("status") != "final"
        ):
            issues.append("queue record is not final/confirmed or has inconsistent IDs")


def validate_datasets(document: dict[str, Any], issues: list[str]) -> dict[str, Any]:
    datasets = keyed(document.get("datasets"), "dataset_id", "datasets", issues)
    for dataset_id, dataset in datasets.items():
        label = f"dataset {dataset_id!r}"
        if not nonempty(dataset.get("name")) or not nonempty(
            dataset.get("version_or_subset")
        ):
            issues.append(f"{label} needs name and version_or_subset")
        if not nonempty(dataset.get("license_or_permission")):
            issues.append(f"{label} needs license_or_permission")
        if not isinstance(dataset.get("cloud_processing_allowed"), bool):
            issues.append(f"{label} needs boolean cloud_processing_allowed")
        if dataset.get("modal_volume") != "datasets":
            issues.append(f"{label} must use Modal Volume 'datasets'")
        modal_path = dataset.get("modal_path")
        if (
            not nonempty(modal_path)
            or modal_path.startswith("/")
            or ".." in str(modal_path).split("/")
        ):
            issues.append(f"{label} needs a safe relative modal_path")
        splits = dataset.get("splits")
        if not isinstance(splits, list) or not splits:
            issues.append(f"{label} needs splits")
            continue
        for index, split in enumerate(splits):
            split_label = f"{label} split[{index}]"
            if not isinstance(split, dict) or not nonempty(split.get("name")):
                issues.append(f"{split_label} needs name")
                continue
            if not isinstance(split.get("sample_count"), int):
                issues.append(f"{split_label} needs integer sample_count")
            files = split.get("files")
            if not isinstance(files, list) or not files:
                issues.append(f"{split_label} needs files")
                continue
            for file_index, file in enumerate(files):
                if not isinstance(file, dict) or not nonempty(file.get("path")):
                    issues.append(f"{split_label} file[{file_index}] needs path")
                    continue
                check_sha(
                    file.get("sha256"),
                    f"{split_label} file[{file_index}].sha256",
                    issues,
                )
    return datasets


def validate_artifacts(
    root: Path, document: dict[str, Any], issues: list[str]
) -> dict[str, Any]:
    artifacts = keyed(document.get("artifacts"), "artifact_id", "artifacts", issues)
    for artifact_id, artifact in artifacts.items():
        label = f"artifact {artifact_id!r}"
        if not nonempty(artifact.get("kind")):
            issues.append(f"{label} needs kind")
        check_reference(root, artifact.get("uri"), f"{label}.uri", issues)
        check_sha(artifact.get("sha256"), f"{label}.sha256", issues)
        if "size_bytes" in artifact and not isinstance(artifact["size_bytes"], int):
            issues.append(f"{label}.size_bytes must be an integer")
    return artifacts


def validate_runs(
    document: dict[str, Any],
    artifacts: dict[str, Any],
    gates: dict[str, Any],
    issues: list[str],
) -> dict[str, Any]:
    runs = keyed(document.get("runs"), "run_id", "runs", issues)
    attempt_groups: dict[str, dict[str, Any]] = {}
    paper_id = document.get("paper_id")
    registered_source_hash = (
        HISTORICAL_RECORD_HASHES.get(paper_id) if isinstance(paper_id, str) else None
    )
    for run_id, run in runs.items():
        label = f"run {run_id!r}"
        if not nonempty(run.get("command")):
            issues.append(f"{label} needs an exact command")
        if isinstance(run.get("exit_code"), bool) or not isinstance(
            run.get("exit_code"), int
        ):
            issues.append(f"{label} exit_code must be an integer")
        checked_references(
            run.get("artifact_ids"),
            artifacts,
            f"{label}.artifact_ids",
            "artifact",
            issues,
        )
        mode, unknown_fields = validate_run_recording(
            run_id,
            run,
            registered_source_hash,
            issues,
        )
        validate_run_stopping(
            run_id,
            run,
            runs,
            gates,
            attempt_groups,
            mode,
            unknown_fields,
            issues,
        )
        compute = run.get("compute")
        if isinstance(compute, dict) and compute.get("platform") == "modal":
            if compute.get("modal_profile") != "repro-sign":
                issues.append(f"Modal {label} must use profile 'repro-sign'")
            gpu_count = compute.get("gpu_count")
            if (
                isinstance(gpu_count, bool)
                or not isinstance(gpu_count, int)
                or gpu_count < 0
            ):
                issues.append(f"Modal {label} needs nonnegative gpu_count")
            elif gpu_count > 0:
                stop_policy = run.get("stop_policy")
                if isinstance(stop_policy, dict):
                    for field in ("max_gpu_hours", "max_cost_chf"):
                        path = f"stop_policy.{field}"
                        if stop_policy.get(field) is None and not (
                            mode == "retrospective" and path in unknown_fields
                        ):
                            issues.append(
                                f"GPU Modal {label} needs {field} or an explicit "
                                "retrospective unknown"
                            )
            cache = compute.get("shared_cache")
            if not isinstance(cache, dict):
                issues.append(f"Modal {label} needs shared_cache")
            else:
                if cache.get("modal_volume") != "huggingface-cache":
                    issues.append(f"Modal {label} must use 'huggingface-cache'")
                if cache.get("mount_path") != "/cache/huggingface":
                    issues.append(
                        f"Modal {label} cache mount must be /cache/huggingface"
                    )
                environment = cache.get("environment", {})
                expected = {
                    "HF_HOME": "/cache/huggingface",
                    "HF_HUB_CACHE": "/cache/huggingface/hub",
                }
                if not isinstance(environment, dict) or any(
                    environment.get(key) != value for key, value in expected.items()
                ):
                    issues.append(f"Modal {label} has invalid cache environment")
    for group_id, group in attempt_groups.items():
        numbers = group["numbers"]
        expected = set(range(1, max(numbers) + 1))
        if numbers != expected:
            issues.append(
                f"attempt group {group_id!r} must be contiguous from attempt 1"
            )
            continue
        for number in range(2, max(numbers) + 1):
            previous_run_id = group["run_by_number"][number - 1]
            current_run_id = group["run_by_number"][number]
            previous_terminal = runs[previous_run_id].get("terminal", {})
            if (
                not isinstance(previous_terminal, dict)
                or not valid_choice(
                    previous_terminal.get("retry_decision"), {"retry", "resume"}
                )
                or previous_terminal.get("next_run_id") != current_run_id
            ):
                issues.append(
                    f"attempt {number} in group {group_id!r} is not linked "
                    "from its preceding attempt"
                )
    return runs


def nested_value(document: dict[str, Any], path: str) -> tuple[bool, Any]:
    value: Any = document
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return False, None
        value = value[part]
    return True, value


def validate_run_recording(
    run_id: str,
    run: dict[str, Any],
    registered_source_hash: str | None,
    issues: list[str],
) -> tuple[str | None, set[str]]:
    label = f"run {run_id!r}"
    recording = run.get("recording")
    if not isinstance(recording, dict):
        issues.append(f"{label} needs recording provenance")
        return None, set()
    for field in ("mode", "source_record_sha256", "unknown_fields"):
        if field not in recording:
            issues.append(f"{label}.recording needs {field}")
    mode = recording.get("mode")
    if not valid_choice(mode, {"contemporaneous", "retrospective"}):
        issues.append(f"{label}.recording.mode is invalid")
        mode = None
    values = recording.get("unknown_fields")
    unknown_fields: set[str] = set()
    if not isinstance(values, list):
        issues.append(f"{label}.recording.unknown_fields must be an array")
    else:
        for index, value in enumerate(values):
            if not nonempty(value) or value not in RETROSPECTIVE_UNKNOWN_FIELDS:
                issues.append(
                    f"{label}.recording.unknown_fields[{index}] is not allowed"
                )
            elif value in unknown_fields:
                issues.append(f"{label}.recording.unknown_fields repeats {value!r}")
            else:
                unknown_fields.add(value)
                exists, field_value = nested_value(run, value)
                if not exists or field_value is not None:
                    issues.append(
                        f"{label}.recording unknown field {value!r} must exist "
                        "and be null"
                    )
    if mode == "contemporaneous":
        if unknown_fields:
            issues.append(f"contemporaneous {label} cannot contain unknown fields")
        if recording.get("source_record_sha256") is not None:
            issues.append(
                f"contemporaneous {label} must have null source_record_sha256"
            )
    elif mode == "retrospective":
        source_hash = recording.get("source_record_sha256")
        check_sha(
            source_hash,
            f"{label}.recording.source_record_sha256",
            issues,
        )
        if registered_source_hash is None:
            issues.append(
                f"retrospective {label} is not registered as a historical migration"
            )
        elif source_hash != registered_source_hash:
            issues.append(
                f"retrospective {label} does not match its registered source record"
            )
        if not nonempty(recording.get("detail")):
            issues.append(f"retrospective {label}.recording needs detail")
    return mode, unknown_fields


def validate_run_stopping(
    run_id: str,
    run: dict[str, Any],
    runs: dict[str, Any],
    gates: dict[str, Any],
    attempt_groups: dict[str, dict[str, Any]],
    recording_mode: str | None,
    unknown_fields: set[str],
    issues: list[str],
) -> None:
    label = f"run {run_id!r}"
    attempt = run.get("attempt")
    number: int | None = None
    maximum: int | None = None
    group_id: str | None = None
    if not isinstance(attempt, dict):
        issues.append(f"{label} needs attempt")
    else:
        group_id = attempt.get("group_id")
        number = attempt.get("number")
        maximum = attempt.get("max_attempts")
        if not nonempty(group_id):
            issues.append(f"{label}.attempt needs group_id")
            group_id = None
        if isinstance(number, bool) or not isinstance(number, int) or number < 1:
            issues.append(f"{label}.attempt.number must be a positive integer")
            number = None
        if maximum is None and (
            recording_mode == "retrospective"
            and "attempt.max_attempts" in unknown_fields
        ):
            pass
        elif isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
            issues.append(
                f"{label}.attempt.max_attempts must be a positive integer or an "
                "explicit retrospective unknown"
            )
            maximum = None
        if number is not None and maximum is not None and number > maximum:
            issues.append(f"{label}.attempt.number exceeds max_attempts")
        if group_id is not None and number is not None:
            group = attempt_groups.setdefault(
                group_id,
                {"max_attempts": maximum, "numbers": set(), "run_by_number": {}},
            )
            if (
                group["max_attempts"] is not None
                and maximum is not None
                and group["max_attempts"] != maximum
            ):
                issues.append(
                    f"attempt group {group_id!r} has inconsistent max_attempts"
                )
            elif group["max_attempts"] is None and maximum is not None:
                group["max_attempts"] = maximum
            if number in group["numbers"]:
                issues.append(f"attempt group {group_id!r} repeats attempt {number}")
            group["numbers"].add(number)
            group["run_by_number"][number] = run_id

    stop_policy = run.get("stop_policy")
    if not isinstance(stop_policy, dict):
        issues.append(f"{label} needs stop_policy")
    else:
        for field in (
            "declared_at_utc",
            "max_wall_time_seconds",
            "max_gpu_hours",
            "max_cost_chf",
        ):
            if field not in stop_policy:
                issues.append(f"{label}.stop_policy needs {field}")
        timestamps: dict[str, datetime | None] = {}
        for value, path in (
            (run.get("started_at_utc"), "started_at_utc"),
            (run.get("finished_at_utc"), "finished_at_utc"),
            (stop_policy.get("declared_at_utc"), "stop_policy.declared_at_utc"),
        ):
            if value is None and (
                recording_mode == "retrospective" and path in unknown_fields
            ):
                timestamps[path] = None
            else:
                timestamps[path] = parse_utc_timestamp(value, f"{label}.{path}", issues)
        started_at = timestamps["started_at_utc"]
        finished_at = timestamps["finished_at_utc"]
        declared_at = timestamps["stop_policy.declared_at_utc"]
        if (
            declared_at is not None
            and started_at is not None
            and declared_at > started_at
        ):
            issues.append(f"{label}.stop_policy was declared after the run started")
        if (
            started_at is not None
            and finished_at is not None
            and finished_at < started_at
        ):
            issues.append(f"{label}.finished_at_utc precedes started_at_utc")
        for field in ("max_wall_time_seconds", "max_gpu_hours", "max_cost_chf"):
            value = stop_policy.get(field)
            path = f"stop_policy.{field}"
            if value is None:
                if field == "max_wall_time_seconds" and not (
                    recording_mode == "retrospective" and path in unknown_fields
                ):
                    issues.append(
                        f"{label}.{path} must be positive or an explicit "
                        "retrospective unknown"
                    )
            elif not finite_number(value, positive=True):
                issues.append(f"{label}.{path} must be positive or null")

    terminal = run.get("terminal")
    if not isinstance(terminal, dict):
        issues.append(f"{label} needs terminal")
        return
    for field in (
        "state",
        "reason_code",
        "detail",
        "failure_class",
        "retry_decision",
        "gate_ids",
    ):
        if field not in terminal:
            issues.append(f"{label}.terminal needs {field}")
    state = terminal.get("state")
    reason_code = terminal.get("reason_code")
    failure_class = terminal.get("failure_class")
    retry_decision = terminal.get("retry_decision")
    if not valid_choice(state, RUN_STATES):
        issues.append(f"{label}.terminal has invalid state")
    elif not valid_choice(reason_code, RUN_REASON_CODES[state]):
        issues.append(f"{label}.terminal reason_code is invalid for state {state!r}")
    if not nonempty(terminal.get("detail")):
        issues.append(f"{label}.terminal needs detail")
    if not valid_choice(retry_decision, RETRY_DECISIONS):
        issues.append(f"{label}.terminal has invalid retry_decision")
    if state == "succeeded":
        if run.get("exit_code") != 0:
            issues.append(f"succeeded {label} must have exit_code 0")
        if failure_class is not None:
            issues.append(f"succeeded {label} must have null failure_class")
        if retry_decision != "not_needed":
            issues.append(f"succeeded {label} must use retry_decision not_needed")
    elif not valid_choice(failure_class, FAILURE_CLASSES):
        issues.append(f"non-successful {label} needs a valid failure_class")
    if (
        isinstance(state, str)
        and state in {"failed", "stopped", "interrupted"}
        and retry_decision == "not_needed"
    ):
        issues.append(f"non-successful {label} cannot use retry_decision not_needed")

    gate_ids = checked_references(
        terminal.get("gate_ids", []),
        gates,
        f"{label}.terminal.gate_ids",
        "gate",
        issues,
    )
    if reason_code == "human_gate_opened":
        if not gate_ids:
            issues.append(f"{label} stopped at a human gate but has no gate_ids")
        elif any(
            gates[gate_id].get("status") != "open"
            for gate_id in gate_ids
            if gate_id in gates
        ):
            issues.append(f"{label} human_gate_opened must reference open gates")
        if retry_decision != "await_gate":
            issues.append(f"{label} human_gate_opened must await_gate")
    if retry_decision == "await_gate" and not gate_ids:
        issues.append(f"{label} retry_decision await_gate needs gate_ids")
    elif retry_decision == "await_gate" and reason_code != "human_gate_opened":
        issues.append(f"{label} await_gate requires reason_code human_gate_opened")

    next_run_id = terminal.get("next_run_id")
    if valid_choice(retry_decision, {"retry", "resume"}):
        if not nonempty(next_run_id) or next_run_id not in runs:
            issues.append(f"{label} retry/resume needs an existing next_run_id")
        else:
            next_attempt = runs[next_run_id].get("attempt")
            if not isinstance(next_attempt, dict):
                issues.append(f"next run {next_run_id!r} needs attempt")
            elif (
                group_id is not None
                and number is not None
                and (
                    next_attempt.get("group_id") != group_id
                    or next_attempt.get("number") != number + 1
                )
            ):
                issues.append(
                    f"{label} next_run_id must be the next attempt in its group"
                )
        if number is not None and maximum is not None and number >= maximum:
            issues.append(f"{label} cannot retry/resume after max_attempts")
    elif next_run_id is not None:
        issues.append(f"{label} has next_run_id without retry/resume")
    if reason_code == "retry_ceiling_reached" and (
        number is None or maximum is None or number != maximum
    ):
        issues.append(f"{label} retry_ceiling_reached requires the final attempt")
    reached_limits = {
        "wall_time_ceiling_reached": "max_wall_time_seconds",
        "gpu_hour_ceiling_reached": "max_gpu_hours",
        "cost_ceiling_reached": "max_cost_chf",
    }
    limit_field = reached_limits.get(reason_code)
    if limit_field and (
        not isinstance(stop_policy, dict) or stop_policy.get(limit_field) is None
    ):
        issues.append(f"{label} {reason_code} requires stop_policy.{limit_field}")
    if (
        valid_choice(
            reason_code,
            {
                "retry_ceiling_reached",
                "gpu_hour_ceiling_reached",
                "cost_ceiling_reached",
                "no_new_hypothesis",
            },
        )
        and retry_decision != "stop"
    ):
        issues.append(f"{label} terminal ceiling/no-hypothesis reason must stop")
    if reason_code == "wall_time_ceiling_reached" and retry_decision not in {
        "stop",
        "resume",
    }:
        issues.append(f"{label} wall-time ceiling must stop or resume")
    no_retry_gate_types = {
        "auth_workspace": {"modal_auth"},
        "license_access": {"data"},
        "budget_protocol": {"compute", "protocol"},
    }
    required_types = (
        no_retry_gate_types.get(failure_class)
        if isinstance(failure_class, str)
        else None
    )
    if required_types is not None:
        if maximum is not None and maximum != 1:
            issues.append(f"{label} failure class forbids retries")
        if not any(
            gates[gate_id].get("type") in required_types for gate_id in gate_ids
        ):
            issues.append(f"{label} failure class needs a matching gate reference")
    if (
        failure_class == "transient_infrastructure"
        and maximum is not None
        and maximum > 4
    ):
        issues.append(f"{label} transient failures allow at most 3 retries")
    if retry_decision == "resume" and failure_class != "interrupted_full_run":
        issues.append(f"{label} resume requires failure_class interrupted_full_run")


def validate_targets(
    document: dict[str, Any],
    datasets: dict[str, Any],
    runs: dict[str, Any],
    artifacts: dict[str, Any],
    gates: dict[str, Any],
    issues: list[str],
) -> tuple[int, int, dict[str, dict[str, Any]]]:
    metrics = keyed(
        document.get("metric_definitions"),
        "metric_id",
        "metric_definitions",
        issues,
    )
    for metric_id, metric in metrics.items():
        for field in (
            "name",
            "direction",
            "implementation",
            "version",
            "aggregation",
            "unit_or_scale",
        ):
            if not nonempty(metric.get(field)):
                issues.append(f"metric {metric_id!r} needs {field}")
    experiments = keyed(
        document.get("experiments"), "experiment_id", "experiments", issues
    )
    for experiment_id, experiment in experiments.items():
        for field in ("system", "checkpoint_rule", "paper_evidence"):
            if not nonempty(experiment.get(field)):
                issues.append(f"experiment {experiment_id!r} needs {field}")
        if experiment.get("dataset_id") not in datasets:
            issues.append(f"experiment {experiment_id!r} references unknown dataset")
    targets = keyed(document.get("targets"), "target_id", "targets", issues)
    produced = 0
    for target_id, target in targets.items():
        label = f"target {target_id!r}"
        for field in ("paper_location", "split"):
            if not nonempty(target.get(field)):
                issues.append(f"{label} needs {field}")
        if target.get("experiment_id") not in experiments:
            issues.append(f"{label} references unknown experiment_id")
        if target.get("metric_id") not in metrics:
            issues.append(f"{label} references unknown metric_id")
        original = target.get("original_value")
        if not finite_number(original):
            issues.append(f"{label} original_value must be numeric")
        result = target.get("result")
        if not isinstance(result, dict) or not valid_choice(
            result.get("status"), TARGET_STATUSES
        ):
            issues.append(f"{label} needs a terminal result")
            continue
        if result["status"] == "produced":
            produced += 1
            reproduced = result.get("reproduced_value")
            difference = result.get("difference")
            if not finite_number(reproduced) or not finite_number(difference):
                issues.append(f"produced {label} needs numeric value and difference")
            elif finite_number(original) and not math.isclose(
                reproduced - original, difference, rel_tol=1e-9, abs_tol=1e-9
            ):
                issues.append(f"{label} difference is not reproduced - original")
            run_ids = checked_references(
                result.get("run_ids"),
                runs,
                f"produced {label}.run_ids",
                "run",
                issues,
                required=True,
            )
            artifact_ids = checked_references(
                result.get("artifact_ids"),
                artifacts,
                f"produced {label}.artifact_ids",
                "artifact",
                issues,
                required=True,
            )
            if result.get("reason") is not None:
                issues.append(f"produced {label} must not have a terminal reason")
            cited_run_artifacts = {
                artifact_id
                for run_id in run_ids
                for artifact_id in (
                    runs[run_id].get("artifact_ids", [])
                    if isinstance(runs[run_id].get("artifact_ids", []), list)
                    else []
                )
                if nonempty(artifact_id)
            }
            if not set(artifact_ids).issubset(cited_run_artifacts):
                issues.append(f"produced {label} artifacts must belong to a cited run")
        else:
            validate_not_produced_result(
                target_id, target, result, runs, artifacts, gates, issues
            )
    return produced, len(targets), targets


def validate_not_produced_result(
    target_id: str,
    target: dict[str, Any],
    result: dict[str, Any],
    runs: dict[str, Any],
    artifacts: dict[str, Any],
    gates: dict[str, Any],
    issues: list[str],
) -> None:
    label = f"target {target_id!r}"
    reason = result.get("reason")
    reason_code: str | None = None
    if not isinstance(reason, dict):
        issues.append(f"not-produced {label} needs reason")
    else:
        reason_code = reason.get("code")
        if not valid_choice(reason_code, TARGET_REASON_PIPELINES):
            issues.append(f"{label} has invalid terminal reason code")
        if not nonempty(reason.get("detail")):
            issues.append(f"{label} terminal reason needs detail")

    result_run_ids = checked_references(
        result.get("run_ids"),
        runs,
        f"{label}.run_ids",
        "run",
        issues,
    )
    checked_references(
        result.get("artifact_ids"),
        artifacts,
        f"{label}.artifact_ids",
        "artifact",
        issues,
    )
    result_gate_ids = checked_references(
        result.get("gate_ids"),
        gates,
        f"{label}.gate_ids",
        "gate",
        issues,
    )
    evidence = result.get("evidence")
    if (
        not isinstance(evidence, list)
        or not evidence
        or not all(nonempty(item) for item in evidence)
    ):
        issues.append(f"{label}.evidence must be an array of non-empty strings")

    gate_types = {
        gate_id: gates[gate_id].get("type")
        for gate_id in result_gate_ids
        if gate_id in gates
    }
    required_gate_types = {
        "data_unavailable": {"data"},
        "data_permission_blocked": {"data"},
        "compute_budget_blocked": {"compute"},
        "compute_infrastructure_blocked": {"compute"},
        "target_ambiguous": {"target"},
        "protocol_ambiguous": {"target", "protocol"},
        "metric_ambiguous": {"target", "protocol"},
        "human_evaluation_required": {"ethics", "protocol", "authority"},
    }
    allowed_gate_types = (
        required_gate_types.get(reason_code) if isinstance(reason_code, str) else None
    )
    if allowed_gate_types and not any(
        isinstance(gate_type, str) and gate_type in allowed_gate_types
        for gate_type in gate_types.values()
    ):
        issues.append(f"{label} reason {reason_code!r} needs a matching gate reference")
    if reason_code == "copied_baseline" and target.get("copied_baseline") is not True:
        issues.append(f"{label} copied_baseline reason requires copied_baseline true")
    if reason_code == "code_not_executable":
        cited_runs = [runs[run_id] for run_id in result_run_ids]
        if not cited_runs or all(
            not isinstance(run.get("terminal"), dict)
            or not valid_choice(
                run["terminal"].get("failure_class"),
                {"deterministic_code", "invalid_run"},
            )
            for run in cited_runs
        ):
            issues.append(
                f"{label} code_not_executable needs deterministic/invalid run evidence"
            )


def validate_gates(
    document: dict[str, Any], issues: list[str]
) -> tuple[dict[str, Any], bool]:
    gates = keyed(document.get("gates", []), "gate_id", "gates", issues)
    if not isinstance(document.get("gates", []), list):
        return {}, False
    open_gate = False
    for gate_id, gate in gates.items():
        label = f"gate {gate_id!r}"
        if not valid_choice(gate.get("type"), GATE_TYPES):
            issues.append(f"{label} has invalid type")
        if not valid_choice(gate.get("status"), {"open", "resolved"}):
            issues.append(f"{label} has invalid status")
        for field in ("reason", "required_action"):
            if not nonempty(gate.get(field)):
                issues.append(f"{label} needs {field}")
        evidence = gate.get("evidence")
        if (
            not isinstance(evidence, list)
            or not evidence
            or not all(nonempty(item) for item in evidence)
        ):
            issues.append(f"{label}.evidence must be an array of non-empty strings")
        if gate.get("type") == "target" and gate.get("status") == "open":
            alternatives = gate.get("alternatives")
            if not isinstance(alternatives, list) or not alternatives:
                issues.append(f"{label} needs unresolved alternatives")
            else:
                for index, alternative in enumerate(alternatives):
                    if not isinstance(alternative, dict) or any(
                        not nonempty(alternative.get(field))
                        for field in (
                            "alternative_id",
                            "description",
                            "paper_evidence",
                            "decision_needed",
                        )
                    ):
                        issues.append(f"{label} alternative[{index}] is incomplete")
        if gate.get("status") == "open":
            open_gate = True
    return gates, open_gate


def validate_readme(
    root: Path,
    pipeline: Any,
    preference: Any,
    issues: list[str],
) -> None:
    try:
        text = (root / "README.md").read_text(encoding="utf-8")
    except OSError as exc:
        issues.append(f"cannot read README.md: {exc}")
        return
    if re.search(r"<[^>]+>", text):
        issues.append("README.md still contains angle-bracket placeholders")
    if not re.search(
        rf"^\*\*Status:\*\*\s+`?{re.escape(str(pipeline))}`?\s*$",
        text,
        re.MULTILINE,
    ):
        issues.append("README.md status does not match reproduction.json")
    if not re.search(
        rf"^\*\*Preference level:\*\*\s+{re.escape(str(preference))}\s*$",
        text,
        re.MULTILINE,
    ):
        issues.append("README.md preference level does not match reproduction.json")


def validate_status_blocker(
    status: dict[str, Any],
    pipeline: Any,
    produced: int,
    target_count: int,
    targets: dict[str, dict[str, Any]],
    issues: list[str],
) -> None:
    if "blocker" not in status:
        issues.append("status needs blocker (an object or null)")
    blocker = status.get("blocker")
    if produced > 0:
        if blocker is not None:
            issues.append("complete/partial status must have null blocker")
        return
    if target_count == 0:
        return
    if not isinstance(blocker, dict):
        issues.append("zero-produced status needs blocker")
        return
    reason_code = blocker.get("reason_code")
    expected_pipeline = (
        TARGET_REASON_PIPELINES.get(reason_code)
        if isinstance(reason_code, str)
        else None
    )
    if expected_pipeline is None:
        issues.append("status.blocker needs a blocking reason_code")
    elif expected_pipeline != pipeline:
        issues.append("status.blocker reason_code does not match pipeline status")
    if not nonempty(blocker.get("detail")):
        issues.append("status.blocker needs detail")
    target_ids = checked_references(
        blocker.get("target_ids"),
        targets,
        "status.blocker.target_ids",
        "target",
        issues,
        required=True,
    )
    for target_id in target_ids:
        target = targets[target_id]
        result = target.get("result", {})
        reason = result.get("reason")
        target_reason = reason.get("code") if isinstance(reason, dict) else None
        if result.get("status") != "not_produced" or target_reason != reason_code:
            issues.append(
                f"status.blocker target {target_id!r} does not carry its reason_code"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paper_directory", type=Path)
    return parser.parse_args()


def main() -> int:
    root = parse_args().paper_directory.resolve()
    issues: list[str] = []
    try:
        document = json.loads((root / "reproduction.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(
            f"validation failed: cannot read reproduction.json: {exc}", file=sys.stderr
        )
        return 2
    if not isinstance(document, dict):
        issues.append("reproduction.json must contain an object")
        document = {}
    if "schema_version" in document:
        issues.append(
            "schema_version is no longer supported; migrate the record to the "
            "single current contract"
        )
    paper_id = document.get("paper_id")
    if not nonempty(paper_id):
        issues.append("reproduction.json needs paper_id")
        paper_id = ""
    validate_assignment(document, issues)
    if not isinstance(document.get("paper"), dict):
        issues.append("paper must be an object")
    sources = keyed(document.get("sources"), "source_id", "sources", issues)
    if not sources:
        issues.append("sources must contain at least one pinned artifact")
    datasets = validate_datasets(document, issues)
    artifacts = validate_artifacts(root, document, issues)
    gates, open_gate = validate_gates(document, issues)
    runs = validate_runs(document, artifacts, gates, issues)
    produced, target_count, targets = validate_targets(
        document,
        datasets,
        runs,
        artifacts,
        gates,
        issues,
    )
    status = document.get("status")
    if not isinstance(status, dict):
        issues.append("status must be an object")
        status = {}
    pipeline = status.get("pipeline")
    if not valid_choice(pipeline, PIPELINE_STATUSES):
        issues.append(f"invalid pipeline status {pipeline!r}")
    numerical_agreement = status.get("numerical_agreement")
    if not valid_choice(numerical_agreement, NUMERICAL_AGREEMENTS):
        issues.append(f"invalid numerical_agreement {numerical_agreement!r}")
    preference = status.get("preference_level")
    if isinstance(preference, bool) or preference not in (1, 2, 3):
        issues.append("preference_level must be 1, 2, or 3")
    if target_count and produced == target_count and pipeline != "complete":
        issues.append("all targets are produced, so pipeline must be complete")
    elif 0 < produced < target_count and pipeline != "partial":
        issues.append("some targets are produced, so pipeline must be partial")
    elif produced == 0 and valid_choice(pipeline, {"complete", "partial"}):
        issues.append("no targets are produced, so pipeline must identify the blocker")
    if target_count == 0:
        issues.append("reproduction must contain at least one target")
    if produced == 0 and target_count and numerical_agreement != "not_assessed":
        issues.append("zero-produced status must use not_assessed")
    validate_status_blocker(
        status,
        pipeline,
        produced,
        target_count,
        targets,
        issues,
    )
    if open_gate and pipeline == "complete":
        issues.append("a complete pipeline cannot retain an open gate")
    validate_readme(root, pipeline, preference, issues)
    if issues:
        print(f"validation failed with {len(issues)} issue(s):", file=sys.stderr)
        for issue in issues:
            print(f"- {issue}", file=sys.stderr)
        return 1
    print(f"validated reproduction: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
