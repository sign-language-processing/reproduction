#!/usr/bin/env python3
"""Validate a completed REPRO-SIGN reproduction's structural invariants."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


ALLOWED_STATUSES = {
    "reproduced",
    "partially_reproduced",
    "blocked_on_data",
    "blocked_on_compute",
    "blocked_on_code",
    "insufficient_information",
}
ALLOWED_GATE_TYPES = {
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
TERMINAL_TARGET_STATUSES = {"produced", "not_produced"}
EXTERNAL_REFERENCE_PREFIXES = ("https://", "http://", "hf://", "s3://", "modal://")
REQUIRED_REPORT_HEADINGS = {
    "Scope and target contract",
    "Source provenance",
    "Results",
    "How to repeat this",
    "Data provenance and permissions",
    "Environment and patches",
    "Execution evidence",
    "Guesses and deviations",
    "Attempts, failures, and dead ends",
    "Candidate flags, ethics, and human evaluation",
    "Author and team contact",
    "Terminal account",
}


def load_json(path: Path, issues: list[str]) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        issues.append(f"missing {path.name}")
    except (OSError, json.JSONDecodeError) as exc:
        issues.append(f"cannot read {path.name}: {exc}")
    return None


def nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def check_reference(root: Path, reference: Any, label: str, issues: list[str]) -> None:
    if not nonempty_string(reference):
        issues.append(f"{label} must be a non-empty path or immutable URL")
        return
    if reference.startswith(EXTERNAL_REFERENCE_PREFIXES):
        return
    if not (root / reference).exists():
        issues.append(f"{label} does not exist: {reference}")


def get_candidate_id(candidate: Any, issues: list[str]) -> str | None:
    if not isinstance(candidate, dict):
        issues.append("candidate.json must be an object")
        return None
    if candidate.get("schema_version") != 1:
        issues.append("candidate.json schema_version must be 1")

    normalized = candidate.get("normalized")
    candidate_id = (
        normalized.get("paper_id")
        if isinstance(normalized, dict)
        else candidate.get("paper_id")
    )
    if not nonempty_string(candidate_id):
        issues.append("candidate.json must contain normalized.paper_id or paper_id")
        return None

    record = candidate.get("record")
    if record is not None:
        if not isinstance(record, dict):
            issues.append("candidate.json record must be an object")
        else:
            if record.get("paper_id") != candidate_id or record.get("id") not in (
                None,
                candidate_id,
            ):
                issues.append("candidate record IDs do not match")
            if (
                record.get("confirmation") != "confirmed"
                or record.get("status") != "final"
            ):
                issues.append("candidate record is not final/confirmed")

    source = candidate.get("source")
    if not isinstance(source, dict) or not re.fullmatch(
        r"[0-9a-f]{64}", str(source.get("sha256", ""))
    ):
        issues.append("candidate source must contain a lowercase SHA-256")
    elif nonempty_string(source.get("path")):
        source_path = Path(source["path"])
        if source_path.exists():
            actual = hashlib.sha256(source_path.read_bytes()).hexdigest()
            if actual != source["sha256"]:
                issues.append("candidate source SHA-256 no longer matches source.path")
    return candidate_id


def validate_targets(
    targets_doc: Any, candidate_id: str | None, issues: list[str]
) -> dict[str, dict[str, Any]]:
    if not isinstance(targets_doc, dict):
        issues.append("targets.json must be an object")
        return {}
    if targets_doc.get("schema_version") != 1:
        issues.append("targets.json schema_version must be 1")
    if candidate_id and targets_doc.get("paper_id") != candidate_id:
        issues.append("targets.json paper_id does not match candidate.json")
    resolution_status = targets_doc.get("resolution_status")
    if resolution_status not in {"resolved", "human_gate"}:
        issues.append(
            "targets.json resolution_status must be 'resolved' or 'human_gate'"
        )
    alternatives = targets_doc.get("unresolved_alternatives")
    if not isinstance(alternatives, list):
        issues.append("targets.json unresolved_alternatives must be an array")
    elif resolution_status == "human_gate" and not alternatives:
        issues.append("human-gated target resolution needs unresolved_alternatives")

    items = targets_doc.get("targets")
    if not isinstance(items, list) or not items:
        issues.append("targets.json must contain at least one target")
        return {}

    targets: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(items):
        label = f"targets[{index}]"
        if not isinstance(item, dict):
            issues.append(f"{label} must be an object")
            continue
        target_id = item.get("target_id")
        if not nonempty_string(target_id):
            issues.append(f"{label}.target_id must be non-empty")
            continue
        if target_id in targets:
            issues.append(f"duplicate target_id {target_id!r}")
            continue
        targets[target_id] = item

        for field in ("paper_location", "system", "paper_evidence"):
            if not nonempty_string(item.get(field)):
                issues.append(f"target {target_id!r} is missing {field}")
        if not isinstance(item.get("dataset"), dict) or not nonempty_string(
            item["dataset"].get("name")
        ):
            issues.append(f"target {target_id!r} is missing dataset identity")
        if not isinstance(item.get("metric"), dict) or not nonempty_string(
            item["metric"].get("name")
        ):
            issues.append(f"target {target_id!r} is missing metric identity")
        if not isinstance(item.get("original_value"), (int, float)):
            issues.append(f"target {target_id!r} original_value must be numeric")
        if item.get("status") not in TERMINAL_TARGET_STATUSES:
            issues.append(
                f"target {target_id!r} has non-terminal status {item.get('status')!r}"
            )
        if item.get("status") == "not_produced" and not nonempty_string(
            item.get("terminal_reason")
        ):
            issues.append(f"target {target_id!r} needs a terminal_reason")
    return targets


def validate_gates(
    root: Path, candidate_id: str | None, issues: list[str]
) -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = []
    gate_directory = root / "evidence" / "gates"
    if not gate_directory.exists():
        return gates
    for path in sorted(gate_directory.glob("*.json")):
        gate = load_json(path, issues)
        if not isinstance(gate, dict):
            continue
        gates.append(gate)
        label = f"gate {path.name!r}"
        if gate.get("schema_version") != 1:
            issues.append(f"{label} schema_version must be 1")
        if candidate_id and gate.get("paper_id") != candidate_id:
            issues.append(f"{label} paper_id does not match candidate.json")
        if not nonempty_string(gate.get("gate_id")):
            issues.append(f"{label} needs gate_id")
        if gate.get("type") not in ALLOWED_GATE_TYPES:
            issues.append(f"{label} has invalid type {gate.get('type')!r}")
        if gate.get("status") not in {"open", "resolved"}:
            issues.append(f"{label} status must be 'open' or 'resolved'")
        for field in (
            "question",
            "why_agent_cannot_resolve",
            "requested_from",
            "requested_action",
        ):
            if not nonempty_string(gate.get(field)):
                issues.append(f"{label} needs {field}")
        if not isinstance(gate.get("evidence"), list) or not gate["evidence"]:
            issues.append(f"{label} needs evidence")
        if gate.get("status") == "resolved":
            for field in ("resolution", "resolved_at_utc", "resolved_by"):
                if not nonempty_string(gate.get(field)):
                    issues.append(f"resolved {label} needs {field}")
    return gates


def load_run_manifests(
    root: Path, run_paths: Any, issues: list[str]
) -> dict[str, dict[str, Any]]:
    if not isinstance(run_paths, list):
        issues.append("metrics.json runs must be an array")
        return {}
    manifests: dict[str, dict[str, Any]] = {}
    for index, relative in enumerate(run_paths):
        if not nonempty_string(relative):
            issues.append(f"metrics runs[{index}] must be a path")
            continue
        path = root / relative
        manifest = load_json(path, issues)
        if not isinstance(manifest, dict):
            continue
        run_id = manifest.get("run_id")
        if not nonempty_string(run_id):
            issues.append(f"{relative} has no run_id")
            continue
        if run_id in manifests:
            issues.append(f"duplicate run_id {run_id!r}")
            continue
        manifests[run_id] = manifest
        if manifest.get("schema_version") != 1:
            issues.append(f"run {run_id!r} schema_version must be 1")
        if not nonempty_string(manifest.get("command")):
            issues.append(f"run {run_id!r} has no exact command")
        if not isinstance(manifest.get("exit_code"), int):
            issues.append(f"run {run_id!r} exit_code must be an integer")
        compute = manifest.get("compute", {})
        if isinstance(compute, dict) and compute.get("platform") == "modal":
            if compute.get("modal_profile") != "repro-sign":
                issues.append(f"Modal run {run_id!r} did not use profile 'repro-sign'")
            shared_cache = compute.get("shared_cache")
            if not isinstance(shared_cache, dict):
                issues.append(f"Modal run {run_id!r} must record shared_cache")
            else:
                if shared_cache.get("modal_volume") != "huggingface-cache":
                    issues.append(
                        f"Modal run {run_id!r} must use Volume 'huggingface-cache'"
                    )
                if shared_cache.get("mount_path") != "/cache/huggingface":
                    issues.append(
                        f"Modal run {run_id!r} must mount the shared cache at '/cache/huggingface'"
                    )
                cache_environment = shared_cache.get("environment")
                if not isinstance(cache_environment, dict):
                    issues.append(
                        f"Modal run {run_id!r} must record Hugging Face cache environment"
                    )
                else:
                    expected_environment = {
                        "HF_HOME": "/cache/huggingface",
                        "HF_HUB_CACHE": "/cache/huggingface/hub",
                    }
                    for name, expected in expected_environment.items():
                        if cache_environment.get(name) != expected:
                            issues.append(
                                f"Modal run {run_id!r} must set {name}={expected}"
                            )
        data = manifest.get("data", [])
        if not isinstance(data, list):
            issues.append(f"run {run_id!r} data must be an array")
        else:
            for data_index, dataset in enumerate(data):
                if not isinstance(dataset, dict):
                    issues.append(
                        f"run {run_id!r} data[{data_index}] must be an object"
                    )
                    continue
                if dataset.get("modal_volume") != "datasets":
                    issues.append(
                        f"run {run_id!r} data[{data_index}] must use Modal Volume 'datasets'"
                    )
                modal_path = dataset.get("modal_path")
                if not nonempty_string(modal_path):
                    issues.append(f"run {run_id!r} data[{data_index}] needs modal_path")
                elif modal_path.startswith("/") or ".." in modal_path.split("/"):
                    issues.append(
                        f"run {run_id!r} data[{data_index}] modal_path must be safe and relative"
                    )
    return manifests


def validate_metrics(
    root: Path,
    metrics: Any,
    candidate_id: str | None,
    targets: dict[str, dict[str, Any]],
    issues: list[str],
) -> tuple[str | None, int | None]:
    if not isinstance(metrics, dict):
        issues.append("metrics.json must be an object")
        return None, None
    if metrics.get("schema_version") != 1:
        issues.append("metrics.json schema_version must be 1")
    paper = metrics.get("paper")
    if not isinstance(paper, dict) or (
        candidate_id and paper.get("paper_id") != candidate_id
    ):
        issues.append("metrics.json paper.paper_id does not match candidate.json")

    overall_status = metrics.get("reproducibility_status")
    if overall_status not in ALLOWED_STATUSES:
        issues.append(f"invalid reproducibility_status {overall_status!r}")
    preference_level = metrics.get("preference_level")
    if preference_level not in (1, 2, 3):
        issues.append("metrics.json preference_level must be 1, 2, or 3")

    datasets = metrics.get("datasets")
    if not isinstance(datasets, list):
        issues.append("metrics.json datasets must be an array")
    else:
        for index, dataset in enumerate(datasets):
            if not isinstance(dataset, dict):
                issues.append(f"metrics datasets[{index}] must be an object")
                continue
            if dataset.get("modal_volume") != "datasets":
                issues.append(
                    f"metrics datasets[{index}] must use Modal Volume 'datasets'"
                )
            modal_path = dataset.get("modal_path")
            if not nonempty_string(modal_path):
                issues.append(f"metrics datasets[{index}] needs modal_path")
            elif modal_path.startswith("/") or ".." in modal_path.split("/"):
                issues.append(
                    f"metrics datasets[{index}] modal_path must be safe and relative"
                )

    manifests = load_run_manifests(root, metrics.get("runs"), issues)
    scores = metrics.get("scores")
    if not isinstance(scores, list):
        issues.append("metrics.json scores must be an array")
        return overall_status, preference_level

    by_target: dict[str, dict[str, Any]] = {}
    for index, score in enumerate(scores):
        if not isinstance(score, dict):
            issues.append(f"scores[{index}] must be an object")
            continue
        target_id = score.get("target_id")
        if not nonempty_string(target_id):
            issues.append(f"scores[{index}].target_id must be non-empty")
            continue
        if target_id in by_target:
            issues.append(f"duplicate score for target {target_id!r}")
            continue
        by_target[target_id] = score

    if set(by_target) != set(targets):
        issues.append(
            "metrics.json must contain exactly one score for every targets.json target_id"
        )

    for target_id, target in targets.items():
        score = by_target.get(target_id)
        if not score:
            continue
        if score.get("status") != target.get("status"):
            issues.append(
                f"target {target_id!r} status differs between targets.json and metrics.json"
            )
        if score.get("status") == "produced":
            if not isinstance(score.get("reproduced"), (int, float)):
                issues.append(
                    f"produced target {target_id!r} needs a numeric reproduced value"
                )
            run_ids = score.get("run_ids")
            if not isinstance(run_ids, list) or not run_ids:
                issues.append(
                    f"produced target {target_id!r} needs at least one run_id"
                )
            else:
                for run_id in run_ids:
                    if run_id not in manifests:
                        issues.append(
                            f"target {target_id!r} references unknown run_id {run_id!r}"
                        )
            artifacts = score.get("raw_metric_artifacts")
            if not isinstance(artifacts, list) or not artifacts:
                issues.append(
                    f"produced target {target_id!r} needs raw_metric_artifacts"
                )
            else:
                for artifact in artifacts:
                    check_reference(
                        root,
                        artifact,
                        f"target {target_id!r} raw metric artifact",
                        issues,
                    )
            original = score.get("original")
            reproduced = score.get("reproduced")
            difference = score.get("difference")
            if all(
                isinstance(value, (int, float))
                for value in (original, reproduced, difference)
            ):
                if not math.isclose(
                    reproduced - original, difference, rel_tol=1e-9, abs_tol=1e-9
                ):
                    issues.append(
                        f"target {target_id!r} difference is not reproduced - original"
                    )
        elif not nonempty_string(score.get("terminal_reason")):
            issues.append(
                f"not-produced target {target_id!r} needs a terminal_reason in metrics.json"
            )

    produced_count = sum(
        target.get("status") == "produced" for target in targets.values()
    )
    if targets and produced_count == len(targets) and overall_status != "reproduced":
        issues.append("all targets are produced, so status must be 'reproduced'")
    if 0 < produced_count < len(targets) and overall_status != "partially_reproduced":
        issues.append(
            "some targets are produced, so status must be 'partially_reproduced'"
        )
    if produced_count == 0 and overall_status in {"reproduced", "partially_reproduced"}:
        issues.append("no targets are produced, so status must identify the blocker")
    return overall_status, preference_level


def validate_report(
    path: Path, status: str | None, preference: int | None, issues: list[str]
) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        issues.append("missing report.md")
        return
    except OSError as exc:
        issues.append(f"cannot read report.md: {exc}")
        return

    if re.search(r"<[^>]+>", text):
        issues.append("report.md still contains angle-bracket placeholders")
    headings = set(re.findall(r"^## (.+?)\s*$", text, flags=re.MULTILINE))
    for heading in sorted(REQUIRED_REPORT_HEADINGS - headings):
        issues.append(f"report.md is missing heading: {heading}")
    if status and status not in text:
        issues.append(
            "report.md does not contain the metrics.json reproducibility status"
        )
    if preference is not None and not re.search(
        rf"Preference level:\*\*\s*`?{preference}`?(?:\s|$)", text
    ):
        issues.append("report.md does not contain the metrics.json preference level")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paper_directory", type=Path)
    return parser.parse_args()


def main() -> int:
    root = parse_args().paper_directory.resolve()
    issues: list[str] = []
    if not root.is_dir():
        print(f"validation failed: not a directory: {root}", file=sys.stderr)
        return 2

    candidate = load_json(root / "candidate.json", issues)
    targets_doc = load_json(root / "targets.json", issues)
    metrics = load_json(root / "metrics.json", issues)

    candidate_id = get_candidate_id(candidate, issues)
    targets = validate_targets(targets_doc, candidate_id, issues)
    gates = validate_gates(root, candidate_id, issues)
    if (
        isinstance(targets_doc, dict)
        and targets_doc.get("resolution_status") == "human_gate"
    ):
        if not any(
            gate.get("type") == "target" and gate.get("status") == "open"
            for gate in gates
        ):
            issues.append(
                "human-gated target resolution needs an open target gate artifact"
            )
    status, preference = validate_metrics(root, metrics, candidate_id, targets, issues)
    if status == "reproduced" and any(gate.get("status") == "open" for gate in gates):
        issues.append("a reproduced attempt cannot retain open gate artifacts")
    validate_report(root / "report.md", status, preference, issues)

    if issues:
        print(f"validation failed with {len(issues)} issue(s):", file=sys.stderr)
        for issue in issues:
            print(f"- {issue}", file=sys.stderr)
        return 1

    print(f"validated reproduction: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
