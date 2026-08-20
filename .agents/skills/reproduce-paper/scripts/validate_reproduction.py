#!/usr/bin/env python3
"""Validate one normalized REPRO-SIGN paper reproduction."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
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


def nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


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


def validate_assignment(document: dict[str, Any], issues: list[str]) -> None:
    assignment = document.get("assignment")
    if not isinstance(assignment, dict):
        issues.append("assignment must be an object")
        return
    kind = assignment.get("kind")
    if kind not in {"queue_record", "direct_user_request"}:
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
    document: dict[str, Any], artifacts: dict[str, Any], issues: list[str]
) -> dict[str, Any]:
    runs = keyed(document.get("runs"), "run_id", "runs", issues)
    for run_id, run in runs.items():
        label = f"run {run_id!r}"
        if not nonempty(run.get("command")):
            issues.append(f"{label} needs an exact command")
        if not isinstance(run.get("exit_code"), int):
            issues.append(f"{label} exit_code must be an integer")
        for artifact_id in run.get("artifact_ids", []):
            if artifact_id not in artifacts:
                issues.append(f"{label} references unknown artifact {artifact_id!r}")
        compute = run.get("compute")
        if isinstance(compute, dict) and compute.get("platform") == "modal":
            if compute.get("modal_profile") != "repro-sign":
                issues.append(f"Modal {label} must use profile 'repro-sign'")
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
    return runs


def validate_targets(
    document: dict[str, Any],
    datasets: dict[str, Any],
    runs: dict[str, Any],
    artifacts: dict[str, Any],
    issues: list[str],
) -> tuple[int, int]:
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
        if not isinstance(original, (int, float)):
            issues.append(f"{label} original_value must be numeric")
        result = target.get("result")
        if not isinstance(result, dict) or result.get("status") not in TARGET_STATUSES:
            issues.append(f"{label} needs a terminal result")
            continue
        if result["status"] == "produced":
            produced += 1
            reproduced = result.get("reproduced_value")
            difference = result.get("difference")
            if not isinstance(reproduced, (int, float)) or not isinstance(
                difference, (int, float)
            ):
                issues.append(f"produced {label} needs numeric value and difference")
            elif isinstance(original, (int, float)) and not math.isclose(
                reproduced - original, difference, rel_tol=1e-9, abs_tol=1e-9
            ):
                issues.append(f"{label} difference is not reproduced - original")
            if not result.get("run_ids"):
                issues.append(f"produced {label} needs run_ids")
            for run_id in result.get("run_ids", []):
                if run_id not in runs:
                    issues.append(f"{label} references unknown run {run_id!r}")
            if not result.get("artifact_ids"):
                issues.append(f"produced {label} needs artifact_ids")
            for artifact_id in result.get("artifact_ids", []):
                if artifact_id not in artifacts:
                    issues.append(
                        f"{label} references unknown artifact {artifact_id!r}"
                    )
        elif not nonempty(result.get("terminal_reason")):
            issues.append(f"not-produced {label} needs terminal_reason")
    return produced, len(targets)


def validate_target_resolution(document: dict[str, Any], issues: list[str]) -> str:
    resolution = document.get("target_resolution")
    if not isinstance(resolution, dict):
        issues.append("target_resolution must be an object")
        return ""
    status = resolution.get("status")
    if status not in {"resolved", "human_gate"}:
        issues.append("target_resolution.status must be resolved or human_gate")
    if not nonempty(resolution.get("assignment_scope")):
        issues.append("target_resolution needs assignment_scope")
    alternatives = resolution.get("unresolved_alternatives")
    if not isinstance(alternatives, list):
        issues.append("target_resolution.unresolved_alternatives must be an array")
        return status
    if status == "resolved" and alternatives:
        issues.append("resolved target_resolution cannot retain alternatives")
    if status == "human_gate" and not alternatives:
        issues.append("human_gate target_resolution needs alternatives")
    for index, alternative in enumerate(alternatives):
        label = f"target alternative[{index}]"
        if not isinstance(alternative, dict):
            issues.append(f"{label} must be an object")
            continue
        for field in (
            "alternative_id",
            "description",
            "paper_evidence",
            "decision_needed",
        ):
            if not nonempty(alternative.get(field)):
                issues.append(f"{label} needs {field}")
    return status


def validate_gates(
    document: dict[str, Any], issues: list[str]
) -> tuple[bool, set[str]]:
    gates = keyed(document.get("gates", []), "gate_id", "gates", issues)
    if not isinstance(document.get("gates", []), list):
        return False, set()
    open_gate = False
    open_types: set[str] = set()
    for gate_id, gate in gates.items():
        label = f"gate {gate_id!r}"
        if gate.get("type") not in GATE_TYPES:
            issues.append(f"{label} has invalid type")
        if gate.get("status") not in {"open", "resolved"}:
            issues.append(f"{label} has invalid status")
        for field in ("reason", "required_action"):
            if not nonempty(gate.get(field)):
                issues.append(f"{label} needs {field}")
        evidence = gate.get("evidence")
        if not isinstance(evidence, list) or not all(
            nonempty(item) for item in evidence
        ):
            issues.append(f"{label}.evidence must be an array of non-empty strings")
        if gate.get("status") == "open":
            open_gate = True
            open_types.add(gate.get("type"))
    return open_gate, open_types


def validate_readme(
    root: Path, paper_id: str, pipeline: Any, preference: Any, issues: list[str]
) -> None:
    try:
        text = (root / "README.md").read_text(encoding="utf-8")
    except OSError as exc:
        issues.append(f"cannot read README.md: {exc}")
        return
    if re.search(r"<[^>]+>", text):
        issues.append("README.md still contains angle-bracket placeholders")
    for value, label in ((paper_id, "paper_id"), (pipeline, "pipeline status")):
        if not nonempty(value) or value not in text:
            issues.append(f"README.md does not contain {label}")
    if preference not in (1, 2, 3) or not re.search(
        rf"Preference level:\*\*\s*`?{preference}`?(?:\s|$)", text
    ):
        issues.append("README.md does not contain preference level")
    headings = set(re.findall(r"^## (.+?)\s*$", text, flags=re.MULTILINE))
    if "Results" not in headings:
        issues.append("README.md is missing Results")
    if not ({"How to reproduce", "How to repeat this"} & headings):
        issues.append("README.md is missing reproduction commands")


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
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        issues.append("reproduction.json must be a schema_version 1 object")
        document = document if isinstance(document, dict) else {}
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
    target_resolution = validate_target_resolution(document, issues)
    datasets = validate_datasets(document, issues)
    artifacts = validate_artifacts(root, document, issues)
    runs = validate_runs(document, artifacts, issues)
    produced, target_count = validate_targets(
        document, datasets, runs, artifacts, issues
    )
    status = document.get("status")
    if not isinstance(status, dict):
        issues.append("status must be an object")
        status = {}
    pipeline = status.get("pipeline")
    if pipeline not in PIPELINE_STATUSES:
        issues.append(f"invalid pipeline status {pipeline!r}")
    numerical_agreement = status.get("numerical_agreement")
    if numerical_agreement not in NUMERICAL_AGREEMENTS:
        issues.append(f"invalid numerical_agreement {numerical_agreement!r}")
    preference = status.get("preference_level")
    if preference not in (1, 2, 3):
        issues.append("preference_level must be 1, 2, or 3")
    if target_count and produced == target_count and pipeline != "complete":
        issues.append("all targets are produced, so pipeline must be complete")
    elif 0 < produced < target_count and pipeline != "partial":
        issues.append("some targets are produced, so pipeline must be partial")
    elif produced == 0 and pipeline in {"complete", "partial"}:
        issues.append("no targets are produced, so pipeline must identify the blocker")
    open_gate, open_gate_types = validate_gates(document, issues)
    if target_resolution == "human_gate" and "target" not in open_gate_types:
        issues.append("human_gate target_resolution needs an open target gate")
    if open_gate and pipeline == "complete":
        issues.append("a complete pipeline cannot retain an open gate")
    validate_readme(root, paper_id, pipeline, preference, issues)
    if issues:
        print(f"validation failed with {len(issues)} issue(s):", file=sys.stderr)
        for issue in issues:
            print(f"- {issue}", file=sys.stderr)
        return 1
    print(f"validated reproduction: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
