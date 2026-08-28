"""End-to-end tests for the reproduction record validator."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

VALIDATOR = Path(__file__).with_name("validate_reproduction.py")
INGESTER = Path(__file__).with_name("ingest_candidate.py")
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
SHA = "0" * 64
HISTORICAL_PAPER_ID = "camgoz-2020-slt"
HISTORICAL_SHA = "9fe456f04eae036309ce538cd0e650299002390772d2d4daddcc815bb29bc15f"


def valid_document() -> dict:
    return {
        "paper_id": "test-paper",
        "assignment": {
            "kind": "direct_user_request",
            "source": {"sha256": SHA},
        },
        "paper": {"title": "Test paper"},
        "sources": [{"source_id": "paper"}],
        "status": {
            "pipeline": "complete",
            "numerical_agreement": "not_fully_reproduced",
            "preference_level": 1,
            "blocker": None,
        },
        "datasets": [
            {
                "dataset_id": "data",
                "name": "Dataset",
                "version_or_subset": "v1",
                "license_or_permission": "Test fixture",
                "cloud_processing_allowed": True,
                "modal_volume": "datasets",
                "modal_path": "test-data",
                "splits": [
                    {
                        "name": "test",
                        "sample_count": 1,
                        "files": [{"path": "manifest.json", "sha256": SHA}],
                    }
                ],
            }
        ],
        "metric_definitions": [
            {
                "metric_id": "accuracy",
                "name": "Accuracy",
                "direction": "higher",
                "implementation": "fixture",
                "version": "1",
                "aggregation": "mean",
                "unit_or_scale": "fraction",
            }
        ],
        "experiments": [
            {
                "experiment_id": "experiment",
                "system": "Fixture",
                "checkpoint_rule": "Only checkpoint",
                "paper_evidence": "Table 1",
                "dataset_id": "data",
            }
        ],
        "targets": [
            {
                "target_id": "accuracy",
                "paper_location": "Table 1",
                "split": "test",
                "experiment_id": "experiment",
                "metric_id": "accuracy",
                "original_value": 0.5,
                "result": {
                    "status": "produced",
                    "reproduced_value": 0.6,
                    "difference": 0.1,
                    "run_ids": ["run-1"],
                    "artifact_ids": ["metrics"],
                },
            }
        ],
        "runs": [
            {
                "run_id": "run-1",
                "command": "python evaluate.py",
                "exit_code": 0,
                "started_at_utc": "2026-08-28T10:00:00Z",
                "finished_at_utc": "2026-08-28T10:01:00Z",
                "artifact_ids": ["metrics"],
                "recording": {
                    "mode": "contemporaneous",
                    "source_record_sha256": None,
                    "unknown_fields": [],
                },
                "attempt": {
                    "group_id": "evaluation",
                    "number": 1,
                    "max_attempts": 1,
                },
                "stop_policy": {
                    "declared_at_utc": "2026-08-28T09:55:00Z",
                    "max_wall_time_seconds": 60,
                    "max_gpu_hours": None,
                    "max_cost_chf": None,
                },
                "terminal": {
                    "state": "succeeded",
                    "reason_code": "completed",
                    "detail": "Evaluation completed.",
                    "failure_class": None,
                    "retry_decision": "not_needed",
                    "gate_ids": [],
                },
            }
        ],
        "artifacts": [
            {
                "artifact_id": "metrics",
                "kind": "metrics",
                "uri": "metrics.json",
                "sha256": "replaced-by-test",
            }
        ],
        "gates": [],
    }


class ValidatorTest(unittest.TestCase):
    def validate(self, document: dict) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metric_bytes = b"{}\n"
            (root / "metrics.json").write_bytes(metric_bytes)
            document = copy.deepcopy(document)
            document["artifacts"][0]["sha256"] = hashlib.sha256(
                metric_bytes
            ).hexdigest()
            (root / "reproduction.json").write_text(
                json.dumps(document), encoding="utf-8"
            )
            (root / "README.md").write_text(
                "**Preference level:** 1\n\n**Status:** `"
                + document["status"]["pipeline"]
                + "`\n",
                encoding="utf-8",
            )
            return subprocess.run(
                ["python3", str(VALIDATOR), str(root)],
                check=False,
                capture_output=True,
                text=True,
            )

    def test_valid_record(self) -> None:
        result = self.validate(valid_document())
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_valid_resume_chain(self) -> None:
        document = valid_document()
        resumed = document["runs"][0]
        resumed["attempt"] = {
            "group_id": "evaluation",
            "number": 2,
            "max_attempts": 2,
        }
        interrupted = copy.deepcopy(resumed)
        interrupted.update(
            {
                "run_id": "run-0",
                "exit_code": 124,
                "artifact_ids": [],
                "attempt": {
                    "group_id": "evaluation",
                    "number": 1,
                    "max_attempts": 2,
                },
                "terminal": {
                    "state": "interrupted",
                    "reason_code": "external_interruption",
                    "detail": "The remote worker timed out after saving a checkpoint.",
                    "failure_class": "interrupted_full_run",
                    "retry_decision": "resume",
                    "next_run_id": "run-1",
                    "gate_ids": [],
                },
            }
        )
        document["runs"].insert(0, interrupted)
        result = self.validate(document)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_valid_blocked_target(self) -> None:
        document = valid_document()
        document["status"] = {
            "pipeline": "insufficient_information",
            "numerical_agreement": "not_assessed",
            "preference_level": 1,
            "blocker": {
                "reason_code": "target_ambiguous",
                "detail": "The paper does not identify the evaluation split.",
                "target_ids": ["accuracy"],
            },
        }
        document["gates"] = [
            {
                "gate_id": "evaluation-split",
                "type": "target",
                "status": "open",
                "reason": "Two materially different evaluation splits remain.",
                "required_action": "Select the authoritative split.",
                "evidence": ["The paper reports only aggregate counts."],
                "alternatives": [
                    {
                        "alternative_id": "split-a",
                        "description": "Use the published count as a random split.",
                        "paper_evidence": "Aggregate count in Table 1.",
                        "decision_needed": "Confirm whether the split was random.",
                    }
                ],
            }
        ]
        document["targets"][0]["result"] = {
            "status": "not_produced",
            "reason": {
                "code": "target_ambiguous",
                "detail": "The exact evaluation split is not published.",
            },
            "run_ids": [],
            "artifact_ids": [],
            "gate_ids": ["evaluation-split"],
            "evidence": ["Table 1 gives counts but no split identifiers."],
        }
        result = self.validate(document)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_run_requires_stop_policy(self) -> None:
        document = valid_document()
        del document["runs"][0]["stop_policy"]
        result = self.validate(document)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("needs stop_policy", result.stderr)

    def test_stop_policy_must_precede_run(self) -> None:
        document = valid_document()
        document["runs"][0]["stop_policy"]["declared_at_utc"] = "2026-08-28T10:01:00Z"
        result = self.validate(document)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("declared after the run started", result.stderr)

    def test_modal_gpu_run_requires_gpu_and_cost_ceilings(self) -> None:
        document = valid_document()
        document["runs"][0]["compute"] = {
            "platform": "modal",
            "modal_profile": "repro-sign",
            "gpu_count": 1,
            "shared_cache": {
                "modal_volume": "huggingface-cache",
                "mount_path": "/cache/huggingface",
                "environment": {
                    "HF_HOME": "/cache/huggingface",
                    "HF_HUB_CACHE": "/cache/huggingface/hub",
                },
            },
        }
        result = self.validate(document)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("needs max_gpu_hours", result.stderr)

    def test_not_produced_requires_coded_reason_and_evidence(self) -> None:
        document = valid_document()
        document["status"] = {
            "pipeline": "insufficient_information",
            "numerical_agreement": "not_assessed",
            "preference_level": 1,
            "blocker": {
                "reason_code": "target_ambiguous",
                "detail": "The target cannot be resolved.",
                "target_ids": ["accuracy"],
            },
        }
        document["targets"][0]["result"] = {
            "status": "not_produced",
            "reason": {
                "code": "invented_reason",
                "detail": "This code is not controlled.",
            },
            "run_ids": [],
            "artifact_ids": [],
            "gate_ids": [],
            "evidence": [],
        }
        result = self.validate(document)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid terminal reason code", result.stderr)
        self.assertIn("evidence must be an array", result.stderr)

    def test_retry_cannot_exceed_declared_maximum(self) -> None:
        document = valid_document()
        run = document["runs"][0]
        run["exit_code"] = 1
        run["terminal"] = {
            "state": "failed",
            "reason_code": "command_failed",
            "detail": "A transient request failed.",
            "failure_class": "transient_infrastructure",
            "retry_decision": "retry",
            "next_run_id": "run-1",
            "gate_ids": [],
        }
        result = self.validate(document)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cannot retry/resume after max_attempts", result.stderr)

    def test_attempt_group_must_be_contiguous_from_one(self) -> None:
        document = valid_document()
        document["runs"][0]["attempt"] = {
            "group_id": "evaluation",
            "number": 2,
            "max_attempts": 2,
        }
        result = self.validate(document)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be contiguous from attempt 1", result.stderr)

    def test_later_attempt_must_be_linked_from_preceding_attempt(self) -> None:
        document = valid_document()
        first = document["runs"][0]
        first["attempt"]["max_attempts"] = 2
        second = copy.deepcopy(first)
        second["run_id"] = "run-2"
        second["attempt"]["number"] = 2
        document["runs"].append(second)
        result = self.validate(document)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("is not linked from its preceding attempt", result.stderr)

    def test_non_success_cannot_use_not_needed(self) -> None:
        document = valid_document()
        run = document["runs"][0]
        run["exit_code"] = 1
        run["terminal"] = {
            "state": "failed",
            "reason_code": "command_failed",
            "detail": "The command failed deterministically.",
            "failure_class": "deterministic_code",
            "retry_decision": "not_needed",
            "gate_ids": [],
        }
        result = self.validate(document)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cannot use retry_decision not_needed", result.stderr)

    def test_malformed_references_report_errors_without_traceback(self) -> None:
        document = valid_document()
        document["runs"][0]["artifact_ids"] = 7
        document["targets"][0]["result"]["run_ids"] = 7
        result = self.validate(document)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be an array", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_no_retry_failure_requires_matching_gate(self) -> None:
        document = valid_document()
        run = document["runs"][0]
        run["exit_code"] = 1
        run["terminal"] = {
            "state": "failed",
            "reason_code": "command_failed",
            "detail": "Modal authentication failed.",
            "failure_class": "auth_workspace",
            "retry_decision": "stop",
            "gate_ids": [],
        }
        result = self.validate(document)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("needs a matching gate reference", result.stderr)

    def test_transient_failure_cannot_prove_code_is_not_executable(self) -> None:
        document = valid_document()
        document["status"] = {
            "pipeline": "blocked_on_code",
            "numerical_agreement": "not_assessed",
            "preference_level": 1,
            "blocker": {
                "reason_code": "code_not_executable",
                "detail": "No executable path was claimed.",
                "target_ids": ["accuracy"],
            },
        }
        document["runs"][0]["exit_code"] = 124
        document["runs"][0]["terminal"] = {
            "state": "interrupted",
            "reason_code": "external_interruption",
            "detail": "A transient network interruption ended the run.",
            "failure_class": "transient_infrastructure",
            "retry_decision": "stop",
            "gate_ids": [],
        }
        document["targets"][0]["result"] = {
            "status": "not_produced",
            "reason": {
                "code": "code_not_executable",
                "detail": "Only a transient interruption was observed.",
            },
            "run_ids": ["run-1"],
            "artifact_ids": [],
            "gate_ids": [],
            "evidence": ["The sole run ended during a network interruption."],
        }
        result = self.validate(document)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("needs deterministic/invalid run evidence", result.stderr)

    def test_gate_evidence_cannot_be_empty(self) -> None:
        document = valid_document()
        document["gates"] = [
            {
                "gate_id": "data",
                "type": "data",
                "status": "resolved",
                "reason": "Data identity was reviewed.",
                "required_action": "None.",
                "evidence": [],
            }
        ]
        result = self.validate(document)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("gate 'data'.evidence", result.stderr)

    def test_retrospective_unknown_fields_are_valid(self) -> None:
        document = valid_document()
        document["paper_id"] = HISTORICAL_PAPER_ID
        run = document["runs"][0]
        run["started_at_utc"] = None
        run["finished_at_utc"] = None
        run["attempt"]["max_attempts"] = None
        run["recording"] = {
            "mode": "retrospective",
            "source_record_sha256": HISTORICAL_SHA,
            "unknown_fields": [
                "started_at_utc",
                "finished_at_utc",
                "attempt.max_attempts",
                "stop_policy.declared_at_utc",
                "stop_policy.max_wall_time_seconds",
            ],
            "detail": "Backfilled from an already-committed report.",
        }
        run["stop_policy"] = {
            "declared_at_utc": None,
            "max_wall_time_seconds": None,
            "max_gpu_hours": None,
            "max_cost_chf": None,
        }
        result = self.validate(document)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_retrospective_record_requires_an_explanation(self) -> None:
        document = valid_document()
        document["paper_id"] = HISTORICAL_PAPER_ID
        recording = document["runs"][0]["recording"]
        recording["mode"] = "retrospective"
        recording["source_record_sha256"] = HISTORICAL_SHA
        recording["detail"] = None
        result = self.validate(document)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("recording needs detail", result.stderr)

    def test_unregistered_record_cannot_claim_retrospective_mode(self) -> None:
        document = valid_document()
        recording = document["runs"][0]["recording"]
        recording.update(
            {
                "mode": "retrospective",
                "source_record_sha256": SHA,
                "detail": "This source is not in the migration registry.",
            }
        )
        result = self.validate(document)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("is not registered as a historical migration", result.stderr)

    def test_contemporaneous_record_cannot_hide_unknowns(self) -> None:
        document = valid_document()
        run = document["runs"][0]
        run["started_at_utc"] = None
        run["recording"]["unknown_fields"] = ["started_at_utc"]
        result = self.validate(document)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cannot contain unknown fields", result.stderr)

    def test_unknown_field_must_be_allowed_and_null(self) -> None:
        document = valid_document()
        document["paper_id"] = HISTORICAL_PAPER_ID
        run = document["runs"][0]
        run["recording"] = {
            "mode": "retrospective",
            "source_record_sha256": HISTORICAL_SHA,
            "unknown_fields": ["terminal.state", "started_at_utc"],
            "detail": "Historical backfill.",
        }
        result = self.validate(document)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("is not allowed", result.stderr)
        self.assertIn("must exist and be null", result.stderr)

    def test_schema_version_is_rejected(self) -> None:
        document = valid_document()
        document["schema_version"] = "retired"
        result = self.validate(document)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("schema_version is no longer supported", result.stderr)

    def test_all_committed_paper_records_use_current_contract(self) -> None:
        paper_directories = sorted((REPOSITORY_ROOT / "papers").glob("*"))
        self.assertTrue(paper_directories)
        for paper_directory in paper_directories:
            with self.subTest(paper=paper_directory.name):
                result = subprocess.run(
                    ["python3", str(VALIDATOR), str(paper_directory)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_candidate_ingestion_creates_current_record_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "candidates.json"
            output = root / "reproduction.json"
            source.write_text(
                json.dumps(
                    [
                        {
                            "paper_id": "paper-1",
                            "confirmation": "confirmed",
                            "status": "final",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            result = subprocess.run(
                ["python3", str(INGESTER), str(source), "paper-1", str(output)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            written = json.loads(output.read_text())
            self.assertEqual(written["paper_id"], "paper-1")
            self.assertEqual(written["assignment"]["kind"], "queue_record")

    def test_candidate_ingestion_preserves_existing_sections(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "candidates.json"
            output = root / "reproduction.json"
            source.write_text(
                json.dumps(
                    [
                        {
                            "paper_id": "paper-1",
                            "confirmation": "confirmed",
                            "status": "final",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            output.write_text(
                json.dumps({"paper_id": "paper-1", "paper": {"title": "Preserved"}}),
                encoding="utf-8",
            )
            result = subprocess.run(
                ["python3", str(INGESTER), str(source), "paper-1", str(output)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                json.loads(output.read_text())["paper"]["title"], "Preserved"
            )

    def test_candidate_ingestion_rejects_retired_version_field(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "candidates.json"
            output = root / "reproduction.json"
            source.write_text(
                json.dumps(
                    [
                        {
                            "paper_id": "paper-1",
                            "confirmation": "confirmed",
                            "status": "final",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            output.write_text(
                json.dumps({"schema_version": "retired", "paper_id": "paper-1"}),
                encoding="utf-8",
            )
            result = subprocess.run(
                ["python3", str(INGESTER), str(source), "paper-1", str(output)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("retired version field", result.stderr)


if __name__ == "__main__":
    unittest.main()
