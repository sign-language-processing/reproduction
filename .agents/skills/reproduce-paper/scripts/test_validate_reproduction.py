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


APP_ID = "ap-" + "A" * 22
OTHER_APP_ID = "ap-" + "b1" * 11


def valid_document() -> dict:
    return {
        "paper_id": "test-paper",
        "agents": [
            {
                "agent_id": "executor",
                "model": "Test model 1.0",
                "harness": "Test application",
                "scope": "Implementation and execution of the fixture experiment.",
                "evidence": ["Fixture session explicitly records both identities."],
            }
        ],
        "assignment": {
            "kind": "direct_user_request",
            "source": {"sha256": SHA},
        },
        "paper": {"title": "Test paper"},
        "sources": [{"source_id": "paper"}],
        "status": {
            "pipeline": "complete",
            "numerical_agreement": "does_not_agree",
            "numerical_agreement_basis": "The fixture values differ by 0.1.",
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
                "in_scope": True,
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
                "agent_ids": ["executor"],
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
        "guesses_and_deviations": [],
        "noncompliance": [],
    }


def readme_section(heading: str, entries: list) -> str:
    rows = "".join(
        f"| `{entry.get('id')}` |\n" for entry in entries if isinstance(entry, dict)
    )
    return f"## {heading}\n\n" + (rows or "None.\n") + "\n"


def readme_text(document: dict) -> str:
    return (
        "**Paper ID:** `"
        + str(document["paper_id"])
        + "`\n\n**Preference level:** 1\n\n**Pipeline status:** `"
        + str(document["status"]["pipeline"])
        + "`\n\n**Numerical agreement:** `"
        + str(document["status"]["numerical_agreement"])
        + "`\n\n"
        + readme_section(
            "Guesses and deviations", document.get("guesses_and_deviations") or []
        )
        + readme_section("Noncompliance", document.get("noncompliance") or [])
    )


def tracker_export(**paper: object) -> dict:
    """A Reproduction Tracker export: one reproductions record, paper expanded."""
    fields = {
        "id": "paper-record",
        "paper_id": "paper-1",
        "status": "final",
        "title": "A paper",
        **paper,
    }
    return {
        "collectionName": "reproductions",
        "id": "reproduction-record",
        "paper": "paper-record",
        "status": "in_progress",
        "assignees": [],
        "url": [],
        "comments": "",
        "expand": {"paper": fields},
    }


class ValidatorTest(unittest.TestCase):
    def validate(
        self, document: dict, readme: str | None = None
    ) -> subprocess.CompletedProcess[str]:
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
            if readme is None:
                readme = readme_text(document)
            (root / "README.md").write_text(readme, encoding="utf-8")
            return subprocess.run(
                ["python3", str(VALIDATOR), str(root)],
                check=False,
                capture_output=True,
                text=True,
            )

    def test_valid_record(self) -> None:
        result = self.validate(valid_document())
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_readme_paper_id_matches_record(self) -> None:
        document = valid_document()
        for name, readme in {
            "missing": readme_text(document).replace("**Paper ID:**", "Paper ID:"),
            "other id": readme_text(document).replace(
                f"`{document['paper_id']}`", "`other-paper`"
            ),
        }.items():
            with self.subTest(name):
                result = self.validate(document, readme)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Paper ID line", result.stderr)

    def test_readme_sections_mirror_entry_ids(self) -> None:
        document = valid_document()
        document["noncompliance"] = [
            {
                "id": "ceiling-late",
                "description": "Ceiling declared after launch.",
                "requirement": "Declare ceilings before launch.",
                "run_ids": [document["runs"][0]["run_id"]],
                "effect": "None on results.",
            }
        ]
        result = self.validate(document)
        self.assertEqual(result.returncode, 0, result.stderr)
        empty = valid_document()
        cases = {
            "id not listed": (
                document,
                readme_text(document).replace("| `ceiling-late` |", "None."),
                "## Noncompliance",
            ),
            "empty list, not None.": (
                empty,
                readme_text(empty).replace(
                    "## Noncompliance\n\nNone.", "## Noncompliance\n\nSee below."
                ),
                "## Noncompliance",
            ),
            "section missing": (
                empty,
                readme_text(empty).replace("## Guesses and deviations", "## Guesses"),
                "## Guesses and deviations",
            ),
        }
        for name, (doc, readme, heading) in cases.items():
            with self.subTest(name):
                result = self.validate(doc, readme)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(heading, result.stderr)

    def test_agent_attribution_is_required(self) -> None:
        for value in (None, [], [{"agent_id": "executor"}]):
            with self.subTest(agents=value):
                document = valid_document()
                document["agents"] = value
                result = self.validate(document)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("agent", result.stderr)

    def test_run_agent_references_are_checked(self) -> None:
        for value in (None, ["missing"], ["executor", "executor"], []):
            with self.subTest(agent_ids=value):
                document = valid_document()
                document["runs"][0]["agent_ids"] = value
                result = self.validate(document)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("agent", result.stderr)

    def test_explicit_unknown_attribution_is_valid(self) -> None:
        document = valid_document()
        document["agents"][0].update(
            {
                "model": None,
                "harness": None,
                "scope": "Historical reproduction; model and application unknown.",
                "evidence": ["Original commit and PR do not record agent identity."],
            }
        )
        document["runs"][0].update(
            {
                "agent_ids": [],
                "agent_attribution_note": "Historical commit/PR evidence does not identify the executor.",
            }
        )
        result = self.validate(document)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_attribution_requires_evidence(self) -> None:
        document = valid_document()
        document["agents"][0]["evidence"] = []
        result = self.validate(document)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("evidence must be", result.stderr)

    def test_multiple_execution_agents_are_valid(self) -> None:
        document = valid_document()
        other = copy.deepcopy(document["agents"][0])
        other.update({"agent_id": "supervisor", "model": "Another model 2.0"})
        document["agents"].append(other)
        document["runs"][0]["agent_ids"].append("supervisor")
        result = self.validate(document)
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
            "numerical_agreement_basis": "No target was produced.",
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

    def guesses_and_deviations_document(self) -> dict:
        document = valid_document()
        document["guesses_and_deviations"] = [
            {
                "id": "seed",
                "kind": "guess",
                "description": "The paper states no seed; seed 0 is used.",
                "used": "seed 0",
                "basis": "No seed is reported anywhere in the paper.",
                "effect": "Single-seed result; variance unknown.",
                "scope": "Random seed only.",
                "status": "adopted before the run",
                "evidence": ["Paper section 4 omits the seed."],
                "affects": ["train"],
                "dataset_ids": ["data"],
            },
            {
                "id": "epochs",
                "kind": "paper_code_conflict",
                "description": "Paper and released config disagree on epochs.",
                "paper_says": "120 epochs",
                "used": "120 epochs, following the paper",
                "related_ids": ["seed"],
            },
            {
                "id": "split-sections",
                "kind": "paper_internal_conflict",
                "description": "Two sections of the paper describe different splits.",
                "paper_says": "Sec. 2.1: 80/20 of all images; Sec. 5.1: 500 per class",
                "used": "500 per class, the protocol tied to the reported table",
            },
        ]
        document["noncompliance"] = [
            {
                "id": "ceilings-not-declared",
                "description": "Launched with only the Modal function timeout.",
                "requirement": "stopping-criteria.md: declare GPU-hour and CHF ceilings.",
                "run_ids": ["run-1"],
                "effect": "Budget control only; no effect on the numbers.",
                "resolution": "Recorded; no re-run.",
            },
            {
                "id": "compute-discussion-skipped",
                "description": "The plan was not discussed before launch.",
                "requirement": "AGENTS.md: discuss runs over 24 GPU-hours first.",
            },
        ]
        return document

    def test_guesses_deviations_and_noncompliance_entries_validate(self) -> None:
        result = self.validate(self.guesses_and_deviations_document())
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_guesses_and_deviations_and_noncompliance_are_required(self) -> None:
        for field in ("guesses_and_deviations", "noncompliance"):
            with self.subTest(field=field):
                document = valid_document()
                del document[field]
                result = self.validate(document)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(f"{field} must be an array", result.stderr)

    def test_guess_deviation_entry_needs_id_kind_and_description(self) -> None:
        cases = {
            "id": "must be an object with id",
            "kind": ".kind must be one of",
            "description": "needs a non-empty description",
        }
        for field, message in cases.items():
            with self.subTest(field=field):
                document = self.guesses_and_deviations_document()
                del document["guesses_and_deviations"][0][field]
                result = self.validate(document)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)

    def test_guess_deviation_rejects_invalid_kind_and_unknown_field(self) -> None:
        document = self.guesses_and_deviations_document()
        entry = document["guesses_and_deviations"][0]
        entry["kind"] = "deviation"
        entry["detail"] = "a legacy spelling"
        result = self.validate(document)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(".kind must be one of", result.stderr)
        self.assertIn("unknown field 'detail'", result.stderr)

    def test_guess_deviation_ids_are_unique(self) -> None:
        document = self.guesses_and_deviations_document()
        document["guesses_and_deviations"][1]["id"] = "seed"
        del document["guesses_and_deviations"][1]["related_ids"]
        result = self.validate(document)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("duplicate id 'seed'", result.stderr)

    def test_guess_deviation_references_must_resolve(self) -> None:
        document = self.guesses_and_deviations_document()
        document["guesses_and_deviations"][0]["dataset_ids"] = ["missing-dataset"]
        document["guesses_and_deviations"][1]["related_ids"] = ["epochs"]
        result = self.validate(document)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("references unknown dataset 'missing-dataset'", result.stderr)
        self.assertIn("references unknown sibling entry 'epochs'", result.stderr)

    def test_noncompliance_needs_requirement_and_known_runs(self) -> None:
        document = self.guesses_and_deviations_document()
        del document["noncompliance"][0]["requirement"]
        document["noncompliance"][0]["run_ids"] = ["missing-run"]
        result = self.validate(document)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("needs a non-empty requirement", result.stderr)
        self.assertIn("references unknown run 'missing-run'", result.stderr)

    def test_legacy_deviation_and_guess_keys_are_rejected(self) -> None:
        document = valid_document()
        document["deviations"] = ["prose"]
        document["guesses"] = []
        document["datasets"][0]["deviations"] = ["prose"]
        result = self.validate(document)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("top-level deviations is no longer supported", result.stderr)
        self.assertIn("top-level guesses is no longer supported", result.stderr)
        self.assertIn("datasets[0].deviations is no longer supported", result.stderr)

    def unset_cache_environment_document(self) -> dict:
        document = valid_document()
        document["runs"][0]["compute"] = {
            "platform": "modal",
            "modal_profile": "repro-sign",
            "modal_app_ids": [APP_ID],
            "gpu_count": 0,
            "shared_cache": {
                "modal_volume": "huggingface-cache",
                "mount_path": "/cache/huggingface",
                "environment": {"HF_HOME": None, "HF_HUB_CACHE": None},
            },
        }
        document["noncompliance"] = [
            {
                "id": "hf-cache-environment-unset",
                "description": "The image did not set the cache variables.",
                "requirement": "AGENTS.md: set HF_HOME and HF_HUB_CACHE on the cache.",
                "run_ids": ["run-1"],
            }
        ]
        return document

    def test_unset_cache_environment_with_noncompliance_validates(self) -> None:
        result = self.validate(self.unset_cache_environment_document())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")

    def test_unset_cache_environment_needs_noncompliance(self) -> None:
        document = self.unset_cache_environment_document()
        document["noncompliance"] = []
        result = self.validate(document)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ran without the cache environment", result.stderr)

    def test_unset_cache_noncompliance_must_name_the_run(self) -> None:
        document = self.unset_cache_environment_document()
        document["noncompliance"][0]["run_ids"] = []
        result = self.validate(document)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ran without the cache environment", result.stderr)

    def test_wrong_cache_environment_value_stays_an_error(self) -> None:
        document = self.unset_cache_environment_document()
        environment = document["runs"][0]["compute"]["shared_cache"]["environment"]
        environment["HF_HOME"] = "/root/.cache/huggingface"
        result = self.validate(document)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("has invalid cache environment", result.stderr)

    def modal_run_document(self) -> dict:
        document = valid_document()
        document["runs"][0]["compute"] = {
            "platform": "modal",
            "modal_profile": "repro-sign",
            "modal_app_ids": [APP_ID],
            "gpu_count": 0,
            "shared_cache": {
                "modal_volume": "huggingface-cache",
                "mount_path": "/cache/huggingface",
                "environment": {
                    "HF_HOME": "/cache/huggingface",
                    "HF_HUB_CACHE": "/cache/huggingface/hub",
                },
            },
        }
        return document

    def assert_invalid(self, document: dict, message: str) -> None:
        result = self.validate(document)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(message, result.stderr)

    def test_modal_run_with_app_ids_validates(self) -> None:
        document = self.modal_run_document()
        document["runs"][0]["compute"]["modal_app_ids"].append(OTHER_APP_ID)
        document["runs"][0]["compute"]["dashboard_url"] = (
            f"https://modal.com/apps/repro-sign/main/{OTHER_APP_ID}"
        )
        result = self.validate(document)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_modal_run_requires_app_ids(self) -> None:
        document = self.modal_run_document()
        del document["runs"][0]["compute"]["modal_app_ids"]
        self.assert_invalid(document, "needs compute.modal_app_ids")

    def test_app_ids_must_be_modal_app_ids(self) -> None:
        for value in ("ap-x", ["unknown"], ["ap-short"], [APP_ID, APP_ID]):
            with self.subTest(value=value):
                document = self.modal_run_document()
                document["runs"][0]["compute"]["modal_app_ids"] = value
                message = (
                    "has duplicates"
                    if value == [APP_ID, APP_ID]
                    else ("must be a list of Modal app IDs")
                )
                self.assert_invalid(document, message)

    def test_missing_app_id_needs_noncompliance(self) -> None:
        document = self.modal_run_document()
        document["runs"][0]["compute"]["modal_app_ids"] = []
        self.assert_invalid(document, "has no app ID")
        document["noncompliance"] = [
            {
                "id": "app-id-not-recorded",
                "description": "The launch output with the app ID was lost.",
                "requirement": "AGENTS.md: record the Modal app ID of every run.",
                "run_ids": ["run-1"],
            }
        ]
        result = self.validate(document)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_every_mentioned_app_id_is_listed(self) -> None:
        document = self.modal_run_document()
        document["runs"][0]["detail"] = f"Resumed in {OTHER_APP_ID}."
        self.assert_invalid(
            document, f"mentions {OTHER_APP_ID} but compute.modal_app_ids omits it"
        )

    def test_run_mentioning_an_app_id_needs_app_ids(self) -> None:
        document = valid_document()
        document["runs"][0]["modal"] = {"dashboard_url": f"https://x/{APP_ID}"}
        self.assert_invalid(document, "needs compute.modal_app_ids")

    def test_run_naming_an_app_must_declare_modal(self) -> None:
        document = self.modal_run_document()
        del document["runs"][0]["compute"]["platform"]
        self.assert_invalid(document, "compute.platform must be 'modal'")

    def test_missing_shared_cache_needs_noncompliance(self) -> None:
        document = self.modal_run_document()
        del document["runs"][0]["compute"]["shared_cache"]
        self.assert_invalid(document, "needs shared_cache, or a noncompliance entry")
        document["noncompliance"] = [
            {
                "id": "hf-cache-not-recorded",
                "description": "The run entry does not record the cache mount.",
                "requirement": "AGENTS.md: every Modal run mounts huggingface-cache.",
                "run_ids": ["run-1"],
            }
        ]
        result = self.validate(document)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_legacy_app_id_fields_are_rejected(self) -> None:
        for where, field in (
            ("compute", "modal_app_id"),
            ("compute", "app_id"),
            ("run", "app_id"),
            ("modal", "app_id"),
        ):
            with self.subTest(where=where, field=field):
                document = self.modal_run_document()
                run = document["runs"][0]
                target = {"run": run, "compute": run["compute"]}.get(where)
                if target is None:
                    target = run.setdefault("modal", {})
                target[field] = APP_ID
                self.assert_invalid(document, "is replaced by compute.modal_app_ids")

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

    def test_cost_ceiling_currency_follows_declaration_date(self) -> None:
        old = self.gpu_run_document("2026-11-19T10:00:00Z")
        new = self.gpu_run_document("2026-11-21T10:00:00Z")
        for document in (old, new):
            result = self.validate(document)
            self.assertEqual(result.returncode, 0, result.stderr)
        old["runs"][0]["stop_policy"]["max_cost_usd"] = 4.0  # both, before cut-off
        self.assertEqual(self.validate(old).returncode, 0)
        cases = {
            "CHF after cut-off": {"max_cost_chf": 5.0},
            "USD ceiling missing": {"max_cost_usd": None},
        }
        for name, change in cases.items():
            with self.subTest(name):
                document = self.gpu_run_document("2026-11-21T10:00:00Z")
                stop_policy = document["runs"][0]["stop_policy"]
                stop_policy.update(change)
                result = self.validate(document)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("max_cost_usd", result.stderr)

    def gpu_run_document(self, started_at: str) -> dict:
        document = valid_document()
        run = document["runs"][0]
        run["started_at_utc"] = started_at
        run["finished_at_utc"] = started_at.replace("T10:00", "T10:30")
        run["stop_policy"]["declared_at_utc"] = started_at.replace("T10:00", "T09:55")
        run["stop_policy"]["max_gpu_hours"] = 1.0
        stop_policy = run["stop_policy"]
        if stop_policy["declared_at_utc"] >= "2026-11-20":
            del stop_policy["max_cost_chf"]
            stop_policy["max_cost_usd"] = 5.0
        else:
            stop_policy["max_cost_chf"] = 5.0
        run["compute"] = {
            "platform": "modal",
            "modal_profile": "repro-sign",
            "modal_app_ids": [APP_ID],
            "gpu_count": 1,
            "shared_cache": {
                "modal_volume": "huggingface-cache",
                "mount_path": "/cache/huggingface",
                "environment": {
                    "HF_HOME": "/cache/huggingface",
                    "HF_HUB_CACHE": "/cache/huggingface/hub",
                },
            },
            "usage": {
                "gpu_hours": 0.5,
                "gpu_hours_basis": "measured",
                "cost_usd": 1.1,
                "cost_basis": "billed",
                "source": "modal billing report --start 2026-11-20 -r h, app ap-x",
            },
        }
        return document

    def test_gpu_run_with_usage_validates(self) -> None:
        result = self.validate(self.gpu_run_document("2026-11-20T10:00:00Z"))
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_gpu_run_requires_usage(self) -> None:
        for started_at in ("2026-08-20T10:00:00Z", "2026-11-20T10:00:00Z"):
            with self.subTest(started_at=started_at):
                document = self.gpu_run_document(started_at)
                del document["runs"][0]["compute"]["usage"]
                result = self.validate(document)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("needs compute.usage", result.stderr)

    def test_billed_usage_needs_app_ids(self) -> None:
        document = self.gpu_run_document("2026-11-20T10:00:00Z")
        document["runs"][0]["compute"]["modal_app_ids"] = []
        document["noncompliance"] = [
            {
                "id": "app-id-not-recorded",
                "description": "The launch output with the app ID was lost.",
                "requirement": "AGENTS.md: record the Modal app ID of every run.",
                "run_ids": ["run-1"],
            }
        ]
        self.assert_invalid(document, "cost_basis 'billed' needs compute.modal_app_ids")
        usage = document["runs"][0]["compute"]["usage"]
        usage.update(gpu_hours_basis="upper_bound", cost_basis="rate_estimate")
        result = self.validate(document)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cpu_run_may_omit_usage(self) -> None:
        document = self.gpu_run_document("2026-11-20T10:00:00Z")
        compute = document["runs"][0]["compute"]
        compute["gpu_count"] = 0
        del compute["usage"]
        result = self.validate(document)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_usage_fields_are_controlled(self) -> None:
        cases = {
            "gpu_hours": (-1, "gpu_hours must be a nonnegative number"),
            "cost_usd": (None, "cost_usd must be a nonnegative number"),
            "gpu_hours_basis": ("guess", "gpu_hours_basis must be one of"),
            "cost_basis": ("invoice", "cost_basis must be one of"),
            "source": ("", "source must be a non-empty string"),
            "cost_chf": (1.0, "unknown field 'cost_chf'"),
        }
        for field, (value, message) in cases.items():
            with self.subTest(field=field):
                document = self.gpu_run_document("2026-11-20T10:00:00Z")
                document["runs"][0]["compute"]["usage"][field] = value
                result = self.validate(document)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)

    def test_gpu_hours_and_cost_only_in_usage(self) -> None:
        for field in ("gpu_hours", "process_gpu_hours", "cost_chf", "cost_note"):
            with self.subTest(field=field):
                document = self.gpu_run_document("2026-11-20T10:00:00Z")
                document["runs"][0]["compute"][field] = 1.0
                result = self.validate(document)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    f"compute.{field}: record GPU-hours and cost only in compute.usage",
                    result.stderr,
                )

    def test_not_produced_requires_coded_reason_and_evidence(self) -> None:
        document = valid_document()
        document["status"] = {
            "pipeline": "insufficient_information",
            "numerical_agreement": "not_assessed",
            "numerical_agreement_basis": "No target was produced.",
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
            "numerical_agreement_basis": "No target was produced.",
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

    def ingest(
        self, export: object, existing: dict | None = None
    ) -> tuple[subprocess.CompletedProcess[str], dict | None]:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "reproductions-paper-1-2026-11-19.json"
            output = root / "reproduction.json"
            source.write_text(json.dumps(export), encoding="utf-8")
            if existing is not None:
                output.write_text(json.dumps(existing), encoding="utf-8")
            result = subprocess.run(
                ["python3", str(INGESTER), str(source), "paper-1", str(output)],
                check=False,
                capture_output=True,
                text=True,
            )
            written = json.loads(output.read_text()) if output.exists() else None
            return result, written

    def test_candidate_ingestion_creates_current_record_shape(self) -> None:
        result, written = self.ingest(tracker_export())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(written["paper_id"], "paper-1")
        self.assertEqual(written["assignment"]["kind"], "queue_record")
        self.assertEqual(written["assignment"]["normalized"]["title"], "A paper")

    def test_candidate_ingestion_rejects_ineligible_exports(self) -> None:
        wrong_relation = tracker_export()
        wrong_relation["paper"] = "other"
        cases = {
            "candidate list": [tracker_export()],
            "paper export": tracker_export()["expand"]["paper"],
            "not final": tracker_export(status="flagged"),
            "other paper": tracker_export(paper_id="paper-2"),
            "wrong relation": wrong_relation,
        }
        for name, export in cases.items():
            with self.subTest(name):
                result, written = self.ingest(export)
                self.assertNotEqual(result.returncode, 0)
                self.assertIsNone(written)

    def test_candidate_ingestion_redacts_email_addresses(self) -> None:
        export = tracker_export(
            finalized_by="reviewer@example.com",
            comments="See reviewer@example.com for the note.",
            status_history=[{"by": "editor@example.org", "after": "final"}],
            expand={
                "datasets": [
                    {
                        "id": "dataset-1",
                        "name": "Example",
                        "assignees": ["owner@example.edu"],
                        "comments": "",
                        "url": [],
                    }
                ]
            },
        )
        export["assignees"] = ["assignee@example.com"]
        result, written = self.ingest(export)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("@", json.dumps(written["assignment"]))
        paper = written["assignment"]["record"]["expand"]["paper"]
        self.assertEqual(paper["finalized_by"], "[redacted]")
        self.assertIn("[redacted]", written["assignment"]["normalized"]["comments"])

    def test_candidate_ingestion_preserves_existing_sections(self) -> None:
        result, written = self.ingest(
            tracker_export(), {"paper_id": "paper-1", "paper": {"title": "Preserved"}}
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(written["paper"]["title"], "Preserved")

    def test_candidate_ingestion_rejects_retired_version_field(self) -> None:
        result, _ = self.ingest(
            tracker_export(), {"schema_version": "retired", "paper_id": "paper-1"}
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("retired version field", result.stderr)

    def test_queue_record_must_be_a_final_tracker_export(self) -> None:
        document = valid_document()
        document["assignment"] = {
            "kind": "queue_record",
            "source": {"path": "export.json", "sha256": "a" * 64},
            "record": tracker_export(paper_id=document["paper_id"]),
        }
        result = self.validate(document)
        self.assertEqual(result.returncode, 0, result.stderr)
        legacy = {
            "paper_id": document["paper_id"],
            "confirmation": "confirmed",
            "status": "final",
        }
        for name, record in {
            "legacy": legacy,
            "not final": tracker_export(
                paper_id=document["paper_id"], status="needs_review"
            ),
            "other paper": tracker_export(),
            "legacy unconfirmed": {**legacy, "confirmation": "pending"},
        }.items():
            with self.subTest(name):
                document["assignment"]["record"] = record
                result = self.validate(document)
                if name == "legacy":
                    self.assertEqual(result.returncode, 0, result.stderr)
                else:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("queue record", result.stderr)


if __name__ == "__main__":
    unittest.main()
