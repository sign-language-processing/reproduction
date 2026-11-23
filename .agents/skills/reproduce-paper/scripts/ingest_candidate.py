#!/usr/bin/env python3
"""Import a reproduction-tracker export into a paper's reproduction record.

The export is the "Download JSON" file from a paper's page in the REPRO-SIGN
Reproduction Tracker (https://github.com/bricksdont/repro-sign-survey-ui): one
`reproductions` record with its paper under `expand.paper`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, NoReturn


email_address_pattern = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")


def fail(message: str) -> NoReturn:
    raise ValueError(message)


def normalize_repositories(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = value.strip()
        return [] if not value or value.upper() == "N/A" else [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return [
            item.strip() for item in value if item.strip() and item.upper() != "N/A"
        ]
    fail("code_repos must be a string, an array of strings, or null")


def normalize_datasets(record: dict[str, Any]) -> list[dict[str, Any]]:
    expand = record.get("expand")
    expanded = expand.get("datasets", []) if isinstance(expand, dict) else []
    if expanded is None:
        expanded = []
    if not isinstance(expanded, list) or not all(
        isinstance(item, dict) for item in expanded
    ):
        fail("expand.datasets must be an array of objects when present")

    datasets = []
    for item in expanded:
        urls = item.get("url", [])
        if isinstance(urls, str):
            urls = [urls]
        if not isinstance(urls, list) or not all(isinstance(url, str) for url in urls):
            fail("each expanded dataset url must be a string or array of strings")
        datasets.append(
            {
                "id": item.get("id"),
                "name": item.get("name"),
                "available": item.get("available"),
                "license": item.get("license"),
                "urls": [url for url in urls if url],
                "comments": item.get("comments", ""),
            }
        )
    return datasets


def normalize_metric_records(record: dict[str, Any]) -> list[dict[str, Any]]:
    expand = record.get("expand")
    metrics = expand.get("metrics", []) if isinstance(expand, dict) else []
    if metrics is None:
        return []
    if not isinstance(metrics, list) or not all(
        isinstance(item, dict) for item in metrics
    ):
        fail("expand.metrics must be an array of objects when present")
    return metrics


def select_paper(record: Any, paper_id: str) -> dict[str, Any]:
    """Check the export and return its paper."""
    if not isinstance(record, dict) or record.get("collectionName") != "reproductions":
        fail(
            "export must be one Reproduction Tracker record "
            "(collectionName 'reproductions'), not a list or a paper export"
        )
    expand = record.get("expand")
    paper = expand.get("paper") if isinstance(expand, dict) else None
    if not isinstance(paper, dict):
        fail("export has no expand.paper")
    if paper.get("paper_id") != paper_id:
        fail(f"export is for paper {paper.get('paper_id')!r}, not {paper_id!r}")
    if record.get("paper") != paper.get("id"):
        fail("export's paper relation does not match expand.paper.id")
    if paper.get("status") != "final":
        fail("paper status must be 'final'")
    return paper


def redact_email_addresses(value: Any) -> Any:
    if isinstance(value, str):
        return email_address_pattern.sub("[redacted]", value)
    if isinstance(value, list):
        return [redact_email_addresses(item) for item in value]
    if isinstance(value, dict):
        return {key: redact_email_addresses(item) for key, item in value.items()}
    return value


def build_assignment(
    source: Path, source_bytes: bytes, record: dict[str, Any], paper: dict[str, Any]
) -> dict[str, Any]:
    return redact_email_addresses(
        {
            "kind": "queue_record",
            "source": {
                "path": source.name,
                "sha256": hashlib.sha256(source_bytes).hexdigest(),
            },
            "normalized": {
                "title": paper.get("title"),
                "year": paper.get("year"),
                "venue": paper.get("venue"),
                "pdf_url": paper.get("pdf_url"),
                "code_repos": normalize_repositories(paper.get("code_repos")),
                "what_to_reproduce": paper.get("what_to_reproduce"),
                "metric_ids": paper.get("metrics", []),
                "metric_records": normalize_metric_records(paper),
                "datasets": normalize_datasets(paper),
                "copied_scores": paper.get("copied_scores"),
                "compute_requirements": paper.get("compute_requirements"),
                "includes_human_evaluation": paper.get("includes_human_evaluation"),
                "potential_ethical_concerns": paper.get("potential_ethical_concerns"),
                "comments": paper.get("comments", ""),
            },
            "record": record,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Reproduction Tracker JSON export")
    parser.add_argument("paper_id", help="exact paper_id to select")
    parser.add_argument("output", type=Path, help="reproduction.json destination")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        source_bytes = args.source.read_bytes()
        record = json.loads(source_bytes)
        paper = select_paper(record, args.paper_id)
        assignment = build_assignment(args.source, source_bytes, record, paper)

        if args.output.exists():
            output = json.loads(args.output.read_text(encoding="utf-8"))
            existing_id = output.get("paper_id")
            # A fresh copy of templates/reproduction.json carries the placeholder.
            if existing_id not in (None, "PAPER_ID", args.paper_id):
                fail(
                    "refusing to update reproduction.json for different paper "
                    f"{existing_id!r}"
                )
            if "schema_version" in output:
                fail(
                    "existing reproduction.json uses the retired version field; "
                    "migrate it to the single current contract first"
                )
        else:
            output = {"paper_id": args.paper_id}
        output["paper_id"] = args.paper_id
        output["assignment"] = assignment

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(f"candidate ingestion failed: {exc}", file=sys.stderr)
        return 2

    print(f"imported final paper {args.paper_id} into {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
