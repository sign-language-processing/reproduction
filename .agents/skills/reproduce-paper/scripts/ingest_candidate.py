#!/usr/bin/env python3
"""Import one final/confirmed candidate into a paper reproduction record."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, NoReturn


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


def select_record(records: Any, paper_id: str) -> dict[str, Any]:
    if not isinstance(records, list):
        fail("candidate export must be a top-level JSON array")
    matches = [
        item
        for item in records
        if isinstance(item, dict) and item.get("paper_id") == paper_id
    ]
    if len(matches) != 1:
        fail(f"expected exactly one record for {paper_id!r}, found {len(matches)}")

    record = matches[0]
    if record.get("id") not in (None, paper_id):
        fail("record id and paper_id do not match")
    if record.get("confirmation") != "confirmed":
        fail("candidate confirmation must be 'confirmed'")
    if record.get("status") != "final":
        fail("candidate status must be 'final'")
    return record


def build_assignment(
    source: Path, source_bytes: bytes, record: dict[str, Any]
) -> dict[str, Any]:
    return {
        "kind": "queue_record",
        "source": {
            "path": str(source.resolve()),
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
        },
        "normalized": {
            "title": record.get("title"),
            "year": record.get("year"),
            "venue": record.get("venue"),
            "pdf_url": record.get("pdf_url"),
            "code_repos": normalize_repositories(record.get("code_repos")),
            "what_to_reproduce": record.get("what_to_reproduce"),
            "metric_ids": record.get("metrics", []),
            "metric_records": normalize_metric_records(record),
            "datasets": normalize_datasets(record),
            "copied_scores": record.get("copied_scores"),
            "compute_requirements": record.get("compute_requirements"),
            "includes_human_evaluation": record.get("includes_human_evaluation"),
            "potential_ethical_concerns": record.get("potential_ethical_concerns"),
            "comments": record.get("comments", ""),
            "flag_reason": record.get("flag_reason", ""),
        },
        "record": record,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="JSON candidate export")
    parser.add_argument("paper_id", help="exact paper_id to select")
    parser.add_argument("output", type=Path, help="reproduction.json destination")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        source_bytes = args.source.read_bytes()
        records = json.loads(source_bytes)
        record = select_record(records, args.paper_id)
        assignment = build_assignment(args.source, source_bytes, record)

        if args.output.exists():
            output = json.loads(args.output.read_text(encoding="utf-8"))
            existing_id = output.get("paper_id")
            if existing_id not in (None, args.paper_id):
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

    print(f"imported final/confirmed candidate {args.paper_id} into {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
