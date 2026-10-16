#!/usr/bin/env python3
"""Re-verify the ub-MOJI coverage findings recorded in reproduction.json.

Reproduces, from scratch, the two independent checks behind
release_coverage_check: (1) file enumeration at every published tag, and
(2) the dataset's own annotations.toml index. Downloads no video.

Requires a Hugging Face token with the accepted ub-MOJI terms, taken from
HF_TOKEN or ~/.cache/huggingface/token.

Usage:
    python3 verify_coverage.py            # all four tags
    python3 verify_coverage.py main       # one revision
"""

import json
import os
import re
import subprocess
import sys

DATASET = "kanglabs/ub-MOJI"
CHECKPOINTS = "kanglabs/ub-hrpro"
PROPOSAL = (
    f"https://huggingface.co/{CHECKPOINTS}/resolve/main/"
    "hrpro/ckpt/ub-moji/HR-Pro/stage1/outputs/proposal.json"
)
TAGS = ["v25.05", "v25.07", "v25.09", "v26.09"]


def token():
    value = os.environ.get("HF_TOKEN")
    if value:
        return value.strip()
    path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(path):
        with open(path) as handle:
            return handle.read().strip()
    sys.exit("No Hugging Face token found; set HF_TOKEN or run `hf auth login`.")


def fetch(url, auth):
    return subprocess.run(
        ["curl", "-sSL", "-H", f"Authorization: Bearer {auth}", url],
        capture_output=True,
        text=True,
    ).stdout


def tree(revision, auth):
    """Page the full recursive file tree for one revision."""
    url = (
        f"https://huggingface.co/api/datasets/{DATASET}/tree/{revision}"
        "?recursive=true&expand=true"
    )
    entries = []
    while url:
        out = subprocess.run(
            ["curl", "-sS", "-D", "-", "-H", f"Authorization: Bearer {auth}", url],
            capture_output=True,
            text=True,
        ).stdout
        head, _, body = out.partition("\r\n\r\n")
        if not body:
            head, _, body = out.partition("\n\n")
        entries.extend(json.loads(body))
        match = re.search(r'[Ll]ink:\s*<([^>]+)>;\s*rel="next"', head)
        url = match.group(1) if match else None
    return entries


def main():
    auth = token()
    revisions = sys.argv[1:] or TAGS

    proposal = json.loads(fetch(PROPOSAL, auth))
    train, held_out = set(proposal["train"]), set(proposal["test"])
    print(f"proposal.json: {len(train)} train + {len(held_out)} held-out IDs\n")
    print(f"{'revision':10s} {'held-out':>10s} {'train':>12s} {'annotated':>10s}")

    for revision in revisions:
        entries = tree(revision, auth)
        videos = {
            os.path.basename(entry["path"]).rsplit(".", 1)[0]
            for entry in entries
            if entry["type"] == "file" and re.search(r"\.mp4$", entry["path"], re.I)
        }
        annotations = fetch(
            f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/annotations.toml",
            auth,
        )
        keys = set(re.findall(r'^\["([^"]+)"\]\s*$', annotations, re.M))
        usable = videos & keys
        print(
            f"{revision:10s} {len(held_out & usable):6d}/{len(held_out):<4d}"
            f"{len(train & usable):8d}/{len(train):<4d}{len(keys):10d}"
        )
        for missing in sorted(held_out - usable):
            state = "no file" if missing not in videos else "no annotation"
            print(f"           unusable: {missing} ({state})")


if __name__ == "__main__":
    main()
