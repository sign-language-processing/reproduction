#!/usr/bin/env bash
set -euo pipefail

# Build the shared MS-ASL tree (official annotations + per-annotation clips cut
# from the still-available YouTube sources), ready to upload as datasets/ms-asl.
# Runs outside Modal because the sources are YouTube videos. Resumable: finished
# videos are recorded under status/ and skipped. The assignee authorized
# acquisition and project-cloud storage on 2026-10-02.
#
#   ./data.sh STAGING_DIR            # build STAGING_DIR/ms-asl
#   modal_repro_sign.sh volume put datasets STAGING_DIR/ms-asl /ms-asl
#
# Needs python3 >= 3.9, a current yt-dlp with a JavaScript runtime (pip install "yt-dlp[default]" deno;
# without one some sources fail with HTTP 403), and ffmpeg (or FFMPEG=/path/to/ffmpeg).
root="${1:?usage: data.sh STAGING_DIR}/ms-asl"
annotations_url="https://download.microsoft.com/download/3/c/a/3ca92c78-1c4a-4a91-a7ee-6980c1d242ec/MS-ASL.zip"
annotations_sha256="a8562008309eea4129e1bc0ed7f654a314fee195227222859657e307b6434c34"

mkdir -p "${root}/annotations" "${root}/clips" "${root}/status"
if [[ ! -f "${root}/annotations/MSASL_train.json" ]]; then
  archive="$(mktemp)"
  trap 'rm -f -- "${archive}"' EXIT
  curl -fsSL -o "${archive}" "${annotations_url}"
  echo "${annotations_sha256}  ${archive}" | sha256sum -c - >/dev/null
  unzip -q -j -o "${archive}" 'MS-ASL/*' -d "${root}/annotations"
fi

exec python3 - "${root}" "${annotations_url}" "${annotations_sha256}" "${FFMPEG:-ffmpeg}" "${WORKERS:-6}" "${LIMIT:-0}" <<'PY'
import hashlib
import json
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

root, annotations_url, annotations_sha256, ffmpeg = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
workers, limit = int(sys.argv[5]), int(sys.argv[6])
FORMAT = "bv*[height<=720][ext=mp4]/bv*[height<=720]/b[height<=720]"
GONE = ("Private video", "Video unavailable", "This video is unavailable", "This video is not available", "has been removed",
        "account associated with this video has been terminated", "This video has been removed",
        "members-only", "copyright", "not available in your country", "violating YouTube")
bot_check = threading.Event()

by_video = defaultdict(list)
for split in ("train", "val", "test"):
    for index, clip in enumerate(json.loads((root / "annotations" / f"MSASL_{split}.json").read_text())):
        video_id = clip["url"].split("v=")[1][:11]
        by_video[video_id].append({
            "split": split, "index": index, "video_id": video_id, "label": clip["label"],
            "text": clip["text"], "start_time": clip["start_time"], "end_time": clip["end_time"],
            "path": f"clips/{split}/{index:05d}.mp4",
        })


def run(command):
    return subprocess.run(command, capture_output=True, text=True)


def fetch(video_id):
    status_file = root / "status" / f"{video_id}.json"
    if status_file.exists() or bot_check.is_set():
        return
    result = {"video_id": video_id, "attempted_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with tempfile.TemporaryDirectory(dir=root / "status") as tmp:
        for attempt in range(1, 4):
            download = run(["yt-dlp", "--no-warnings", "--no-playlist", "-f", FORMAT, "-o", f"{tmp}/src.%(ext)s",
                            "--print", "after_move:%(height)s %(fps)s %(format_id)s",
                            f"https://www.youtube.com/watch?v={video_id}"])
            error = download.stderr.strip().splitlines()[-1][:300] if download.stderr.strip() else ""
            if download.returncode == 0 or any(marker in error for marker in GONE):
                break
            if "not a bot" in error or "Sign in to confirm" in error:
                # Never work around bot detection: stop the whole run and leave this video unrecorded.
                bot_check.set()
                return
            time.sleep(20 * attempt)
        result["attempts"] = attempt
        if download.returncode != 0:
            gone = any(marker in error for marker in GONE)
            result.update(status="unavailable" if gone else "failed", error=error)
            if gone:
                status_file.write_text(json.dumps(result))
            else:
                print(f"transient failure, will retry on rerun: {video_id}: {error}", file=sys.stderr)
            return
        source = next(Path(tmp).glob("src.*"))
        result["source_format"] = download.stdout.strip().splitlines()[-1]
        clips = {}
        for clip in by_video[video_id]:
            target = root / clip["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            cut = run([ffmpeg, "-nostdin", "-loglevel", "error", "-y", "-ss", str(clip["start_time"]),
                       "-to", str(clip["end_time"]), "-i", str(source), "-an", "-c:v", "libx264", "-crf", "18",
                       "-preset", "veryfast", "-pix_fmt", "yuv420p", str(target)])
            if cut.returncode == 0 and target.stat().st_size > 0:
                clips[clip["path"]] = hashlib.sha256(target.read_bytes()).hexdigest()
            else:
                target.unlink(missing_ok=True)
                clips[clip["path"]] = None
        result.update(status="downloaded", clips=clips)
        status_file.write_text(json.dumps(result))


video_ids = sorted(by_video)[:limit] if limit else sorted(by_video)
with ThreadPoolExecutor(workers) as pool:
    list(pool.map(fetch, video_ids))
if bot_check.is_set():
    sys.exit("YouTube asked for bot verification; stopped without working around it. Rerun later to resume.")

statuses = {p.stem: json.loads(p.read_text()) for p in (root / "status").glob("*.json")}
pending = [v for v in video_ids if v not in statuses]
rows, counts = [], Counter()
for video_id in sorted(by_video):
    status = statuses.get(video_id, {"status": "not_attempted"})
    for clip in by_video[video_id]:
        sha256 = status.get("clips", {}).get(clip["path"])
        state = "ok" if sha256 else ("cut_failed" if status["status"] == "downloaded" else status["status"])
        rows.append({**clip, "status": state, "sha256": sha256})
        counts[(clip["split"], state)] += 1
rows.sort(key=lambda r: (r["split"], r["index"]))
manifest = {
    "dataset": "MS-ASL", "citation": "Vaezi Joze and Koller, BMVC 2019",
    "annotations": {"url": annotations_url, "sha256": annotations_sha256, "license": "C-UDA 0.1"},
    "videos": "Third-party YouTube uploads; clips are stored for computational use on project storage only and must not be redistributed.",
    "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "tools": {"yt_dlp": run(["yt-dlp", "--version"]).stdout.strip(),
              "ffmpeg": run([ffmpeg, "-version"]).stdout.splitlines()[0]},
    "processing": f"Source format '{FORMAT}'; each annotation cut by start_time/end_time, re-encoded libx264 crf 18, audio dropped, full frame kept (signer box stays in annotations).",
    "video_counts": dict(Counter(s["status"] for s in statuses.values())) | {"total": len(by_video), "pending": len(pending)},
    "clip_counts": {split: {state: n for (s, state), n in sorted(counts.items()) if s == split} for split in ("train", "val", "test")},
    "clips": rows,
}
(root / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
print(json.dumps({k: manifest[k] for k in ("video_counts", "clip_counts")}, indent=1))
if pending:
    sys.exit(f"{len(pending)} videos failed transiently; rerun to retry them.")
PY
