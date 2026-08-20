#!/usr/bin/env python3
"""Build the temporal lead-in clip corpus that the tracking-cascade arm evaluates on."""

import csv
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np


# === Config ===
LIMIT = None           # int or None (smoke test limit)
LEAD_SECONDS = 5.0
WORKERS = 4
RETRY_FAILED = False
UPLOAD = False

LEAD_SECONDS = 5.0	# lead-in per clip; the eval loop can use any suffix of this, so a shorter horizon needs no re-extraction
CRF = 20	# x264 quality for the lead-in frames; they drive the tracker only, never the score
PRESET = "veryfast"
WORKERS = 12	# network-bound, so well above the core count
MATCH_DIFF_MAX = 8.0	# mean abs diff accepting a tail frame as the annotated one; measured 1.7-2.8 for a true match vs 13-34 for neighbours
# Seeking to a non-frame-boundary time makes the -t window end mid-interval, which dropped
# the annotated frame on 1 of the first 8 clips. Overshoot past T, then find it by match.
TAIL_SECONDS = 0.25	# extra tail past T, so the annotated frame is never the one that rounds away
TAIL_SEARCH_FRAMES = 12	# tail frames searched for the match when the exact timestamp is known
SHEET_TAIL_SECONDS = 1.5	# extra tail for the few frames whose timestamp comes from the 1s-truncated sheet
RETRIES = 3
URL_TTL_S = 1500	# presigned CDN URLs last ~1h; refresh well inside that

# ── Paths ───────────────────────────────────
GCS_BUCKET = "laparoscopic-seg-REPLACE_K_NUMBER"
GCS_ROOT = f"gs://{GCS_BUCKET}/temporalTrainingFrames"
HF_REPO = "visurg/LEMON"
SHEET_ID = "REPLACE_SHEET_ID"
DELETED_STATUSES = ("deleted", "invalid", "duplicate", "excluded")	# matches the training notebook's retirement filter

HERE = Path(__file__).resolve().parent
# Deliberately outside the repo: the clip corpus is ~10GB and is published to GCS, not committed.
WORK = Path(os.environ.get("CLIP_WORK_DIR", Path.home() / "lemon_clip_build"))
CLIPS_DIR = WORK / "clips"
JPEG_DIR = WORK / "jpegs"
PLAN_CSV = WORK / "sampling_plan.csv"
SHEET_CSV = WORK / "sheet.csv"
MANIFEST = WORK / "manifest.csv"

MANIFEST_FIELDS = [
    "key", "video_id", "annotator", "frame_id", "t_source", "timestamp_sec",
    "clip_start_sec", "lead_actual_s", "fps", "n_frames", "target_idx",
    "match_diff", "match_margin", "native_w", "native_h", "jpeg_w", "jpeg_h",
    "clip_bytes", "status",
]


def hf_token():
    for p in ("~/.cache/huggingface/token", "~/.huggingface/token"):
        f = Path(p).expanduser()
        if f.exists():
            return f.read_text().strip()
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not tok:
        sys.exit("No HF token found; run `hf auth login` or set HF_TOKEN")
    return tok


def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        sys.exit("No ffmpeg: install it or `pip install imageio-ffmpeg`")



FF = ffmpeg_exe()
TOKEN = hf_token()


# ── Source metadata ─────────────────────────

def fetch_inputs():
    """Cache sampling_plan.csv (exact frame positions) and the annotation sheet (statuses)."""
    WORK.mkdir(parents=True, exist_ok=True)
    if not PLAN_CSV.exists():
        subprocess.run(["gsutil", "-q", "cp",
                        f"gs://{GCS_BUCKET}/pipeline/sampling_plan.csv", str(PLAN_CSV)], check=True)
    if not SHEET_CSV.exists():
        subprocess.run(["curl", "-sL", "-o", str(SHEET_CSV),
                        f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"], check=True)


def sync_jpegs():
    """Pull the annotated JPEGs locally; every clip is verified against one of them."""
    JPEG_DIR.mkdir(parents=True, exist_ok=True)
    for sub in ("JPEGImages", "NoTools"):
        dest = JPEG_DIR / sub
        if dest.exists() and any(dest.rglob("*.jpg")):
            continue
        dest.mkdir(parents=True, exist_ok=True)
        print(f"  rsync {sub} ...")
        subprocess.run(["gsutil", "-m", "-q", "rsync", "-r",
                        f"gs://{GCS_BUCKET}/trainingFrames/{sub}", str(dest)], check=True)


def index_jpegs():
    """(video_id, frame_id) -> local JPEG path, across both the annotated and NoTools roots."""
    idx = {}
    for p in JPEG_DIR.rglob("frame_*.jpg"):
        m = re.match(r"frame_(\d+)$", p.stem)
        if m:
            idx[(p.parent.name, int(m.group(1)))] = p
    return idx


def frame_key(video_id, frame_id):
    """Join key used throughout the training notebook: '<videoId>/frame_XXXXX'."""
    return f"{video_id}/frame_{frame_id:05d}"


def sheet_seconds(ts):
    parts = str(ts).strip().split(":")
    if len(parts) != 3 or not all(p.strip().isdigit() for p in parts):
        return None
    h, m, s = (int(p) for p in parts)
    return h * 3600 + m * 60 + s


def build_worklist():
    """One work item per annotated frame that survives the notebook's status filter.

    Prefers sampling_plan's normalized_position (exact). Falls back to the sheet's
    1s-truncated timestamp for the handful of rows the plan does not cover, and widens
    that clip's search tail so the pixel match still pins the exact frame.
    """
    plan = {int(r["frame_id"]): r for r in csv.DictReader(open(PLAN_CSV))}
    jpegs = index_jpegs()
    items, skipped = [], {"retired": 0, "no_jpeg": 0, "no_timestamp": 0}

    for row in csv.DictReader(open(SHEET_CSV)):
        status = str(row.get("Status", "")).lower()
        if any(w in status for w in DELETED_STATUSES):
            skipped["retired"] += 1
            continue
        m = re.search(r"frame_(\d+)", row.get("File Name", ""))
        if not m:
            continue
        frame_id = int(m.group(1))
        video_id = row["Video ID"].strip()
        jpeg = jpegs.get((video_id, frame_id))
        if jpeg is None:
            skipped["no_jpeg"] += 1
            continue

        p = plan.get(frame_id)
        # A plan row for a different video means the id spaces disagree for this frame;
        # trust the sheet's own Video ID and fall back to its timestamp.
        if p is not None and p["youtubeId"].strip() == video_id:
            t_source, norm_pos, t_sheet = "plan", float(p["normalized_position"]), None
        else:
            t_sheet = sheet_seconds(row.get("Timestamp"))
            if t_sheet is None:
                skipped["no_timestamp"] += 1
                continue
            t_source, norm_pos = "sheet", None

        items.append({
            "key": frame_key(video_id, frame_id),
            "video_id": video_id,
            "annotator": jpeg.parent.parent.name,
            "frame_id": frame_id,
            "t_source": t_source,
            "norm_pos": norm_pos,
            "t_sheet": t_sheet,
            "jpeg": str(jpeg),
        })
    return items, skipped


# ── Video access ────────────────────────────

def resolve_url(video_id):
    """HEAD-resolve the HF pointer to a presigned CDN URL.

    Deliberately a HEAD: resolving with a Range request returns a URL whose policy pins
    that exact byte range, and every later range request against it 403s.
    """
    url = (f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/"
           f"{urllib.parse.quote(video_id)}.mp4")
    out = subprocess.run(
        ["curl", "-sIL", "-H", f"Authorization: Bearer {TOKEN}", url,
         "-o", os.devnull, "-w", "%{url_effective}"],
        capture_output=True, text=True, timeout=120)
    final = out.stdout.strip()
    if not final.startswith("http"):
        raise RuntimeError(f"could not resolve {video_id}")
    return final


def probe(url):
    """(duration_s, fps) from ffmpeg's own header parse; no ffprobe dependency."""
    r = subprocess.run([FF, "-hide_banner", "-nostdin", "-i", url],
                       capture_output=True, text=True, timeout=300)
    d = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", r.stderr)
    f = re.search(r"(\d+(?:\.\d+)?)\s+fps", r.stderr)
    if not d or not f:
        raise RuntimeError(f"probe failed: {r.stderr[-300:]}")
    h, m, s = d.groups()
    return int(h) * 3600 + int(m) * 60 + float(s), float(f.group(1))


def extract(url, start, dur, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FF, "-hide_banner", "-nostdin", "-v", "error",
        "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5",
        "-ss", f"{start:.3f}", "-i", url, "-t", f"{dur:.3f}",
        "-c:v", "libx264", "-crf", str(CRF), "-preset", PRESET, "-an",
        "-y", str(out_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if r.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr[-300:]}")


def verify(clip_path, jpeg_path, search_all):
    """Locate the annotated frame inside the clip by pixel match.

    Returns (n_frames, target_idx, best_diff, margin_to_next_best, w, h). The margin is
    carried so an ambiguous match in a near-static scene can be audited later rather
     than silently trusted.
    """
    gt = cv2.imread(str(jpeg_path))
    if gt is None:
        raise RuntimeError(f"unreadable JPEG {jpeg_path}")
    cap = cv2.VideoCapture(str(clip_path))
    keep = 10_000 if search_all else TAIL_SEARCH_FRAMES
    tail, n = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        tail.append(frame)
        if len(tail) > keep:
            tail.pop(0)
    cap.release()
    if not tail:
        raise RuntimeError("clip decoded to zero frames")

    h, w = tail[-1].shape[:2]
    base = n - len(tail)
    diffs = []
    for f in tail:
        if f.shape != gt.shape:
            diffs.append(float("inf"))
            continue
        diffs.append(float(np.abs(f.astype(np.int16) - gt.astype(np.int16)).mean()))
    order = np.argsort(diffs)
    best = int(order[0])
    margin = (diffs[int(order[1])] - diffs[best]) if len(order) > 1 else float("nan")
    return n, base + best, diffs[best], margin, w, h


def build_one(item, lead_seconds):
    """Extract and verify one clip. Returns a manifest row; never raises."""
    row = {k: "" for k in MANIFEST_FIELDS}
    row.update(key=item["key"], video_id=item["video_id"], annotator=item["annotator"],
               frame_id=item["frame_id"], t_source=item["t_source"])
    out = CLIPS_DIR / item["annotator"] / item["video_id"] / f"frame_{item['frame_id']:05d}.mp4"

    last_err = ""
    for attempt in range(RETRIES):
        try:
            url = resolve_url(item["video_id"])
            duration, fps = probe(url)
            if item["t_source"] == "plan":
                t = item["norm_pos"] * duration
                tail = TAIL_SECONDS
            else:
                t = float(item["t_sheet"])
                tail = SHEET_TAIL_SECONDS
            start = max(0.0, t - lead_seconds)
            extract(url, start, (t - start) + tail, out)
            n, idx, diff, margin, w, h = verify(out, item["jpeg"], search_all=(item["t_source"] == "sheet"))
            gt = cv2.imread(item["jpeg"])
            row.update(timestamp_sec=f"{t:.4f}", clip_start_sec=f"{start:.4f}",
                       lead_actual_s=f"{t - start:.4f}", fps=f"{fps:.4f}",
                       n_frames=n, target_idx=idx, match_diff=f"{diff:.4f}",
                       match_margin=f"{margin:.4f}", native_w=w, native_h=h,
                       jpeg_w=gt.shape[1], jpeg_h=gt.shape[0],
                       clip_bytes=out.stat().st_size,
                       status="ok" if diff <= MATCH_DIFF_MAX else "unverified")
            return row
        except Exception as e:                                  # noqa: BLE001 - row carries the reason
            last_err = f"{type(e).__name__}: {e}"[:200]
            time.sleep(2 * (attempt + 1))

    if out.exists():
        out.unlink()
    row["status"] = f"failed: {last_err}"
    return row


# === Main ===

def load_manifest():
    if not MANIFEST.exists():
        return {}
    with open(MANIFEST) as f:
        return {r["key"]: r for r in csv.DictReader(f)}


def main():

    WORK.mkdir(parents=True, exist_ok=True)
    if UPLOAD:
        subprocess.run(["gsutil", "-m", "rsync", "-r", str(CLIPS_DIR), f"{GCS_ROOT}/clips"], check=True)
        subprocess.run(["gsutil", "-q", "cp", str(MANIFEST), f"{GCS_ROOT}/manifest.csv"], check=True)
        print(f"uploaded to {GCS_ROOT}")
        return

    print("Preparing inputs ...")
    fetch_inputs()
    sync_jpegs()
    items, skipped = build_worklist()
    print(f"worklist       : {len(items)} frames  (skipped {skipped})")

    done = load_manifest()
    pending = []
    for it in items:
        prev = done.get(it["key"])
        if prev is None:
            pending.append(it)
        elif not prev["status"].startswith("ok") and RETRY_FAILED:
            pending.append(it)
        elif prev["status"].startswith("ok") and not (
                CLIPS_DIR / it["annotator"] / it["video_id"] / f"frame_{it['frame_id']:05d}.mp4").exists():
            pending.append(it)                                  # manifest says ok but the clip is gone
    if LIMIT:
        pending = pending[:LIMIT]
    print(f"already done   : {sum(1 for r in done.values() if r['status'].startswith('ok'))}")
    print(f"to build       : {len(pending)}  (lead {LEAD_SECONDS}s, {WORKERS} workers)")
    if not pending:
        return

    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    is_new = not MANIFEST.exists()
    t0 = time.time()
    ok = bad = 0
    with open(MANIFEST, "a", newline="") as mf:
        w = csv.DictWriter(mf, fieldnames=MANIFEST_FIELDS)
        if is_new:
            w.writeheader()
            mf.flush()
        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            futs = {pool.submit(build_one, it, LEAD_SECONDS): it for it in pending}
            for i, fut in enumerate(as_completed(futs), 1):
                row = fut.result()
                w.writerow(row)
                mf.flush()                                      # a crash must not lose finished work
                if row["status"].startswith("ok"):
                    ok += 1
                else:
                    bad += 1
                    print(f"  [{row['status'][:70]}] {row['key']}")
                if i % 25 == 0 or i == len(pending):
                    el = time.time() - t0
                    rate = i / el
                    print(f"  {i}/{len(pending)}  ok={ok} bad={bad}  "
                          f"{rate*60:.0f} clips/min  eta {(len(pending)-i)/rate/60:.0f} min")

    total_mb = sum(p.stat().st_size for p in CLIPS_DIR.rglob("*.mp4")) / 1e6
    print(f"\ndone           : ok={ok} bad={bad}  stored {total_mb/1000:.2f} GB in {(time.time()-t0)/60:.1f} min")
    print(f"manifest       : {MANIFEST}")
    print(f"next           : python3 {Path(__file__).name} --upload")


if __name__ == "__main__":
    main()
