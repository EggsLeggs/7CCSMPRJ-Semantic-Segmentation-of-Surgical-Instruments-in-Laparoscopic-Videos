"""Sample LEMON videos into annotation frames, stratified by procedure, robotic, and resolution."""
import json
import os
import csv
import math
import random
import subprocess
import logging
from collections import defaultdict

# === Config ===
ANNOTATORS = [] # ADD
FRAMES_PER_ANNOTATOR = 1000
TOTAL_FRAMES = FRAMES_PER_ANNOTATOR * len(ANNOTATORS) 
TEMPORAL_QUARTILES = 4   # (early, mid-early, mid-late, late)
JITTER_PCT = 0.02
RANDOM_SEED = 42 # 42 was used in the dissertation
LABELS_JSON = "LEMON_labels.json"
VIDEO_DIR = "./lemon_data"
OUTPUT_DIR = "./extracted_frames"
PLAN_CSV = "sampling_plan.csv"
DRY_RUN = False  # True writes the csv only, no ffmpeg

# === Logging ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("lemon_sampler.log"),
    ],
)
log = logging.getLogger(__name__)


# === Sampling ===
def sample_frames_from_stratum(videos: list[dict], n_frames: int, rng: random.Random) -> list[dict]:
    # 1. Pick videos
    n_videos_needed = math.ceil(n_frames / TEMPORAL_QUARTILES)
    n_videos_needed = min(n_videos_needed, len(videos))
    selected_videos = rng.sample(videos, n_videos_needed)
    base_per_video = n_frames // n_videos_needed
    extra = n_frames % n_videos_needed
    sampled_frames = []
    for i, video in enumerate(selected_videos):
        frames_for_this = base_per_video + (1 if i < extra else 0) # normalized pos 0-1, we dont have frame counts yet

        # 2. Place each frame in a temporal quartile, with jitter
        positions = []
        for j in range(frames_for_this):
            quartile = j % TEMPORAL_QUARTILES
            quartile_start = quartile / TEMPORAL_QUARTILES
            quartile_end = (quartile + 1) / TEMPORAL_QUARTILES

            base_pos = quartile_start + (quartile_end - quartile_start) * ((j // TEMPORAL_QUARTILES + 0.5) / max(1, math.ceil(frames_for_this / TEMPORAL_QUARTILES)))
            jitter = rng.uniform(-JITTER_PCT, JITTER_PCT)
            pos = max(0.01, min(0.99, base_pos + jitter))  # clamp so not exactly 0/1
            positions.append(pos)

        for pos in sorted(positions):
            sampled_frames.append({
                "youtubeId": video["youtubeId"],
                "procedureName": video["procedureName"],
                "robotic": video["robotic"],
                "resolution": video["resolution"],
                "title": video["title"],
                "normalized_position": pos,
            })

    return sampled_frames


# === FFmpeg ===
def get_video_duration(video_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


def extract_frame(video_path: str, timestamp_sec: float, output_path: str) -> bool:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{timestamp_sec:.3f}",
        "-i", str(video_path),
        "-frames:v", "1",
        "-q:v", "2",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


# === Main ===
def main():
    rng = random.Random(RANDOM_SEED)

    # 1. Load labels
    log.info(f"Loading labels from {LABELS_JSON}...")
    with open(LABELS_JSON) as f:
        labels = json.load(f)
    log.info(f"Loaded {len(labels)} videos")

    # 2. Stratify by procedure, robotic, resolution
    log.info(f"Building strata from {len(labels)} videos...")
    strata = defaultdict(list)
    for video in labels:
        primary_proc = video["procedureName"][0] # basis strata on first procedure only
        w, h = video["resolution"]
        res_class = "1280x720" if w == 1280 and h == 720 else "other"
        key = (primary_proc, video["robotic"], res_class)
        strata[key].append(video)
    log.info(f"Found {len(strata)} unique strata")

    # 3. Allocate frames
    log.info(f"Calculating allocation for {TOTAL_FRAMES} frames...")
    total_videos = sum(len(v) for v in strata.values())
    raw_alloc = {key: (len(videos) / total_videos) * TOTAL_FRAMES for key, videos in strata.items()}
    floor_alloc = {k: max(1, int(v)) for k, v in raw_alloc.items()}
    allocated = sum(floor_alloc.values())
    remainder = {k: raw_alloc[k] - floor_alloc[k] for k in raw_alloc}
    deficit = TOTAL_FRAMES - allocated
    if deficit > 0:
        sorted_by_remainder = sorted(remainder, key=remainder.get, reverse=True)
        for i in range(deficit):
            floor_alloc[sorted_by_remainder[i]] += 1
    elif deficit < 0:
        # over-allocated from min=1, trim from biggest
        sorted_by_size = sorted(floor_alloc, key=floor_alloc.get, reverse=True)
        for i in range(-deficit):
            floor_alloc[sorted_by_size[i]] -= 1
    allocation = floor_alloc

    log.info("Stratum allocation:")
    for key in sorted(allocation, key=allocation.get, reverse=True)[:15]:
        proc, robotic, res = key
        log.info(f"  {proc} | {'robotic' if robotic else 'laparoscopic'} | {res}: "
                 f"{len(strata[key])} videos → {allocation[key]} frames")
    if len(allocation) > 15:
        log.info(f"  ... and {len(allocation) - 15} more strata")

    # 4. Sample frames from each stratum
    all_frames = []
    for key, n_frames in allocation.items():
        all_frames.extend(sample_frames_from_stratum(strata[key], n_frames, rng))
    log.info(f"Sampled {len(all_frames)} frames total")

    # 5. Shuffle and assign annotators
    rng.shuffle(all_frames)
    for i, frame in enumerate(all_frames):
        frame["annotator"] = ANNOTATORS[i % len(ANNOTATORS)]
    all_frames.sort(key=lambda f: (f["annotator"], f["youtubeId"], f["normalized_position"]))
    for i, frame in enumerate(all_frames):
        frame["frame_id"] = i + 1

    # 6. Write sampling plan
    fieldnames = [
        "frame_id", "youtubeId", "procedureName", "robotic", "resolution",
        "normalized_position", "annotator", "title",
    ]
    with open(PLAN_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for frame in all_frames:
            row = {**frame}
            row["procedureName"] = ", ".join(row["procedureName"])
            row["resolution"] = f"{row['resolution'][0]}x{row['resolution'][1]}"
            writer.writerow(row)
    log.info(f"Wrote to {PLAN_CSV}")

    if DRY_RUN:
        log.info(f"Dry run done. Check {PLAN_CSV}")
        return

    # 7. Extract frames
    log.info(f"Extracting {len(all_frames)} frames from videos in {VIDEO_DIR}...")
    duration_cache = {}
    successful = 0
    failed = 0
    for i, frame in enumerate(all_frames):
        vid = frame["youtubeId"]
        video_path = os.path.join(VIDEO_DIR, f"{vid}.mp4")
        if vid not in duration_cache:
            duration_cache[vid] = get_video_duration(video_path)
        timestamp_sec = frame["normalized_position"] * duration_cache[vid]
        local_path = os.path.join(
            OUTPUT_DIR, frame["annotator"], vid, f"frame_{frame['frame_id']:05d}.jpg"
        )
        if extract_frame(video_path, timestamp_sec, local_path):
            successful += 1
        else:
            failed += 1
        if (i + 1) % 100 == 0:
            log.info(f"  Progress: {i + 1}/{len(all_frames)} ({failed} failed)")
    log.info(f"Extracted {successful}, {failed} failed")
    log.info(f"Full run done. Check {PLAN_CSV} and frames in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
