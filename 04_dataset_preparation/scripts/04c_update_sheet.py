"""Push sampling plan rows to the Annotation Tracker Google Sheet."""
import csv
import json
import subprocess
import logging

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

# === Config ===
SPREADSHEET_ID = "" # ADD
SHEET_NAME = "Annotation Tracker"
GCS_FRAME_PREFIX = "frames"
SERVICE_ACCOUNT_KEY = "" # ADD
PLAN_CSV = "sampling_plan.csv"
VIDEO_DIR = "./lemon_data"
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

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


# === FFmpeg ===
def get_duration(video_path):
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return -1
    return float(json.loads(result.stdout)["format"]["duration"])


def secs_to_timestamp(s):
    h, m, s = int(s // 3600), int((s % 3600) // 60), int(s % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# === Main ===
def main():
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_KEY, scopes=SCOPES)
    sheet = build("sheets", "v4", credentials=creds).spreadsheets()

    # 1. Read sampling plan
    with open(PLAN_CSV) as f:
        frames = list(csv.DictReader(f))
    log.info(f"Read {len(frames)} frames from {PLAN_CSV}")

    # 2. Get video durations
    video_ids = set(f["youtubeId"] for f in frames)
    log.info(f"Getting durations for {len(video_ids)} videos...")
    duration_cache = {}
    for i, vid in enumerate(video_ids):
        duration_cache[vid] = get_duration(f"{VIDEO_DIR}/{vid}.mp4")
        if (i + 1) % 100 == 0:
            log.info(f"  ffprobe progress: {i + 1}/{len(video_ids)}")
    failed = sum(1 for d in duration_cache.values() if d <= 0)
    log.info(f"Got durations: {len(video_ids) - failed} ok, {failed} failed")

    # 3. Build rows
    rows = []
    for frame in frames:
        vid = frame["youtubeId"]
        annotator = frame["annotator"]
        frame_id = frame["frame_id"]
        robotic = frame["robotic"] == "True"
        file_name = f"{GCS_FRAME_PREFIX}/{annotator}/{vid}/frame_{int(frame_id):05d}.jpg"
        pos = float(frame["normalized_position"])
        dur = duration_cache.get(vid, -1)
        timestamp = secs_to_timestamp(pos * dur) if dur > 0 else ""  # blank if ffprobe failed

        rows.append([
            int(frame_id),                        # A: Frame ID
            vid,                                  # B: Video ID
            file_name,                            # C: File Name
            timestamp,                            # D: Timestamp
            frame["procedureName"],               # E: Surgery Type
            frame["resolution"],                  # F: Video Resolution
            robotic,                              # G: Robotic (checkbox)
            annotator,                            # H: Annotator
            "Not Started",                        # I: Status
            "",                                   # J: Challenges
            False,                                # K: Tools Visible
            "",                                   # L: Notes
        ])

    # 4. Append to sheet
    result = sheet.values().append(
        spreadsheetId=SPREADSHEET_ID,
        range=f"'{SHEET_NAME}'!A2",
        valueInputOption="USER_ENTERED",
        insertDataOption="INSERT_ROWS",
        body={"values": rows},
    ).execute()
    updated = result.get("updates", {}).get("updatedRows", 0)
    log.info(f"Wrote {updated} rows to Google Sheet")


if __name__ == "__main__":
    main()
