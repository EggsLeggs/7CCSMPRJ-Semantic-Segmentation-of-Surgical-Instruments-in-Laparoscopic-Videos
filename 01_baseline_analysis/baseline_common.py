"""Shared constants and functions for the baseline strata analysis notebooks."""

from pathlib import Path
from typing import Union
from urllib.request import urlretrieve
import re
import subprocess
import warnings

import numpy as np
import pandas as pd
from IPython.display import display

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 240)
pd.set_option("display.width", 180)
pd.set_option("display.max_colwidth", 120)

# === Figure palette ===
# One scheme across the dissertation: red carries the series the figure is about, and every other
# series takes a distinct hue rather than a second red, because red-on-red does not separate in
# print. Kept in step with the same block in 02_results_comparison/compare_common.py.
PRIMARY = "red"
SECONDARY = "#6c757d"
RULE = "#333333"
ACCENT = "#2a9d8f"
ACCENT_WARM = "#f4a261"
SERIES = [PRIMARY, "#457b9d", ACCENT_WARM, ACCENT, "#9c6ade", "#c45c26", "#1d3557"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
SEQUENTIAL_CMAP = "Reds"

# === Evaluation ===
MASK_PIXELS = 512 * 512   # masks were evaluated at 512 x 512 in the training notebook

METRICS = ["iou", "dice", "precision", "hd95", "hd", "msd", "asd", "tool_pixels", "fp_pixels"]
META_COLS = ["Video ID", "Surgery Type", "Video Resolution", "Robotic", "Annotator", "Challenges", "No Tools", "Timestamp"]
HIGHER_IS_BETTER = {"iou", "dice", "precision"}
LOWER_IS_BETTER = {"hd95", "hd", "msd", "asd", "fp_pixels"}

# === Group size thresholds ===
MIN_GROUP_FRAMES = 25   # frame-level groups smaller than this are excluded from ranked breakdowns
MIN_GROUP_VIDEOS = 5    # video-level groups smaller than this are excluded from ranked breakdowns

# === Size class labels ===
# Frame-level tool-area cuts, matching 00_model_training size_cls / SIZE_BUCKETS:
# tiny < 4.5%, small < 8%, medium < 19%, large >= 19%
SIZE_BAND_EDGES = (4.5, 8.0, 19.0)
SIZE_LABELS = {
    0: "0_none",
    1: "1_tiny",
    2: "2_small",
    3: "3_medium",
    4: "4_large",
}


def size_band_from_area_pct(area_pct: pd.Series) -> pd.Series:
    """Map tool_area_pct (%) to 0..4 using the training notebook cuts."""
    pct = pd.to_numeric(area_pct, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    out = np.zeros(len(pct), dtype=int)
    lo, mid, hi = SIZE_BAND_EDGES
    out[(pct > 0) & (pct < lo)] = 1
    out[(pct >= lo) & (pct < mid)] = 2
    out[(pct >= mid) & (pct < hi)] = 3
    out[pct >= hi] = 4
    return pd.Series(out, index=area_pct.index)

# === Thinness class labels ===
# Frame-level thin/wide split, from the mean local width the training pipeline measured with a
# distance transform. The cut is the 25th percentile of that distribution, stored per run as
# thin_cut_px (22.81 px at 512 x 512). Used as scored rather than recomputed: 03b measures the
# same idea after a nearest-neighbour resize and lands about 5% higher.
THIN_LABELS = {
    "none": "0_none",
    "thin": "1_thin",
    "wide": "2_wide",
}


# === Failure flag columns ===
FAILURE_FLAG_COLS = [
    "failure_severe_iou", "failure_weak_iou", "failure_boundary_outlier",
    "failure_low_precision", "failure_fp_outlier", "failure_no_tool_fp",
]

NO_TOOLS_UNRECONCILED = "No Tools (sheet, unreconciled)"  # training export name for the sheet flag


def _gsutil(*args: str) -> None:
    r = subprocess.run(["gsutil", *args], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError((r.stderr or r.stdout or "gsutil failed").strip())


def _run_cache(run_dir: str, cache_dir: Path) -> Path:
    """Per-run subdirectory so switching RUN_NAME/RUN_ID cannot reuse another arm's CSV."""
    dest = Path(cache_dir).joinpath(*Path(run_dir.rstrip("/")).parts[-2:])
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def fetch_per_image_results(run_dir: str, cache_dir: Path) -> Path:
    """Download {run_dir}/per_image_results.csv and fix the No Tools column name."""
    cache_dir = _run_cache(run_dir, cache_dir)
    raw = cache_dir / "per_image_results.csv"
    if not raw.exists():
        _gsutil("-q", "cp", f"{run_dir.rstrip('/')}/per_image_results.csv", str(raw))
    df = pd.read_csv(raw)
    df.columns = df.columns.str.strip()
    if "No Tools" not in df.columns and NO_TOOLS_UNRECONCILED in df.columns:
        df = df.rename(columns={NO_TOOLS_UNRECONCILED: "No Tools"})
    out = cache_dir / "per_image_results.normalized.csv"
    df.to_csv(out, index=False)
    return out


def fetch_results_json(run_dir: str, cache_dir: Path) -> Path:
    """Download {run_dir}/results.json, the fold means and the run's own configuration."""
    dest = _run_cache(run_dir, cache_dir) / "results.json"
    if not dest.exists():
        _gsutil("-q", "cp", f"{run_dir.rstrip('/')}/results.json", str(dest))
    return dest


def fetch_annotations(sheet_id: str, cache_dir: Path) -> Path:
    """Download the annotator spreadsheet CSV export into cache_dir."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / "annotations.csv"
    if not dest.exists():
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
        urlretrieve(url, dest)
    return dest


def fetch_predicted_masks(run_dir: str, cache_dir: Path) -> Path:
    """Rsync {run_dir}/PredictedMasks into cache_dir (01b gallery)."""
    dest = _run_cache(run_dir, cache_dir) / "PredictedMasks"
    if not dest.exists() or not any(dest.iterdir()):
        dest.mkdir(parents=True, exist_ok=True)
        _gsutil("-m", "-q", "rsync", "-r", f"{run_dir.rstrip('/')}/PredictedMasks", str(dest))
    return dest


def fetch_frame_data(frames_gcs: str, cache_dir: Path) -> Path:
    """Rsync JPEGImages + SegmentationClass from frames_gcs into cache_dir (large)."""
    cache_dir = Path(cache_dir)
    dest = cache_dir / "lap_data"
    root = frames_gcs.rstrip("/")
    for sub in ("JPEGImages", "SegmentationClass"):
        subdir = dest / sub
        if not subdir.exists() or not any(subdir.iterdir()):
            subdir.mkdir(parents=True, exist_ok=True)
            _gsutil("-m", "-q", "rsync", "-r", f"{root}/{sub}", str(subdir))
    return dest


# === Cleaning helpers ===

def key_from_path(value: str) -> Union[str, float]:
    """Builds a join key from an image path, e.g. video03/frame_0142."""
    if pd.isna(value):
        return np.nan
    parts = Path(str(value)).parts
    if len(parts) >= 2:
        return f"{parts[-2]}/{Path(parts[-1]).stem}"
    return Path(str(value)).stem


def norm_bool(value) -> Union[bool, float]:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return np.nan


def clean_label(value, empty: str = "Unknown") -> str:
    if pd.isna(value):
        return empty
    text = str(value).strip()
    return text if text else empty


def split_multi(value) -> list:
    # challenge labels can be comma, semicolon, slash or pipe separated
    text = clean_label(value, empty="")
    if not text:
        return []
    return [part.strip() for part in re.split(r"[,;/|]+", text) if part.strip()]


def collapse_rare(series: pd.Series, top_n: int = 20, min_count: int = 20, other_label: str = "Other") -> pd.Series:
    """Buckets low-frequency categories into 'Other' so tables/charts stay readable."""
    cleaned = series.map(lambda x: clean_label(x))
    counts = cleaned.value_counts(dropna=False)
    keep = counts[counts >= min_count].head(top_n).index
    return cleaned.where(cleaned.isin(keep), other_label)


def add_mean_ci(table: pd.DataFrame, value_col: str, std_col: str, n_col: str, prefix: str) -> pd.DataFrame:
    # normal approximation, fine at these sample sizes
    n = table[n_col].replace(0, np.nan)
    se = table[std_col] / np.sqrt(n)
    table[f"{prefix}_ci_low"] = table[value_col] - 1.96 * se
    table[f"{prefix}_ci_high"] = table[value_col] + 1.96 * se
    return table


def summarize_groups(df: pd.DataFrame, group_col, min_rows: int = MIN_GROUP_FRAMES, min_videos: int = 1, sort: bool = True) -> pd.DataFrame:
    """One-factor summary table for group_col: mean/median IoU, failure rates,
    95% CI. Groups below min_rows frames or min_videos videos are dropped,
    so sparse strata don't get over-read. Sorted worst-first by default.
    """
    group_cols = [group_col] if isinstance(group_col, str) else list(group_col)
    data = df.copy()
    for col in group_cols:
        data[col] = data[col].map(lambda x: clean_label(x))

    grouped = data.groupby(group_cols, dropna=False)
    out = grouped.agg(
        n_frames=("key", "size"),
        n_videos=("Video ID", "nunique"),
        iou_mean=("iou", "mean"),
        iou_std=("iou", "std"),
        iou_median=("iou", "median"),
        dice_mean=("dice", "mean"),
        precision_mean=("precision", "mean"),
        hd95_mean=("hd95", "mean"),
        msd_mean=("msd", "mean"),
        asd_mean=("asd", "mean"),
        tool_pixels_mean=("tool_pixels", "mean"),
        fp_pixels_mean=("fp_pixels", "mean"),
        fp_area_pct_mean=("fp_area_pct", "mean"),
        failure_any_rate=("failure_any", "mean"),
        severe_iou_rate=("failure_severe_iou", "mean"),
        weak_iou_rate=("failure_weak_iou", "mean"),
        boundary_outlier_rate=("failure_boundary_outlier", "mean"),
        precision_low_rate=("failure_low_precision", "mean"),
        fp_outlier_rate=("failure_fp_outlier", "mean"),
        no_tool_fp_rate=("failure_no_tool_fp", "mean"),
        failure_score_mean=("failure_score", "mean"),
    ).reset_index()

    out = out[(out["n_frames"] >= min_rows) & (out["n_videos"] >= min_videos)].copy()
    out["delta_iou_vs_global"] = out["iou_mean"] - data["iou"].mean()
    out["delta_failure_vs_global"] = out["failure_any_rate"] - data["failure_any"].mean()
    out = add_mean_ci(out, "iou_mean", "iou_std", "n_frames", "iou_mean")
    if sort:
        out = out.sort_values(["failure_any_rate", "iou_mean", "n_frames"], ascending=[False, True, False])
    return out


def display_and_save(table: pd.DataFrame, name: str, output_dir: Path, n: int = 30) -> None:
    path = output_dir / f"{name}.csv"
    table.to_csv(path, index=False)
    print(f"Saved {path}")
    display(table.head(n))


# === Load and pair ===

def load_and_pair(results_path: Path, annotations_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reads the results and annotation CSVs, validates required columns, and
    joins them on frame key. Only "Completed" annotations are joined; the
    rest are returned separately as invalid_annotations, not discarded.
    Returns (frame_df, metadata_mismatches, invalid_annotations).
    """
    results = pd.read_csv(results_path)
    results.columns = results.columns.str.strip()
    if "No Tools" not in results.columns and NO_TOOLS_UNRECONCILED in results.columns:
        results = results.rename(columns={NO_TOOLS_UNRECONCILED: "No Tools"})
    annotations_raw = pd.read_csv(annotations_path)
    annotations_raw.columns = annotations_raw.columns.str.strip()

    required_results = {"key", "iou", "dice", "precision", "hd95", "msd", "asd", "tool_pixels", "fp_pixels", *META_COLS}
    missing_results = sorted(required_results - set(results.columns))
    if missing_results:
        raise ValueError(f"Missing required result columns: {missing_results}")

    for col in ["Robotic", "No Tools"]:
        results[col] = results[col].map(norm_bool)
        if col in annotations_raw.columns:
            annotations_raw[col] = annotations_raw[col].map(norm_bool)

    for col in METRICS:
        if col in results.columns:
            results[col] = pd.to_numeric(results[col], errors="coerce")

    # only "Completed" annotations are trustworthy metadata - in-progress rows
    # are excluded from the join, not just filtered afterwards
    annotations_raw["key"] = annotations_raw["File Name"].map(key_from_path)
    annotations_completed = annotations_raw[annotations_raw["Status"].fillna("").eq("Completed")].copy()
    annotation_dupes = annotations_completed["key"].duplicated().sum()
    annotations_join = annotations_completed.drop_duplicates("key", keep="last").copy()

    join_cols = ["key", "Status", "Notes"]
    for col in META_COLS:
        if col in annotations_join.columns:
            annotations_join[f"{col}_annotation"] = annotations_join[col]
            join_cols.append(f"{col}_annotation")

    frame_df = results.merge(annotations_join[join_cols], on="key", how="left", validate="m:1")
    frame_df["annotation_matched"] = frame_df["Status"].notna()

    for col in META_COLS:
        if col in frame_df.columns and frame_df[col].dtype == object:
            frame_df[col] = frame_df[col].map(lambda x: clean_label(x, empty=""))

    # counts, per column, how many rows disagree between the two sources - informational only, doesn't filter or raise
    mismatch_rows = []
    for col in META_COLS:
        ann_col = f"{col}_annotation"
        if ann_col in frame_df.columns:
            left = frame_df[col].map(lambda x: clean_label(x, empty="")).astype(str).str.lower()
            right = frame_df[ann_col].map(lambda x: clean_label(x, empty="")).astype(str).str.lower()
            comparable = frame_df[ann_col].notna()
            mismatch_rows.append({"column": col, "mismatches": int((comparable & (left != right)).sum())})
    metadata_mismatches = pd.DataFrame(mismatch_rows)

    invalid_annotations = annotations_raw[annotations_raw["Status"].fillna("").ne("Completed")].copy()

    print(f"Results rows: {len(results):,}")
    print(f"Raw annotation rows: {len(annotations_raw):,}")
    print(f"Completed annotation rows: {len(annotations_completed):,}")
    print(f"Completed annotation duplicate keys dropped before join: {annotation_dupes:,}")
    print(f"Paired rows: {len(frame_df):,}")
    print(f"Annotation match rate for result rows: {frame_df['annotation_matched'].mean():.3f}")

    return frame_df, metadata_mismatches, invalid_annotations


# === Derived columns ===

def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Adds cleaned booleans, area/ratio metrics, parsed challenge and resolution
    fields, and the frame-level size and thinness classes to df.
    """
    df = df.copy()

    # Misc - normalisation of the spreadsheet and guards against annotation errors
    df["No Tools"] = df["No Tools"].map(norm_bool)
    df["Robotic"] = df["Robotic"].map(norm_bool)
    df["no_tools_bool"] = df["No Tools"].fillna(False).astype(bool)
    df["has_tool_pixels"] = df["tool_pixels"].fillna(0).gt(0)
    df["has_tool"] = (~df["no_tools_bool"]) & df["has_tool_pixels"]
    df["tool_area_pct"] = 100 * df["tool_pixels"] / MASK_PIXELS
    df["fp_area_pct"] = 100 * df["fp_pixels"] / MASK_PIXELS
    df["fp_to_tool_ratio"] = df["fp_pixels"] / df["tool_pixels"].replace(0, np.nan)
    df["pred_fp_share"] = df["fp_pixels"] / (df["fp_pixels"] + df["tool_pixels"]).replace(0, np.nan)

    # Challenges - splits challenge strings into arrays
    df["challenge_tokens"] = df["Challenges"].map(split_multi)
    df["has_challenge"] = df["challenge_tokens"].map(bool)
    df["challenge_combo"] = df["challenge_tokens"].map(lambda xs: " | ".join(sorted(xs)) if xs else "None")

    # Resolutions - regex for resolution strings (e.g., 1920x1080) to derive height, width and ar
    resolution_parts = df["Video Resolution"].map(lambda x: re.match(r"^\s*(\d+)\s*x\s*(\d+)\s*$", str(x)))
    df["resolution_width"] = resolution_parts.map(lambda m: int(m.group(1)) if m else np.nan)
    df["resolution_height"] = resolution_parts.map(lambda m: int(m.group(2)) if m else np.nan)
    df["resolution_pixels"] = df["resolution_width"] * df["resolution_height"]
    df["resolution_aspect"] = df["resolution_width"] / df["resolution_height"]

    # Timestamps - converts to number of seconds
    timestamp = pd.to_timedelta(df["Timestamp"], errors="coerce")
    df["timestamp_seconds"] = timestamp.dt.total_seconds()

    # Size classes
    df["size_cls"] = size_band_from_area_pct(df["tool_area_pct"])
    df["size_cls_label"] = df["size_cls"].map(SIZE_LABELS)

    # Thinness classes - taken from the run, which is the cut every thin number is scored on
    if "thin_class" in df.columns:
        df["thin_cls_label"] = df["thin_class"].map(clean_label).str.lower().map(THIN_LABELS).fillna("0_none")
        df["width_px"] = pd.to_numeric(df.get("width_px"), errors="coerce")
    return df


# === Failure flags ===

def _failure_tags_row(row):
    tags = [col.replace("failure_", "") for col in FAILURE_FLAG_COLS if bool(row[col])]
    return ", ".join(tags) if tags else "none"


def add_failure_flags(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Adds per-frame failure flags (boolean), a combined failure_any/failure_score,
    and a human-readable failure_tags string to df.
    Returns (df, cutoff_table, failure_thresholds) where cutoff_table records the
    formatted threshold used for each flag, and failure_thresholds holds the raw
    numeric values for reuse (e.g. plotting threshold lines).
    """# mostly data-relative percentiles, except the 0.50 IoU cutoff, which is absolute
    weak_iou_threshold = df["iou"].quantile(0.10)
    boundary_threshold = df["hd95"].quantile(0.90)
    precision_threshold = df["precision"].quantile(0.10)
    fp_threshold = df["fp_pixels"].quantile(0.90)
    fp_share_threshold = df["pred_fp_share"].quantile(0.90)
    notool_fp_values = df.loc[df["no_tools_bool"], "fp_pixels"].dropna()
    notool_fp_threshold = max(100.0, float(notool_fp_values.quantile(0.75))) if len(notool_fp_values) else max(100.0, float(fp_threshold))

    df["failure_severe_iou"] = df["iou"].le(0.50)
    df["failure_weak_iou"] = df["iou"].le(weak_iou_threshold)
    df["failure_boundary_outlier"] = df["hd95"].ge(boundary_threshold)
    df["failure_low_precision"] = df["precision"].le(precision_threshold)
    df["failure_fp_outlier"] = df["fp_pixels"].ge(fp_threshold) | df["pred_fp_share"].ge(fp_share_threshold)
    df["failure_no_tool_fp"] = df["no_tools_bool"] & df["fp_pixels"].ge(notool_fp_threshold)

    # failures are multi-label - a frame can trip more than one flag at once
    df["failure_any"] = df[FAILURE_FLAG_COLS].any(axis=1)
    df["failure_score"] = df[FAILURE_FLAG_COLS].sum(axis=1)
    df["failure_tags"] = df.apply(_failure_tags_row, axis=1)

    cutoff_table = pd.DataFrame([
        {"flag": "failure_severe_iou", "threshold": "iou <= 0.50"},
        {"flag": "failure_weak_iou", "threshold": f"iou <= p10 = {weak_iou_threshold:.4f}"},
        {"flag": "failure_boundary_outlier", "threshold": f"hd95 >= p90 = {boundary_threshold:.2f}"},
        {"flag": "failure_low_precision", "threshold": f"precision <= p10 = {precision_threshold:.4f}"},
        {"flag": "failure_fp_outlier", "threshold": f"fp_pixels >= p90 = {fp_threshold:.1f} or fp share >= p90 = {fp_share_threshold:.3f}"},
        {"flag": "failure_no_tool_fp", "threshold": f"No Tools and fp_pixels >= {notool_fp_threshold:.1f}"},
    ])
    failure_thresholds = {
        "weak_iou": weak_iou_threshold,
        "boundary": boundary_threshold,
        "precision": precision_threshold,
    }
    return df, cutoff_table, failure_thresholds