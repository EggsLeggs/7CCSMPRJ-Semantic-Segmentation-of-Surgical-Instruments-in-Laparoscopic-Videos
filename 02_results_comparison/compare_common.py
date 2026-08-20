"""Compare two scored runs: alignment, paired statistics, tables and figures.

Fetching and the strata vocabulary come from 01_baseline_analysis/baseline_common.py.
"""
import json
import sys
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "01_baseline_analysis"))
from baseline_common import (
    LOWER_IS_BETTER, MASK_PIXELS, SIZE_BAND_EDGES,
    display_and_save, fetch_per_image_results, fetch_results_json, split_multi,
)

LOWER_IS_BETTER = LOWER_IS_BETTER | {"fn_pixels", "miss"}
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")


# === Config ===
BUCKET = "gs://training-checkpoints-REPLACE_K_NUMBER"
EXPERIMENTS = f"{BUCKET}/experiments"
SHEET_ID = "REPLACE_SHEET_ID"
CACHE_DIR = Path.home() / ".cache" / "laparoscopic_seg" / "comparison"
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"

PRIMARY_THRESHOLD = 0.50  # the operating point results.json reports
MIN_STRATUM = 25          # below this, report the stratum but do not read it as evidence
N_BOOT = 2000
BOOT_SEED = 0
EPS = 1e-3                # smaller per-frame change than this is a tie
DECIMALS = 4              # dissertation quotes four decimal places

PRIMARY = "red"           # matching the 03a / 03b histograms
SECONDARY = "#6c757d"     # the reference that series is read against
RULE = "#333333"          # zero lines and thresholds
ARM_SERIES = [PRIMARY, "#457b9d", "#f4a261", "#2a9d8f", "#9c6ade", "#c45c26", "#1d3557"]
ARM_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]

BASELINE = "baseline-seed42-clean"
FT_SEED = "baseline-seed2026"  # every fine-tune and the cascade were seeded from this one


# === Runs ===
@dataclass(frozen=True)
class Run:
    path: str
    label: str
    family: str
    varies: str
    reference: str = ""
    weights_from: str = ""
    erroneous: str = ""


RUNS = {
    BASELINE: Run(
        "baseline-seed42-clean_cv_Unet_resnet34_bce_lr1e-04_p20/20260813-232105",
        "Baseline (seed 42)", "reference", "reference run"),
    FT_SEED: Run(
        "baseline-seed2026_cv_Unet_resnet34_bce_lr1e-04_p20/20260812-205137",
        "Baseline (seed 2026)", "reference", "SEED 42 to 2026", reference=BASELINE),
    "bce_tversky": Run(
        "tversky2_cv_balsweep_Unet_resnet34_bce_tversky_lr1e-04_p20/20260808-013552",
        "BCE-Tversky", "loss", "LOSS bce to bce_tversky", reference=BASELINE,
        erroneous="threshold grid stops at 0.70, and this run's precision does not reach the "
                  "baseline's until 0.805, so matched precision cannot be read on it. "
                  "Superseded by tversky-rescore-hi, which re-scored the same weights."),
    "tversky-rescore-hi": Run(
        "tversky-rescore-hi_cv_Unet_resnet34_bce_tversky_lr1e-04_p20/20260814-104220",
        "BCE-Tversky (grid to 0.95)", "loss", "threshold grid extended",
        reference=BASELINE, weights_from="bce_tversky"),
    "bce_dice": Run(
        "bce-dice_cv_Unet_resnet34_bce_dice_lr1e-04_p20/20260808-123132",
        "BCE-Dice", "loss", "LOSS bce to bce_dice", reference=BASELINE),
    "synaug_low": Run(
        "synthetic-augmentation-low-cv_cv_synaug-low_Unet_resnet34_bce_lr1e-04_p20/20260809-213637",
        "Synthetic aug, safe", "augmentation", "augmentation preset, rebuilt recipe",
        reference=BASELINE,
        erroneous="confounded: the preset builder rebuilt the recipe instead of extending it, "
                  "so this run also trained with less brightness and elastic augmentation than "
                  "the baseline. Superseded by synaug-fixed."),
    "synaug_fatal": Run(
        "synthetic-augmentation-fatal-cv_cv_synaug-fatal_Unet_resnet34_bce_lr1e-04_p20/20260809-213800",
        "Synthetic aug, extreme", "augmentation", "augmentation preset, rebuilt recipe",
        reference=BASELINE,
        erroneous="confounded by the same preset builder defect as synaug_low. "
                  "Superseded by synaug-fixed."),
    "synaug-fixed": Run(
        "synaug-fixed_cv_synaug-low_additive_Unet_resnet34_bce_lr1e-04_p20/20260813-140837",
        "Synthetic aug, additive", "augmentation", "augmentation preset, additive recipe",
        reference=BASELINE),
    "static-tiling": Run(
        "static-tiling-cv_cv_Unet_resnet34_bce_lr1e-04_p20/20260810-010618",
        "Static tiling", "tiling", "native-resolution tile grid", reference=BASELINE),
    "cascade-redetect-clean": Run(
        "cascade-redetect-clean_cv_cascade-fused_redetect_Unet_resnet34_bce_lr1e-04_p20/20260813-230613",
        "Cascade on baseline weights", "cascade", "test-time cascade",
        reference=FT_SEED, weights_from=FT_SEED),
    "cascade-roift-neg-redetect": Run(
        "cascade-roift-neg-redetect_cv_cascade-fused_redetect_Unet_resnet34_bce_lr1e-04_p20/20260815-011358",
        "Cascade on ROI-tuned weights", "cascade", "test-time cascade",
        reference="roift-neg-clean", weights_from="roift-neg-clean"),
    "roift-control-ff": Run(
        "roift-control-ff_cv_roift_Unet_resnet34_bce_lr1e-04_p20/20260814-144749",
        "Fine-tune control, 15 ep", "finetune", "15 epochs at LR/10, no crops",
        reference=FT_SEED, weights_from=FT_SEED),
    "roift-clean": Run(
        "roift-clean_cv_roift_Unet_resnet34_bce_lr1e-04_p20/20260813-233143",
        "ROI fine-tune, 15 ep", "finetune", "ROI_FT_P 0.5, no negatives",
        reference="roift-control-ff", weights_from=FT_SEED),
    "roift-neg-clean": Run(
        "roift-neg-clean_cv_roift_Unet_resnet34_bce_lr1e-04_p20/20260814-105438",
        "ROI fine-tune + negatives 0.3", "finetune", "ROI_FT_NEG_P 0.0 to 0.3",
        reference="roift-control-ff", weights_from=FT_SEED),
    "roift-neg-strong": Run(
        "roift-neg-strong_cv_roift_Unet_resnet34_bce_lr1e-04_p20/20260814-152647",
        "ROI fine-tune + negatives 0.5", "finetune", "ROI_FT_NEG_P 0.0 to 0.5",
        reference="roift-control-ff", weights_from=FT_SEED),
    "roift-small-neg": Run(
        "roift-small-neg_cv_roift_Unet_resnet34_bce_lr1e-04_p20/20260814-143954",
        "ROI fine-tune, size-biased", "finetune", "ROI_FT_SMALL_W 1.0 to 2.0",
        reference="roift-control-ff", weights_from=FT_SEED),
    "roift-tversky": Run(
        "roift-tversky_cv_roift_Unet_resnet34_bce_tversky_lr1e-04_p20/20260814-104911",
        "ROI fine-tune from BCE-Tversky", "finetune", "seed weights and loss",
        reference="roift-control-ff", weights_from="bce_tversky"),
    "ft-long-control": Run(
        "ft-long-control_cv_roift_Unet_resnet34_bce_lr1e-04_p20/20260815-012030",
        "Fine-tune control, 50 ep", "finetune", "15 to 50 epochs, no crops",
        reference="roift-control-ff", weights_from=FT_SEED),
    "roift-long": Run(
        "roift-long_cv_roift_Unet_resnet34_bce_lr1e-04_p20/20260815-012339",
        "ROI fine-tune, 50 ep", "finetune", "15 to 50 epochs, ROI_FT_P 0.5",
        reference="ft-long-control", weights_from=FT_SEED),
    "roift-long-seed2026": Run(
        "roift-long-seed2026_cv_roift_Unet_resnet34_bce_lr1e-04_p20/20260816-004733",
        "ROI fine-tune, 50 ep, repeat seed", "finetune", "SEED only",
        reference="roift-long", weights_from=FT_SEED),
}

# one run per family, for the summary forest; the fine-tunes are read against their own control
SUMMARY_ARMS = [
    "tversky-rescore-hi", "synaug-fixed", "static-tiling",
    "cascade-redetect-clean", "roift-control-ff", "roift-long",
]


def registry(keys=None) -> pd.DataFrame:
    rows = []
    for k in (keys or RUNS):
        spec = RUNS[k]
        rows.append({
            "key": k,
            "label": spec.label,
            "family": spec.family,
            "varies": spec.varies,
            "read against": spec.reference or "-",
            "erroneous": "yes" if spec.erroneous else "",
        })
    return pd.DataFrame(rows)


def run_dir(key: str) -> str:
    if key not in RUNS:
        raise KeyError(f"unknown run {key!r}; declared runs: {sorted(RUNS)}")
    return f"{EXPERIMENTS}/{RUNS[key].path}"


def label(key: str) -> str:
    name = RUNS[key].label
    if RUNS[key].erroneous:
        name += " [ERRONEOUS]"
    return name


_RESULTS = {}


def results(key: str) -> dict:
    if key not in _RESULTS:
        path = fetch_results_json(run_dir(key), CACHE_DIR)
        _RESULTS[key] = json.loads(path.read_text())
    return _RESULTS[key]


def thin_cut(key: str) -> float:
    return float(results(key)["config"]["thin_cut_px"])


def declared_weights_source(meta: dict) -> str:
    # cascade-redetect-clean is logged as a re-score of the clean baseline; its config disagrees
    config = meta.get("config", {})
    for section, key in (("tracking_cascade", "weights_from"), ("roi_finetune", "seeded_from")):
        source = (config.get(section) or {}).get(key)
        if not source:
            continue
        run_id = str(source).rstrip("/").split("/")[-1]
        for name, spec in RUNS.items():
            if spec.path.endswith(run_id):
                return name
        return str(source)
    return ""


def warnings_for(reference: str, arms: list[str]) -> list[str]:
    notes = []
    for key in [reference] + list(arms):
        if RUNS[key].erroneous:
            notes.append(f"ERRONEOUS  {key}: {RUNS[key].erroneous}")
    ref_source = declared_weights_source(results(reference))
    for arm in arms:
        source = declared_weights_source(results(arm))
        if source and source not in (reference, ref_source):
            notes.append(
                f"lineage    {arm}: carries the weights of {source}, not of the "
                f"reference {reference}, so the difference includes whatever "
                f"separates those two checkpoint sets"
            )
    return notes


# === Strata ===
SIZE_ORDER = ["tiny", "small", "med", "large"]  # training cuts: <4.5%, <8%, <19%, >=19%
THIN_ORDER = ["thin", "wide"]                   # cut at each run's stored thin_cut_px
DEGRADATIONS = ["Smoke", "Blood", "Motion Blur", "Underexposure", "Overexposure"]

METRIC_LABEL = {
    "iou": "IoU",
    "dice": "Dice",
    "precision": "Precision",
    "recall": "Recall",
    "hd95": "boundary error (95th percentile, px)",
    "hd": "boundary error (px)",
    "msd": "mean boundary error (px)",
    "asd": "average boundary error (px)",
}


def metric_label(metric: str) -> str:
    return METRIC_LABEL.get(metric, metric.replace("_", " "))


def stratum_label(name: str) -> str:
    if name.startswith("size "):
        return f"{name[5:]} tools"
    if name.startswith("scene "):
        return name[6:].lower()
    return name


# === Loading ===
def stored_thresholds(df: pd.DataFrame) -> list[float]:
    # early grids stop at 0.70; read whatever this run actually stored
    return sorted(float(c[3:]) for c in df.columns if c.startswith("tp@"))


def problematic_keys() -> set[str]:
    # empty until the hand review marks frames; currently excludes nothing
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = CACHE_DIR / "annotation_sheet.csv"
    if not dest.exists():
        url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"
        dest.write_bytes(urllib.request.urlopen(url, timeout=120).read())
    sheet = pd.read_csv(dest)
    marked = sheet["Status"].astype(str).str.lower().str.contains("problematic", na=False)
    keys = (
        sheet["Video ID"].astype(str)
        + "/"
        + sheet["File Name"].astype(str).map(lambda x: Path(str(x)).stem)
    )
    return set(keys[marked])


def load_runs(keys: list[str], exclude_problematic: bool = False):
    # 1. Fetch each run
    raw = {}
    for key in keys:
        df = pd.read_csv(fetch_per_image_results(run_dir(key), CACHE_DIR))
        raw[key] = df.drop_duplicates("key", keep="last").set_index("key")

    # 2. Align on the first run's frame order. A mismatch raises.
    ref = raw[keys[0]]
    order = ref.index.tolist()
    frames = {}
    for key in keys:
        df = raw[key]
        if set(df.index) != set(order):
            raise ValueError(f"{key!r} is not frame-aligned with {keys[0]!r}")
        df = df.reindex(order)
        if not df["fold"].equals(ref["fold"]):
            raise ValueError(f"{key!r} was scored on different folds from {keys[0]!r}")
        for col in ("tool_pixels", "size_class", "thin_class"):
            if not df[col].equals(ref[col]):
                raise ValueError(f"{key!r} disagrees with {keys[0]!r} on ground truth {col!r}")
        df["fp_area_pct"] = 100.0 * df["fp_pixels"] / MASK_PIXELS
        df["miss"] = df["pred_empty"].astype(float)
        frames[key] = df

    cuts = {key: thin_cut(key) for key in keys}
    if len(set(cuts.values())) > 1:
        raise ValueError(f"runs disagree on the thin/wide cut: {cuts}")

    # 3. Shared ground truth, one row per frame
    gt = pd.DataFrame(index=ref.index)
    gt["fold"] = ref["fold"]
    gt["video"] = ref["Video ID"].fillna("unknown")
    gt["size_class"] = ref["size_class"]
    gt["thin_class"] = ref["thin_class"]
    gt["has_tool"] = ~ref["gt_empty"].astype(bool)
    tokens = ref["Challenges"].fillna("").map(split_multi)
    for token in sorted({t for ts in tokens for t in ts}):
        gt[f"ch::{token}"] = tokens.map(lambda ts, t=token: t in ts)
    gt["any_degradation"] = gt[[f"ch::{t}" for t in DEGRADATIONS if f"ch::{t}" in gt]].any(axis=1)
    gt["clean_scene"] = tokens.map(len).eq(0)

    # 4. Drop hand-reviewed annotation errors if asked
    if exclude_problematic:
        dropped = problematic_keys() & set(gt.index)
        keep = ~gt.index.isin(dropped)
        gt = gt[keep]
        frames = {k: df[keep] for k, df in frames.items()}
        gt.attrs["excluded"] = len(dropped)
    else:
        gt.attrs["excluded"] = 0
    return frames, gt


def build_strata(gt: pd.DataFrame) -> dict:
    tool = gt["has_tool"].to_numpy()
    strata = {"all tool frames": tool}
    for size in SIZE_ORDER:
        strata[f"size {size}"] = tool & (gt["size_class"] == size).to_numpy()
    for thin in THIN_ORDER:
        strata[f"{thin} tools"] = tool & (gt["thin_class"] == thin).to_numpy()
    strata["tiny or thin"] = tool & (
        (gt["size_class"] == "tiny") | (gt["thin_class"] == "thin")
    ).to_numpy()
    for token in DEGRADATIONS:
        if f"ch::{token}" in gt:
            strata[f"scene {token}"] = tool & gt[f"ch::{token}"].to_numpy()
    strata["scene any degradation"] = tool & gt["any_degradation"].to_numpy()
    strata["scene clean"] = tool & gt["clean_scene"].to_numpy()
    strata["tool-free frames"] = ~tool
    return strata


# === Statistics ===
def cluster_bootstrap_ci(values, clusters, n_boot=N_BOOT, seed=BOOT_SEED, alpha=0.05):
    # resample whole videos, not frames, so a long video cannot dominate
    v = np.asarray(values, dtype=float)
    c = np.asarray(clusters)[np.isfinite(v)]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan
    codes, uniques = pd.factorize(c)
    sums = np.bincount(codes, weights=v, minlength=len(uniques))
    counts = np.bincount(codes, minlength=len(uniques)).astype(float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(uniques), size=(n_boot, len(uniques)))
    means = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def paired_p(diff: np.ndarray) -> float:
    # Wilcoxon: per-frame differences are non-normal and zero-inflated
    d = diff[np.isfinite(diff)]
    if d.size < 10 or np.allclose(d, 0):
        return np.nan
    try:
        return float(wilcoxon(d, zero_method="wilcox").pvalue)
    except ValueError:
        return np.nan


def _counts(df: pd.DataFrame, t: float):
    tp = df[f"tp@{t:.2f}"].to_numpy(float)
    fp = df[f"fp@{t:.2f}"].to_numpy(float)
    fn = df[f"fn@{t:.2f}"].to_numpy(float)
    return tp, fp, fn


def _rates(df: pd.DataFrame, t: float):
    tp, fp, fn = _counts(df, t)
    iou = np.divide(tp, tp + fp + fn, out=np.full_like(tp, np.nan), where=(tp + fp + fn) > 0)
    prec = np.divide(tp, tp + fp, out=np.full_like(tp, np.nan), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.full_like(tp, np.nan), where=(tp + fn) > 0)
    return iou, prec, rec


def _curve(df: pd.DataFrame, mask: np.ndarray, thrs: list[float], pooled: bool) -> np.ndarray:
    out = []
    for t in thrs:
        if pooled:
            tp, fp, fn = (x[mask].sum() for x in _counts(df, t))
            prec = tp / (tp + fp) if tp + fp else np.nan
            rec = tp / (tp + fn) if tp + fn else np.nan
            out.append((prec, rec))
        else:
            _, prec, rec = _rates(df, t)
            out.append((float(np.nanmean(prec[mask])), float(np.nanmean(rec[mask]))))
    return np.array(out)


# === Tables ===
HEADLINE_KEYS = [
    "iou", "dice", "precision", "recall", "hd95", "msd",
    "iou_tiny", "iou_small", "iou_med", "iou_large", "iou_thin", "iou_wide",
    "recall_tiny", "recall_thin", "notool_clean_rate", "notool_fp_mean", "miss_rate",
]


def headline(keys: list[str]) -> pd.DataFrame:
    rows = []
    for key in keys:
        meta = results(key)
        means = meta["mean_std"]
        row = {"run": key, "label": label(key), "run_id": meta["run_id"]}
        for name in HEADLINE_KEYS:
            row[name] = means[name][0] if name in means else np.nan
            row[f"{name}_sd"] = means[name][1] if name in means else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def paired_deltas(frames, gt, strata, reference: str, arms: list[str],
                  metrics=("iou", "recall", "precision", "hd95"), n_boot=N_BOOT) -> pd.DataFrame:
    rows = []
    for stratum, mask in strata.items():
        m = np.asarray(mask, dtype=bool)
        if m.sum() < MIN_STRATUM:
            continue
        videos = gt["video"].to_numpy()[m]
        for arm in arms:
            for metric in metrics:
                if metric not in frames[arm]:
                    continue
                b = frames[reference][metric].to_numpy(float)[m]
                a = frames[arm][metric].to_numpy(float)[m]
                d = a - b
                ok = np.isfinite(d)
                lo, hi = cluster_bootstrap_ci(d, videos, n_boot=n_boot)
                lower_better = metric in LOWER_IS_BETTER
                if not np.isfinite(lo):
                    verdict = "n/a"
                elif lo * hi <= 0:
                    verdict = "no change"
                elif (lo > 0) != lower_better:
                    verdict = "better"
                else:
                    verdict = "worse"
                rows.append({
                    "stratum": stratum,
                    "run": arm,
                    "label": label(arm),
                    "metric": metric,
                    "n_frames": int(m.sum()),
                    "n_videos": int(pd.unique(videos).size),
                    "reference": float(np.nanmean(b)),
                    "arm_value": float(np.nanmean(a)),
                    "delta": float(np.nanmean(d)),
                    "median": float(np.nanmedian(d[ok])) if ok.any() else np.nan,
                    "ci_low": lo,
                    "ci_high": hi,
                    "p": paired_p(d),
                    "improved_pct": float(100 * np.mean(d[ok] > EPS)) if ok.any() else np.nan,
                    "worsened_pct": float(100 * np.mean(d[ok] < -EPS)) if ok.any() else np.nan,
                    "verdict": verdict,
                })
    return pd.DataFrame(rows)


def threshold_sweep(frames, strata, reference: str, arms: list[str]) -> pd.DataFrame:
    tool = strata["all tool frames"]
    tiny = strata["size tiny"]
    rows = []
    for key in [reference] + list(arms):
        df = frames[key]
        for t in stored_thresholds(df):
            iou, prec, rec = _rates(df, t)
            fp = df[f"fp@{t:.2f}"].to_numpy(float)
            rows.append({
                "run": key,
                "label": label(key),
                "threshold": t,
                "iou": float(np.nanmean(iou[tool])),
                "precision": float(np.nanmean(prec[tool])),
                "recall": float(np.nanmean(rec[tool])),
                "iou_tiny": float(np.nanmean(iou[tiny])),
                "recall_tiny": float(np.nanmean(rec[tiny])),
                "notool_fp_px": float(fp[~tool].mean()),
            })
    return pd.DataFrame(rows)


def matched_precision(frames, strata, reference: str, arms: list[str],
                      conventions=("per-frame", "pooled")) -> pd.DataFrame:
    usable = {
        k: v for k, v in strata.items()
        if k != "tool-free frames" and np.asarray(v).sum() >= MIN_STRATUM
    }
    ref_thrs = stored_thresholds(frames[reference])
    rows = []
    for convention in conventions:
        pooled = convention == "pooled"
        for stratum, mask in usable.items():
            m = np.asarray(mask, dtype=bool)
            # 1. Reference precision at 0.50 is the target the arm has to match
            bp = _curve(frames[reference], m, ref_thrs, pooled)
            target = float(np.interp(PRIMARY_THRESHOLD, ref_thrs, bp[:, 0]))
            base_recall = float(np.interp(PRIMARY_THRESHOLD, ref_thrs, bp[:, 1]))
            for arm in arms:
                # 2. Read the arm at the threshold where its precision equals that target
                thrs = stored_thresholds(frames[arm])
                ap = _curve(frames[arm], m, thrs, pooled)
                # precision rises with threshold, so the grid is already increasing for np.interp
                matched_thr = float(np.interp(target, ap[:, 0], np.asarray(thrs)))
                arm_recall = float(np.interp(target, ap[:, 0], ap[:, 1]))
                recall_at_50 = float(np.interp(PRIMARY_THRESHOLD, thrs, ap[:, 1]))
                rows.append({
                    "convention": convention,
                    "stratum": stratum,
                    "run": arm,
                    "label": label(arm),
                    "n_frames": int(m.sum()),
                    "target_precision": target,
                    "matched_threshold": matched_thr,
                    "reference_recall": base_recall,
                    "arm_recall_matched": arm_recall,
                    "delta_recall_matched": arm_recall - base_recall,
                    "delta_recall_at_0.50": recall_at_50 - base_recall,
                    "out_of_grid": not (ap[:, 0].min() <= target <= ap[:, 0].max()),
                })
    return pd.DataFrame(rows)


def matched_distribution(frames, strata, reference: str, arms: list[str],
                         keys=("all tool frames", "size tiny", "thin tools"),
                         metric: str = "recall") -> pd.DataFrame:
    subset = {k: strata[k] for k in keys if np.asarray(strata[k]).sum() >= MIN_STRATUM}
    matched = matched_precision(frames, subset, reference, arms, conventions=("per-frame",))
    idx = 2 if metric == "recall" else 0
    ref_vals = _rates(frames[reference], PRIMARY_THRESHOLD)[idx]
    rows = []
    for stratum in subset:
        m = np.asarray(strata[stratum], dtype=bool)
        for arm in arms:
            sel = (matched["stratum"] == stratum) & (matched["run"] == arm)
            thr = float(matched.loc[sel, "matched_threshold"].iloc[0])
            grid = stored_thresholds(frames[arm])
            lo = max([t for t in grid if t <= thr], default=grid[0])
            hi = min([t for t in grid if t >= thr], default=grid[-1])
            w = 0.0 if hi == lo else (thr - lo) / (hi - lo)
            vals = (1 - w) * _rates(frames[arm], lo)[idx] + w * _rates(frames[arm], hi)[idx]
            d = (vals - ref_vals)[m]
            d = d[np.isfinite(d)]
            rows.append({
                "stratum": stratum,
                "run": arm,
                "label": label(arm),
                "metric": metric,
                "matched_threshold": thr,
                "n_frames": int(d.size),
                "mean": float(d.mean()),
                "median": float(np.median(d)),
                "improved_pct": float(100 * np.mean(d > EPS)),
                "worsened_pct": float(100 * np.mean(d < -EPS)),
                "p": paired_p(d),
            })
    return pd.DataFrame(rows)


def lofo_threshold(frames, gt, strata, reference: str, arms: list[str]) -> pd.DataFrame:
    # pick the threshold on four folds, score the fifth, rotate
    tool = strata["all tool frames"]
    fold = gt["fold"].to_numpy()
    rows = []
    for objective, obj_mask in (("overall IoU", tool), ("tiny IoU", strata["size tiny"])):
        for key in [reference] + list(arms):
            thrs = stored_thresholds(frames[key])
            cache = {t: _rates(frames[key], t) for t in thrs}
            iou_out = np.full(len(gt), np.nan)
            picked = []
            for f in sorted(set(fold)):
                train = obj_mask & (fold != f)
                test = fold == f
                best = max(thrs, key=lambda t: np.nanmean(cache[t][0][train]))
                picked.append(best)
                iou_out[test] = cache[best][0][test]
            if PRIMARY_THRESHOLD in cache:
                at_50 = cache[PRIMARY_THRESHOLD][0]
            else:
                at_50 = np.full(len(gt), np.nan)
            rows.append({
                "objective": objective,
                "run": key,
                "label": label(key),
                "thresholds": "/".join(f"{t:g}" for t in picked),
                "iou_lofo": float(np.nanmean(iou_out[tool])),
                "iou_at_0.50": float(np.nanmean(at_50[tool])),
                "gain": float(np.nanmean(iou_out[tool]) - np.nanmean(at_50[tool])),
            })
    return pd.DataFrame(rows)


def tool_free(frames, strata, reference: str, arms: list[str]) -> pd.DataFrame:
    notool = strata["tool-free frames"]
    ref_fp = frames[reference]["fp_pixels"].to_numpy(float)[notool]
    rows = []
    for key in [reference] + list(arms):
        fp = frames[key]["fp_pixels"].to_numpy(float)[notool]
        rows.append({
            "run": key,
            "label": label(key),
            "n_frames": int(notool.sum()),
            "fp_px_mean": float(np.mean(fp)),
            "fp_px_median": float(np.median(fp)),
            "clean_rate": float(np.mean(fp < 100)),
            "clean_rate_json": results(key)["mean_std"]["notool_clean_rate"][0],
            "delta_fp_px": float(np.mean(fp) - np.mean(ref_fp)),
            "p": paired_p(fp - ref_fp),
        })
    return pd.DataFrame(rows)


def error_distribution(frames, strata, key: str) -> dict:
    df = frames[key]
    tool = strata["all tool frames"]
    deficit = float((1.0 - df["iou"].to_numpy(float)[tool]).sum())
    names = (
        [f"size {s}" for s in SIZE_ORDER]
        + [f"{t} tools" for t in THIN_ORDER]
        + ["tiny or thin"]
    )
    rows = []
    for name in names:
        m = np.asarray(strata[name], dtype=bool)
        iou = df["iou"].to_numpy(float)[m]
        tool_px = df["tool_pixels"].to_numpy(float)[m]
        rows.append({
            "stratum": name,
            "n_frames": int(m.sum()),
            "share_of_tool_frames_pct": 100.0 * m.sum() / tool.sum(),
            "iou": float(np.nanmean(iou)),
            "recall": float(np.nanmean(df["recall"].to_numpy(float)[m])),
            "precision": float(np.nanmean(df["precision"].to_numpy(float)[m])),
            "msd_px": float(np.nanmean(df["msd"].to_numpy(float)[m])),
            "hd95_px": float(np.nanmean(df["hd95"].to_numpy(float)[m])),
            "tool_px_mean": float(np.nanmean(tool_px)),
            "fp_per_tool_px": float(df["fp_pixels"].to_numpy(float)[m].sum() / tool_px.sum()),
            "fn_per_tool_px": float(df["fn_pixels"].to_numpy(float)[m].sum() / tool_px.sum()),
            "share_of_iou_deficit_pct": 100.0 * float((1.0 - iou).sum()) / deficit,
        })
    fp_tool = float(df["fp_pixels"].to_numpy(float)[tool].sum())
    fp_notool = float(df["fp_pixels"].to_numpy(float)[~tool].sum())
    where = pd.DataFrame([
        {
            "frames": "tool-present",
            "n": int(tool.sum()),
            "fp_pixels": fp_tool,
            "share_of_fp_mass_pct": 100.0 * fp_tool / (fp_tool + fp_notool),
        },
        {
            "frames": "tool-free",
            "n": int((~tool).sum()),
            "fp_pixels": fp_notool,
            "share_of_fp_mass_pct": 100.0 * fp_notool / (fp_tool + fp_notool),
        },
    ])
    return {"by_stratum": pd.DataFrame(rows), "fp_location": where}


def master_table(keys=None) -> pd.DataFrame:
    rows = []
    for key in (keys or list(RUNS)):
        meta = results(key)
        means = meta["mean_std"]
        rows.append({
            "run": key,
            "label": RUNS[key].label,
            "family": RUNS[key].family,
            "varies": RUNS[key].varies,
            "weights_from": RUNS[key].weights_from,
            "run_id": meta["run_id"],
            "iou": means["iou"][0],
            "iou_sd": means["iou"][1],
            "dice": means["dice"][0],
            "precision": means["precision"][0],
            "recall": means["recall"][0],
            "iou_tiny": means["iou_tiny"][0],
            "iou_small": means["iou_small"][0],
            "iou_med": means["iou_med"][0],
            "iou_large": means["iou_large"][0],
            "iou_thin": means["iou_thin"][0],
            "iou_wide": means["iou_wide"][0],
            "recall_tiny": means["recall_tiny"][0],
            "recall_thin": means["recall_thin"][0],
            "hd95": means["hd95"][0],
            "msd": means["msd"][0],
            "notool_clean_rate": means["notool_clean_rate"][0],
            "notool_fp_mean": means["notool_fp_mean"][0],
            "miss_rate": means["miss_rate"][0],
            "epochs": "/".join(str(e) for e in meta.get("epochs_trained", [])),
            "erroneous": RUNS[key].erroneous,
        })
    return pd.DataFrame(rows)


def cascade_diagnostics(frames, strata, arm: str, reference: str) -> dict:
    df = frames[arm]
    ref = frames[reference]
    if "cascade_tier" not in df:
        raise ValueError(f"{arm!r} carries no cascade diagnostics")
    tool = np.asarray(strata["all tool frames"], dtype=bool)
    delta = df["iou"].to_numpy(float) - ref["iou"].to_numpy(float)

    tiers = []
    for tier, sub in df.groupby("cascade_tier"):
        m = df.index.isin(sub.index)
        tiers.append({
            "tier": tier,
            "n_frames": int(m.sum()),
            "share_pct": 100.0 * m.sum() / len(df),
            "delta_iou": float(np.nanmean(delta[m])),
            "delta_iou_tool_frames": float(np.nanmean(delta[m & tool])),
            "mean_zoom": float(np.nanmean(df.loc[m, "cascade_roi_zoom"])),
        })

    zoom = df["cascade_roi_zoom"].to_numpy(float)
    fired = zoom > 1.0
    precision_delta = df["precision"].to_numpy(float) - ref["precision"].to_numpy(float)
    rows = [{
        "zoom": "no crop",
        "n_frames": int((~fired & tool).sum()),
        "delta_iou": float(np.nanmean(delta[~fired & tool])),
        "delta_precision": float(np.nanmean(precision_delta[~fired & tool])),
    }]
    for lo, hi in [(1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 10.0)]:
        m = tool & fired & (zoom >= lo) & (zoom < hi)
        if m.sum() < MIN_STRATUM:
            continue
        rows.append({
            "zoom": f"{lo:g} to {hi:g}",
            "n_frames": int(m.sum()),
            "delta_iou": float(np.nanmean(delta[m])),
            "delta_precision": float(np.nanmean(precision_delta[m])),
        })
    return {"by_tier": pd.DataFrame(tiers), "by_zoom": pd.DataFrame(rows)}


def save_table(table: pd.DataFrame, name: str, out_dir: Path, show: bool = False,
               n: int = 25) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    if show:
        display_and_save(table.round(6), name, out_dir, n=n)
    else:
        table.round(6).to_csv(out_dir / f"{name}.csv", index=False)
    return table.round(DECIMALS)


def save_fig(fig, name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _style(reference: str, arms: list[str], key: str):
    if key == reference:
        return SECONDARY, "o"
    i = list(arms).index(key)
    return ARM_SERIES[i % len(ARM_SERIES)], ARM_MARKERS[i % len(ARM_MARKERS)]


# === Figures ===
def figure_forest(paired: pd.DataFrame, reference: str, arms: list[str], out_dir: Path,
                  metric: str = "iou") -> Path:
    table = paired[paired["metric"] == metric]
    # IoU on a tool-free frame is a different quantity, and sits in the tool-free table
    strata = [s for s in dict.fromkeys(table["stratum"]) if s != "tool-free frames"]
    height = max(3.2, 0.30 * len(strata) + 0.10 * len(strata) * len(arms))
    fig, ax = plt.subplots(figsize=(8.2, height))
    offsets = np.linspace(-0.32, 0.32, len(arms)) if len(arms) > 1 else [0.0]
    for j, arm in enumerate(arms):
        sub = table[table["run"] == arm].set_index("stratum").reindex(strata)
        colour, marker = _style(reference, arms, arm)
        ax.errorbar(
            sub["delta"], np.arange(len(strata)) + offsets[j],
            xerr=[sub["delta"] - sub["ci_low"], sub["ci_high"] - sub["delta"]],
            fmt=marker, ms=4, lw=1.2, capsize=2.5, color=colour, label=label(arm),
        )
    ax.axvline(0, color=RULE, lw=1)
    ax.set_yticks(np.arange(len(strata)))
    ax.set_yticklabels([stratum_label(s) for s in strata], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(
        f"Change in {metric_label(metric)} vs {RUNS[reference].label}"
        f"    (bars show the 95% confidence range)",
        fontsize=9,
    )
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    return save_fig(fig, f"forest_{metric}", out_dir)


def figure_stratum_bars(frames, strata, reference: str, arms: list[str], out_dir: Path) -> Path:
    bands = [f"size {s}" for s in SIZE_ORDER] + [f"{t} tools" for t in THIN_ORDER]
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    width = 0.8 / (len(arms) + 1)
    for j, key in enumerate([reference] + list(arms)):
        iou = frames[key]["iou"].to_numpy(float)
        vals = [float(np.nanmean(iou[np.asarray(strata[b], dtype=bool)])) for b in bands]
        colour, _ = _style(reference, arms, key)
        ax.bar(np.arange(len(bands)) + j * width, vals, width, color=colour,
               edgecolor="white", label=label(key))
    ax.set_xticks(np.arange(len(bands)) + width * len(arms) / 2)
    ax.set_xticklabels([stratum_label(b) for b in bands])
    ax.set_ylabel("IoU")
    ax.set_ylim(0.5, 0.95)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return save_fig(fig, "iou_by_stratum", out_dir)


def figure_operating_curves(sweep: pd.DataFrame, matched: pd.DataFrame, reference: str,
                            arms: list[str], out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for key in [reference] + list(arms):
        sub = sweep[sweep["run"] == key]
        colour, marker = _style(reference, arms, key)
        axes[0].plot(sub["recall"], sub["precision"], marker=marker, ms=3.5, color=colour,
                     label=label(key))
        axes[1].plot(sub["threshold"], sub["iou"], marker=marker, ms=3.5, color=colour,
                     label=label(key))
    overall = matched[matched["stratum"] == "all tool frames"]
    if len(overall):
        axes[0].axhline(overall["target_precision"].iloc[0], color=RULE, lw=0.9, ls="--")
    axes[0].set_xlabel("Recall (averaged per frame, frames containing tools)")
    axes[0].set_ylabel("Precision")
    axes[0].set_title(
        "Precision against recall\ndashed line: the precision the runs are compared at",
        fontsize=10,
    )
    axes[1].axvline(PRIMARY_THRESHOLD, color=RULE, lw=0.9, ls=":")
    axes[1].set_xlabel("Decision threshold")
    axes[1].set_ylabel("IoU")
    axes[1].set_title(
        "IoU against decision threshold\ndotted line: the threshold quoted in the text",
        fontsize=10,
    )
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    return save_fig(fig, "operating_curves", out_dir)


def figure_delta_distribution(frames, strata, reference: str, arms: list[str], stratum: str,
                              out_dir: Path, metric: str = "iou") -> Path:
    mask = np.asarray(strata[stratum], dtype=bool)
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    for arm in arms:
        d = (frames[arm][metric].to_numpy(float)[mask]
             - frames[reference][metric].to_numpy(float)[mask])
        d = d[np.isfinite(d)]
        colour, _ = _style(reference, arms, arm)
        ax.hist(
            d, bins=60, range=(-0.3, 0.3), histtype="step", lw=1.4, color=colour,
            label=(
                f"{label(arm)}    mean {d.mean():+.4f}, median {np.median(d):+.4f}, "
                f"{100 * np.mean(d > EPS):.1f}% of frames improved"
            ),
        )
    ax.axvline(0, color=RULE, lw=1)
    ax.set_xlabel(
        f"Change in {metric_label(metric)} per frame, {stratum_label(stratum)}"
        f"    ({int(mask.sum()):,} frames)"
    )
    ax.set_ylabel("Number of frames")
    # one long label per run, so the legend sits above the axes rather than over the peak
    ax.legend(frameon=False, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return save_fig(fig, f"delta_distribution_{stratum.replace(' ', '_')}_{metric}", out_dir)


def figure_error_distribution(by_stratum: pd.DataFrame, key: str, out_dir: Path) -> Path:
    bands = by_stratum[by_stratum["stratum"].str.startswith("size ")].copy()
    bands["stratum"] = bands["stratum"].str.replace("size ", "", regex=False)
    x = np.arange(len(bands))

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.7))
    axes[0].bar(x, bands["iou"], color=PRIMARY, edgecolor="white")
    axes[0].set_ylim(0.5, 0.95)
    axes[0].set_ylabel("IoU")
    axes[0].set_title("How well each tool size is segmented", fontsize=10)

    width = 0.38
    axes[1].bar(x - width / 2, bands["share_of_tool_frames_pct"], width, color=SECONDARY,
                edgecolor="white", label="share of all frames")
    axes[1].bar(x + width / 2, bands["share_of_iou_deficit_pct"], width, color=PRIMARY,
                edgecolor="white", label="share of the total error")
    axes[1].set_ylabel("Percentage of frames containing tools")
    axes[1].set_title("Which tool sizes cause the most error", fontsize=10)
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].bar(x, bands["fp_per_tool_px"], color=PRIMARY, edgecolor="white")
    axes[2].set_ylabel("Wrongly predicted pixels per tool pixel")
    axes[2].set_title("Error relative to how big the tool is", fontsize=10)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(bands["stratum"])
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"Where the error sits: {label(key)}", fontsize=10)
    fig.tight_layout()
    return save_fig(fig, "error_distribution", out_dir)


DOSE_ARMS = {"roift-clean": 0.0, "roift-neg-clean": 0.3, "roift-neg-strong": 0.5}
DOSE_CONTROL = "roift-control-ff"


def figure_dose_response(out_dir: Path) -> Path:
    table = master_table(list(DOSE_ARMS) + [DOSE_CONTROL]).set_index("run")
    control = table.loc[DOSE_CONTROL]
    treated = table.loc[list(DOSE_ARMS)]
    dose = list(DOSE_ARMS.values())

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    axes[0].plot(dose, treated["notool_clean_rate"], marker="o", color=PRIMARY)
    axes[0].axhline(control["notool_clean_rate"], color=SECONDARY, ls="--", lw=1,
                    label="matched-budget control, no crops")
    axes[0].set_ylabel("Tool-free frames predicted clean")
    axes[1].plot(dose, treated["iou"], marker="o", color=PRIMARY, label="ROI-crop fine-tune")
    axes[1].axhline(control["iou"], color=SECONDARY, ls="--", lw=1,
                    label="matched-budget control")
    axes[1].set_ylabel("IoU")
    for ax in axes:
        ax.set_xlabel("Share of training crops containing no tool")
        ax.set_xticks(dose)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return save_fig(fig, "negatives_dose_response", out_dir)


# === Drivers ===
DISTRIBUTION_STRATA = ("size tiny", "thin tools")  # the strata the hypotheses are about


def output_dir(reference: str, arms: list[str], cleaned: bool = False, name: str = None,
               root: Path = OUTPUT_ROOT) -> Path:
    stem = name or "-and-".join(arms)
    suffix = "_cleaned" if cleaned else ""
    return Path(root) / f"{stem}_vs_{reference}{suffix}"


def compare_runs(reference: str, arms: list[str], cleaned: bool = False, name: str = None,
                 root: Path = OUTPUT_ROOT, show: bool = False) -> Path:
    frames, gt = load_runs([reference] + [a for a in arms if a != reference], cleaned)
    strata = build_strata(gt)
    out = output_dir(reference, arms, cleaned, name, root)
    tables = out / "tables"
    figures = out / "figures"
    for note in warnings_for(reference, arms):
        print(f"  {note}")

    # 1. Tables
    save_table(headline([reference] + list(arms)), "headline_fold_means", tables, show)
    diagnostics = error_distribution(frames, strata, reference)
    save_table(diagnostics["by_stratum"], "reference_error_by_stratum", tables, show)
    save_table(diagnostics["fp_location"], "reference_fp_location", tables, show)
    paired = paired_deltas(frames, gt, strata, reference, arms)
    save_table(paired, "paired_deltas", tables, show)
    sweep = save_table(
        threshold_sweep(frames, strata, reference, arms), "threshold_sweep", tables, show,
    )
    matched = matched_precision(frames, strata, reference, arms)
    save_table(matched, "matched_precision", tables, show)
    save_table(
        matched_distribution(frames, strata, reference, arms), "matched_distribution",
        tables, show,
    )
    save_table(lofo_threshold(frames, gt, strata, reference, arms), "threshold_lofo", tables, show)
    save_table(tool_free(frames, strata, reference, arms), "tool_free", tables, show)

    # 2. Figures
    figure_forest(paired, reference, arms, figures)
    figure_stratum_bars(frames, strata, reference, arms, figures)
    figure_operating_curves(sweep, matched, reference, arms, figures)
    figure_error_distribution(diagnostics["by_stratum"], reference, figures)
    for stratum in DISTRIBUTION_STRATA:
        if np.asarray(strata[stratum]).sum() >= MIN_STRATUM:
            figure_delta_distribution(frames, strata, reference, arms, stratum, figures)
    if all("cascade_tier" in frames[a] for a in arms):
        for arm in arms:
            diag = cascade_diagnostics(frames, strata, arm, reference)
            save_table(diag["by_tier"], f"cascade_tiers_{arm}", tables, show)
            save_table(diag["by_zoom"], f"cascade_zoom_{arm}", tables, show)
    return out


def declared_pairs() -> list:
    return [(spec.reference, [key]) for key, spec in RUNS.items() if spec.reference]


def build_all_outputs(cleaned: bool = False, root: Path = OUTPUT_ROOT) -> list:
    made = []
    # 1. Every declared pair
    for reference, arms in declared_pairs():
        made.append(compare_runs(reference, arms, cleaned, root=root))
        print(f"  {made[-1].name}")
    # 2. Summary forest of one run per family
    made.append(compare_runs(BASELINE, SUMMARY_ARMS, cleaned, name="all-arms", root=root))
    print(f"  {made[-1].name}")
    # 3. Master table and dose-response, which are not a pair
    every = Path(root) / "all-runs"
    save_table(master_table(), "master_table", every / "tables")
    figure_dose_response(every / "figures")
    made.append(every)
    print(f"  {every.name}")
    return made
