"""Cleaned re-evaluation: re-aggregate every scored arm with hand-reviewed annotation errors excluded."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sys
import importlib
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / '01_baseline_analysis'))
from baseline_common import key_from_path

# === Config ===
NO_UPLOAD = True             # if True, skip the GCS upload
CANDIDATES = 400             # size of the review candidate pool

SHEET_ID = "REPLACE_SHEET_ID"
BUCKET = "gs://training-checkpoints-REPLACE_K_NUMBER"
EXPERIMENTS = f"{BUCKET}/experiments"
OUT_PREFIX = f"{BUCKET}/analysis/cleaned_reeval"
CACHE = Path.home() / ".cache/laparoscopic_seg/cleaned_reeval"

# The status substring that marks an annotation error. Matched case-insensitively, the
# same way section 6b matches its retirement words.
PROBLEM_WORD = "problematic"

# Every arm with a usable per_image_results.csv, as of 16 Aug. Kept here rather than
# discovered, so the re-evaluation covers a fixed declared set and a missing arm is an
# error rather than a silent omission. Mirrors archive/00_model_training/run_log.md.
RUNS = {
	"baseline-seed2026":          "baseline-seed2026_cv_Unet_resnet34_bce_lr1e-04_p20/20260812-205137",
	"baseline-seed42-clean":      "baseline-seed42-clean_cv_Unet_resnet34_bce_lr1e-04_p20/20260813-232105",
	"bce_tversky":                "tversky2_cv_balsweep_Unet_resnet34_bce_tversky_lr1e-04_p20/20260808-013552",
	"tversky-rescore-hi":         "tversky-rescore-hi_cv_Unet_resnet34_bce_tversky_lr1e-04_p20/20260814-104220",
	"bce_dice":                   "bce-dice_cv_Unet_resnet34_bce_dice_lr1e-04_p20/20260808-123132",
	"synaug_low":                 "synthetic-augmentation-low-cv_cv_synaug-low_Unet_resnet34_bce_lr1e-04_p20/20260809-213637",
	"synaug_fatal":               "synthetic-augmentation-fatal-cv_cv_synaug-fatal_Unet_resnet34_bce_lr1e-04_p20/20260809-213800",
	"synaug-fixed":               "synaug-fixed_cv_synaug-low_additive_Unet_resnet34_bce_lr1e-04_p20/20260813-140837",
	"static-tiling":              "static-tiling-cv_cv_Unet_resnet34_bce_lr1e-04_p20/20260810-010618",
	"cascade-redetect-clean":     "cascade-redetect-clean_cv_cascade-fused_redetect_Unet_resnet34_bce_lr1e-04_p20/20260813-230613",
	"cascade-roift-neg-redetect": "cascade-roift-neg-redetect_cv_cascade-fused_redetect_Unet_resnet34_bce_lr1e-04_p20/20260815-011358",
	"roift-clean":                "roift-clean_cv_roift_Unet_resnet34_bce_lr1e-04_p20/20260813-233143",
	"roift-tversky":              "roift-tversky_cv_roift_Unet_resnet34_bce_tversky_lr1e-04_p20/20260814-104911",
	"roift-neg-clean":            "roift-neg-clean_cv_roift_Unet_resnet34_bce_lr1e-04_p20/20260814-105438",
	"roift-control-ff":           "roift-control-ff_cv_roift_Unet_resnet34_bce_lr1e-04_p20/20260814-144749",
	"roift-small-neg":            "roift-small-neg_cv_roift_Unet_resnet34_bce_lr1e-04_p20/20260814-143954",
	"roift-neg-strong":           "roift-neg-strong_cv_roift_Unet_resnet34_bce_lr1e-04_p20/20260814-152647",
	"ft-long-control":            "ft-long-control_cv_roift_Unet_resnet34_bce_lr1e-04_p20/20260815-012030",
	"roift-long":                 "roift-long_cv_roift_Unet_resnet34_bce_lr1e-04_p20/20260815-012339",
	"roift-long-seed2026":        "roift-long-seed2026_cv_roift_Unet_resnet34_bce_lr1e-04_p20/20260816-004733",
}

METRICS = ["iou", "dice", "precision", "recall"]
STRATA = [("tiny", "size_class"), ("small", "size_class"), ("med", "size_class"),
          ("large", "size_class"), ("thin", "thin_class"), ("wide", "thin_class")]


# === Core ===




def fetch_sheet() -> pd.DataFrame:
	url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"
	return pd.read_csv(io.BytesIO(urllib.request.urlopen(url, timeout=120).read()))


def fetch_per_image(arm: str, rel: str) -> pd.DataFrame:
	CACHE.mkdir(parents=True, exist_ok=True)
	dest = CACHE / f"{arm}.csv"
	if not dest.exists():
		src = f"{EXPERIMENTS}/{rel}/per_image_results.csv"
		if subprocess.run(["gsutil", "-q", "cp", src, str(dest)]).returncode != 0:
			raise RuntimeError(f"could not fetch {src}")
	return pd.read_csv(dest)


def aggregate(df: pd.DataFrame, drop: set) -> dict:
	"""Overall and per-stratum means on tool frames, before and after the exclusion."""
	keep = ~df["key"].isin(drop)
	tool = ~df["gt_empty"].astype(bool)
	notool = df["gt_empty"].astype(bool)
	out = {}
	for label, mask in [("original", pd.Series(True, index=df.index)), ("cleaned", keep)]:
		row = {}
		m = tool & mask
		for metric in METRICS:
			row[metric] = float(df.loc[m, metric].mean())
		row["n_tool"] = int(m.sum())
		for name, col in STRATA:
			s = m & (df[col] == name)
			row[f"iou_{name}"] = float(df.loc[s, "iou"].mean()) if s.any() else float("nan")
			row[f"n_{name}"] = int(s.sum())
		nt = notool & mask
		row["notool_fp_mean"] = float(df.loc[nt, "fp_pixels"].mean()) if nt.any() else float("nan")
		row["n_notool"] = int(nt.sum())
		out[label] = row
	out["delta"] = {k: out["cleaned"][k] - out["original"][k]
	                for k in out["original"] if not k.startswith("n_") and k != "n_tool"}
	return out


def candidate_pool(frames: dict, n: int) -> pd.DataFrame:
	"""Worst frames by IoU across the union of arms, the defensible review pool.

	Drawn from every arm rather than one, because a mis-annotated frame is
	mis-annotated regardless of which model scored it badly, and a pool built from a
	single arm's ranking is not independent of that arm.
	"""
	parts = []
	for arm, df in frames.items():
		tool = ~df["gt_empty"].astype(bool)
		sub = df.loc[tool, ["key", "iou", "size_class", "thin_class", "tool_pixels"]].copy()
		sub["arm"] = arm
		parts.append(sub)
	allf = pd.concat(parts, ignore_index=True)
	agg = (allf.groupby("key")
	           .agg(worst_iou=("iou", "min"), mean_iou=("iou", "mean"),
	                arms_below_0_3=("iou", lambda s: int((s < 0.3).sum())),
	                size_class=("size_class", "first"), thin_class=("thin_class", "first"),
	                tool_pixels=("tool_pixels", "first"))
	           .reset_index())
	return agg.sort_values(["arms_below_0_3", "mean_iou"], ascending=[False, True]).head(n)


# === Main ===

def main() -> int:

	stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
	outdir = CACHE / stamp
	outdir.mkdir(parents=True, exist_ok=True)

	print("sheet   : fetching")
	sheet = fetch_sheet()
	status = sheet["Status"].astype(str).str.lower()
	sheet["key"] = (sheet["Video ID"].astype(str) + "/"
	                + sheet["File Name"].astype(str).map(lambda x: Path(str(x)).stem))
	flagged = set(sheet.loc[status.str.contains(PROBLEM_WORD, na=False), "key"])
	noted = set(sheet.loc[sheet["Notes"].notna(), "key"])
	print(f"          {len(sheet)} rows, {len(flagged)} marked {PROBLEM_WORD!r}, "
	      f"{len(noted)} carrying a free-text note")

	print("arms    : loading per-frame results")
	frames = {}
	for arm, rel in RUNS.items():
		df = fetch_per_image(arm, rel)
		df["key"] = df["image"].map(key_from_path)
		frames[arm] = df
	sizes = {len(df) for df in frames.values()}
	assert len(sizes) == 1, f"arms disagree on frame count: {sizes}"
	keys = [set(df["key"]) for df in frames.values()]
	assert all(k == keys[0] for k in keys), "arms disagree on which frames were scored"
	print(f"          {len(frames)} arms, {sizes.pop()} frames each, key sets identical")

	drop = flagged & keys[0]
	print(f"exclude : {len(drop)} frames present in the scored set")

	results = {
		"generated_utc": datetime.now(timezone.utc).isoformat(),
		"sheet_id": SHEET_ID,
		"problem_word": PROBLEM_WORD,
		"n_sheet_rows": int(len(sheet)),
		"n_flagged_in_sheet": int(len(flagged)),
		"n_flagged_in_scored_set": int(len(drop)),
		"n_notes_in_sheet": int(len(noted)),
		"excluded_keys": sorted(drop),
		"runs": RUNS,
		"arms": {arm: aggregate(df, drop) for arm, df in frames.items()},
	}

	rows = []
	for arm, res in results["arms"].items():
		rows.append({"arm": arm,
		             "iou_original": res["original"]["iou"], "iou_cleaned": res["cleaned"]["iou"],
		             "iou_delta": res["delta"]["iou"],
		             "iou_tiny_original": res["original"]["iou_tiny"],
		             "iou_tiny_cleaned": res["cleaned"]["iou_tiny"],
		             "iou_tiny_delta": res["delta"]["iou_tiny"],
		             "n_tool_original": res["original"]["n_tool"],
		             "n_tool_cleaned": res["cleaned"]["n_tool"]})
	table = pd.DataFrame(rows).sort_values("iou_cleaned", ascending=False)

	(outdir / "03k_cleaned_reeval.json").write_text(json.dumps(results, indent=2) + "\n")
	table.to_csv(outdir / "03k_cleaned_reeval.csv", index=False)

	lines = [f"# Cleaned re-evaluation, {stamp} UTC", "",
	         f"Sheet rows {len(sheet)}, marked {PROBLEM_WORD!r} {len(flagged)}, "
	         f"of which {len(drop)} are in the scored set.", ""]
	if not drop:
		lines += ["**No frame carries the status, so the cleaned set equals the original set and "
		          "every delta below is zero by construction.** The review candidate pool is "
		          "written alongside this file so the labelling can proceed.", ""]
	lines += ["| arm | IoU original | IoU cleaned | delta | iou_tiny original | iou_tiny cleaned | delta |",
	          "| --- | --- | --- | --- | --- | --- | --- |"]
	for _, r in table.iterrows():
		lines.append(f"| `{r.arm}` | {r.iou_original:.4f} | {r.iou_cleaned:.4f} | {r.iou_delta:+.4f} "
		             f"| {r.iou_tiny_original:.4f} | {r.iou_tiny_cleaned:.4f} | {r.iou_tiny_delta:+.4f} |")
	lines += ["", f"Frames scored per arm: {table.n_tool_original.iloc[0]} tool frames original, "
	              f"{table.n_tool_cleaned.iloc[0]} cleaned.", ""]
	if drop:
		spread = table["iou_delta"].abs()
		lines += ["## Bias check", "",
		          "Exclusions are the same frames for every arm, so what matters is whether one arm "
		          "gains disproportionately.", "",
		          f"Largest gain {table.iou_delta.max():+.4f} (`{table.loc[table.iou_delta.idxmax(),'arm']}`), "
		          f"smallest {table.iou_delta.min():+.4f} (`{table.loc[table.iou_delta.idxmin(),'arm']}`), "
		          f"spread {spread.max() - spread.min():.4f}.", "",
		          "A spread comparable to the effect sizes being claimed means the exclusion is "
		          "helping one arm rather than cleaning the measurement, and the comparison should "
		          "be reported on both sets."]
	(outdir / "03k_cleaned_reeval.md").write_text("\n".join(lines) + "\n")

	pool = candidate_pool(frames, CANDIDATES)
	pool.to_csv(outdir / "review_candidates.csv", index=False)
	print(f"pool    : {len(pool)} candidate frames written, "
	      f"{int((pool.arms_below_0_3 > 0).sum())} scored below 0.30 by at least one arm")

	print(f"\nwrote   : {outdir}")
	for f in sorted(outdir.iterdir()):
		print(f"          {f.name}  {f.stat().st_size:,} bytes")

	if not NO_UPLOAD:
		dest = f"{OUT_PREFIX}/{stamp}/"
		rc = subprocess.run(["gsutil", "-m", "-q", "cp", "-r", f"{outdir}/.", dest]).returncode
		if rc != 0:
			print(f"\nGCS upload FAILED (exit {rc}); files are still in {outdir}")
			return 1
		print(f"\nuploaded: {dest}")

	if not drop:
		print("\nNOTE: nothing was excluded, so the cleaned numbers equal the originals. Apply the")
		print(f"      {PROBLEM_WORD!r} status in the sheet, then re-run this to get real deltas.")
	return 0


if __name__ == "__main__":
	sys.exit(main())
