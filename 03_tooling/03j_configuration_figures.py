"""Chapter figures for baseline configuration: learning-rate sweep and tool-free oversampling shares."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# === Config ===
OUT_DIR = "notes/writeup/figures"
RUN = "baseline-seed42-clean_cv_Unet_resnet34_bce_lr1e-04_p20/20260813-232105"

BUCKET = "gs://training-checkpoints-REPLACE_K_NUMBER"
SWEEP_JSON = f"{BUCKET}/_latest_sweep.json"	# the sweep every arm's lr_source names by RUN_ID


CACHE = Path.home() / ".cache/laparoscopic_seg/configuration"

PRIMARY = "red"	# one red across the dissertation, matching the 03a and 03b histograms
SECONDARY = "#6c757d"
RULE = "#333333"
# Ordered series for a swept variable: warm at one end, cool at the other, so position in the
# palette carries the ordering of the swept value.
SWEEP_SERIES = [PRIMARY, "#f4a261", "#2a9d8f", "#457b9d", "#1d3557"]
SWEEP_MARKERS = ["o", "s", "^", "D", "v"]


# === Core ===

def fetch(src: str, name: str) -> dict:
	CACHE.mkdir(parents=True, exist_ok=True)
	dest = CACHE / name
	if not dest.exists() and subprocess.run(["gsutil", "-q", "cp", src, str(dest)]).returncode:
		raise SystemExit(f"could not fetch {src}")
	return json.loads(dest.read_text())


def lr_sweep_figure(out_dir: Path) -> Path:
	"""Best validation IoU per rate, beside the per-epoch curve for every rate on the grid.

	One fold at a reduced epoch budget, so what the sweep establishes is the ordering across
	the grid and not an absolute score; the full runs land well above every value here.
	"""
	meta = fetch(SWEEP_JSON, "lr_sweep.json")
	grid = sorted(meta["grid"], key=lambda point: point["lr"])
	best = max(point["best_val_iou"] for point in grid)
	winner = next(point for point in grid if point["lr"] == meta["best_lr"])

	fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
	axes[0].plot([p["lr"] for p in grid], [p["best_val_iou"] for p in grid], marker="o", color=PRIMARY)
	axes[0].scatter([winner["lr"]], [winner["best_val_iou"]], s=110, facecolors="none",
	                edgecolors=RULE, zorder=3, label=f"selected, {winner['lr']:.0e}")
	axes[0].set_xscale("log")
	axes[0].set_xlabel("Learning rate")
	axes[0].set_ylabel("Best validation IoU")
	axes[0].set_title("Best validation IoU per learning rate", fontsize=10)
	axes[0].legend(frameon=False, fontsize=8)

	# Ascending rate, so the winner and its near-tie take the two warm ends of the palette and
	# the three rates the sweep rejects fall away into the cool end.
	for i, point in enumerate(grid):
		history = point["val_iou_hist"]
		axes[1].plot(range(1, len(history) + 1), history, lw=1.4, ms=4,
		             marker=SWEEP_MARKERS[i % len(SWEEP_MARKERS)],
		             color=SWEEP_SERIES[i % len(SWEEP_SERIES)], label=f"{point['lr']:.0e}")
	axes[1].set_xlabel("Epoch")
	axes[1].set_ylabel("Validation IoU")
	axes[1].set_title("Validation IoU per epoch, one line per rate", fontsize=10)
	axes[1].legend(frameon=False, fontsize=8, ncol=2, title="learning rate", title_fontsize=8)

	for ax in axes:
		ax.grid(alpha=0.3)
	# Phrased without an article before the epoch count, which would otherwise need to agree
	# with a number read out of the run.
	fig.suptitle(f"Learning-rate sweep: {len(grid)} points on fold {meta['fold']}, reduced budget "
	             f"of {meta['epochs']} epochs, run {meta['run_id']}", fontsize=10)
	fig.tight_layout()

	path = out_dir / "lr_sweep.png"
	fig.savefig(path, dpi=200, bbox_inches="tight")
	plt.close(fig)
	for point in grid:
		mark = "  selected" if point["lr"] == meta["best_lr"] else ""
		print(f"  lr {point['lr']:.0e}  best_val_iou {point['best_val_iou']:.4f}  "
		      f"deficit {best - point['best_val_iou']:.4f}  "
		      f"best epoch {int(np.argmax(point['val_iou_hist'])) + 1}{mark}")
	if meta["best_lr"] in (grid[0]["lr"], grid[-1]["lr"]):
		print("  warning: the winner sits at the edge of the grid, so the range was too narrow")
	return path


def notool_share_figure(out_dir: Path, run: str) -> Path:
	"""The sampler's realised tool-free share per fold, against its target and the natural rate."""
	meta = fetch(f"{BUCKET}/experiments/{run}/results.json", "baseline_results.json")
	config, data = meta["config"], meta["data"]
	realised = [100 * share for share in config["sampler_notool_share_per_fold"]]
	target = 100 * config["sampler_notool_target"]
	natural = 100 * data["n_notool"] / data["n_pairs"]

	fig, ax = plt.subplots(figsize=(6.4, 3.8))
	ax.bar(range(len(realised)), realised, color=PRIMARY, edgecolor="white",
	       label="realised in training")
	ax.axhline(target, color=RULE, ls="--", lw=1, label=f"target {target:.0f}%")
	ax.axhline(natural, color=SECONDARY, ls=":", lw=1, label=f"natural {natural:.1f}%")
	ax.set_xticks(range(len(realised)))
	ax.set_xlabel("Fold")
	ax.set_ylabel("Share of training samples with no tool (%)")
	ax.set_title("What the tool-free oversampling delivered per fold", fontsize=10)
	# Headroom above the tallest bar, so the legend sits in empty axes rather than over the data.
	ax.set_ylim(0, max(max(realised), target) * 1.42)
	ax.grid(alpha=0.3, axis="y")
	ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=3)
	fig.tight_layout()

	path = out_dir / "notool_share.png"
	fig.savefig(path, dpi=200, bbox_inches="tight")
	plt.close(fig)
	print(f"  natural {natural:.1f}%  target {target:.0f}%  realised "
	      + " ".join(f"{s:.1f}%" for s in realised))
	return path


# === Main ===

def main() -> int:

	out_dir = Path(OUT_DIR)
	out_dir.mkdir(parents=True, exist_ok=True)
	print("learning-rate sweep")
	print(f"  wrote {lr_sweep_figure(out_dir)}")
	print("tool-free sampler share")
	print(f"  wrote {notool_share_figure(out_dir, RUN)}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
