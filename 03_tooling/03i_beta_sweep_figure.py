#!/usr/bin/env python3
"""Chapter figure for the Tversky beta sweep: recall rises with beta, IoU on tiny tools does not."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt

# === Config ===
RUN_NAME = "tversky2_cv_balsweep_Unet_resnet34_bce_tversky_lr1e-04_p20/20260808-013552"
OUT_PATH = "notes/writeup/figures/beta_sweep_recall_vs_iou.png"

BUCKET = "gs://training-checkpoints-REPLACE_K_NUMBER"


CACHE = Path.home() / ".cache/laparoscopic_seg/beta_sweep"

# left panel is the mechanism, right panel is the outcome measure it fails to move
PANELS = [
	("recall_tiny", "Recall on tiny tools", "the dial the loss is designed to turn"),
	("iou_tiny", "IoU on tiny tools", "the outcome measure it has to convert into"),
]


# === Core ===

def load(run: str) -> dict:
	CACHE.mkdir(parents=True, exist_ok=True)
	dest = CACHE / f"{run.replace('/', '_')}.json"
	if not dest.exists():
		src = f"{BUCKET}/experiments/{run}/loss_balance_results.json"
		if subprocess.run(["gsutil", "-q", "cp", src, str(dest)]).returncode:
			# The sweep block is also embedded in the arm's results.json, so fall back to it
			# rather than failing when only the CV artefacts were kept.
			alt = CACHE / f"{run.replace('/', '_')}_results.json"
			if subprocess.run(["gsutil", "-q", "cp",
			                   f"{BUCKET}/experiments/{run}/results.json", str(alt)]).returncode:
				raise SystemExit(f"no sweep artefacts under {BUCKET}/experiments/{run}")
			embedded = json.loads(alt.read_text())["config"].get("loss_balance_sweep")
			if not embedded:
				raise SystemExit(f"{run} has no loss_balance_sweep block; was the sweep run?")
			dest.write_text(json.dumps(embedded))
	return json.loads(dest.read_text())


# === Main ===

def main() -> int:

	data = load(RUN_NAME)
	agg, runs = data["by_beta"], data.get("runs", [])
	betas = [g["beta"] for g in agg]
	anchor = float(data.get("best_beta", 0.7))

	fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
	for ax, (key, title, subtitle) in zip(axes, PANELS):
		means = [g[key] for g in agg]
		spread = [g.get(f"{key}_spread", 0.0) for g in agg]
		ax.fill_between(betas, [m - s / 2 for m, s in zip(means, spread)],
		                [m + s / 2 for m, s in zip(means, spread)],
		                alpha=0.15, color="red", label="seed spread")
		ax.plot(betas, means, "o-", color="red", label="seed mean")
		if runs:
			ax.scatter([r["beta"] for r in runs], [r[key] for r in runs],
			           c="#6c757d", s=16, zorder=3, label="per seed")
		ax.axvline(0.5, ls=":", c="#333333")
		ax.annotate("alpha = beta = 0.5,\nTversky reduces to Dice",
		            xy=(0.5, 1.0), xycoords=("data", "axes fraction"),
		            xytext=(5, -4), textcoords="offset points", fontsize=7, color="#333333",
		            va="top", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
		ax.axvline(anchor, ls="--", c="#333333", lw=1)
		ax.set_xlabel("Tversky beta (alpha = 1 - beta)")
		# The stored key is the column name, not a label a reader should be shown.
		ax.set_ylabel(title)
		ax.set_title(subtitle, fontsize=10)
		ax.grid(alpha=0.3)
		ax.legend(fontsize=7, frameon=False)

	delta = agg[-1]["recall_tiny"] - agg[0]["recall_tiny"]
	fig.suptitle(f"Tversky beta sweep: recall on tiny tools moves {delta:+.3f} across the grid "
	             f"while IoU on tiny tools does not follow (fold 0, reduced epoch budget)", fontsize=10)
	fig.tight_layout()

	out = Path(OUT_PATH)
	out.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out, dpi=200)
	print(f"recall_tiny {agg[0]['recall_tiny']:.4f} at beta {betas[0]:.2f} -> "
	      f"{agg[-1]['recall_tiny']:.4f} at beta {betas[-1]:.2f}  ({delta:+.4f})")
	print(f"iou_tiny    {agg[0]['iou_tiny']:.4f} -> {agg[-1]['iou_tiny']:.4f}  "
	      f"(seed noise floor {max(g.get('iou_tiny_spread', 0.0) for g in agg):.4f})")
	print(f"selection verdict: {data.get('reason', 'not recorded')}")
	print(f"wrote {out}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
