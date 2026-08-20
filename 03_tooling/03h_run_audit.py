#!/usr/bin/env python3
"""Artefact-level audit of every run: folds.json, checkpoint metadata, and no unintended checkpoint sharing."""

from __future__ import annotations

import base64
import io
import json
import pickle
import re
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib
_reeval = importlib.import_module('03k_cleaned_reeval')
RUNS = _reeval.RUNS
EXPERIMENTS = _reeval.EXPERIMENTS  # single declared run list, do not duplicate it

# === Config ===
DEEP = False           # if True, also open every checkpoint (downloads ~9GB)
ONLY_ARMS = []         # restrict to these arm names, or empty for all

CACHE = Path.home() / ".cache/laparoscopic_seg/run_audit"
FOLDS_MD5 = "a7fb12152395ffa3e40674e7ca3f466c"

# How each arm was produced, which decides what "correct" means for it. Mirrors archive/00_model_training/run_log.md.
# rescore: copied finished weights and only re-evaluated them, so the weights must match
# the source byte for byte. finetune: section 12e, per_fold_best is empty by design.
MODES = {
	"tversky-rescore-hi":         ("rescore", "bce_tversky"),
	"cascade-redetect-clean":     ("rescore", "baseline-seed42-clean"),
	"cascade-roift-neg-redetect": ("rescore", "roift-neg-clean"),
	"roift-clean":                ("finetune", "baseline-seed2026"),
	"roift-tversky":              ("finetune", "bce_tversky"),
	"roift-neg-clean":            ("finetune", "baseline-seed2026"),
	"roift-control-ff":           ("finetune", "baseline-seed2026"),
	"roift-small-neg":            ("finetune", "baseline-seed2026"),
	"roift-neg-strong":           ("finetune", "baseline-seed2026"),
	"ft-long-control":            ("finetune", "baseline-seed2026"),
	"roift-long":                 ("finetune", "baseline-seed2026"),
	"roift-long-seed2026":        ("finetune", "baseline-seed2026"),
}

IOU_TOL = 1e-4   # results.json rounds through json, so exact float equality is too strict


# === Core ===

def gs_md5(uri: str) -> str | None:
	"""MD5 from GCS object metadata. No download, so this is cheap enough to run on everything."""
	r = subprocess.run(["gsutil", "stat", uri], capture_output=True, text=True)
	if r.returncode:
		return None
	m = re.search(r"Hash \(md5\):\s*(\S+)", r.stdout)
	if not m:
		return None
	return base64.b64decode(m.group(1)).hex()


def fetch(uri: str, dest: Path) -> Path | None:
	dest.parent.mkdir(parents=True, exist_ok=True)
	if dest.exists():
		return dest
	if subprocess.run(["gsutil", "-q", "cp", uri, str(dest)]).returncode:
		return None
	return dest


def _stub(*args, **kwargs):
	return None


class _ScalarUnpickler(pickle.Unpickler):
	"""Reads a checkpoint's scalars without torch installed.

	A .pt is a zip whose data.pkl holds the saved dict; the tensors are separate members
	loaded through persistent_load. Stubbing the torch and numpy classes and returning
	None for every storage leaves the plain Python entries (epoch, val_iou) intact, so
	the audit runs on a laptop rather than needing the training environment.
	"""

	def find_class(self, module: str, name: str):
		if module.startswith(("torch", "numpy")):
			return _stub
		return super().find_class(module, name)

	def persistent_load(self, pid):
		return None


def ckpt_meta(uri: str, dest: Path) -> dict | None:
	"""epoch and val_iou from inside a checkpoint. Needs the file, hence --deep only."""
	path = fetch(uri, dest)
	if path is None:
		return None
	with zipfile.ZipFile(path) as z:
		member = next(n for n in z.namelist() if n.endswith("data.pkl"))
		blob = _ScalarUnpickler(io.BytesIO(z.read(member))).load()
	return {"epoch": blob.get("epoch"), "val_iou": blob.get("val_iou")}


def audit_run(arm: str, rel: str, deep: bool, md5s: dict) -> dict:
	run_dir = f"{EXPERIMENTS}/{rel}"
	mode, source = MODES.get(arm, ("train", None))
	problems: list[str] = []

	folds_md5 = gs_md5(f"{run_dir}/folds.json")
	if folds_md5 is None:
		problems.append("no folds.json")
	elif folds_md5 != FOLDS_MD5:
		problems.append(f"folds.json md5 {folds_md5[:8]} != {FOLDS_MD5[:8]}")

	for k in range(5):
		md5s[(arm, k)] = gs_md5(f"{run_dir}/best_fold{k}.pt")
	if any(md5s[(arm, k)] is None for k in range(5)):
		problems.append("missing checkpoint(s)")

	if mode == "rescore" and source in RUNS:
		src_dir = f"{EXPERIMENTS}/{RUNS[source]}"
		for k in range(5):
			if md5s[(arm, k)] != gs_md5(f"{src_dir}/best_fold{k}.pt"):
				problems.append(f"fold {k} weights differ from {source}")

	passed = 0
	if deep and mode != "rescore":
		res = fetch(f"{run_dir}/results.json", CACHE / arm / "results.json")
		best = json.loads(res.read_text()).get("per_fold_best", []) if res else []
		for k in range(5):
			meta = ckpt_meta(f"{run_dir}/best_fold{k}.pt", CACHE / arm / f"best_fold{k}.pt")
			if meta is None:
				problems.append(f"fold {k} unreadable")
				continue
			if mode == "finetune":
				# per_fold_best is empty by design here, so the check is that the epoch is
				# inside the fine-tune budget rather than a full-training epoch number.
				if meta["epoch"] is not None and meta["epoch"] <= 50:
					passed += 1
				else:
					problems.append(f"fold {k} epoch {meta['epoch']} outside the fine-tune budget")
			elif k < len(best):
				same_ep = meta["epoch"] == best[k].get("epoch")
				same_iou = abs((meta["val_iou"] or 0) - (best[k].get("val_iou") or 0)) < IOU_TOL
				if same_ep and same_iou:
					passed += 1
				else:
					problems.append(
						f"fold {k} checkpoint says epoch {meta['epoch']} / IoU "
						f"{meta['val_iou']:.4f}, results say epoch {best[k].get('epoch')} / "
						f"IoU {best[k].get('val_iou')}")
			else:
				problems.append(f"fold {k} has no per_fold_best entry")

	return {"arm": arm, "mode": mode, "consistent": f"{passed}/5" if deep and mode != "rescore" else "-",
	        "problems": problems}


# === Main ===

def main() -> int:

	wanted = ONLY_ARMS or list(RUNS)
	unknown = [a for a in wanted if a not in RUNS]
	if unknown:
		print(f"unknown arm(s): {unknown}\nknown: {list(RUNS)}")
		return 2

	md5s: dict = {}
	rows = [audit_run(a, RUNS[a], DEEP, md5s) for a in wanted]

	# Cross-run duplicate scan: the check that actually named the 7 Aug incident. A
	# re-scoring arm copies its source's weights on purpose, so that pairing is expected
	# and is filtered out; anything else sharing bytes is the failure mode being hunted.
	expected = {frozenset((arm, src)) for arm, (mode, src) in MODES.items() if mode == "rescore"}
	by_hash: dict = {}
	for (arm, k), h in md5s.items():
		if h:
			by_hash.setdefault(h, []).append((arm, k))
	shared = {}
	for h, members in by_hash.items():
		if len(members) < 2:
			continue
		arms = {a for a, _ in members}
		if len(arms) == 2 and frozenset(arms) in expected:
			continue
		shared[h] = [f"{a}:fold{k}" for a, k in members]

	print(f"{'arm':30s} {'mode':9s} {'consistent':11s} verdict")
	print("-" * 96)
	failures = 0
	for r in rows:
		verdict = "ok" if not r["problems"] else "; ".join(r["problems"])
		failures += bool(r["problems"])
		print(f"{r['arm']:30s} {r['mode']:9s} {r['consistent']:11s} {verdict}")

	print("")
	if shared:
		print("UNEXPECTED SHARED CHECKPOINTS, two runs hold the same bytes:")
		for h, v in shared.items():
			print(f"  {h[:12]}  {', '.join(v)}")
	else:
		print("no unexpected checkpoint sharing between runs")

	if not DEEP:
		print("\nrun again with --deep for the checkpoint-versus-results consistency check")
	print(f"\n{len(rows) - failures}/{len(rows)} arms clean")
	return 1 if (failures or shared) else 0


if __name__ == "__main__":
	raise SystemExit(main())
