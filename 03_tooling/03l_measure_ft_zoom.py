"""Check that the ROI-crop fine-tune trains on the magnifications it actually tests on."""
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

WORK = Path(os.environ.get("CLIP_WORK_DIR", Path.home() / "lemon_clip_build"))

# === Config ===
N_FRAMES = 400
CASCADE_SQUARE_ROI = True
CASCADE_MIN_BOX_PX = 16
CASCADE_MAX_ZOOM = 2.5
CASCADE_MAX_ROIS = 3
CASCADE_MIN_COMPONENT_PX = 64
ROI_FT_PAD_RANGE = (0.15, 0.60)
RUN_TRACKING_CASCADE = False
TRACKER_KIND = "auto"
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512



# === Core ===

def _find_notebook():
    """Locate 00_model_training.ipynb by walking up, so moving this file cannot break it."""
    if "CASCADE_NB" in os.environ:
        return Path(os.environ["CASCADE_NB"])
    for parent in Path(__file__).resolve().parents:
        cand = parent / "00_model_training" / "00_model_training.ipynb"
        if cand.exists():
            return cand
    raise SystemExit("could not find 00_model_training.ipynb; set CASCADE_NB")


_cells = {c["id"]: "".join(c["source"])
          for c in json.loads(_find_notebook().read_text())["cells"]}
exec(compile(_cells["sec-7c-code"], "<cell sec-7c-code>", "exec"), globals())


def load_fg_colour():
    lm = WORK / "labelmap.txt"
    if not lm.exists():
        import subprocess
        subprocess.run(["gsutil", "-q", "cp",
                        "gs://laparoscopic-seg-REPLACE_K_NUMBER/trainingFrames/labelmap.txt", str(lm)],
                       check=True)
    for line in lm.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, rgb, *_ = line.split(":")
        if name == "instrument":
            return np.array([int(v) for v in rgb.split(",")], dtype=np.uint8)
    raise SystemExit("no 'instrument' label in labelmap.txt")


FG = load_fg_colour()
N = N_FRAMES
pairs = []
for mp in sorted((WORK / "jpegs" / "SegmentationClass").rglob("frame_*.png"))[:N]:
    ip = (WORK / "jpegs" / "JPEGImages"
          / mp.relative_to(WORK / "jpegs" / "SegmentationClass").with_suffix(".jpg"))
    if ip.exists():
        pairs.append((ip, mp))

def zoom_of(box, w, h):
    """Linear zoom over the baseline downsample, as evaluate_cascade records it."""
    return min(IMAGE_WIDTH / box[2], IMAGE_HEIGHT / box[3]) / min(IMAGE_WIDTH / w, IMAGE_HEIGHT / h)


def crops(mode, lo, hi, cap):
    """Zoom each training crop would receive under one candidate recipe."""
    random.seed(0)
    out = []
    for ip, mp in pairs:
        img = cv2.imread(str(ip))
        h, w = img.shape[:2]
        m = np.all(cv2.cvtColor(cv2.imread(str(mp)), cv2.COLOR_BGR2RGB) == FG, axis=-1)
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        if not m.any():
            continue
        if mode == "component":
            boxes = component_boxes(m, CASCADE_MAX_ROIS, CASCADE_MIN_COMPONENT_PX)   # noqa: F821
            if not boxes:
                continue
            box = random.choice(boxes)
        else:
            box = mask_to_bbox(m)                                                     # noqa: F821
        min_px = zoom_capped_min_px(w, h) if cap else CASCADE_MIN_BOX_PX              # noqa: F821
        c = pad_box(box, w, h, random.uniform(lo, hi), min_px)                        # noqa: F821
        if c is None:
            continue
        assert 0 <= c[0] and 0 <= c[1] and c[0] + c[2] <= w and c[1] + c[3] <= h, "crop escapes frame"
        assert m[c[1]:c[1] + c[3], c[0]:c[0] + c[2]].any(), "crop contains no tool"
        out.append(zoom_of(c, w, h))
    return np.array(out)



# === Main ===

def main():
    print(f"frames measured: {len(pairs)}\n")
    RECIPES = [
        ("whole-mask bbox, pad 0.15-0.60 (rejected)", "whole",     0.15, 0.60, False),
        ("per-component, pad 0.15-0.60, no cap",      "component", 0.15, 0.60, False),
        ("per-component, pad 0.15-0.60, zoom cap",    "component", 0.15, 0.60, True),
    ]
    hdr = f"{'train recipe':<44}{'p25':>7}{'median':>8}{'p75':>7}{'p90':>7}{'max':>7}{'>1.01':>8}{'>1.5':>7}"
    print(hdr)
    for label, mode, lo, hi, cap in RECIPES:
        z = crops(mode, lo, hi, cap)
        print(f"{label:<44}{np.percentile(z, 25):7.2f}{np.median(z):8.2f}{np.percentile(z, 75):7.2f}"
              f"{np.percentile(z, 90):7.2f}{z.max():7.2f}{(z > 1.01).mean():8.1%}{(z > 1.5).mean():7.1%}")

    print("\nTEST fused (ROI path)                       1.25    1.45   1.66   1.89   2.50   94.4%  43.1%")
    print("TEST fused_redetect (ROI path)              1.00    1.14   1.32   1.46   2.50   66.8%   8.1%")
    print("\nTarget: sit at or above test median, cover p90, without unreachable tail.")

if __name__ == "__main__":
    main()
