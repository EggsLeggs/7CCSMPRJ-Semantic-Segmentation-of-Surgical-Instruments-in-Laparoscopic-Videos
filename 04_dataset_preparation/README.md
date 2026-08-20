# Dataset preparation

This directory contains a few scripts that downloaded the LEMON videos, sampled and extracted them into frames, and assigned those frames to annotators. Each script was designed to be run once and is included here for reproducibility of the dataset. Scripts that used random functions were given a SEED to ensure reproducibility of the sample (42 was used in the dissertation).

These scripts ultimately produce three artefacts: a sampling plan, a log of the sampling process, and an enriched Google Sheet file used by the annotators to track progress and record challenges.

## Structure

### Scripts

| Path | Role |
| ------ | ------ |
| [`scripts/04a_download.py`](scripts/04a_download.py) | Downloads the `visurg/LEMON` videos from the Hugging Face dataset into `./lemon_data`. |
| [`scripts/04b_sampler.py`](scripts/04b_sampler.py) | Reads metadata from `LEMON_labels.json`. Uses the strata of procedure, robotic, and resolution to create a weighted sample of 5000 frames to annotate, divided 1000 each to five annotators. Extracts each frame with ffmpeg into `<annotator>/<youtubeId>/frame_<id>.jpg`, and wrote `artefacts/04b_sampling_plan.csv` and `artefacts/04b_lemon_sampler.log`. |
| [`scripts/04c_update_sheet.py`](scripts/04c_update_sheet.py) | Using the `artefacts/04b_sampling_plan.csv`, updates the Annotation Tracker Google Sheet with frame details and assignments. |

### Artefacts

| Path | Role |
| ------ | ------ |
| [`artefacts/04b_sampling_plan.csv`](artefacts/04b_sampling_plan.csv) | Local dry run export of the sampling plan and annotator assignment. Contains a few columns that are not in the Google Sheet, such as `video_title` as they are not relevant to the dissertation. `../03_tooling/03f_temporal_clip_builder.py` depends on the GCS copy of this file to build the temporal clips in the cascade experiments as it contains the full precision frame position `normalized_position`. |
| [`artefacts/04b_lemon_sampler.log`](artefacts/04b_lemon_sampler.log) | Local export of the log from the sampler script, including the per-stratum allocation table. |
| `artefacts/04c_annotation_tracker.xlsx` | Local export of the Annotation Tracker Google Sheet, used by annotators to track progress and record challenges. Status' may be out of sync with the live sheet as `../03_tooling/03g_frame_review_tool.ipynb` may update frame statuses. The current sheet is available at <https://docs.google.com/spreadsheets/d/REPLACE_SHEET_ID/edit?usp=sharing> |

## How to run

### Prerequisites

- Python 3.10+ with [`scripts/requirements.txt`](scripts/requirements.txt) installed
- `hf auth login` to authenticate with Hugging Face
- `ffmpeg` and `ffprobe` on the PATH
- Download `labels.json` from the [visurg-ai/LEMON](https://github.com/visurg-ai/LEMON) GitHub repo
- Verify and fill the config section of the scripts before running

### Running order

Run them in this order. Each full run depends on the previous script:

1. [`scripts/04a_download.py`](scripts/04a_download.py)
2. [`scripts/04b_sampler.py`](scripts/04b_sampler.py)
3. [`scripts/04c_update_sheet.py`](scripts/04c_update_sheet.py)

## Notes

- The way `scripts/04b_sampler.py` is written is not ideal, it requires every single video to be downloaded locally which takes a lot of time and an enormous amount of storage. If you are thinking of running this, it would be better to modify the script to download each video into a temporary directory, or into memory and then processed and deleted as needed.
- `scripts/04b_sampler.py` can started as a dry run via the config. When this is set, the videos don't need to be downloaded.
- Some of these scripts were cleaned up for the dissertation submission. The logic is identical though, including the less than ideal way of downloading the videos.
