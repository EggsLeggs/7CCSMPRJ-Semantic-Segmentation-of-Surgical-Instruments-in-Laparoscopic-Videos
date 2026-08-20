# Laparoscopic instrument segmentation

This repository contains the source code for an MSc dissertation project on semantic segmentation of surgical instruments in laparoscopic video.

## Dissertation hypothesis

We hypothesise that the failures of a laparoscopic tool-segmentation model are not spread evenly across
the data, but concentrated within a small, identifiable set of conditions: tools that are small or thin,
and scenes degraded by smoke, blood, or poor exposure. Within these conditions the model errs in both
directions, missing the tool entirely in some frames and over-segmenting onto surrounding tissue in others.
Because these conditions can be identified frame by frame, we hypothesise that interventions targeted
directly at them will improve worst-case robustness in these weak regions more effectively than optimising
for average-case performance alone, without degrading performance elsewhere. These interventions include:
a weighted BCE–Tversky loss to explicitly balance recall against precision; targeted synthetic augmentation
to expose the model to degraded conditions during training; and a tracking-based cascade that crops to
the tool’s tracked region to increase effective resolution, with progressively added geometric safety margins
and recovery logic to protect against tracking failure. We evaluate each of these individually against the
baseline, with combined interventions considered as a possible extension. We expect this to hold using only
the existing endoscopic video and annotations, without additional data, sensors, or model capacity.

## Dataset

- [visurg/LEMON](https://huggingface.co/datasets/visurg/LEMON) on Hugging Face, with labels from the [visurg-ai/LEMON](https://github.com/visurg-ai/LEMON) GitHub repo
- [Annotator spreadsheet](https://docs.google.com/spreadsheets/d/REPLACE_SHEET_ID/edit?usp=sharing)

## Structure

This repository is split into numbered sections, ordered by how frequently they are used. For example the contents of the`04_dataset_preparation/` directory are only used once to prepare the dataset for training, whereas the contents of the`00_model_training/` directory contains the core notebook for running the experiments.

| Path | Contents |
| ------ | ------------- |
| `00_model_training/` | Notebook used to train and evaluate experiment arms (baseline and each intervention) |
| `01_baseline_analysis/` | Notebooks used to analyse the results of the baseline |
| `02_results_comparison/` | Notebooks used to compare results from different experiment arms |
| `03_tooling/` | One-off tooling used to analyse the data or results for either the dissertation or for configuring experiments |
| `04_dataset_preparation/` | Scripts used to sample the dataset and assign frames to annotators |

Instead of listing each file in this readme, which would be excessively long and unclear, a full table of contents is provided in a `README.md` under each section's directory.

## Notes

- This is covered in more detail under `00_model_training/README.md`, but its important to note that `00_model_training/00_model_training.ipynb` has only been tested on Google Colab Enterprise. It is probably quite easy to adapt it to run locally but it wasn't done for the dissertation due to time constraints and a lack of suitable hardware to test on. In its current state, it is not recommended to run it outside of Google Colab Enterprise.
- Apart from `00_model_training/00_model_training.ipynb`, the code in each section should easily be able to be run locally, unless specified otherwise in a `README.md` or the file itself.
- All frames and results are stored in a GCS bucket. To prevent any accidental abuse, this bucket is private and requires authentication to access. If you would like access to the bucket, please reach out to me and I will sort something out.
- Most of the notebooks under sections `01`, `02`, and `03` normally need access to the GCS bucket with results. However, to make this repository reproducible without bucket access, the lightweight `results.json` and `per_image_results.csv` files have been bundled locally into `00_model_training/artefacts/`. You can run the analysis notebooks using these local files without needing to connect to GCS.
- I haven't vendored third-party libraries in this repository. If you want to run the notebooks locally, you will need to install the packages imported by the notebooks in your environment. A `requirements.txt` is provided in each section's directory, or installed via a cell in the notebooks for convenience.
- Copy the local pre-commit hook before committing: `cp pre-commit .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit` (it scans staged files for secrets and blocks the commit if it finds any; it does not rewrite files).
