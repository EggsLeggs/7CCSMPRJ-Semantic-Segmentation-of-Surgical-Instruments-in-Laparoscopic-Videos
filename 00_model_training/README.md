# Model Training

This directory contains the primary training notebook used to configure, train, and evaluate the segmentation models.

## Structure

### Notebooks

| Path | Role |
| ------ | ------ |
| [`00_model_training.ipynb`](00_model_training.ipynb) | The central training pipeline. Handles cross-validation, hyperparameter sweeps, model checkpointing, and evaluation across various metrics. |

### Artefacts

| Path | Role |
| ------ | ------ |
| [`artefacts/`](artefacts/) | Results from the training runs. |

## Project storage

Data is read from a GCS bucket and a spreadsheet. Results are written to a second GCS bucket.

### Input dataset (GCS)
Manually combined labelled frames from the LEMON dataset. Labelling was done in CVAT and exported with CVAT's Segmentation Mask format (Pascal VOC-based). Five annotators labelled frames; each person's exported masks were programmatically combined into one collection.

```
gs://laparoscopic-seg-<KCL_ID>/trainingFrames/
    JPEGImages/
    SegmentationClass/
    SegmentationObject/   <- (instance-segmentation masks; unused for semantic segmentation)
    ImageSets/
        Segmentation/
            default.txt
    NoTools/              <- (addition that breaks spec but helps with training)
    labelmap.txt
```

### Metadata spreadsheet
A Google Sheet set via `SHEET_ID` in the config cell, with per-video annotator metadata (Video ID, Robotic, Challenges, No Tools, Status).

- **Stratification**: `Video ID` groups frames by video for leak-safe splitting; `Challenges` and `Robotic` balance the 5-fold split.
- **Retiring frames**: frames with `Status` `deleted`/`invalid`/`duplicate`/`excluded` are dropped.
- **No-tool disagreements**: some annotators mislabelled tool presence; when the sheet's `No Tools` flag disagrees with the mask, the mask wins.

### Output checkpoints (GCS)
`RUN_NAME` and `RUN_ID` come from the config cell.
```
gs://training-checkpoints-<KCL_ID>/experiments/{RUN_NAME}/{RUN_ID}/
    best_fold{0..4}.pt      <- best checkpoint per fold, by val_iou
    fold{0..4}_scores.json  <- per-fold summary + history, written as each fold finishes
    sweep_results.json      <- LR sweep grid (if RUN_SWEEP)
    results.json            <- final run summary: config + per-fold metrics + mean±std
    training_curves.png
    per_image_results.csv   <- one row per scored frame, incl. per-threshold TP/FP/FN counts
    folds.json              <- the splits this run used; reused by later arms via FOLDS_JSON
    PredictedMasks/         <- per-frame predicted masks (if EXPORT_PRED_MASKS)
    last_cell_figs.zip      <- best/median/worst panels from the final cell
```
