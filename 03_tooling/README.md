# Tooling

This directory contains the notebooks and scripts used as tooling for the dissertation. These are included here for reproducibility as many support the dissertation's methodology and findings.

## Structure

### Notebooks

| Path | Role |
| ------ | ------ |
| [`03a_tool_size_distribution.ipynb`](03a_tool_size_distribution.ipynb) | Justifies the size thresholds used for `size_cls`, against equal-width and equal-count alternatives. |
| [`03b_tool_thinness_distribution.ipynb`](03b_tool_thinness_distribution.ipynb) | Justifies the thinness threshold used for `thin_cls`, against slenderness as an alternative. |
| [`03c_leak_guard.ipynb`](03c_leak_guard.ipynb) | Difference-hash scan over every frame and the fold leak-guard assertions. |
| [`03d_empirical_tuning_tool.ipynb`](03d_empirical_tuning_tool.ipynb) | Visual sweep that set the synthetic degradation strengths, stepping each operation up on real frames until the instrument stops being identifiable by eye. |
| [`03e_synthetic_aug_comparison.ipynb`](03e_synthetic_aug_comparison.ipynb) | Compares the four synthetic-augmentation probe runs and generates the multi-run bar and scatter figures for the evaluation chapter. |

### Scripts

| Path | Role |
| ------ | ------ |
| [`03f_temporal_clip_builder.py`](03f_temporal_clip_builder.py) | Rebuilds a 5-second lead-in clip for every annotated frame at native resolution, and pixel-matches each clip's tail against the stored annotated JPEG. The cascade cannot run without these. |
| [`03h_run_audit.py`](03h_run_audit.py) | Checks that the epoch and validation IoU inside each `best_fold{k}.pt` match that run's `results.json`. |
| [`03i_beta_sweep_figure.py`](03i_beta_sweep_figure.py) | Plots tiny-tool recall rising with Tversky beta, beside tiny-tool IoU failing to follow. |
| [`03j_configuration_figures.py`](03j_configuration_figures.py) | Plots the learning-rate sweep used to pick the baseline rate, and the per-fold tool-free share that oversampling actually delivered against its target. |
| [`03l_measure_ft_zoom.py`](03l_measure_ft_zoom.py) | Checks that the ROI-crop fine-tune trains on the magnifications the cascade actually produces at test time. |

### Scripts and Notebooks (unused)

| Path | Role |
| ------ | ------ |
| [`03g_frame_review_tool.ipynb`](03g_frame_review_tool.ipynb) | Scrolls through the worst-scoring frames so annotation errors can be flagged by hand. |
| [`03k_cleaned_reeval.py`](03k_cleaned_reeval.py) | Re-aggregates every scored arm with hand-reviewed annotation errors excluded, and writes the review candidate pool. |

## How to run

These notebooks and scripts are designed to be run locally.

### Prerequisites (these may not all be required)

- Authenticate with your Google account if you need to fetch new data from GCS. Otherwise, the results are already bundled locally in `../00_model_training/artefacts/`, so GCS access is no longer strictly required to run these notebooks.
- Connect the notebooks to the Jupyter Notebook runtime. To run Jupyter Notebook locally:
  - Install JupyterLab `pip install jupyterlab`
  - Start the JupyterLab server `jupyter lab`
- Verify and fill the config section of the notebooks before running

### Running the notebooks

If all the prerequisites are met, you should be able to run the notebooks. The notebooks should be able to run all the cells at once.

## Notes

- This section relies on the helper scripts in section `01` to analyse the results.
- `03f_temporal_clip_builder.py` needs the LEMON source videos and a lot of storage, unlike everything else here.
