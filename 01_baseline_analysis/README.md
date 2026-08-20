# Baseline analysis

This directory contains the notebooks used to analyse the results of the baseline. This analysis formed the basis of the dissertation's hypothesis and the subsequent experiments.

## Structure

### Notebook

| Path | Role |
| ------ | ------ |
| [`01a_baseline_failure_analysis.ipynb`](01a_baseline_failure_analysis.ipynb) | Analysis of the baseline results, broken down by annotator, surgery type, resolution, tool size, tool thinness and challenge label. |
| [`01b_baseline_smalltool_failure_analysis.ipynb`](01b_baseline_smalltool_failure_analysis.ipynb) | Further analysis of the baseline results, given weaknesses identified in `01a`. |

### Helper Scripts

| Path | Role |
| ------ | ------ |
| [`baseline_common.py`](baseline_common.py) | Shared constants and helpers used by `01a` and `01b`. This is also used by notebooks in sections `02` and `03`. |

## How to run

These notebooks are designed to be run locally.

### Prerequisites

- Authenticate with your Google account if you need to fetch new data from GCS. Otherwise, the results are already bundled locally in `../00_model_training/artefacts/`, so GCS access is no longer strictly required to run these notebooks.
- Connect the notebooks to the Jupyter Notebook runtime. To run Jupyter Notebook locally:
  - Install JupyterLab `pip install jupyterlab`
  - Start the JupyterLab server `jupyter lab`
- Verify and fill the config section of the notebooks before running

### Running the notebooks

If all the prerequisites are met, you should be able to run the notebooks. The notebooks should be able to run all the cells at once.

## Notes

- The strata are the ones the runs were scored under, not recomputed here. Size bands use the training cuts on frame area (tiny < 4.5%, small < 8%, medium < 19%, large >= 19%); the thin/wide band is the run's own `thin_class`, cut at the 25th percentile of mean local width (`thin_cut_px`, 22.81 px at 512 x 512). `03a` and `03b` justify both cuts against the alternatives.
