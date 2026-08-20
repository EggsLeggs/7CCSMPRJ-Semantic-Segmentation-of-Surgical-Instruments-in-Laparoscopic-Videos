# Results comparison

This directory contains the notebooks used to compare the results of the different experiment arms. This comparison forms the basis of the dissertation's findings and conclusions.

## Structure

### Notebooks

| Path | Role |
| ------ | ------ |
| [`02a_results_comparison.ipynb`](02a_results_comparison.ipynb) | Comparison of two scored runs or the whole set of scored runs. Breaks down the results by tool size, tool thinness, scene challenge and tool-free frames. |
| [`02b_cascade_mechanism.ipynb`](02b_cascade_mechanism.ipynb) | Used to check the cascade mechanism is working as intended and how it performs under different conditions. |

### Outputs

| Path | Role |
| ------ | ------ |
| [`outputs/`](outputs/) | Tables and figures from the comparison notebooks. |

## How to run

These notebooks are designed to be run locally.

### Prerequisites

- Authenticate with your Google account if you need to fetch new data from GCS. Otherwise, the results are already bundled locally in `../00_model_training/artefacts/`, so GCS access is no longer strictly required to run these notebooks.
- Connect the notebooks to the Jupyter Notebook runtime. To run Jupyter Notebook locally:
  - Install JupyterLab `pip install jupyterlab`
  - Start the JupyterLab server `jupyter lab`
- Verify and fill the config section of the notebooks before running (especially for `02b`, which requires you to configure the arms you want to compare).

### Running the notebooks

If all the prerequisites are met, you should be able to run the notebooks. The notebooks should be able to run all the cells at once.

## Notes

- This section relies on the helper script in section `01` for the shared fetching, caching and strata definitions.
- Tables and figures are written to `outputs/<arm>_vs_<reference>/` so a number can be quoted without re-running the notebook.
- The learning-rate sweep and tool-free sampler figures describe how one run was configured rather than how two runs compare, so they live in [`../03_tooling/03j_configuration_figures.py`](../03_tooling/03j_configuration_figures.py).
