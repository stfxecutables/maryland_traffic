# Files and Directories

## Results

The results from the initial major `df-analyze` run using only the LightGBM model
are located in `results/traffic_results_subset`.


## Downloading Data

Data should be downloaded from the [Montgomery Count
Website](https://data.montgomerycountymd.gov/Public-Safety/Traffic-Violations/4mse-ku6q/about_data)
and saved in the project root as `traffic_violations_complete.csv` (see
`clean_data.py`).


## Code Files

### Python Scripts

- *`clean_data.py`*: performs cleaning of the raw data, and places outputs in
  `traffic_data`
- *`summary.py`*: summarizes the results of the full job (see ["Job
  Scripts"](#job-scripts) below) into various tables (some printed to stdout,
  some saved)

### Job Scripts

Contains the actual scripts for running on Compute Canada. Will require setting
up a container to run `df-analyze`.
