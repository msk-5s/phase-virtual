# **phase-virtual**

This repository contains the source code for recreating the research in "Virtual Voltage Measurements via Averaging for Phase Identification in Power Distribution Systems". The dataset can be downloaded from [Kaggle](https://www.kaggle.com/msk5sdata/phase-svd). Alternatively, the dataset in this work can be recreated from scratch using the [phase-svd-opendss](https://github.com/msk-5s/phase-svd-opendss.git) repository.

## Requirements
    - Python 3.8+ (64-bit)
    - See requirements.txt file for the required python packages.

## Folders
`data/`
: The voltage magnitude dataset, default ckt5 load profiles, synthetic load profiles, and metadata should be placed in this folder (download it from [Kaggle](https://www.kaggle.com/msk5sdata/phase-svd)).

`results/`
: This folder contains the phase identification results for different window widths and run counts. These are the results reported in the paper.

## Running
The `run.py` script can be used to run phase identification for a given snapshot in the year of data. `run_suite.py` can be used to run phase identification across the entire year of data for different window widths and noise percents. If you have access to a computing cluster, then the `base_submit_suite.sh` script can be used to run `run_suite.py` as an array job. See the script comments for details.
> **NOTE: `run_suite.py` will save results to the `results/` folder.**
