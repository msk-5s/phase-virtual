#!/bin/bash
# <Place HPC tool specific headers here>

# SPDX-License-Identifier: MIT

# This bash script is for submitting an array job to a High Performance Computing (HPC) tool such
# as SLURM. Depending on the tool being used, you may only need to change `SLURM_ARRAY_TASK_ID`
# to the environment variable that is approriate for your HPC tool. Prepend any tool specific
# headers at line 2 above.

python3 run_suite.py ${SLURM_ARRAY_TASK_ID}
