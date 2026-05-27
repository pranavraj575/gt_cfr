#!/bin/bash

#SBATCH --partition=cpu_epyc7282
#SBATCH --time=180:00:00
#SBATCH --exclude=marvel-0-29

echo "${@}"
source ~/.bashrc
conda activate gt_cfr

"${@}"