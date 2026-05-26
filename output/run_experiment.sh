#!/bin/bash

#SBATCH --partition=cpu_epyc7282
#SBATCH --time=180:00:00
#SBATCH --exclude=marvel-0-29,marvel-1-[19,21,23,25,27]

echo "${@}"
source ~/.bashrc
conda activate gt_cfr

"${@}"