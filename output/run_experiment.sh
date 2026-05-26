#!/bin/bash

#SBATCH --partition=cpu_epyc7282
#SBATCH --time=180:00:00
#SBATCH --mem=48G
#SBATCH --exclude=marvel-0-29,marvel-1-[19,21,23,25,27]

og_args=$@
new_args=${og_args//QUOTE/\"}
echo python xdo.py $new_args

source ~/.bashrc
conda activate gt_cfr
python xdo.py $new_args
