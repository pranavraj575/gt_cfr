#!/bin/bash

#SBATCH --partition=cpu_epyc7282
#SBATCH --time=180:00:00
#SBATCH --mem=48G
#SBATCH --exclude=marvel-0-29,marvel-1-[19,21,23,25,27]
#SBATCH --output=./my_script_$1.out
#SBATCH --error=./my_script_$1.err

python xdo.py $1
