#!/bin/bash
#SBATCH -J SV
#SBATCH --array=1-100
#SBATCH --output=output/array_%A_%a.out
#SBATCH --error=output/array_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=10:00:00


python3 test_load.py $SLURM_ARRAY_TASK_ID /home/users/ianpg/git_cloned/SWIlarge/data/PriorModel/varlist.pkl

