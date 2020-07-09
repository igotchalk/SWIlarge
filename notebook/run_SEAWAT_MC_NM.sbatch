#!/bin/bash
#SBATCH -J NM_SEAWAT
#SBATCH --array=1-500
#SBATCH --output=/scratch/users/ianpg/SWIlarge/work/output/NM/sbatch/array_%A_%a.out
#SBATCH --error=/scratch/users/ianpg/SWIlarge/work/output/NM/sbatch/array_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=10:00:00

python3 run_SEAWAT_MC_NM_0707.py $SLURM_ARRAY_TASK_ID /home/users/ianpg/git_cloned/SWIlarge/data/PriorModel/varlist.pkl $SLURM_ARRAY_JOB_ID

