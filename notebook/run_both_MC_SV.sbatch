#!/bin/bash
#SBATCH -J SV_both
#SBATCH --array=1-500
#SBATCH --output=/scratch/users/ianpg/SWIlarge/work/output/SV/sbatch/array_%A_%a.out
#SBATCH --error=/scratch/users/ianpg/SWIlarge/work/output/SV/sbatch/array_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=15:00:00

python3 run_SEAWAT_MC_SV_0715.py $SLURM_ARRAY_TASK_ID /home/users/ianpg/git_cloned/SWIlarge/data/PriorModel/varlist.pkl $SLURM_ARRAY_JOB_ID

python3 run_AEM_SV_MC.py $SLURM_ARRAY_TASK_ID /home/users/ianpg/git_cloned/SWIlarge/data/PriorModel/varlist.pkl $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_JOB_ID
