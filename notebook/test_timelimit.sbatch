#!/bin/bash
#SBATCH -J test_tl
#SBATCH --output=output/array_%A_%a.out
#SBATCH --error=output/array_%A_%a.err
#SBATCH --time=04:00:00

python3 utils.py