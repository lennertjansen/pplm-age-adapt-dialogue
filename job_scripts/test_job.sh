#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=ExampleJob
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=04:00:00
#SBATCH --mem=32000M
#SBATCH --output=slurm_output_%A.out

module purge
module load 2019
module load Python/3.7.5-foss-2019b

# Your job starts in the directory where you call sbatch
cd $HOME/code/msc-ai-thesis

# Run your code
python klad2.py