#!/bin/bash

#SBATCH -p gpu_titanrtx_shared ## Select the partition. This one is almost always free, and has TitanRTXes (much RAM)
# #SBATCH -p gpu_shared ## This one is much cheaper on your credits, but has longer queue times and less RAM/is slower (GTX1080)
# #SBATCH --ntasks=1 ## not relevant
#SBATCH --time=8:30:00 ## Max time your script runs for (max is 5-00:00:00)
# #SBATCH --time=0:05:00
#SBATCH --gres=gpu:1 ## This means "I want one gpu"
#SBATCH --array=508-540%5 ## Array job number. Will run the same script with different array numbers, %5 means only five at a time
#SBATCH --job-name=irm ## job name for calling 'squeue -u dnobbe'
#SBATCH -o /home/dnobbe/oodg-experiments/SLURM/output/IRM-B.%j_%a.out ## this is where the terminal output is printed to. %j is root job number, %a array number

## COMMENT: So above here we give some of the settings for our job.
## You run it with the command 'sbatch lennert_IRMv1-B.sh'
## I've added an explanation after each setting
## They need the #SBATCH in front. making it '# #SBATCH' effectively comments it out

## You can run multiple runs in parallel using the array setting above.
## I use the array number to index into a text file that has the arguments for that run
## that's what happens here: 

declare -i exp_number=${SLURM_ARRAY_TASK_ID} 
param_file='SLURM/scripts/runfiles/IRM-B.txt';
run_params=`sed -n ${exp_number}p ${param_file}`; 
# Check if not a comment in param file
if [[ $run_params =~ "#" ]]; then
    echo "Exiting, settings line commented"
    exit 1 
fi
## This also supports commenting in the arguments text file
## it reads out the line at exp_number, and exits if it starts with a #


## normal conda stuff for lisa
module load 2020
module load Miniconda3
source activate base # Required to activate conda
#source activate oodg12
# conda activate thesis_lisa # this works nowadays
export LD_LIBRARY_PATH=~/.conda/envs/oodg/lib:$LD_LIBRARY_PATH ## Needed this to compile, you might not need this

cd ~/oodg-experiments/
nvidia-smi ## This prints your gpu status before running
# ln -s /home/dnobbe/oodg-experiments/detectron2/configs  /home/dnobbe/oodg-experiments/detectron2/detectron2/model_zoo/configs

# Select parameters to run with
# param_file='SLURM/scripts/runfiles/V-REx-3.txt';
# run_params=`sed -n 36p ${param_file}`; 

## At the ${run_params} I insert the arguments from the text file
python experiments/general/train.py --eval_period 200 --min_iter 2000 --wandb_project oodg0 \
 ${run_params} --array_job_number ${exp_number} 
