#!/bin/bash

#SBATCH --partition=gpu_short
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=debugger_blog_bert_fr
#SBATCH --time=1:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=lennertjansen95@gmail.com
#SBATCH -o /home/lennertj/code/msc-ai-thesis/SLURM/output/debugger_blog_bert_fr.%A.out ## this is where the terminal output is printed to. %j is root job number, %a array number. try %j_%a ipv %A (job id)

# Loading all necessary modules.
echo "Loading modules..."
module purge
module load 2020
#module load eb
#module load Python/3.7.5-foss-2019b
#module load Miniconda3
module load Anaconda3/2020.02

# Activate the conda environment
echo "Activating conda environment..."
source /home/lennertj/miniconda3/etc/profile.d/conda.sh
source activate base
conda info --envs
source activate /home/lennertj/miniconda3/envs/thesis_lisa2

# Change directories
echo "Changing directory"
cd $HOME/code/PPLM
#cd $HOME/code/msc-ai-thesis

# Run your code
echo "Running python code..."
# declare an array variable
declare -a arr=("bert-base-uncased")
#
# --dataset_fp '/home/lennertj/code/msc-ai-thesis/data/bnc/bnc_rb_full_generic_pplm.txt' \
# --dataset_fp '/home/lennertj/code/msc-ai-thesis/data/blogs_kaggle/blog_small_generic_pplm.txt' \

## now loop through the above array
for i in "${arr[@]}"
do
  for j in 1 2
  do

    python run_pplm_discrim_train.py --dataset 'generic' \
          --dataset_fp '/home/lennertj/code/msc-ai-thesis/data/blogs_kaggle/blog_small_generic_pplm.txt' \
          --epochs 2 \
          --batch_size 32 \
          --log_interval 20000 \
          --pretrained_model "$i"
  done
done

#for seed in 2021
#do
#  echo 'Starting new seed:'
#  echo "$seed"
#
#  python train_classifiers.py \
#         --data 'blog' \
#         --model_type 'bert' \
#         --mode 'train' \
#         --seed "$seed" \
#         --batch_size 8 \
#         --embedding_dim 128 \
#         --hidden_dim 256 \
#         --num_layers 2 \
#         --bidirectional \
#         --batch_first \
#         --epochs 1 \
#         --lr 0.001 \
#         --early_stopping_patience 3 \
#         --train_frac 0.75 \
#         --val_frac 0.15 \
#         --test_frac 0.1 \
#         --log_interval 1000 \
#         --no_tb
#done