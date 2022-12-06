#!/bin/bash

##SBATCH --partition=gpu
#SBATCH -p gpu_titanrtx_shared ## Select the partition. This one is almost always free, and has TitanRTXes (much RAM)
#SBATCH --nodes=1
##SBATCH --gpus-per-node=1
#SBATCH --job-name=ctg_YOUNG_prompt_100mcw_baseline_bow_fb_and_miu_young_and_old
#SBATCH --time=5-00:00:00 ## Max time your script runs for (max is 5-00:00:00 | 5 days)
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=lennertjansen95@gmail.com
#SBATCH -o /home/lennertj/code/msc-ai-thesis/SLURM/output/ctg_explore/ctg_YOUNG_prompt_100mcw_baseline_bow_fb_and_miu_young_and_old.%A.out ## this is where the terminal output is printed to. %j is root job number, %a array number. try %j_%a ipv %A (job id)

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
#cd $HOME/code/PPLM
cd $HOME/code/msc-ai-thesis

# Run your code
echo "Running python code..."

## for BoW-based
#for seed in 2021
#do
#  for length in 6 12 24 30 36 42 48 54 60
#  do
#    python plug_play/run_pplm.py \
#           --pretrained_model 'gpt2-medium' \
#           --num_samples 30 \
#           --bag_of_words 'plug_play/wordlists/bnc_rb_WS_100_mi_unigrams_old.txt' \
#           --length $length \
#           --seed $seed \
#           --sample \
#           --class_label 1 \
#           --verbosity "quiet" \
#           --uncond
#  done
#done

#declare -a arr=("Can you tell me about your last holidays?<|endoftext|>" "Can you tell me about your favorite food?<|endoftext|>" "Can you tell me about your hobbies?<|endoftext|>" "Once upon a time" "The last time" "The city")
#
#
## for discrim-based
#for i in "${arr[@]}"
#do
#  for length in 16 24 32
#  do
#    for seed in 2021 2022 2023
#    do
#      python plug_play/run_pplm.py \
#             --pretrained_model 'gpt2-medium' \
#             --cond_text "$i" \
#             --num_samples 2 \
#             --discrim 'generic' \
#             --length $length \
#             --seed $seed \
#             --sample \
#             --class_label 1 \
#             --verbosity "quiet" \
#             --discrim_weights "plug_play/discriminators/gpt2_incl_sw_nac/generic_pm_gpt2-medium_ml_512_lr_0.0001_classifier_head_epoch_19.pt" \
#             --discrim_meta "plug_play/discriminators/gpt2_incl_sw_nac/generic_pm_gpt2-medium_ml_512_lr_0.0001_classifier_head_meta.json"
#    done
#  done
#done



# Weights and metadata for discriminators

# gpt2-medium
#--discrim_weights "plug_play/discriminators/gpt2_incl_sw_nac/generic_pm_gpt2-medium_ml_512_lr_0.0001_classifier_head_epoch_19.pt" \
#--discrim_meta "plug_play/discriminators/gpt2_incl_sw_nac/generic_pm_gpt2-medium_ml_512_lr_0.0001_classifier_head_meta.json"

# microsoft/dialogpt-medium
#--discrim_weights "plug_play/discriminators/dialogpt-medium/generic_pm_microsoft-DialoGPT-medium_ml_512_lr_0.0001_classifier_head_epoch_17.pt" \
#--discrim_meta "plug_play/discriminators/dialogpt-medium/generic_pm_microsoft-DialoGPT-medium_ml_512_lr_0.0001_classifier_head_meta.json"

# for length in 6 12 24 30 36 42 48 54 60 # Lengths used for main results table until now (Monday, 27 September 2021)

# declare an array variable
#declare -a arr=("gpt2-medium")
#declare -a arr=("bert-base-uncased")

# data fps
# --dataset_fp '/home/lennertj/code/msc-ai-thesis/data/bnc/bnc_rb_full_generic_pplm.txt' \
# --dataset_fp '/home/lennertj/code/msc-ai-thesis/data/blogs_kaggle/blog_generic_pplm.txt' \

## now loop through the above array
#for i in "${arr[@]}"
#do
#  for j in 1 2 3 4 5
#  do
#
#    python run_pplm_discrim_train.py --dataset 'generic' \
#          --dataset_fp '/home/lennertj/code/msc-ai-thesis/data/blogs_kaggle/blog_generic_pplm.txt' \
#          --epochs 5 \
#          --batch_size 16 \
#          --log_interval 20000 \
#          --pretrained_model "$i"
#  done
#done

#for seed in 6 8 9 10 11 12 13
#do
#  echo 'Starting new seed:'
#  echo "$seed"
#
#  python train_classifiers.py \
#         --data 'bnc_rb' \
#         --model_type 'bert' \
#         --mode 'train' \
#         --seed "$seed" \
#         --batch_size 4 \
#         --embedding_dim 128 \
#         --hidden_dim 256 \
#         --num_layers 2 \
#         --batch_first \
#         --epochs 10 \
#         --lr 0.001 \
#         --early_stopping_patience 3 \
#         --train_frac 0.75 \
#         --val_frac 0.15 \
#         --test_frac 0.1 \
#         --log_interval 10000 \
#         --no_tb
#done

#python plug_play/run_pplm.py --pretrained_model 'microsoft/DialoGPT-medium' --cond_text 'Hello, how are you?<|endoftext|>' --num_samples 30 --discrim 'generic' --length 10 --seed 2021 --sample --class_label 1 --verbosity "verbose" --discrim_weights "plug_play/discriminators/dialogpt-medium/generic_pm_microsoft-DialoGPT-medium_ml_512_lr_0.0001_classifier_head_epoch_17.pt" --discrim_meta "plug_play/discriminators/dialogpt-medium/generic_pm_microsoft-DialoGPT-medium_ml_512_lr_0.0001_classifier_head_meta.json"

## Skeleton for job script loop

#############
# BoW-based #
#############

# for pretrained model in [gpt2, dialogpt]
## for control-attribute in [uncontrolled, young, old]
## --> if uncontrolled set stepsize and/or num_iterations to zero
## --> if young/old set class_label to 0/1, and wordlist to Y-FB/O-FB
### for conditioning in [unprompted, prompted]
### --> if unprompted set --uncond
### --> if prompted set prompt to ???

declare -a pretrained_models=("gpt2-medium" "microsoft/DialoGPT-medium")
#declare -a pretrained_models=("microsoft/DialoGPT-medium")
declare -a attributes=("baseline" "young" "old" "young_miu" "old_miu")
#declare -a conditions=("unprompted" "prompted")
declare -a conditions=("prompted")

#TODO: CHANGE PROMPT-TYPE VARIABLE ACCORDINGLY
#declare -a neutral_prompts=("Hey." "Hello, tell me about your latest holiday." "Hi, how's it going?" "Can we talk?" "Good weather we're having.") #TODO: CHANGE PROMPT-TYPE VARIABLE ACCORDINGLY
#prompt_type="neutral_prompt"
#declare -a old_prompts=("Hello, tell me about yourself." "Hello, how are you?" "I had a splendid weekend." "Good afternoon." "Tell me about your family.") #TODO: CHANGE PROMPT-TYPE VARIABLE ACCORDINGLY
#prompt_type="old_prompt"
declare -a young_prompts=("Awesome! I actually haven't been there. When did you go?" "Can I add you on Facebook?" "Do you have any hobbies?" "What do you wanna eat?" "What are your plans this week?") #TODO: CHANGE PROMPT-TYPE VARIABLE ACCORDINGLY
prompt_type="young_prompt"


for pretrained_model in "${pretrained_models[@]}"
do

  for attribute in "${attributes[@]}"
  do

    case "$attribute" in

      "baseline")
        bow="plug_play/wordlists/bnc_rb_ws_100_most_common.txt"
        label=0
        ;;

      "young")
        bow="plug_play/wordlists/bnc_young_mcwu_ws_pct_85.txt"
        label=0
        ;;

      "young_miu")
        bow="plug_play/wordlists/bnc_rb_WS_100_mi_unigrams_young.txt"
        label=0
        ;;

      "old")
        bow="plug_play/wordlists/bnc_old_mcwu_ws_pct_85.txt"
        label=1
        ;;

      "old_miu")
        bow="plug_play/wordlists/bnc_rb_WS_100_mi_unigrams_old.txt"
        label=1
        ;;

      *)
        echo -n "unknown"
        ;;
    esac

    for condition in "${conditions[@]}"
    do
      if [ "$condition" = "unprompted" ]
      then
        echo "Configuration --> pm: $pretrained_model | attribute: $attribute | bow: $bow |class_label: $label | prompt: $condition |"

        python plug_play/run_pplm.py \
          --pretrained_model "$pretrained_model" \
          --uncond \
          --num_samples 30 \
          --bag_of_words "$bow" \
          --length 50 \
          --seed 2021 \
          --sample \
          --class_label $label \
          --verbosity "quiet"

      else
        echo "Configuration --> pm: $pretrained_model | attribute: $attribute | bow: $bow | class_label: $label | prompt: Tell me about your holidays. Sure! I went to Greece and had a very fun time. |"

        for length in 6 12 18 24 30 36 42 48 54 60
        do

          for prompt in "${young_prompts[@]}"
          do

            python plug_play/run_pplm.py \
              --pretrained_model "$pretrained_model" \
              --cond_text "$prompt" \
              --num_samples 6 \
              --bag_of_words "$bow" \
              --length $length \
              --seed 2021 \
              --sample \
              --class_label $label \
              --prompt_type "$prompt_type" \
              --verbosity "quiet"

          done

        done
      fi
    done

  done

done


#### Before running python script, echo current configuration

#################
# Discrim-based #
#################

# for pretrained model in [gpt2, dialogpt]
# --> if gpt2/dialogpt set discrim_weights and discrim_meta appropriately
## for control-attribute in [uncontrolled, young, old]
## --> if uncontrolled set stepsize and/or num_iterations to zero
## --> if young/old set class_label to 0/1
### for conditioning in [unprompted, prompted]
### --> if unprompted set --uncond
### --> if prompted set prompt to ???

#declare -a pretrained_models=("gpt2-medium" "microsoft/DialoGPT-medium")
#declare -a attributes=("uncontrolled" "young" "old")
##declare -a attributes=("uncontrolled")
##declare -a conditions=("unprompted" "prompted")
#declare -a conditions=("prompted")

#TODO: CHANGE PROMPT-TYPE VARIABLE ACCORDINGLY
#declare -a neutral_prompts=("Hey." "Hello, tell me about your latest holiday." "Hi, how's it going?" "Can we talk?" "Good weather we're having.") #TODO: CHANGE PROMPT-TYPE VARIABLE ACCORDINGLY
#prompt_type="neutral_prompt"
#declare -a old_prompts=("Hello, tell me about yourself." "Hello, how are you?" "I had a splendid weekend." "Good afternoon." "Tell me about your family.") #TODO: CHANGE PROMPT-TYPE VARIABLE ACCORDINGLY
#prompt_type="old_prompt"
#declare -a young_prompts=("Awesome! I actually haven't been there. When did you go?" "Can I add you on Facebook?" "Do you have any hobbies?" "What do you wanna eat?" "What are your plans this week?") #TODO: CHANGE PROMPT-TYPE VARIABLE ACCORDINGLY
#prompt_type="young_prompt"

#for pretrained_model in "${pretrained_models[@]}"
#do
#
#  # set discriminator weights
#  [ "$pretrained_model" = "gpt2-medium" ] &&
#    discrim_weights="plug_play/discriminators/gpt2_incl_sw_nac/generic_pm_gpt2-medium_ml_512_lr_0.0001_classifier_head_epoch_19.pt" ||
#    discrim_weights="plug_play/discriminators/dialogpt-medium/generic_pm_microsoft-DialoGPT-medium_ml_512_lr_0.0001_classifier_head_epoch_17.pt"
#
#  # set discriminator meta-data
#  [ "$pretrained_model" = "gpt2-medium" ] &&
#    discrim_meta="plug_play/discriminators/gpt2_incl_sw_nac/generic_pm_gpt2-medium_ml_512_lr_0.0001_classifier_head_meta.json" ||
#    discrim_meta="plug_play/discriminators/dialogpt-medium/generic_pm_microsoft-DialoGPT-medium_ml_512_lr_0.0001_classifier_head_meta.json"
#
#  for attribute in "${attributes[@]}"
#  do
#    [ "$attribute" = "uncontrolled" ] && stepsize=0 || stepsize=0.02
#    [ "$attribute" = "uncontrolled" ] && num_iterations=0 || num_iterations=3
#
#    [ "$attribute" = "uncontrolled" ] || [ "$attribute" = "young" ] && label=0 || label=1
#
#    for condition in "${conditions[@]}"
#    do
#      if [ "$condition" = "unprompted" ]
#      then
#        echo "Configuration --> pm: $pretrained_model | attribute: $attribute | weights: $discrim_weights | meta: $discrim_meta |class_label: $label | prompt: $condition | stepsize: $stepsize | iterations: $num_iterations |"
#
#        for length in 6 12 18 24 30 36 42 48 54 60
#        do
#
#          python plug_play/run_pplm.py \
#               --pretrained_model "$pretrained_model" \
#               --uncond \
#               --num_samples 30 \
#               --discrim 'generic' \
#               --length $length \
#               --seed 2021 \
#               --sample \
#               --stepsize $stepsize \
#               --num_iterations $num_iterations \
#               --class_label $label \
#               --verbosity "quiet" \
#               --discrim_weights "$discrim_weights" \
#               --discrim_meta "$discrim_meta"
#
#        done
#
#      else
#        echo "Configuration --> pm: $pretrained_model | attribute: $attribute | weights: $discrim_weights | meta: $discrim_meta | class_label: $label | prompt: $prompt | stepsize: $stepsize | iterations: $num_iterations |"
#
#        for length in 6 12 18 24 30 36 42 48 54 60
#        do
#
#          for prompt in "${young_prompts[@]}"
#          do
#
#            python plug_play/run_pplm.py \
#                 --pretrained_model "$pretrained_model" \
#                 --cond_text "$prompt" \
#                 --num_samples 6 \
#                 --discrim 'generic' \
#                 --length $length \
#                 --seed 2021 \
#                 --sample \
#                 --stepsize $stepsize \
#                 --num_iterations $num_iterations \
#                 --class_label $label \
#                 --verbosity "quiet" \
#                 --discrim_weights "$discrim_weights" \
#                 --discrim_meta "$discrim_meta" \
#                 --prompt_type "$prompt_type"
#          done
#
#        done
#      fi
#    done
#
#  done
#
#done