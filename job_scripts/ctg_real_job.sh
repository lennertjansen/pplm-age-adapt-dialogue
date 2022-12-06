#!/bin/bash

##SBATCH --partition=gpu
#SBATCH -p gpu_titanrtx_shared ## Select the partition. This one is almost always free, and has TitanRTXes (much RAM)
#SBATCH --nodes=1
##SBATCH --gpus-per-node=1
#SBATCH --job-name=scripted_dialogue_generation_hobby_turnwise_OLD_utterance_classification
#SBATCH --time=5-00:00:00 ## Max time your script runs for (max is 5-00:00:00 | 5 days)
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=lennertjansen95@gmail.com
#SBATCH -o /home/lennertj/code/msc-ai-thesis/SLURM/output/scripted_dialogue_generation_hobby_turnwise_OLD_utterance_classification.%A.out ## this is where the terminal output is printed to. %j is root job number, %a array number. try %j_%a ipv %A (job id)

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
#declare -a arr=("Tell me about your holidays. Sure! I went to Greece and had a very fun time." "Tell me about your favorite food. Of course. I love pasta!." "Have you seen the news lately? I can't believe what I saw.")
#declare -a arr=("Hello, how are you?<|endoftext|>")

# food topic prompts
#declare -a young_prompts=("Tell me about your favourite food. I love sushi and Japanese food in general. What about you?" "Anything salmon. I am not so much of a spicy food lover, however. I love food too." "Me too! Meat and fish all the way." "I agree! A lot of great fish and meat. And it is also nice and lean." "Tell me about your favourite food. Too many to choose. I think one of my favorites for sure is a traditional stuffed pigeon from home. We eat pigeon a lot over there and it's usually stuffed with the most amazing herb rice! I usually eat it for special ocassions though, as it is quite hard to prepare and takes quite some time." "Hmm.. Probably a traditional egyptian one too. It's called konafa." "Tell me about yours?" "That's always a good idea! Me too. THe more local/fresh, the better. Tell me about your best dining experience!" "Tell me about your favourite food. My favorite food is fried chicken with French fries." "We share something in common." "I would describe myself as a fun-loving person. What about you?" "What is your favorite drink?" "I love apple juice.")
#declare -a old_prompts=("Tell me about your favourite food. Well, I am very fun with Mediterranean food, and especially food with herbs. Lot of fruits, nuts, especially in the summer weather." "I love seafood and fish, except I do not like chips! I much prefer olive fish accompanied with green salad." "I actually I am not fond of cheese and it is not a food that I include in my diet" "It's not different at all. My favourite food is what I usually consume." "Tell me about your favourite food. Healthy or non healthy food, which would you prefer me to talk about?" "Well I love fish, all kinds of fish, I try and eat meat sparingly, love fruit and vegetables, so I ought to be very healthy, but unfortunately I have a sweet tooth for Chocolate." "Well I love Cadburys, especially Fruit and Nut Chocolate, thinking the fruit and nut is somehow healthy for me. I also like Dark Chocolate as I know that should be better than milk chocolate, especially for the heart." "My allergies, well I thought that was typed as arteries, my arteries are probably clogged with chocolate, so not as healthy as I ought to be." "Salmon, which is fish, prawn, which is fish also, in terms of meat/animal, lamb, chicken and beef/steak. But as I mentioned earlier I try and eat more fruit and vegetables." "Tell me about your favourite food. I enjoy spicy foods. I have a curry quite a lot." "I also enjoy fish and chips sometimes , but not too often as it is fattening." "You need to have vegtables with your chips as it adds to the taste." "Do you like halloumi? Its one of my favorites.")

# hobby topic prompts
#declare -a young_prompts=("Tell me about your hobbies. Recently I have been riding horses a lot. It es great in summer as you get to spend time in nature." "I understand. Being outdoors is a great escape from a busy lifestyle." "So you no longer have a full-time job? Yes, I can imagine. Especially if your family lives far away." "What do you mean?" "That sounds lovely. I'm not so much of a hiker myself, but nice to hear." "Tell me about your hobbies. I love too cook. I also love to play football, but my knees are not in the best shape. So, us talking about food has been really nice!" "I grew up in Egypt for the first half of my life, and then I moved to Budapest with my brother and mother. You?" "Oh nice. I am in Amsterdam currently. I have been playing football since I was 8 years old. I love playing in the midfield... do you play any sports?" "Nice! Holland has some of the best footballers. What position did you play?" "Nice! My cousin was a centre-back. He was the best I'd ever seen. What other hobbies do you have?" "Tell me about your hobbies. My main hobbies are listening to music, watching movies and also dancing. What about you?" "What is your favorite genre in music?" "I am not a fan of metal." "My favorite song is 'Saturday Night' by Khalid." "My favorite movie as a child was 'Home Alone'. What about you?")
#declare -a old_prompts=("Tell me about your hobbies. I have many hobbies. My oldest hobby is socializing and interacting with people. That's for me a hobby, not something that you just do." "Well, my hobby is helping people in a more fundamental way. By adjusting their nutritional needs and help them with specific issues such as fertility." "Sharing knowledge is fundamental to life and has a positive effect for the recipient as well for the given." "Tell me about your hobbies. Well I don't get much time for hobbies, but when I do, especially on holiday I love to Scuba Dive, I love flying, I enjoy technology, I ought to buy a drone, to capture lots of aerial shots and I enjoy skiing, but last time I fell and broke my wrist which was painful." "What kind of games do you enjoy? I like card games, strategy games, which require interaction with people. I did used to play computer games, but prefer to see the people I play against." "I like Chess, because it's a strategy game and requires thinking in advance of your opponents moves." "Hobbies help me relax from day to day work, but again finding the time for any personal hobbies is difficult, so playing game of chess can be equally rewarding, but unfortunately my son now beats me most of the time, I taught him too well." "Tell me about your hobbies. I enjoy badminton and chess which I play. I enjoy watching football also." "Good, what position do you play." "Do you score a lot of goals.Do you help out in defence as well." "Im too old to play for a team now. Do you have a favorite player.")

## for discrim-based
#for prompt in "${young_prompts[@]}"
#do
#  echo "NEW PROMPT ALERT WOOOWOOOOWOOOOOOWOOOOOOO"
#  for seed in 2021
#  do
#    for length in 45
#    do
#      python plug_play/run_pplm.py \
#             --pretrained_model 'gpt2-medium' \
#             --num_samples 3 \
#             --bag_of_words "plug_play/wordlists/bnc_young_mcwu_ws_pct_85.txt" \
#             --length $length \
#             --seed $seed \
#             --sample \
#             --class_label 0 \
#             --verbosity "quiet" \
#             --cond_text "$prompt"
#    done
#  done
#done

#declare -a arr=("Tell me about your holidays. Sure! I went to Greece and had a very fun time." "Tell me about your favorite food. Of course. I love pasta!." "Have you seen the news lately? I can't believe what I saw.")
#declare -a arr=("Hello, how are you?<|endoftext|>")
#
#
## for discrim-based
#for prompt in "${arr[@]}"
#do
#  for seed in 2021
#    do
#
#    for length in 6 12 24 30 36 42 48 54 60
#    do
#
#      python plug_play/run_pplm.py \
#             --pretrained_model 'microsoft/DialoGPT-medium' \
#             --cond_text "$prompt" \
#             --num_samples 30 \
#             --discrim 'generic' \
#             --length $length \
#             --seed $seed \
#             --sample \
#             --class_label 0 \
#             --verbosity "quiet" \
#             --discrim_weights "plug_play/discriminators/dialogpt-medium/generic_pm_microsoft-DialoGPT-medium_ml_512_lr_0.0001_classifier_head_epoch_17.pt" \
#             --discrim_meta "plug_play/discriminators/dialogpt-medium/generic_pm_microsoft-DialoGPT-medium_ml_512_lr_0.0001_classifier_head_meta.json" \
#             --num_iterations 0 \
#             --stepsize 0
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

#python plug_play/run_pplm.py --pretrained_model 'microsoft/DialoGPT-medium' --cond_text 'Hello, how are you? Fine, thanks. And you?' --num_samples 5 --discrim 'generic' --length 15 --seed 2021 --sample --class_label 0 --verbosity "verbose" --discrim_weights "plug_play/discriminators/dialogpt-medium/generic_pm_microsoft-DialoGPT-medium_ml_512_lr_0.0001_classifier_head_epoch_17.pt" --discrim_meta "plug_play/discriminators/dialogpt-medium/generic_pm_microsoft-DialoGPT-medium_ml_512_lr_0.0001_classifier_head_meta.json" --num_iterations 0 --stepsize 0

# plug_play/discriminators/gpt2_incl_sw_nac/generic_pm_gpt2-medium_ml_512_lr_0.0001_classifier_head_meta.json
# plug_play/discriminators/gpt2_incl_sw_nac/generic_pm_gpt2-medium_ml_512_lr_0.0001_classifier_head_epoch_19.pt

# plug_play/wordlists/bnc_young_mcwu_ws_pct_85.txt

# python plug_play/run_pplm.py --pretrained_model 'gpt2-medium' --cond_text 'A: Hello, how are you? B: Fine, thanks. And you? A:' --num_samples 5 --discrim 'generic' --length 15 --seed 2021 --sample --class_label 1 --verbosity "verbose" --discrim_weights "plug_play/discriminators/gpt2_incl_sw_nac/generic_pm_gpt2-medium_ml_512_lr_0.0001_classifier_head_epoch_19.pt" --discrim_meta "plug_play/discriminators/gpt2_incl_sw_nac/generic_pm_gpt2-medium_ml_512_lr_0.0001_classifier_head_meta.json" --num_iterations 0 --stepsize 0
# python plug_play/run_pplm.py --pretrained_model 'gpt2-medium' --bag_of_words "plug_play/wordlists/bnc_young_mcwu_ws_pct_85.txt" --cond_text 'A: Hello, how are you? B: Fine, thanks. And you? A:' --num_samples 5 --length 15 --seed 2021 --sample --class_label 0 --verbosity "verbose"

#declare -a young_utts=("Tell me about your favourite food. I love sushi and Japanese food in general. What about you?" "I love sushi in general." "Anything salmon. I am not so much of a spicy food lover, however. I love food too." "So when I saw this amazing dish from one of my favorite foodie friends in town, I was excited to try it!" "Me too! Meat and fish all the way." "I love it!" "I agree! A lot of great fish and meat. And it is also nice and lean." "What I miss in a fish is some nice crunch to it." "Tell me about your favourite food. Too many to choose. I think one of my favorites for sure is a traditional stuffed pigeon from home. We eat pigeon a lot over there and it's usually stuffed with the most amazing herb rice! I usually eat it for special ocassions though, as it is quite hard to prepare and takes quite some time." "I also love making my own, as I love to try and make my own dishes that are unique and fun to eat." "Hmm.. Probably a traditional egyptian one too. It's called konafa." "This one is a bit more unusual." "Tell me about yours?" "My name is Chris, and I'm a freelance writer living in Toronto." "That's always a good idea! Me too. THe more local/fresh, the better. Tell me about your best dining experience!" "We were at Cinco de Mayo, and we were so impressed with the tacos at Tacolote that we had to go to the other side." "Tell me about your favourite food. My favorite food is fried chicken with French fries." "Fried chicken is my favourite food." "We share something in common." "We all feel that we are in a difficult spot, and we all want help." "I would describe myself as a fun-loving person. What about you?" "I am a fun-loving person and am happy with that." "What is your favorite drink?" "We love our drinks, but we love drinking all the time too." "I love apple juice." "Not just because it comes out clean, refreshing and tasty and has a good balance of sweetness from the apple.")
#declare -a old_utts=("Tell me about your favourite food. Well, I am very fun with Mediterranean food, and especially food with herbs. Lot of fruits, nuts, especially in the summer weather." "I love all kinds of seafood, especially mussels." "I love seafood and fish, except I do not like chips! I much prefer olive fish accompanied with green salad." "I also like to add grilled salmon to my salad as it is quite healthy." "I actually I am not fond of cheese and it is not a food that I include in my diet" "I am not very fond of cheese and I do not like any dairy products that contain cheese in any way whatsoever." "It's not different at all. My favourite food is what I usually consume." "I love steak and chicken but sometimes I have to try it for something different because it tastes good, so sometimes I eat something different." "Tell me about your favourite food. Healthy or non healthy food, which would you prefer me to talk about?" "Healthy is the most difficult to talk about." "Well I love fish, all kinds of fish, I try and eat meat sparingly, love fruit and vegetables, so I ought to be very healthy, but unfortunately I have a sweet tooth for Chocolate." "I also like to drink milk, coffee and tea as it is quite healthy." "Well I love Cadburys, especially Fruit and Nut Chocolate, thinking the fruit and nut is somehow healthy for me. I also like Dark Chocolate as I know that should be better than milk chocolate, especially for the heart." "But I don't think I can eat a chocolate bar and not like milk chocolate, especially when it tastes of cream milk chocolate." "My allergies, well I thought that was typed as arteries, my arteries are probably clogged with chocolate, so not as healthy as I ought to be." "But I was thinking about it in a different light, and my blood work came back fine, but I still have some redness on my face and on my lips." "Salmon, which is fish, prawn, which is fish also, in terms of meat/animal, lamb, chicken and beef/steak. But as I mentioned earlier I try and eat more fruit and vegetables." "I also eat more whole milk, milk products, cheese and eggs." "Tell me about your favourite food. I enjoy spicy foods. I have a curry quite a lot." "But I don't really enjoy the sweet and sour food that we like in India, especially in the south of India." "I also enjoy fish and chips sometimes , but not too often as it is fattening." "I always like to add some fresh herbs to the fish for a nice change." "You need to have vegtables with your chips as it adds to the taste." "This is why I recommend to add them to a pan and heat it up." "Do you like halloumi? Its one of my favorites." "I love it because when I eat it its not greasy and it has some flavor that I love.")
#

# hobby topic utterances
#declare -a young_utts=("Tell me about your hobbies. Recently I have been riding horses a lot. It es great in summer as you get to spend time in nature." "My brother is a horse rider." "I understand. Being outdoors is a great escape from a busy lifestyle." "I'm always happy to have the sun and a warm place to sleep and warm up in the shade and sun." "So you no longer have a full-time job? Yes, I can imagine. Especially if your family lives far away." "I have been in a few different jobs over the years and my family's lives have gotten much more complicated than I thought they would." "What do you mean?" "I'm not talking about the people who say they have a problem with it, but the people who think you've changed your mind about it after seeing the videos or reading my posts on it." "That sounds lovely. I'm not so much of a hiker myself, but nice to hear." "I know you've been doing some hiking in the park, and it must be pretty hard work on those long backpacking days." "Tell me about your hobbies. I love too cook. I also love to play football, but my knees are not in the best shape. So, us talking about food has been really nice!" "I like to cook for myself and my friends as well." "I grew up in Egypt for the first half of my life, and then I moved to Budapest with my brother and mother. You?" "You have to remember that I grew up in the 1970s, when the Soviet Union was still the dominant superpower in this part of the world." "Oh nice. I am in Amsterdam currently. I have been playing football since I was 8 years old. I love playing in the midfield... do you play any sports?" "I have played in a few different sports but football is the one that I am most obsessed with." "Nice! Holland has some of the best footballers. What position did you play?" "I played centre forward for the Dutch National Team as a 19 year old, I had to work hard to get to where I am now." "Nice! My cousin was a centre-back. He was the best I'd ever seen. What other hobbies do you have?" "I do have quite a few other hobbies, but I've never been able to find time to do them." "Tell me about your hobbies. My main hobbies are listening to music, watching movies and also dancing. What about you?" "My most popular hobby is dancing!" "What is your favorite genre in music?" "I love everything from punk rock to hip-hop." "I am not a fan of metal." "My tastes don't include it." "My favorite song is 'Saturday Night' by Khalid." "This one is so fun and catchy with great lyrics." "My favorite movie as a child was 'Home Alone'. What about you?" "The new movie 'Carnival of Monsters 2' is being directed by Steven Spielberg and stars the likes of Brad Pitt, Emma Stone, James Franco, and Kristen Stewart as the titular monsters.")
declare -a old_utts=("Tell me about your hobbies. I have many hobbies. My oldest hobby is socializing and interacting with people. That's for me a hobby, not something that you just do." "My most popular hobby is traveling." "Well, my hobby is helping people in a more fundamental way. By adjusting their nutritional needs and help them with specific issues such as fertility." "This can be as simple as giving them a small dose of a nutrient such as vitamin A or vitamin B6 to help them with their needs for a few days to see how they feel and adjust accordingly." "Sharing knowledge is fundamental to life and has a positive effect for the recipient as well for the given." "This can only be achieved through the sharing of knowledge." "Tell me about your hobbies. Well I don't get much time for hobbies, but when I do, especially on holiday I love to Scuba Dive, I love flying, I enjoy technology, I ought to buy a drone, to capture lots of aerial shots and I enjoy skiing, but last time I fell and broke my wrist which was painful." "So I don't have much time for hobbies, so I like to play video games." "What kind of games do you enjoy? I like card games, strategy games, which require interaction with people. I did used to play computer games, but prefer to see the people I play against." "My favorite genre is board games." "I like Chess, because it's a strategy game and requires thinking in advance of your opponents moves." "This means that it requires you to know what moves your opponent's are going to make, and how to anticipate those moves so you can predict your next move." "Hobbies help me relax from day to day work, but again finding the time for any personal hobbies is difficult, so playing game of chess can be equally rewarding, but unfortunately my son now beats me most of the time, I taught him too well." "My brother is playing chess too, but I don't play with him often because I'm busy with work (and I've already got a job)." "Tell me about your hobbies. I enjoy badminton and chess which I play. I enjoy watching football also." "My brother is a football fan." "Good, what position do you play." "I'm a very good player and a very smart player." "Do you score a lot of goals.Do you help out in defence as well." "I'm always interested in how you play and how your teammates are playing and how they are performing." "Im too old to play for a team now. Do you have a favorite player." "No, I don't.")

# for discrim-based
for prompt in "${old_utts[@]}"
do
  echo "NEW PROMPT ABI"
  python plug_play/prompt_classifier.py --std --age_cat 1 --prompt "$prompt"
done