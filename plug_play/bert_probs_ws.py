# Imports
# Imports
import argparse
import pdb


from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
import numpy as np
import os, sys
import re
from nltk.corpus import stopwords
from datetime import datetime
from torch.utils.data import DataLoader
p = os.path.abspath('.')
sys.path.insert(1, p)
from classifiers import TextClassificationBERT
from dataset import BncDataset, PadSequence

def preprocess_col(df):

    data_size = len(df)

    # Remove all non-alphabetical characters
    df['clean_text'] = df['text'].apply(lambda x: re.sub(r'[^A-Za-z]+',' ', x))

    # make all letters lowercase
    df['clean_text'] = df['clean_text'].apply(lambda x: x.lower())

    # remove whitespaces from beginning or ending
    df['clean_text'] = df['clean_text'].apply(lambda x: x.strip())

    # remove stop words
    # stopwords_dict = set(stopwords.words('english')) # use set (hash table) data structure for faster lookup
    # df['clean_text'] = df['clean_text'].apply(lambda x: ' '.join([words for words in x.split() if words not in stopwords_dict]))

    # Remove instances empty strings
    df.drop(df[df.clean_text == ''].index, inplace = True)

    # number of datapoints removed by all pre-processing steps
    dropped_instances = data_size - len(df)
    data_size = len(df)

    # df['age_cat'] = df['label'].apply(age_to_cat)

    # # rename column
    # df.rename(columns={'label': 'age_cat'}, inplace=True)

    return df

def bert_pred(model, criterion, device, data_loader, data='bnc_rb', writer=None, global_iteration=0, set='validation',
                         print_metrics=True, plot_cm=False, save_fig=True, show_fig=False, model_type='bert', mode='train'):
    # For Confucius matrix
    y_pred = []
    y_true = []
    probabilities = []

    # set model to evaluation mode
    model.eval()

    # initialize loss and number of correct predictions
    set_loss = 0
    total_correct = 0

    # start eval timer
    eval_start_time = datetime.now()

    with torch.no_grad():
        for iteration, (batch_inputs, batch_labels, batch_lengths) in enumerate(data_loader):

            # move everything to device
            batch_inputs, batch_labels, batch_lengths = batch_inputs.to(device), batch_labels.to(device), \
                                                        batch_lengths.to(device)


            loss, text_fea = model(batch_inputs, batch_labels)
            batch_probs = torch.softmax(text_fea, axis=1)
            set_loss += loss

            predictions = torch.argmax(text_fea, 1)


            # batch_pred = [int(item[0]) for item in predictions.tolist()]
            # batch_pred = predictions.tolist()
            # ## OLD
            # if model_type == 'lstm':
            #     y_pred.extend(batch_pred)
            # elif model_type == 'bert':
            #     y_pred.extend(predictions.tolist())

            y_pred.extend(predictions.tolist()) #New
            y_true.extend(batch_labels.tolist())
            probabilities.extend(batch_probs.tolist())

            total_correct += predictions.eq(batch_labels.view_as(predictions)).sum().item()

        # # average losses and accuracy
        # set_loss /= len(data_loader.dataset)
        # accuracy = total_correct / len(data_loader.dataset)
        # if print_metrics:
        #     print('-' * 91)
        #     print(
        #         "| " + set + " set "
        #         "| time {}"
        #         "| loss: {:.5f} | Accuracy: {}/{} ({:.5f})".format(
        #             datetime.now() - eval_start_time, set_loss, total_correct, len(data_loader.dataset), accuracy
        #         )
        #     )
        #     print('-' * 91)
        #
        # if writer:
        #     if set == 'validation':
        #         writer.add_scalar('Accuracy/val', accuracy, global_iteration)
        #         writer.add_scalar('Loss/val', set_loss, global_iteration)
        #
        # print(91 * '-')

        if mode == 'tvt':
            f1_scores = f1_score(y_true, y_pred, average=None)

            return set_loss, accuracy, f1_scores
        else:
            return probabilities

# Set up bert model stuff

def new_bert_probs(file_path):

    # list of csv output file paths

    # for csv_path in list_of_paths:

    criterion = torch.nn.CrossEntropyLoss()  # combines LogSoftmax and NLL

    # LJ: Initialize, move to device, and load saved model
    bert_model = TextClassificationBERT(num_classes=2)
    bert_model.to(device)
    # bert_model_path = 'bert_bnc_rb_case_analysis_seed_4_BEST.pt'  # TODO: CHANGE TO BERT TRAINED WITH STOPWORDS!!!!!
    bert_model_path = 'bert_bnc_rb_ws_ca_seed_4_24_Sep_2021_12_51_09.pt'  # Trained With stopwords
    bert_model.load_state_dict(torch.load(bert_model_path, map_location=device))

    # LJ: Setup data stuff
    # LJ: BERT tokenizer
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', truncation=True,
                                                   max_length=500)  # truncation only considers sequences of max 512 tokens (same as original BERT implementation)

    prompt_df = pd.DataFrame({'text': [prompt], "age_cat": [age_cat]})

    # LJ: preprocess sequences for bert
    bert_df = preprocess_col(prompt_df)[['clean_text', 'age_cat']]

    # LJ: re-rename column because im stupid
    # bert_df.rename(columns={'label': 'age_cat'}, inplace=True)

    # LJ: create BncDataset instance
    bert_dataset = BncDataset(df=bert_df, tokenizer=bert_tokenizer)

    # LJ: construct dataloader
    bert_loader = DataLoader(dataset=bert_dataset,
                             batch_size=4,
                             shuffle=False,
                             collate_fn=PadSequence())

    # LJ: use bert to make predictions and save assigned probabilities of belonging to young or old age group
    bert_probs = bert_pred(model=bert_model, criterion=criterion, device=device, data_loader=bert_loader,
                           save_fig=False, set='test')
    bert_probs = np.array(bert_probs)  # for easier slicing
    young_probs = bert_probs[:, 0]
    old_probs = bert_probs[:, 1]
    print(f"Young: {young_probs}")
    print(f"Old: {old_probs}")

    pass


if __name__ == "__main__":

    ## call new_bert_probs(file_path=csv_path)