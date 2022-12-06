# Import statements
import pandas as pd
import numpy as np
import csv
import pdb

# Import for n gram
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for cool plotting
import re # for regular expression
import nltk # natural language processing toolkit
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix, classification_report
from langdetect import detect, detect_langs # for language detection
# from tqdm.notebook import tqdm, trange
from tqdm import tqdm
import time
import math
from pathlib import Path
import pickle
from collections import Counter
import spacy
import argparse
from datetime import datetime

from preprocessing import preprocess_df
from utils import make_confusion_matrix
from nltk.corpus import stopwords

# Imports for BERT
import torch
from classifiers import TextClassificationBERT
from transformers import BertTokenizer
from dataset import BncDataset, PadSequence
from torch.utils.data import DataLoader


# Seeds for reproducibility
SEED = 7
torch.manual_seed(SEED)
np.random.seed(SEED)

TEST_FRAC = 0.10


def add_id_to_csv(input_fp='data_preprocessing/BNC/subsets/bnc_subset_19_29_vs_50_plus_nfiles_0_w_gender_topics.csv',
                  output_fp='data/bnc/bnc_rb_w_id_gender_topics.csv'):

    df = pd.read_csv(input_fp)

    df.insert(0, 'text_id', range(len(df)))

    # Save dataframe to csv
    df.to_csv(
        output_fp,
        index=False
    )


def preprocess_col(df, data='blog'):

    data_size = len(df)

    # Remove all non-alphabetical characters
    # df['clean_text'] = df['text'].apply(lambda x: re.sub(r'[^A-Za-z]+',' ', x))

    # incl. stopwords setting
    df['clean_text'] = df['text']

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

    if data == 'blog':
        # Add labels for age categories
        def age_to_cat(age):
            '''Returns age category label for given age number.'''

            if 13 <= int(age) <= 17:
                return 0 #'13-17'
            elif 23 <= int(age) <= 27:
                return 1 #'23-27'
            elif 33 <= int(age):
                return 2 #'33-47'
            else:
                raise ValueError("Given age not in one of pre-defined age groups.")

        df['age_cat'] = df['age'].apply(age_to_cat)

    elif data == 'bnc' or data == 'bnc_rb':

        # Add labels for age categories
        def age_to_cat(label):
            '''Returns age category label for given age number.'''

            if label == '19_29':
                return 0  # '13-17'
            elif label == '50_plus':
                return 1  # '23-27'
            else:
                raise ValueError("Given age not in one of pre-defined age groups.")

        df['age_cat'] = df['label'].apply(age_to_cat)

        # # rename column
        # df.rename(columns={'label': 'age_cat'}, inplace=True)

    return df


# Evaluate performance
def print_evaluation_scores(labels, preds):
    print(f"Accuracy: {accuracy_score(labels, preds)}")
    print(f"F1 score: {f1_score(labels, preds, average = None)}") # outputs F1 per class
    print(f"Average precision: {average_precision_score(labels, preds, average = 'micro')}")
    print(f"Average recall: {recall_score(labels, preds, average = 'micro')}")
    print(classification_report(labels, preds, digits=5, zero_division=0))
    # print(f"Confusion Matrix: {confusion_matrix(labels.argmax(axis=1), preds.argmax(axis=1))}")


def train_test_ngram(train_df, test_df, n_grams=[3], seeds=[SEED], dataset='bnc_rb'):

    overall_start_time = time.time()

    # results dict
    accs_all = {}
    if dataset == 'blog':
        class_labels_list = ['13-17', '23-27', '33-47']
    elif dataset == 'bnc' or dataset == 'bnc_rb':
        class_labels_list = ['19_29', '50_plus']

    test_accs = {}
    test_f1s = {}

    for n_gram in n_grams:
        test_accs[n_gram] = {}
        test_f1s[n_gram] = {}
        # for class_label in class_labels_list:
        #     test_f1s[n_gram][class_label] = {}

    print("Starting training and testing loops...")
    for seed in tqdm(seeds, desc = "Seed loop."):

        for n in tqdm(n_grams, desc = "n gram loop."):

            # Split data into features/ X and labels / Y
            # X = data['clean_data']
            # Y = data['labels']

            # n-gram model
            vectorizer = CountVectorizer(binary = True, ngram_range = (1, n))

            # fit vectorization model
            concat_df = pd.concat([train_df, test_df])
            X = vectorizer.fit_transform(concat_df['clean_text'])

            X_train = X[:-len(test_df)]
            X_test = X[-len(test_df):]

            # Binarize the labels for prediction
            if dataset == 'blog':
                # binarizer = MultiLabelBinarizer(classes = sorted(label_counts.keys()))
                binarizer = LabelBinarizer()
            elif dataset == 'bnc_rb' or dataset == 'bnc':
                binarizer = LabelBinarizer()

            # Y = binarizer.fit_transform(train_df.age_cat)
            Y_train = train_df.label
            Y_test = test_df.label

            # label_counts.keys()

            # Split data into train and test sets
            # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size)

            # Fit logistic regression model
            start_time = time.time()
            model = LogisticRegression(solver = 'lbfgs', multi_class='ovr', max_iter = 1000000)
            model = OneVsRestClassifier(model)
            # model = MultiOutputClassifier(model)
            model.fit(X_train, Y_train)
            print(f"Fitting model took {time.time() - start_time} seconds.")

            # make predictions on test set
            Y_pred = model.predict(X_test)

            # Y_pred_inversed = binarizer.inverse_transform(Y_pred)
            # Y_test_inversed = binarizer.inverse_transform(Y_test)

            print("=" * 81)

            print(f"n = {n}")
            print(f"seed = {seed}")
            print_evaluation_scores(Y_test, Y_pred)

            test_accs[n][seed] = accuracy_score(Y_test, Y_pred)
            test_f1s[n][seed] = f1_score(Y_test, Y_pred, average=None)

            # for label_idx in range(len(class_labels_list)):
            #     test_f1s[n][class_labels_list[label_idx]][seed] = f1_score(Y_test, Y_pred, average=None)[label_idx]

            if n in accs_all:
                accs_all[n].append(accuracy_score(Y_test, Y_pred))
            else:
                accs_all[n] = [accuracy_score(Y_test, Y_pred)]

            # Print most informative features
            # if n == 1:
            #     print("Most informative features per age-group.")
            #     print_top_n_thresh(vectorizer = vectorizer, clf = model,
            #                 class_labels = class_labels_list, n_feat = 20)

            print("-" * 81)
    #         print("Some failure cases.")
    # #         predictions = model.predict(inputs)
    #         for i, (x, pred, label) in enumerate(zip(X_test, Y_pred, Y_test)):
    #             if (pred != label).any():
    #                 print(f"pred: {pred}")
    #                 print(f"label: {label}")
    #                 pred_cat = binarizer.classes_[np.where(pred == 1)[0][0]]
    #                 label_cat = binarizer.classes_[np.where(label == 1)[0][0]]
    #                 print(data['clean_data'][i], 'has been classified as ', pred_cat, 'and should be ', label_cat)

            print("=" * 81)


            int_labels = [label for label in range(len(class_labels_list))]
            cm = confusion_matrix(Y_test, Y_pred, labels=int_labels)
            # make_confusion_matrix(cf=cm, categories=class_labels_list, title=f'Confusion Matrix for {dataset} on Test set',
            #                       num_labels=int_labels, y_true=Y_test, y_pred=Y_pred, figsize=FIGSIZE)
            cur_datetime = datetime.now().strftime('%d_%b_%Y_%H_%M_%S')
            # plt.savefig(f"{FIGDIR}{dataset}/cm_{n}_gram_{dataset}_dt_{cur_datetime}.png",
            #             bbox_inches='tight')

    #         most_informative_feature_for_class(vectorizer = vectorizer, classifier = model, class_labels = class_labels_list, n=10)

            df = pd.read_csv('bnc_rb_10p_testset_case_analysis_ws.csv')

            df.insert(len(df.columns), 'trigram_pred', Y_pred)

            # Save dataframe to csv
            df.to_csv(
                'bnc_rb_10p_testset_case_analysis_ws.csv',
                index=False
            )


    # print average metrics
    print(89*'-')
    print(89 * '-')
    print("PRINTING AVERAGE METRICS")
    for n_gram in n_grams:
        n_gram_accs = []
        n_gram_f1s = []
        for seed in seeds:
            n_gram_accs.append(test_accs[n_gram][seed])
            n_gram_f1s.append(test_f1s[n_gram][seed])

        print(f"| n = {n_gram} | Average accuracy = {np.mean(n_gram_accs)} | Acc std = {np.std(n_gram_accs)} "
              f"| Average f1s = {np.mean(n_gram_f1s, axis=0)} | F1s std = {np.std(n_gram_f1s, axis=0)} |")

    overall_end_time = time.time()


    print(f"Done with everything. Took {overall_end_time - overall_start_time} seconds.")

def test_bert(model, criterion, device, data_loader, data='bnc_rb', writer=None, global_iteration=0, set='validation',
                         print_metrics=True, plot_cm=False, save_fig=True, show_fig=False, model_type='bert', mode='train'):
    # For Confucius matrix
    y_pred = []
    y_true = []

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

            total_correct += predictions.eq(batch_labels.view_as(predictions)).sum().item()

        # average losses and accuracy
        set_loss /= len(data_loader.dataset)
        accuracy = total_correct / len(data_loader.dataset)
        if print_metrics:
            print('-' * 91)
            print(
                "| " + set + " set "
                "| time {}"
                "| loss: {:.5f} | Accuracy: {}/{} ({:.5f})".format(
                    datetime.now() - eval_start_time, set_loss, total_correct, len(data_loader.dataset), accuracy
                )
            )
            print('-' * 91)

        if writer:
            if set == 'validation':
                writer.add_scalar('Accuracy/val', accuracy, global_iteration)
                writer.add_scalar('Loss/val', set_loss, global_iteration)

        print(91 * '-')
        print(34 * '-' + ' Classification Report ' + 34 * '-')
        labels = [label for label in range(data_loader.dataset.num_classes)]
        print(classification_report(y_true, y_pred, labels=labels, digits=5, zero_division=0))

        print(91 * '-')
        print('| Confusion Matrix |')
        # cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='all')
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        # df_confusion = pd.DataFrame(cm * len(y_true))
        df_confusion = pd.DataFrame(cm)
        print("    Predicted")
        print(df_confusion)
        print("True -->")

        # print(cm * len(y_true))

        if plot_cm:

            if data == 'bnc' or 'bnc_rb':
                tick_labels = ['19_29', '50_plus']
            elif data == 'blog':
                tick_labels = ['13-17', '23-27', '33-47']
            make_confusion_matrix(cf=cm, categories=tick_labels, title=f'Confusion Matrix for {data} on {set} set',
                                  num_labels=labels, y_true=y_true, y_pred=y_pred, figsize=FIGSIZE)

            if save_fig:
                cur_datetime = datetime.now().strftime('%d_%b_%Y_%H_%M_%S')
                plt.savefig(f"{FIGDIR}{data}/cm_{model_type}_{set}_dt_{cur_datetime}.png",
                            bbox_inches='tight')
            if show_fig:
                plt.show()


        if mode == 'tvt':
            f1_scores = f1_score(y_true, y_pred, average=None)

            return set_loss, accuracy, f1_scores
        else:
            return set_loss, accuracy, y_pred

if __name__ == '__main__':

    # Add ID row for all text samples.
    # add_id_to_csv()

    file_path = 'data/bnc/bnc_rb_w_id_gender_topics.csv'

    ######## Split data into training, validation and test set
    df = pd.read_csv(file_path, encoding="utf-8")  # to keep no. unique chars consistent across platforms
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)  # shuffle data set before subsampling

    # preprocess text data
    df = preprocess_col(df, data='bnc_rb')

    # rename column
    df.rename(columns={'label': 'age_cat',
                       'age_cat' : 'label'}, inplace=True)

    # In case you want a subset for faster debugging
    # subset_size = 1000
    # if subset_size:
    #     df = df.iloc[:subset_size]
    #     df.reset_index(drop=True, inplace=True)  # Reset index after subsetting

    # set proportions for data splits
    train_frac = 0.75
    val_frac = 0.15
    test_frac = 0.1

    train_df, val_df, test_df = np.split(df, [int(train_frac * len(df)),
                                              int((1 - test_frac) * len(df))])

    # reset indices of subsets
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    # pdb.set_trace()

    # save the data sets
    # train_df.to_csv(
    #     'data/bnc/ca_splits/bnc_rb_ca_trainset_case_analysis_ws.csv',
    #     index=False
    # )
    # val_df.to_csv(
    #     'data/bnc/ca_splits/bnc_rb_ca_valset_case_analysis_ws.csv',
    #     index=False
    # )
    test_df.to_csv(
        'bnc_rb_10p_testset_case_analysis_ws.csv',
        index=False
    )

    concat_train_df = pd.concat([train_df, val_df])
    train_test_ngram(train_df=concat_train_df, test_df=test_df)

    ####### Now test bert

    # loss criterion
    criterion = torch.nn.CrossEntropyLoss()  # combines LogSoftmax and NLL

    # Set device
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    device = torch.device(device)
    print(f"Device: {device}")

    # Initialize and load saved model
    model = TextClassificationBERT(num_classes=2)
    model.to(device)
    # model_path = 'bert_bnc_rb_case_analysis_seed_4_BEST.pt'
    model_path = 'models/bert/bert_bnc_rb_ws_ca_seed_4_24_Sep_2021_12_51_09.pt' # for lisa
    # model_path = 'bert_bnc_rb_ws_ca_seed_4_24_Sep_2021_12_51_09.pt'  # for local
    model.load_state_dict(torch.load(model_path))

    # Setup data stuff
    # BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', truncation=True,
                                              max_length=500)  # truncation only considers sequences of max 512 tokens (same as original BERT implementation)

    test_preprocessed = test_df[['clean_text', 'label']]

    # re-rename column because im stupid
    test_preprocessed.rename(columns={'label': 'age_cat'}, inplace=True)

    test_dataset = BncDataset(df=test_preprocessed, tokenizer=tokenizer)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=4,
                             shuffle=False,
                             collate_fn=PadSequence())

    _, _, bert_pred = test_bert(model=model, criterion=criterion, device=device, data_loader=test_loader,
                                save_fig=False, set='test')

    df = pd.read_csv('bnc_rb_10p_testset_case_analysis_ws.csv')

    df.insert(len(df.columns), 'bert_pred', bert_pred)

    # Save dataframe to csv
    df.to_csv(
        'bnc_rb_10p_testset_case_analysis_final_ws.csv',
        index=False
    )

    # main()