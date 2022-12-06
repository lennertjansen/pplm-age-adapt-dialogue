# Import statements
import os  # for directory operations
import numpy as np  # for numerical/linear algebra methods
import pandas as pd  # for data(frame) processing
import pdb # for debudding
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

FIGDIR = 'figures/'
FIGSIZE = (15, 8)

def train_classifiers(dataset,
                      dataset_fp,
                      subset_size,
                      n_grams,
                      seeds,
                      test_size):

    overall_start_time = time.time()

    if dataset == 'bnc_rb':
        # Read raw data
        raw_data = pd.read_csv('data/bnc/bnc_subset_19_29_vs_50_plus_nfiles_0_rand_balanced.csv')

        # prepocess data
        data = preprocess_df(df=raw_data, data='bnc_rb')

        # change column names so everything works later
        data.rename(columns={"clean_text": "clean_data",
                             "age_cat": "labels"}, inplace=True)
    elif dataset == 'bnc':
        raw_data = pd.read_csv('data/bnc/bnc_subset_19_29_vs_50_plus_nfiles_0.csv')

        # prepocess data
        data = preprocess_df(df=raw_data, data='bnc')

        # change column names so everything works later
        data.rename(columns={"clean_text": "clean_data",
                             "age_cat": "labels"}, inplace=True)
    elif dataset == 'blog':

        raw_data = pd.read_csv('data/blogs_kaggle/blogtext.csv')

        # prepocess data
        data = preprocess_df(df=raw_data, data='blog')

        # change column names so everything works later
        data.rename(columns={"clean_text": "clean_data",
                             "age_cat": "labels"}, inplace=True)

        # preproc_file = Path("./data/blogs_kaggle/blogger_preprocessed_data_FAKE.csv")

        # # Pre-process raw data if pre-processed data doesn't exist
        # try:
        #     preproc_abs_path = preproc_file.resolve(strict=True)
        # except FileNotFoundError:
        #     # doesn't exist
        #
        #     # Read and load dataset
        #     print("Reading raw data...")
        #     data = pd.read_csv("./data/blogs_kaggle/blogtext.csv")
        #     print("Done reading raw data.")
        #
        #
        #     # Subsetting data
        #     # perc_df = 0.00020 # fraction of dataset to take
        #     # sub_sample = math.ceil(perc_df * data.shape[0])
        #
        #     if subset_size != -1:
        #         # Chosen to train and test model(s) on subset of size subset_size
        #
        #         #shuffle data set before subsampling
        #         data = data.sample(frac=1).reset_index(drop=True)
        #         data = data[:subset_size]
        #
        #     print(f"Dataset size before preprocessing: {data.shape[0]}")
        #
        #     print("Preprocessing data...")
        #     # Removing all unwanted text/characters from data['text'] column
        #     # Remove all non-alphabetical characters
        #     data['clean_data'] = data['text'].apply(lambda x: re.sub(r'[^A-Za-z]+',' ', x))
        #
        #     # Make all letters lower case
        #     data['clean_data'] = data['clean_data'].apply(lambda x: x.lower())
        #
        #     # Remove white space from beginning and end of string
        #     data['clean_data'] = data['clean_data'].apply(lambda x: x.strip())
        #
        #     # Remove instances empty strings
        #     before_rm_empty = len(data)
        #     data.drop(data[data.clean_data == ''].index, inplace = True)
        #
        #     print(f'{before_rm_empty - len(data)} empty string instances removed.')
        #
        #     # Remove texts that are probably not English by filtering blogs that dont contain at least one of the top 50 most used English words
        #     # create dict with most common English words
        #     top_en_words = {}
        #     with open('./data/wordlists/top1000english.txt') as f:
        #         count = 1
        #         for line in f:
        #             key = line.split()[0].lower()
        #             top_en_words[key] = count
        #             count += 1
        #
        #             # Stop at top 50 words. Idea taken from DialoGPT paper.
        #             if count > 50:
        #                 break
        #
        #
        #     data['top_50_en'] = data['clean_data'].apply(lambda x : True if not set(x.split()).isdisjoint(top_en_words) else False)
        #
        #     def top_lang_detect(text):
        #
        #         detected_langs = detect_langs(text)
        #
        #         return detected_langs[0].lang
        #
        #
        #     def top_prob_detect(text):
        #
        #         detected_langs = detect_langs(text)
        #
        #         return detected_langs[0].prob
        #
        #     start_time = time.time()
        #     data['top_lang'] = data['clean_data'].apply(top_lang_detect)
        #     print(f"Top lang detection took {time.time() - start_time} seconds")
        #     start_time = time.time()
        #     data['top_prob'] = data['clean_data'].apply(top_prob_detect)
        #     print(f"Top lang prob lang detection took {time.time() - start_time} seconds")
        #
        #     # Remove rows without one of top50 most common english words
        #     before_top50_removal = len(data)
        #     data.drop(data[data['top_50_en'] == False].index, inplace = True)
        #     print(f"{before_top50_removal - len(data)} instances dropped")
        #
        #     before_top_lang = len(data)
        #     data.drop(data[data['top_lang'] != 'en'].index, inplace = True)
        #     print(f'{before_top_lang - len(data)} instances dropped.')
        #
        #     before_top_prob = len(data)
        #     data.drop(data[data['top_prob'] < 0.9].index, inplace = True)
        #     print(f'{before_top_prob - len(data)} instances dropped.')
        #
        #     # Remove stop words
        #     stopwords = set(nltk.corpus.stopwords.words('english')) # use set (hash table) data structure for faster lookup
        #
        #     # also add urllink and nbsp to set of words to remove
        #     stopwords.update(['urllink', 'nbsp'])
        #
        #     data['clean_data'] = data['clean_data'].apply(lambda x: ' '.join([words for words in x.split() if words not in stopwords]))
        #
        #     print("Done preprocessing data.")
        #
        #     print("Saving preprocessed dataframe to csv...")
        #     # save pre-processed dataframe to csv
        #     data.to_csv("./data/blogs_kaggle/blogger_preprocessed_data.csv")
        #
        # else:
        #     # exists
        #     # Read and load dataset
        #     print("Reading preprocessed data...")
        #     data = pd.read_csv("./data/blogs_kaggle/blogger_preprocessed_data.csv")
        #     print("Done reading preprocessed data.")
        #     # data = data[['clean_data', 'labels']]
        #
        # print(f"Dataset size after preprocessing: {data.shape[0]}")
        #
        # # Drop columns that are uninformative for writing style (i.e., ID and date)
        # data.drop(['id', 'date'], axis = 1, inplace = True)
        #
        # # Add labels for age categories
        # def age_to_cat(age):
        #     '''Returns age category label for given age number.'''
        #
        #     if 13 <= int(age) <= 17:
        #         return '13-17'
        #     elif 23 <= int(age) <= 27:
        #         return '23-27'
        #     elif 33 <= int(age):
        #         return '33-47'
        #     else:
        #         print(int(age))
        #         raise ValueError("Given age not in one of pre-defined age groups.")
        #
        #
        # data['age_cat'] = data['age'].apply(age_to_cat)
        #
        # # Merge all possibly interesting labels into one column
        # data['labels'] = data.apply(lambda col: [col['gender'], str(col['age']), col['topic'], col['sign']], axis = 1)
        #
        # # Only keep age as label
        # # data['labels'] = data.apply(lambda col: [str(col['age'])], axis = 1) # TODO: Why keep age as string?
        # # data['labels'] = data.apply(lambda col: [col['age']], axis = 1)
        # data['labels'] = data.apply(lambda col: [col['age_cat']], axis = 1)
        #
        # # Reduce dataframe to only contain cleaned blogs and list of labels
        # data = data[['clean_data', 'labels']]

    # results dict
    accs_all = {}
    if dataset == 'blog':
        class_labels_list = ['13-17', '23-27', '33-47']
    elif dataset == 'bnc' or dataset == 'bnc_rb':
        class_labels_list = ['19_29', '50_plus']

    # Evaluate performance
    def print_evaluation_scores(labels, preds):
        print(f"Accuracy: {accuracy_score(labels, preds)}")
        print(f"F1 score: {f1_score(labels, preds, average = None)}") # outputs F1 per class
        print(f"Average precision: {average_precision_score(labels, preds, average = 'micro')}")
        print(f"Average recall: {recall_score(labels, preds, average = 'micro')}")
        print(classification_report(labels, preds, digits=5, zero_division=0))
        # print(f"Confusion Matrix: {confusion_matrix(labels.argmax(axis=1), preds.argmax(axis=1))}")


    # def print_top_n(vectorizer, clf, class_labels, n_feat = 10):
    #     """Prints features with the highest coefficient values, per class"""
    #     feature_names = vectorizer.get_feature_names()
    #     for i, class_label in enumerate(class_labels):
    #         topn = np.argsort(clf.estimators_[i].coef_)[0][-n_feat:]
    #         print("%s: %s" % (class_label,
    #               " ".join(feature_names[j] for j in topn)))

    # spacy english tokenizer
    # spacy_eng = spacy.load("en_core_web_sm")

    # def tokenizer_eng(text):
    #     text = str(text)
    #     return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]
    #
    # token_counter = Counter()
    # for sentence in data.clean_data:
    #     for word in tokenizer_eng(sentence):
    #         token_counter.update([word])
    #
    # min_thresh = 3000
    # trunc_counter = {x: count for x, count in token_counter.items() if count >= min_thresh}

    # TODO: FIX REVERSE ORDERING BUG. SEE NOTEBOOK FOR RPA
    # def print_top_n_thresh(vectorizer, clf, class_labels, n_feat = 100,
    #                        counter = trunc_counter):
    #     """Prints features with the highest coefficient values, per class"""
    #     feature_names = vectorizer.get_feature_names()
    #     for i, class_label in enumerate(class_labels):
    #         topn = np.argsort(clf.estimators_[i].coef_)[0][-n_feat:]
    #         print("%s: %s" % (class_label,
    #               " ".join(feature_names[j] for j in topn if feature_names[j] in counter)))

    # def print_top_n_thresh(vectorizer, clf, class_labels, n_feat = 100,
    #                    counter = trunc_counter):
    #     """Prints features with the highest coefficient values, per class"""
    #     feature_names = vectorizer.get_feature_names()
    #     for i, class_label in enumerate(class_labels):
    #         topn = np.argsort(clf.estimators_[i].coef_)[0][-n_feat:]
    #         topn = topn[::-1]  # Reverse order of arg s.t. features with high coefficients appear first
    #         print("%s: %s" % (class_label,
    #               " ".join(feature_names[j] for j in topn if feature_names[j] in counter)))
    #
    # def most_informative_feature_for_class(vectorizer, classifier, class_labels, n=10):
    #     #labelid = list(classifier.classes_).index(classlabel)
    #     feature_names = vectorizer.get_feature_names()
    #     for i, class_label in enumerate(class_labels):
    #         topn = sorted(zip(classifier.estimators_[i].coef_[0], feature_names))[-n:]
    #
    #         for coef, feat in topn:
    #             print(class_label, feat, coef)

    test_accs = {}
    test_f1s = {}

    for n_gram in n_grams:
        test_accs[n_gram] = {}
        test_f1s[n_gram] = {}
        # for class_label in class_labels_list:
        #     test_f1s[n_gram][class_label] = {}



    print("Starting training and testing loops...")
    for seed in tqdm(seeds, desc = "Seed loop."):

        # set seed for reproducibility
        np.random.seed(seed)

        # shuffle dataframe
        data = data.sample(frac=1).reset_index(drop=True)


        for n in tqdm(n_grams, desc = "n gram loop."):

            # Split data into features/ X and labels / Y
            X = data['clean_data']
            Y = data['labels']

            # n-gram model
            vectorizer = CountVectorizer(binary = True, ngram_range = (1, n))

            # fit model
            X = vectorizer.fit_transform(X)

            # # check out a sample of the uni- and bigrams
            # print(vectorizer.get_feature_names()[:10])

            # Get label counts
            label_counts = {}
            if dataset == 'blog':
                # for labels in data.labels.values:
                #     for label in labels:
                #         if label in label_counts:
                #             label_counts[label] += 1
                #         else:
                #             label_counts[label] = 1
                for label in data.labels.values:
                    if label in label_counts:
                        label_counts[label] += 1
                    else:
                        label_counts[label] = 1
            elif dataset == 'bnc_rb' or dataset == 'bnc':
                for label in data.labels.values:
                    if label in label_counts:
                        label_counts[label] += 1
                    else:
                        label_counts[label] = 1

            # Binarize the labels for prediction
            if dataset == 'blog':
                # binarizer = MultiLabelBinarizer(classes = sorted(label_counts.keys()))
                binarizer = LabelBinarizer()
            elif dataset == 'bnc_rb' or dataset == 'bnc':
                binarizer = LabelBinarizer()

            Y = binarizer.fit_transform(data.labels)

            label_counts.keys()

            # Split data into train and test sets
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size)

            # if n == 1:
            #     # save splits and vectorizer
            #     save_file_splits_vzer = f"splits_vzer_{n}_gram_seed_{seed}"
            #     pickle.dump((vectorizer, X_train, X_test, Y_train, Y_test),
            #                 open(save_file_splits_vzer, 'wb'))


            # Fit logistic regression model
            start_time = time.time()
            model = LogisticRegression(solver = 'lbfgs', multi_class='ovr', max_iter = 1000000)
            model = OneVsRestClassifier(model)
            # model = MultiOutputClassifier(model)
            model.fit(X_train, Y_train)
            print(f"Fitting model took {time.time() - start_time} seconds.")

            # save the classifier
            # save_file_name = f"logit_{n}_gram_seed_{seed}"
            # pickle.dump(model, open(save_file_name, 'wb'))

            # make predictions on test set
            Y_pred = model.predict(X_test)

            Y_pred_inversed = binarizer.inverse_transform(Y_pred)
            Y_test_inversed = binarizer.inverse_transform(Y_test)

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

            # UNCOMMENT FOLLOWING LINES FOR CM PLOTS
            # int_labels = [label for label in range(len(class_labels_list))]
            # cm = confusion_matrix(Y_test, Y_pred, labels=int_labels)
            # make_confusion_matrix(cf=cm, categories=class_labels_list, title=f'Confusion Matrix for {dataset} on Test set',
            #                       num_labels=int_labels, y_true=Y_test, y_pred=Y_pred, figsize=FIGSIZE)
            # cur_datetime = datetime.now().strftime('%d_%b_%Y_%H_%M_%S')
            # plt.savefig(f"{FIGDIR}{dataset}/cm_{n}_gram_{dataset}_dt_{cur_datetime}.png",
            #             bbox_inches='tight')

    #         most_informative_feature_for_class(vectorizer = vectorizer, classifier = model, class_labels = class_labels_list, n=10)

    # def plot_accuracies(accs, show = False):
    #
    #     means = [np.mean(accs[n]) for n in range(1, len(accs) + 1)]
    #     # print(np.mean(means))
    #     stds = [np.std(accs[n]) for n in range(1, len(accs) + 1)]
    #
    #     x_pos = np.arange(len(accs))
    #     x_labels = list(accs.keys())
    #
    #     # Build the plot
    #     fig, ax = plt.subplots()
    #     ax.bar(x_pos, means, yerr=stds, align='center', alpha=0.5, ecolor='black', capsize=10)
    #     ax.set_ylabel('Mean classification accuracy.')
    #     ax.set_xlabel("$n$")
    #     ax.set_xticks(x_pos)
    #     ax.set_xticklabels(x_labels)
    #     ax.set_title('Age group prediction accuracy for various n-gram models.')
    #     ax.yaxis.grid(True)
    #
    #     # Save the figure and show
    #     plt.tight_layout()
    #     plt.savefig('figures/bar_plot_with_error_bars_10000.png')
    #
    #     if show:
    #         plt.show()

    # plot_accuracies(accs_all)

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


    # pdb.set_trace()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description = "Train n gram logistic classifiers."
    )

    # Command line arguments
    parser.add_argument("--dataset", type=str, default='blog',
                        choices=("blog", "bnc", "bnc_rb"),
                        help="Dataset to train n-gram logistic classifiers on."
                             "Dataset must be .csv format.")
    parser.add_argument("--dataset_fp", type=str, default="",
                        help="File path of the dataset to use. "
                             "Needed only in case of generic datadset")
    parser.add_argument("--subset_size", type=int, default=-1,
                        help="Desired size of subset. -1: full dataset")
    parser.add_argument("--n_grams", type=int, default=[3], nargs='+',
                        help="Size(s) of n-gram models.")
    parser.add_argument("--seeds", type=int, default=[1], nargs='+',
                        help="Seeds to set for reproducibility."
                             "NB: number of seeds is the number of runs to " + \
                             "train and test stochastic models.")
    parser.add_argument("--test_size", type=float, default = 0.2,
                        help="Fraction of preprocessed data(sub)set to reserve " + \
                             "for testing.")

    args = parser.parse_args()

    train_classifiers(**vars(args))
