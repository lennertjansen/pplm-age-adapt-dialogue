# Handles pre-processing of data and implementation of dataset type.
#
# Manual Vocabulary based on: https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/custom_dataset_txt/loader_customtext.py
# Date created: 1 March 2021
################################################################################
# What do I want to achieve?
# Main goal: Convert text to numerical values
# To do so, wee need:
# 1. A vocabulary that maps every word to an index
# 2. To set up a PyTorch Dataset type to load the data
# 3. Set up padding for each batch so every sequence in a batch has equal length


# Import statements
import pdb
from pdb import set_trace # for easier debugging
import os # for path file reading/loading
import numpy as np # for linear algebra and numerical methods
import pandas as pd # for easier csv parsing
from torch import is_tensor
from torch.utils.data import Dataset, DataLoader
from nltk.corpus import stopwords
import re
import torch
from torch.nn.utils.rnn import pad_sequence  # pad batch
from custom_tokenizers import Vocabulary, WordTokenizer
from itertools import islice
from tokenizers import Tokenizer
from preprocessing import preprocess_df

from transformers import BertTokenizer, BertTokenizerFast

class BlogDataset(Dataset):
    '''
    Text dataset type. Inherits functionality from data.Dataset.
    '''

    def __init__(self, df, tokenizer):
        """
        ...
        """
        #
        # #TODO Pre-processing steps happen here -- implement them st the less
        # # standard ones can be switched on and off to evaluate their impact on
        # # performance:
        # # Remove all non-alphabetical characters

        #
        # # TODO: filter out non-English
        #
        # # TODO: mask age-disclosing utterances


        self.df = df
        self.tokenizer = tokenizer

        # # TODO: make this dynamic.
        self.num_classes = 3
        self.vocab_size = tokenizer.vocab_size

    def __len__(self):
        '''
        Overrides the inherited method s.t. len(dataset) returns the size of the
        data set.
        '''
        return len(self.df)

    def __getitem__(self, index):
        '''
        Overrides the inherited method s.t. indexing dataset[i] will get the
        i-th sample. Also handles reading of examples (as opposed to init()),
        which is more memory efficient, as all examples are not stored in
        memory at once, but as they are required.
        '''

        # Convert index to list if its a tensor
        if is_tensor(index):
            index = index.tolist()

        encoded_input = self.tokenizer.encode(self.df.clean_text.iloc[index], add_special_tokens=True, truncation=True,
                                              max_length=512)
        label = self.df.age_cat.iloc[index]

        return torch.tensor(encoded_input), torch.tensor(label)

class BncDataset(Dataset):

    def __init__(self, df, tokenizer):
        """
        ...
        """
        #
        # #TODO Pre-processing steps happen here -- implement them st the less
        # # standard ones can be switched on and off to evaluate their impact on
        # # performance:
        # # Remove all non-alphabetical characters

        #
        # # TODO: filter out non-English
        #
        # # TODO: mask age-disclosing utterances


        self.df = df
        self.tokenizer = tokenizer

        # # TODO: make this dynamic.
        self.num_classes = 2
        self.vocab_size = tokenizer.vocab_size

    def __len__(self):
        '''
        Overrides the inherited method s.t. len(dataset) returns the size of the
        data set.
        '''
        return len(self.df)

    def __getitem__(self, index):
        '''
        Overrides the inherited method s.t. indexing dataset[i] will get the
        i-th sample. Also handles reading of examples (as opposed to init()),
        which is more memory efficient, as all examples are not stored in
        memory at once, but as they are required.
        '''

        # Convert index to list if its a tensor
        if is_tensor(index):
            index = index.tolist()

        encoded_input = self.tokenizer.encode(self.df.clean_text.iloc[index], add_special_tokens=True, truncation=True,
                                              max_length=512)
        label = self.df.age_cat.iloc[index]

        return torch.tensor(encoded_input), torch.tensor(label)


class MyCollate:
    def __init__(self, pad_index):
        self.pad_index = pad_index

    def __call__(self, batch):

        blogs = [blog[0].unsqueeze(0) for blog in batch]
        blogs = torch.cat(blogs, dim = 0)
        blogs = pad_sequence(blogs, batch_first = False, padding_value = self.pad_index)

        return blogs

class PadSequence:
    def __init__(self, pad_index=0):
        self.pad_index = pad_index

    def __call__(self, batch):
        # Let's assume that each element in "batch" is a tuple (data, label).
        # Sort the batch in the descending order
        # sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        sorted_batch = batch

        # sorted_batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
        # Get each sequence and pad it
        sequences = [x[0] for x in sorted_batch]

        sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=self.pad_index)

        # Also need to store the length of each sequence
        # This is later needed in order to unpad the sequences
        lengths = torch.LongTensor([len(x) for x in sequences])

        # Don't forget to grab the labels of the *sorted* batch
        labels = torch.LongTensor(list(map(lambda x: x[1], sorted_batch)))

        return sequences_padded, labels, lengths

def padded_collate(batch, pad_idx=0):
    """Pad sentences, return sentences and labels as LongTensors."""
    blogs, labels = zip(*batch)
    lengths = [len(s) for s in blogs]
    max_length = max(lengths)
    # Pad each sentence with zeros to max_length
    padded_sentences = [blog + [pad_idx] * (max_length - len(blog)) for blog in blogs]
    # padded_targets = [s + [pad_idx] * (max_length - len(s)) for s in targets]

    return torch.LongTensor(padded_sentences), torch.LongTensor(labels), lengths



def get_datasets(subset_size=None,
                 file_path = 'data/blogs_kaggle/blogtext.csv',
                 train_frac = 0.7,
                 test_frac = 0.2,
                 val_frac = 0.1,
                 seed=2021,
                 data='blog',
                 model_type='lstm'):

    """
    :param subset_size: (int) number of datapoints to take as subset. If None, full dataset is taken.
    :param file_path: (str) full file path to .csv dataset
    :param train_frac: (float) fraction of data(sub)set's observations to reserve for training
    :param test_frac: (float) fraction of data(sub)set's observations to reserve for validation/model selection
    :param val_frac: (float) fraction of data(sub)set's observations to reserve for testing
    :param seed: (int) random state/seed for reproducibility of shuyffling dataset.
    :return: train/val/test splits as BlogDataset types.
    """

    # set seed for reproducibility of the shuffling
    np.random.seed(seed)

    try:
        # Assert that the given splits are valid, i.e., sum up to one
        assert np.sum([train_frac, test_frac, val_frac]) == 1
    except:
        # ensure they do
        val_frac = 1 - train_frac - test_frac

        # Assert that the given splits are valid, i.e., sum up to one
        assert np.sum([train_frac, test_frac, val_frac]) == 1


    # Read in dataframe and do the shuffling and splitting here. s.t. you only have to read it once
    assert os.path.splitext(file_path)[1] == ".csv"  # check if file csv format
    df = pd.read_csv(file_path, encoding="utf-8")  # to keep no. unique chars consistent across platforms
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle data set before subsampling

    if subset_size:
        df = df.iloc[:subset_size]
        df.reset_index(drop=True, inplace=True)  # Reset index after subsetting

    train_df, val_df, test_df = np.split(df, [int(train_frac * len(df)),
                                              int((1 - test_frac) * len(df))])

    # Temporary fix for case analysis....
    # train_preprocessed = pd.read_csv('data/bnc/ca_splits/bnc_rb_ca_trainset_case_analysis_ws.csv', encoding="utf-8")
    # val_preprocessed = pd.read_csv('data/bnc/ca_splits/bnc_rb_ca_valset_case_analysis_ws.csv', encoding="utf-8")
    # test_preprocessed = pd.read_csv('data/bnc/ca_splits/bnc_rb_ca_testset_case_analysis_ws.csv', encoding="utf-8")
    #
    # # reset indices of subsets
    # train_preprocessed.reset_index(drop=True, inplace=True)
    # val_preprocessed.reset_index(drop=True, inplace=True)
    # test_preprocessed.reset_index(drop=True, inplace=True)

    # TODO: uncomment this after you fixed the BERT dataset bug ###
    # reset indices of subsets
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    train_preprocessed = preprocess_df(train_df, data=data)
    val_preprocessed = preprocess_df(val_df, data=data)
    test_preprocessed = preprocess_df(test_df, data=data)
    ###############################################################



    if model_type == 'lstm':
        tokenizer = WordTokenizer(train_preprocessed.clean_text)
    elif model_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', truncation=True,
                                                  max_length=500) # truncation only considers sequences of max 512 tokens (same as original BERT implementation)

    if data == 'blog':
        # return the three splits as BlogDataset types
        train_dataset = BlogDataset(df=train_preprocessed, tokenizer=tokenizer)
        val_dataset = BlogDataset(df=val_preprocessed, tokenizer=tokenizer)
        test_dataset = BlogDataset(df=test_preprocessed, tokenizer=tokenizer)
    elif data == 'bnc' or data == 'bnc_rb':
        # return the three splits as BlogDataset types
        train_dataset = BncDataset(df=train_preprocessed, tokenizer=tokenizer)
        val_dataset = BncDataset(df=val_preprocessed, tokenizer=tokenizer)
        test_dataset = BncDataset(df=test_preprocessed, tokenizer=tokenizer)

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":

    # # create dataset instance
    # dataset = BlogDataset()
    #
    # # TODO: add collate function for batching that also returns lengths
    # data_loader = DataLoader(dataset, batch_size = 2, collate_fn = PadSequence())
    #
    # for a in islice(data_loader, 10):
    #     print(a)
    bnc_path = 'data/bnc/bnc_subset_19_29_vs_50_plus_nfiles_0.csv'
    bnc_rb_path = 'data/bnc/bnc_subset_19_29_vs_50_plus_nfiles_0_rand_balanced.csv'
    train_dataset, val_dataset, test_dataset = get_datasets(subset_size=None, file_path=bnc_rb_path, data='bnc_rb',
                                                            model_type='bert')
    input, label = train_dataset[34]
    pdb.set_trace()
