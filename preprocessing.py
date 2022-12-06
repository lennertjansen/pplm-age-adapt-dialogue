import pdb

from nltk.corpus import stopwords
import pandas as pd
import re

def preprocess_df(df, data='blog'):

    data_size = len(df)

    #TODO Pre-processing steps happen here -- implement them st the less
    # standard ones can be switched on and off to evaluate their impact on
    # performance:
    # Remove all non-alphabetical characters
    # df['clean_text'] = df['text'].apply(lambda x: re.sub(r'[^A-Za-z]+',' ', x))
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

    # TODO: filter out non-English

    # TODO: mask age-disclosing utterances

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

    # import pdb
    # pdb.set_trace()
    return df[['clean_text', 'age_cat']]
    # return df[['clean_text', 'label']]
