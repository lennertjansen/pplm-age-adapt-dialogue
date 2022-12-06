import csv
import pdb
from pdb import set_trace
import pandas as pd
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
    df['clean_text'] = df['text'] # uncomment this and comment line above if you want incl. non-alph chars

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


def main():

    # bnc_path = 'data/bnc/bnc_subset_19_29_vs_50_plus_nfiles_0_rand_balanced.csv'
    blog_path = 'data/blogs_kaggle/blogtext.csv'
    print('Loading data...')
    df = pd.read_csv(blog_path, encoding="utf-8")  # to keep no. unique chars consistent across platforms

    print('Preprocessing data...')
    df_pp = preprocess_df(df=df, data='blog')
    # df_pp.index = range(0, len(df_pp))  # you can define your range as required
    df_pp.reset_index(drop=True, inplace=True)


    print("Writing data to tsv...")
    with open('blogs_incl_stopwords_nonalph_generic_pplm.txt', 'w') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')
        for row_num in range(len(df_pp)):
            # writer.writerow([df.loc[row_num].label, df.loc[row_num].text])
            writer.writerow([df_pp.loc[row_num].age_cat, df_pp.loc[row_num].clean_text])
            # if row_num > 10000:
            #     break

if __name__ == '__main__':
    main()