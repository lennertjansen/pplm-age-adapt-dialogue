from collections import Counter
from string import punctuation
import nltk
import pandas as pd
import numpy as np
import pdb
import matplotlib.pyplot as plt



def df_to_wordlist(df, top_k=None, age=None):

    # set of stopwords
    stopwords = set(nltk.corpus.stopwords.words('english')) # 0(1) lookups

    # frequency counter without stopwords
    without_stp  = Counter()

    # remove apostrophe (') from punctuation so tokens like "i'm" stay
    punc = punctuation.translate(str.maketrans('', '', "'"))
    numbers = '0123456789'
    # any(p in ts for p in punctuation)

    # extract text from dataframe
    if not age:
        text = df.text
    else:
        text = df.text[df.label == age]

    for line in text:
        # split line into tokens
        spl = line.split()

        # update count off all words in the line that are not in stopwords
        # without_stp.update(w.lower().rstrip(punctuation) for w in spl if w not in stopwords and not any(p in w for p in punctuation) and not any(p in w for p in numbers)) # exclude stopwords
        without_stp.update(w.lower() for w in spl if not any(p in w for p in punctuation) and not any(p in w for p in numbers)) # keep stopwords

    # return a list with top ten most common words from each
    if top_k:
        return [(word, count) for word, count in without_stp.most_common(top_k)]
    else:
        return [(word, count) for word, count in without_stp.most_common()]



if __name__ == '__main__':

    bnc_rb_path = 'data/bnc/bnc_subset_19_29_vs_50_plus_nfiles_0_rand_balanced.csv'
    bnc_path = 'data/bnc/bnc_subset_19_29_vs_50_plus_nfiles_0.csv'
    print('Loading data...')
    df = pd.read_csv(bnc_rb_path, encoding="utf-8")  # to keep no. unique chars consistent across platforms

    most_common_words = df_to_wordlist(df, top_k=500)
    # with open('plug_play/wordlists/bnc_rb_ws_500_most_common.txt', 'w') as f:
    #     for word, count in most_common_words:
    #         f.write("%s\n" % word)

    df_full = pd.read_csv(bnc_path, encoding="utf-8")  # to keep no. unique chars consistent across platforms
    mcw_young = df_to_wordlist(df_full, age='19_29')
    mcw_old = df_to_wordlist(df_full, age='50_plus')

    freqs_young = [count for word, count in mcw_young]
    sum_freqs_young = sum(freqs_young)

    freqs_old = [count for word, count in mcw_old]
    sum_freqs_old = sum(freqs_old)

    perc_young = 0
    zipf_young = []
    for word, count in mcw_young:

        if perc_young < .8:
            zipf_young.append((word, count))
            perc_young += (count / sum_freqs_young)
        else:
            break

    perc_old = 0
    zipf_old = []
    for word, count in mcw_old:

        if perc_old < .8:
            zipf_old.append((word, count))
            perc_old += (count / sum_freqs_old)
        else:
            break

    # indices = np.arange(len(mcw_young[:200]))
    # freqs = [count for word, count in mcw_young[:200]]
    # words = [word for word, count in mcw_young[:200]]
    # plt.bar(indices, freqs, color='r')
    # plt.xticks(indices, words, rotation='vertical')
    # plt.tight_layout()
    # plt.show()



    cutoff = 3000

    zipf_young_words = [word for word, count in zipf_young]
    zipf_old_words = [word for word, count in zipf_old]

    young_words_unique = np.setdiff1d(zipf_young_words, zipf_old_words)
    old_words_unique = np.setdiff1d(zipf_old_words, zipf_young_words)


    # remove word from zipf sets if they are members of the union
    zipf_young = [(word, count) for word, count in zipf_young if word in young_words_unique]
    zipf_old = [(word, count) for word, count in zipf_old if word in old_words_unique]

    sum_young = sum([count for word, count in zipf_young])
    sum_old = sum([count for word, count in zipf_old])

    zipf_young = [(word, float(count) / sum_young) for word, count in zipf_young]
    zipf_old = [(word, float(count) / sum_old) for word, count in zipf_old]


    max_percentile = 0.8

    mcw_young_unique = []
    percen_young = 0

    mcw_old_unique = []
    percen_old = 0

    for word, prob in zipf_young:
        if percen_young < max_percentile:
            mcw_young_unique.append((word, prob))
            percen_young += prob
        else:
            break

    for word, prob in zipf_old:
        if percen_old < max_percentile:
            mcw_old_unique.append((word, prob))
            percen_old += prob
        else:
            break

    print(f"Young mcwu len: {len(mcw_young_unique)}")
    print(f"Old mcwu len: {len(mcw_old_unique)}")
    with open(f'plug_play/wordlists/bnc_young_mcwu_ws_pct_{int(100*max_percentile)}.txt', 'w') as f:
        for word, prob in mcw_young_unique:
            f.write("%s\n" % word)

    with open(f'plug_play/wordlists/bnc_old_mcwu_ws_pct_{int(100*max_percentile)}.txt', 'w') as f:
        for word, prob in mcw_old_unique:
            f.write("%s\n" % word)

    pdb.set_trace()

