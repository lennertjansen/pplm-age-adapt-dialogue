import numpy as np
import pdb
import re

with open('plug_play/wordlists/bnc_young_mcwu.txt', 'r') as f:
    young_list = [line.strip() for line in f]

with open('plug_play/wordlists/bnc_old_mcwu.txt', 'r') as f:
    old_list = [line.strip() for line in f]

with open('plug_play/texts/unperturbed2.txt', 'r') as f:
    unp_texts = [[word.lower() for word in re.sub("[^\w]", " ",  line).split()] for line in f]

# NB: these were generated with stepsize 0.03
with open('plug_play/texts/old2.txt', 'r') as f:
    old_texts = [[word.lower() for word in re.sub("[^\w]", " ",  line).split()] for line in f]

# NB: these were generated with stepsize 0.03
with open('plug_play/texts/young3.txt', 'r') as f:
    young_texts = [[word.lower() for word in re.sub("[^\w]", " ",  line).split()] for line in f]


def wl_ctrl(wordlist, text):

    if isinstance(wordlist, list):
        # convert wordlist to dict for faster lookup
        word_dict = {}.fromkeys(wordlist)
    elif isinstance(wordlist, dict):
        pass
    else:
        raise TypeError("Wordlist is not of type list or dict.")

    # make list of words from wordlist in text
    listwords_in_text = [word for word in text if word in word_dict]

    numer = len(listwords_in_text) + 1
    denom = len(text) * len(wordlist)

    # result = (np.log(len(listwords_in_text) + 1)) / np.log(len(text))
    # result = np.log((len(listwords_in_text) + 1) / len(text))

    # result = numer / denom
    result = np.log(numer)/np.log(denom)
    # result = np.log(numer / denom)

    # return result1, result2, result3
    return result


ot_ol = []
yt_yl = []

ot_yl = []
yt_ol = []

ut_yl = []
ut_ol = []


for text in old_texts:
    ot_yl.append(wl_ctrl(young_list, text))
    ot_ol.append(wl_ctrl(old_list, text))

for text in young_texts:
    yt_yl.append(wl_ctrl(young_list, text))
    yt_ol.append(wl_ctrl(old_list, text))

for text in unp_texts:
    ut_yl.append(wl_ctrl(young_list, text))
    ut_ol.append(wl_ctrl(old_list, text))


print('-' * 100)
print('-' * 100)
print(f'Young wordlist + Unp. texts || Mean: {np.mean(ut_yl)} | Std: {np.std(ut_yl)}')
print(f'Old wordlist +  Unp. texts || Mean: {np.mean(ut_ol)} | Std: {np.std(ut_ol)}')
print('-' * 100)
print(f'Young wordlist + Old texts || Mean: {np.mean(ot_yl)} | Std: {np.std(ot_yl)}')
print(f'Old wordlist + Old texts || Mean: {np.mean(ot_ol)} | Std: {np.std(ot_ol)}')
print('-' * 100)
print(f'Young wordlist + Young texts || Mean: {np.mean(yt_yl)} | Std: {np.std(yt_yl)}')
print(f'Old wordlist + Young texts || Mean: {np.mean(yt_ol)} | Std: {np.std(yt_ol)}')
print('-' * 100)
print('-' * 100)

pdb.set_trace()



