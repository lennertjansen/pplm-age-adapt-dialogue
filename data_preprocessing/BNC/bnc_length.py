import argparse
import os
from collections import defaultdict, Counter
import pandas as pd
import spacy
from nltk import word_tokenize as nltk_tokenize
from transformers import RobertaTokenizer

# from src.BNC import Corpus
from BNC import Corpus

nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'tagger'])

def spacy_tokenize(text):
    return [x.text.strip() for x in nlp(text)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", default="../../data/bnc2014spoken-xml/spoken", type=str,
        help="The input data directory."
    )
    parser.add_argument(
        "--out_path", default = "output", type=str,
        help="The output data directory."
    )
    parser.add_argument(
        "--cutoff", default=1, type=int,
        help="The sentence position frequency cut-off. Skip sentence positions with frequency lower than this value."
    )
    parser.add_argument(
        "--tokenizer", type=str,
        help="Tokenizer: 'nltk', 'spacy', or 'huggingface'."
    )
    parser.add_argument("--n", type=int, default = 100, help = "Number of files to load. 0: all")
    args = parser.parse_args()

    c = Corpus(
        untagged_path=os.path.join(args.data_path, "untagged"),
        tagged_path=os.path.join(args.data_path, "tagged"),
        n = args.n
    )

    if args.tokenizer == 'huggingface':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    sentence_lengths = []
    sentence_positions = []
    sentences = []
    for conv in c.conversations.values():
        turn = 0
        for u in conv.utterances:
            if not u.sentence:
                continue
            turn += 1
            sentences.append(u.sentence)
            sentence_positions.append(turn)

            if args.tokenizer == 'spacy':
                sentence_lengths.append(len(spacy_tokenize(u.sentence)))
            elif args.tokenizer == 'nltk':
                sentence_lengths.append(len(nltk_tokenize(u.sentence)))
            elif args.tokenizer == 'huggingface':
                inputs = tokenizer(
                    u.sentence,
                    return_tensors='pt',
                    add_special_tokens=False
                )
                sentence_lengths.append(inputs['input_ids'].shape[1])

    print("Number of utterances:", len(sentences))
    print("Max utterance position: ", max(sentence_positions))

    # Check frequency of each sentence position
    pos_counter = Counter(sentence_positions)
    print('Positions and their frequency:')
    print([(p, pos_counter[p]) for p in sorted(list(pos_counter.keys()))])

    # Check the highest sentence position for each frequency value
    highest_pos_by_freq = defaultdict(int)
    for turn, fr in pos_counter.items():
        if turn > highest_pos_by_freq[fr]:
            for _fr in range(1, fr + 1):
                highest_pos_by_freq[_fr] = turn

    # Only consider poisitions with at least k items in the test set (k=10 in Keller 2004)
    print('Frequency cut-off:', args.cutoff,
          '  Lowest position with this frequency:', highest_pos_by_freq[args.cutoff])

    tmp_sentences, tmp_positions, tmp_lengths = [], [], []
    for sentence, position, length in zip(sentences, sentence_positions, sentence_lengths):
        if pos_counter[position] >= args.cutoff:
            tmp_sentences.append(sentence)
            tmp_positions.append(position)
            tmp_lengths.append(length)

    tmp_dataset = list(zip(tmp_sentences, tmp_positions, tmp_lengths))
    tmp_tokens = [len(s.split()) for s in tmp_sentences]
    tmp_dataset = [x for (_, x) in sorted(zip(tmp_tokens, tmp_dataset), reverse=True)]

    positions = [x[1] for x in tmp_dataset]
    lengths = [x[2] for x in tmp_dataset]

    df = pd.DataFrame({
        'position': positions,
        'length': lengths
    })
    out_file_name = os.path.join(args.out_path, 'bnc_{}_{}'.format(
        args.cutoff,
        args.tokenizer
    ))
    df.to_csv(
        '{}.zip'.format(out_file_name),
        index=False,
        compression=dict(
            method='zip', archive_name='{}.csv'.format(out_file_name))
    )
