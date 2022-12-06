import argparse
import os
from collections import defaultdict, Counter
import pandas as pd
import spacy
from nltk import word_tokenize as nltk_tokenize
from BNC import Corpus
import pdb
from tqdm import tqdm

def main(args):

    # Create Corpus instance based on given data path and number of files to load
    bnc_path = args.raw_data_dir
    corpus = Corpus(
        untagged_path=os.path.join(bnc_path, "untagged"),
        tagged_path=os.path.join(bnc_path, "tagged"),
        n=args.no_files,
        add_speaker_id=True
    )

    # column names and instantiation of subset output dataframe
    keys = ['speaker_id', 'gender', 'topics', "conv_id", 'turn', 'text', 'length', 'label']
    df = pd.DataFrame(columns=keys)

    N = args.no_speakers

    for conv_id, conv in tqdm(corpus.conversations.items()):
        if conv.n_speakers == N:

            # TODO: this now assumes N is 2. Make this work for all N?
            speaker1_id = conv.speaker_ids[0]
            speaker2_id = conv.speaker_ids[1]

            same_group1 = (corpus.speakers[speaker1_id].age_range in args.group1 and
                           corpus.speakers[speaker2_id].age_range in args.group1)
            same_group2 = (corpus.speakers[speaker1_id].age_range in args.group2 and
                           corpus.speakers[speaker2_id].age_range in args.group2)

            # if same_group1:
            #     for u in conv.utterances:
            #
            #         df.loc[0 if pd.isnull(df.index.max()) else df.index.max() + 1] = [u.speaker_id] + \
            #                                                                          [conv_id] + \
            #                                                                          [u.turn] + [u.transcription] + \
            #                                                                          [len(u.tokens)] + [args.label1]
            # elif same_group2:
            #     for u in conv.utterances:
            #         df.loc[0 if pd.isnull(df.index.max()) else df.index.max() + 1] = [u.speaker_id] + \
            #                                                                          [conv_id] + \
            #                                                                          [u.turn] + [u.transcription] + \
            #                                                                          [len(u.tokens)] + [args.label2]

            if same_group1:
                for u in conv.utterances:

                    try:
                        gender = corpus.speakers[u.speaker_id].gender
                    except KeyError:
                        if u.speaker_id == 'UNKMALE':
                            gender = 'M'
                        elif u.speaker_id == 'UNKFEMALE':
                            gender = 'F'

                    df.loc[0 if pd.isnull(df.index.max()) else df.index.max() + 1] = [u.speaker_id] + \
                                                                                     [gender] + \
                                                                                     [conv.topics] + \
                                                                                     [conv_id] + \
                                                                                     [u.turn] + [u.transcription] + \
                                                                                     [len(u.tokens)] + [args.label1]


            elif same_group2:
                for u in conv.utterances:

                    try:
                        gender = corpus.speakers[u.speaker_id].gender
                    except KeyError:
                        if u.speaker_id == 'UNKMALE':
                            gender = 'M'
                        elif u.speaker_id == 'UNKFEMALE':
                            gender = 'F'

                    df.loc[0 if pd.isnull(df.index.max()) else df.index.max() + 1] = [u.speaker_id] + \
                                                                                     [gender] + \
                                                                                     [conv.topics] + \
                                                                                     [conv_id] + \
                                                                                     [u.turn] + [u.transcription] + \
                                                                                     [len(u.tokens)] + [args.label2]


    # Balance out dataset
    # Option 0) Remove 19_29 utterances until no. 19_29 utterances == no. 50_plus utterances, without taking into account
    # unique SID's

    # Randomly drop 19_29 rows until dataset is balanced
    while df['label'].value_counts()['19_29'] != df['label'].value_counts()['50_plus']:
        df = df.drop(df[df['label'] == '19_29'].sample(n=1).index)

    print(f"Df length after balancing: {len(df)}")
    nu_19 = df[df.label == '19_29']['speaker_id'].nunique()
    nu_50 = df[df.label == '50_plus']['speaker_id'].nunique()
    print(f'No. unique speakers in range 19_29: {nu_19}')
    print(f'No. unique speakers in range 50_plus: {nu_50}')



    pdb.set_trace()

    # Save dataframe to csv
    df.to_csv(
        args.save_dir +
        f'bnc_subset_{args.label1}_vs_{args.label2}_nfiles_{args.no_files}_w_gender_topics.csv',
        index=False
    )


def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--no_files', type=int, default=500,
        help="Maximum number of files (conversations) to load. 0: all. Default=500. Max=1251"
    )
    parser.add_argument(
        '--raw_data_dir', type=str, default='data/bnc2014spoken-xml/spoken/'
    )
    parser.add_argument(
        '--save_dir', type=str, default='data_preprocessing/BNC/subsets/'
    )
    parser.add_argument(
        '--no_speakers', type=int, default=2,
        help="Number of speakers in conversations"
    )
    #TODO: also output an accompanying metadata csv file
    parser.add_argument(
        '--meta', action='store_true', help='Output metadata .csv file.'
    )
    parser.add_argument(
        '--shuffle', action='store_true', help='Shuffle rows of output dataframe.'
    )
    parser.add_argument(
        '--group1', nargs='+', help='Age ranges to include in first group.', default=['19_29']
    )
    parser.add_argument(
        '--label1', type=str, default='19_29', help='Name/label for group 1.'
    )
    parser.add_argument(
        '--group2', nargs='+', help='Age ranges to include in first group.',
        default=['50_59', '60_69', '70_79', '80_89', '90_99']
    )
    parser.add_argument(
        '--label2', type=str, default='50_plus', help='Name/label for group 2.'
    )

    # TODO: Add arguments to specify:
    # 1): number of (regrouped) age categories to keep; e.g., 2 or 3
    # 2): How these should be split/regrouped; e.g., 19-29 vs. 50 plus
    # ---> this could also be done by asking for lists containing the regrouped age categories
    # 3) Utterance granularity: single utterance, one round (i.e., two turns/utterances), ..., full conversation

    return parser.parse_args()

if __name__ == '__main__':

    # Parse command line arguments
    args = parse_arguments()

    main(args)