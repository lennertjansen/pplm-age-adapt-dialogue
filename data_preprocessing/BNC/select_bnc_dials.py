
import argparse
import os
from collections import defaultdict, Counter
import pandas as pd
import spacy
from nltk import word_tokenize as nltk_tokenize

from BNC import Corpus

import pdb

def main(args):

    bnc_path = args.data_path

    corpus = Corpus(
        untagged_path=os.path.join(bnc_path, "untagged"),
        tagged_path=os.path.join(bnc_path, "tagged"),
        n=args.no_files,
        add_speaker_id=True
    )

    relevant_conversation_ids = []
    N = args.no_speakers

    # age_cat = args.age_cat

    # out_dict = {}
    #
    # # pdb.set_trace()
    #
    # for conv_id, conv in corpus.conversations.items():
    #     for speaker_id in conv.speaker_ids:
    #         if speaker_id not in out_dict:
    #             # speaker_dict = {}
    #             # speaker_dict['age_range'] = corpus.speakers[speaker_id].age_range
    #             # speaker_dict['topics'] = set(conv.topics)
    #             # out_dict[speaker_id] = speaker_dict
    #             # out_dict[speaker_id] = [corpus.speakers[speaker_id].age_range, set(conv.topics)]
    #             out_dict[speaker_id] = corpus.speakers[speaker_id].age_range
    #         else:
    #             # out_dict[speaker_id]['topics'].update(conv.topics)
    #             # out_dict[speaker_id][1].update(conv.topics)
    #             pass
    #
    # out_df = pd.DataFrame(list(out_dict.items()))
    #
    # # out_df = pd.DataFrame.from_dict({(i, j): out_dict[i][j]
    # #                         for i in out_dict.keys()
    # #                         for j in out_dict[i].keys()},orient='index')
    # out_df.to_csv(
    #     'speakers_dict.csv',
    #     index=False
    # )

    # for age_cat in ["11_18", "19_29", "30_39", "40_49", "50_59", "60_69", "70_79", '80_89', '90_99']:
    #     for age_cat2 in ["11_18", "19_29", "30_39", "40_49", "50_59", "60_69", "70_79", '80_89', '90_99']:

    for conv_id, conv in corpus.conversations.items():
        if conv.n_speakers == N:
            # age_ranges_list = list(conv.speakers_age_ranges.values())
            # if age_ranges_list[0] == age_cat and age_ranges_list[1] == age_cat2:
            relevant_conversation_ids.append(conv_id)
            pdb.set_trace()
    for conv_id in relevant_conversation_ids:
        print(100 * "=")
        print('|' + 98 * " " + "|")
        print('|' + 98 * " " + "|")
        print('|' + 98 * " " + "|")
        print('|' + 98 * " " + "|")
        print(100 * "=")
        print(f"Speaker age ranges: {corpus.conversations[conv_id].speakers_age_ranges}")
        for u in corpus.conversations[conv_id].utterances:
            print(u.sentence)


    if args.meta:

        keys = ["conv_id", "n_utterances", "n_tokens", "topics", "type", 'speaker1_id', "speaker1_age_cat",
                'speaker1_gender',
                'speaker1_educ', 'speaker2_id', "speaker2_age_cat", 'speaker2_gender', 'speaker2_educ']
        df = pd.DataFrame(columns=keys)

        for conv_id, conv in corpus.conversations.items():
            if conv.n_speakers == N:
                dial_meta_dict = {}
                dial_meta_dict.fromkeys(keys)
                # age_ranges_list = list(conv.speakers_age_ranges.values())

                speaker1_id = conv.speaker_ids[0]
                speaker2_id = conv.speaker_ids[1]

                speaker1_age_range = corpus.speakers[speaker1_id].age_range
                speaker2_age_range = corpus.speakers[speaker2_id].age_range

                speaker1_gender = corpus.speakers[speaker1_id].gender
                speaker2_gender = corpus.speakers[speaker2_id].gender

                speaker1_education = corpus.speakers[speaker1_id].education
                speaker2_education = corpus.speakers[speaker2_id].education

                df.loc[0 if pd.isnull(df.index.max()) else df.index.max() + 1] = [conv_id] + [conv.n_utterances] +\
                                                                                 [conv.n_tokens] + [conv.topics] + \
                                                                                 [conv.type] + [speaker1_id] + \
                                                                                 [speaker1_age_range] + \
                                                                                 [speaker1_gender] + [speaker1_education]+\
                                                                                 [speaker2_id] + \
                                                                                 [speaker2_age_range] + [speaker2_gender] +\
                                                                                 [speaker2_education]


        if args.no_files == 0:
            args.no_files = 'all'

        df.to_csv(
            f'data_preprocessing/BNC/metadata/metadata_nfiles_{args.no_files}_nspeakers_{args.no_speakers}.csv',
            index=False
        )




if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--age_cat', type = str, default = '30_39',
        choices = ["11_18", "19_29", "30_39", "40_49", "50_59", "60_69", "70_79"],
        help = "Desired age category of all speakers in queried conversations."
    )
    parser.add_argument(
        '--no_speakers', type = int, default = 2,
        help = "Number of speakers in conversations"
    )
    parser.add_argument(
        '--no_files', type = int, default = 500,
        help = "Maximum number of files to load. 0: all"
    )
    parser.add_argument(
        '--data_path', type = str, default = 'data/bnc2014spoken-xml/spoken/'
    )
    parser.add_argument(
        '--meta', action='store_true', help='Output metadata .csv file.'
    )


    args = parser.parse_args()

    main(args)
