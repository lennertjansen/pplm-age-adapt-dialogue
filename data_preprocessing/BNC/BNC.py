import os
import xml.etree.ElementTree as ET
import random
import re
from string import ascii_uppercase
from tqdm import tqdm

# Added by Lennert
import pdb


SPEAKER_NAMES = [c for c in ascii_uppercase]


class Token(object):
    def __init__(self, w_element):
        self.form = w_element.text
        self.lemma = w_element.get("lemma")
        self.pos = w_element.get("pos")
        self.word_class = w_element.get("class")
        self.semantic_tag = w_element.get("usas")


class Utterance(object):
    def __init__(self, tagged_u_element, untagged_u_element, speaker_id2name):
        assert untagged_u_element.get("n") == untagged_u_element.get("n")
        self.turn = int(untagged_u_element.get("n"))

        assert untagged_u_element.get("who") == untagged_u_element.get("who")
        self.speaker_id = untagged_u_element.get("who")

        assert untagged_u_element.get("trans") == untagged_u_element.get("trans")
        self.overlap_with_previous = untagged_u_element.get("trans") == "overlap"

        assert untagged_u_element.get("whoConfidence") == untagged_u_element.get("whoConfidence")
        self.confident_attribution = untagged_u_element.get("whoConfidence") == "high"

        self.tokens = []
        # skipping transcription tags; they can be found in self.transcription
        for w_element in tagged_u_element.findall("w"):
            tok = Token(w_element)
            self.tokens.append(tok)

        transcription = ET.tostring(untagged_u_element).decode("utf-8")
        transcription = re.sub(r"<u .+\">", "", transcription)
        transcription = re.sub(r"</u>", "", transcription)
        transcription = re.sub(r" {2,}", " ", transcription)
        transcription = transcription.rstrip("\n")
        transcription = transcription.strip()

        # utterance including trascription tags
        self.transcription = transcription

        # tokens-only utterance without transcription tags
        sentence = re.sub(r"</?[^<]*>", "", transcription)
        sentence = re.sub(r" {2,}", " ", sentence)
        sentence = sentence.rstrip("\n")
        sentence = sentence.strip()
        if speaker_id2name:
            if self.speaker_id in speaker_id2name:
                name = speaker_id2name[self.speaker_id]
            else:
                # ['UNKFEMALE', 'UNKMALE', 'UNKMULTI']
                rand_idx = random.randint(0, len(speaker_id2name) - 1)
                name = list(speaker_id2name.values())[rand_idx]
            sentence = '{}: {}'.format(name, sentence)

        self.sentence = sentence


class Speaker(object):
    def __init__(self, speaker_element):
        self.id = speaker_element.get("id")
        self.age = "unk"
        if speaker_element.find("exactage").text:
            self.age = speaker_element.find("exactage").text
        self.age_range = speaker_element.find("agerange").text
        self.gender = speaker_element.find("gender").text
        self.nationality = speaker_element.find("nat").text
        self.birthplace = speaker_element.find("birthplace").text
        self.birthcountry = speaker_element.find("birthcountry").text
        self.l1 = speaker_element.find("l1").text
        self.education = speaker_element.find("edqual").text
        self.social_grade = speaker_element.find("socgrade").text
        self.social_status = speaker_element.find("nssec").text
        self.core_speaker = speaker_element.find("in_core").text == "y"
        self.dialects = [
            speaker_element.find("dialect_l1").text,
            speaker_element.find("dialect_l2").text,
            speaker_element.find("dialect_l3").text,
            speaker_element.find("dialect_l4").text
        ]

    def __eq__(self, other):
        if self.id != other.id:
            return False
        if self.age != other.age:
            return False
        if self.age_range != other.age_range:
            return False
        if self.gender != other.gender:
            return False
        if self.nationality != other.nationality:
            return False
        if self.birthplace != other.birthplace:
            return False
        if self.birthcountry != other.birthcountry:
            return False
        if self.education != other.education:
            return False
        if self.social_grade != other.social_grade:
            return False
        if self.social_status != other.social_status:
            return False
        if self.core_speaker != other.core_speaker:
            return False
        if self.dialects != other.dialects:
            return False
        return True


class Conversation(object):

    def __init__(self, untagged_root, tagged_root, speakers, speakers2conv, add_speaker_id):
        assert untagged_root.get("id") == tagged_root.get("id")
        self.id = untagged_root.get("id")

        header = untagged_root.find("header")

        self.rec_length = header.find("rec_length").text
        self.rec_date = header.find("rec_date").text
        self.rec_year = header.find("rec_year").text
        self.rec_period = header.find("rec_period").text
        self.rec_location = header.find("rec_loc").text

        self.relationships = []
        if header.find("relationships").text:
            self.relationships = header.find("relationships").text.lower().split(", ")

        self.topics = []
        if header.find("topics").text:
            self.topics = header.find("topics").text.lower().split(", ")

        self.activity = []
        if header.find("activity").text:
            self.activity = header.find("activity").text.lower()

        self.type = []
        if header.find("conv_type").text:
            self.type = header.find("conv_type").text.lower().split(", ")

        self.revised_conventions = header.find("conventions").text == "Revised"
        self.in_sample = header.find("in_sample").text == "y"
        self.transcriber = header.find("transcriber").text

        self.n_speakers = int(header.find("n_speakers").text)
        self.speaker_ids = header.find("list_speakers").text.split(" ")
        if add_speaker_id:
            self.speaker_id2name = {s_id: SPEAKER_NAMES[i] for i, s_id in enumerate(self.speaker_ids)}
        else:
            self.speaker_id2name = None

        self.speakers_ages_exact = {}
        self.speakers_age_ranges = {}

        speaker_ids_tmp = []
        for sp_elem in header.find("speakerInfo").findall("speaker"):
            speaker = Speaker(sp_elem)

            assert speaker.id in self.speaker_ids
            speaker_ids_tmp.append(speaker.id)

            if speaker.id in speakers:
                assert speaker == speakers[speaker.id]
            else:
                speakers[speaker.id] = speaker

            if speaker.id in speakers2conv:
                speakers2conv[speaker.id].append(self.id)
            else:
                speakers2conv[speaker.id] = [self.id]

            self.speakers_age_ranges[speaker.id] = speaker.age_range
            if speaker.age:
                self.speakers_ages_exact[speaker.id] = speaker.age

        assert len(speaker_ids_tmp) == self.n_speakers

        tagged_utterances = tagged_root.findall("u")
        untagged_utterances = untagged_root.find("body").findall("u")
        assert len(tagged_utterances) == len(untagged_utterances)

        self.utterances = []
        for tagged_u_elem, untagged_u_elem in zip(tagged_utterances, untagged_utterances):
            u = Utterance(tagged_u_elem, untagged_u_elem, self.speaker_id2name)
            self.utterances.append(u)

    @property
    def n_utterances(self):
        return len(self.utterances)

    @property
    def n_tokens(self):
        return sum([len(utterance.tokens) for utterance in self.utterances])

class Corpus(object):
    def __init__(
            self,
            untagged_path="../../data/bnc2014spoken-xml/spoken/untagged",
            tagged_path="../../data/bnc2014spoken-xml/spoken/tagged",
            n=0,
            add_speaker_id=False
    ):
        """
        A container for the 2014 version of the BNC Corpus of spoken conversations.
        :param untagged_path: the directory path of the untagged XML files
        :param tagged_path: the directory path of the tagged XML files
        :param n: the maximum number of files to load (default 0: load all files)
        :param add_speaker_id: whether to prepend utterances with speaker identifiers (e.g. 'A: yeah')
        """
        self.conversations = {}
        self.speakers = {}
        self.speakers2conv = {}

        # collect all untagged XML filenames
        untagged_ids = []
        for root, dirs, files in os.walk(untagged_path):
            for name in files:
                untagged_ids.append(name)

        # collect all tagged XML filenames
        tagged_ids = []
        for root, dirs, files in os.walk(tagged_path):
            for name in files:
                tagged_ids.append(name)

        if not (untagged_ids and tagged_ids):
            raise ValueError('No BNC files found. Check data path.')

        # same number of tagged and untagged?
        assert len(untagged_ids) == len(tagged_ids)
        print("Number of XML files:", len(untagged_ids))

        # does every tagged file have its untagged counterpart?
        tagged_ids.sort()
        untagged_ids.sort()
        n_conv = 0
        for u, t in zip(untagged_ids, tagged_ids):
            u_id = u[:-len(".xml")]
            t_id = t[:-len("-tgd.xml")]
            assert (t_id == u_id)
            n_conv += 1
            if n_conv >= n:
                break

        for u, t in tqdm(zip(untagged_ids, tagged_ids), total=n_conv):
            u_tree = ET.parse(os.path.join(untagged_path, u))
            t_tree = ET.parse(os.path.join(tagged_path, t))

            conv = Conversation(u_tree.getroot(), t_tree.getroot(), self.speakers, self.speakers2conv, add_speaker_id)

            self.conversations[conv.id] = conv

            if n and len(self.conversations) >= n:
                break

    def __len__(self):
        return len(self.conversations)
