# This code was taken from the tutorial posted on Canvas
import torch
from collections import defaultdict, Counter
import spacy # for tokenizer
# TODO: use huggingface tokenizers and experiment with BPE, WordPiece, and SentencePiece encodings: https://huggingface.co/transformers/tokenizer_summary.html

# spacy english tokenizer
spacy_eng = spacy.load("en_core_web_sm")

class Vocabulary:
    '''
    Manual vocabulary using spacy tokenizer.
    '''

    def __init__(self, freq_threshold):

        # integer to string mappings (and vice versa) with tokens for padding,
        # start of sentence, end of sentence, and unknown
        self.itos = {0: "<PAD>", 1: "<BOS>", 2: "<EOS>", 3: "<UNK>"} # int to str dict
        self.stoi = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3} # str to int dict

        # frequency threshold for token to be added to vocabulary
        self.freq_threshold = freq_threshold

    def __len__(self):
        ''' Returns length (aka size) of vocabulary.'''
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}

        index = 4 # start integer index at 4 because 0 through 3 are special tokens

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                # add dict entry for word if not yet in dict, else increment
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

                # add word to vocab if it passes frequency threshold
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = index
                    self.itos[index] = word
                    index += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]


class WordTokenizer:
    """
    Simple word tokenizer with same interface as Huggingface tokenizer.
    """

    pad_token = '[PAD]'
    bos_token = '[BOS]'
    eos_token = '[EOS]'
    unk_token = '[UNK]'
    special_tokens = [pad_token, bos_token, eos_token, unk_token]
    remove_in_decode = {pad_token, bos_token, eos_token} # i.e., keep unkown (unk) token when decoding

    def __init__(self, data, max_vocab_size=100000):
        if max_vocab_size < len(self.special_tokens):
            raise ValueError("Minimum vocab size is {}.".format(self.special_tokens))
        self.max_vocab_size = max_vocab_size
        self.w2i, self.i2w = self.train_on_data(data, max_vocab_size)

        self.pad_token_id = self.w2i[self.pad_token]
        self.bos_token_id = self.w2i[self.bos_token]
        self.eos_token_id = self.w2i[self.eos_token]
        self.unk_token_id = self.w2i[self.unk_token]

    @property
    def vocab_size(self):
        return len(self.w2i)

    def encode(self, x, add_special_tokens=True, truncation=False, max_length=512):
        """
        Turn a sentence into a list of tokens. if add_special_tokens is True,
        add a start and stop token.

        Args:
            x (str): sentence to tokenize.
            add_special_tokens (bool, optional): if True, add a bos and eos token.
                Defaults to True.

        Returns:
            list: list of integers.
        """
        encoded = [self.w2i.get(w, self.unk_token_id) for w in x.split()]

        if add_special_tokens:
            if truncation:
               if len(encoded) > (max_length - 2):
                   encoded = encoded[:(max_length - 2)] # take into account the two special tokens BOS and EOS

            encoded = [self.bos_token_id] + encoded + [self.eos_token_id]
        else:
            if truncation:
                if len(encoded) > max_length:
                    encoded = encoded[:max_length]
        return encoded

    def decode(self, x, skip_special_tokens=True):
        """
        Turn a list or torch.Tensor back into a sentence.
        If skip_special_tokens is True, all tokens in self.remove_in_decode are removed.

        Args:
            x (Iterable): Iterable or torch.Tensor of tokens.
            skip_special_tokens (bool, optional): Remove special tokens (leave [UNK]).
                Defaults to True.

        Returns:
            str: decoded sentence.
        """
        if isinstance(x, torch.Tensor):
            x = [x.detach().cpu().numpy().tolist()]

        decoded = [self.i2w[i] for i in x]
        if skip_special_tokens:
            decoded = [t for t in decoded if t not in self.remove_in_decode]
        return ' '.join(decoded)

    def train_on_data(self, data, max_vocab_size=None):
        """
        Train this tokenizer on a list of sentences.
        Method, split sentences, aggragate word counts, make a word to index (w2i)
        and index to word (i2w) dictionary from the max_vocab_size most common words.

        Args:
            data (Iterable): Iterable of strings, where each string is a sentence.
            max_vocab_size (int, optional): If defined, only keep the max_vocab_size most common words in the vocabulary.
                Defaults to None.

        Returns:
            tuple: w2i, i2w dicts
        """
        word_counts = Counter()
        for sentence in data:
            word_counts.update(sentence.split())

        # Make vocabularies, sorted alphabetically
        w2i = defaultdict(lambda: len(w2i))
        i2w = dict()

        # Default tokens
        for t in self.special_tokens:
            i2w[w2i[t]] = t

        # Give each word a token
        if max_vocab_size:
            words = [w[0] for w in word_counts.most_common(max_vocab_size - len(self.special_tokens))]
        else:
            words = list(word_counts.keys())
        for word in sorted(words):
            i2w[w2i[word]] = word

        return dict(w2i), i2w
