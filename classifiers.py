import pdb

import torch.nn as nn
import torch
from pdb import set_trace

from transformers import BertForSequenceClassification, BertConfig, AutoModel

class TextClassificationLSTM(nn.Module):

    def __init__(self, batch_size, vocab_size, embedding_dim, hidden_dim,
                 num_classes, num_layers, bidirectional = False,
                 dropout = 0, device = 'cpu',
                 batch_first = True):
        """Text classification LSTM module.

        Args
        ----
        batch_size (int): No. datapoints simultaneously fed through model.
        voacb_size (int): No. unique tokens in dictionary.
        embedding_dim (int): No. dimensions to represent each token with.
        hidden_dim (int): Dimensionality of hidden layer.
        num_classes (int): size of output layer.
        num_layers (int): No. layers to be stacked in LSTM.
        bidirectional (bool): If True, introduces bidirectional LSTM, i.e.,
            sequence will be processed in both directions simultaneously.
            Default: False
        dropout (float): probability of dropping out connections in all but
            last layer. Default: 0
        device (str): Hardware device to store/run model on. Default: cpu
        batch_first (bool): Processing input assuming first dimension is
            batch dimension. Default: True

        ----------------
        For inspiration:
            https://www.analyticsvidhya.com/blog/2020/01/first-text-classification-in-pytorch/
            https://towardsdatascience.com/lstm-text-classification-using-pytorch-2c6c657f8fc0 (GOEIE)
            https://towardsdatascience.com/text-classification-with-pytorch-7111dae111a6
            https://towardsdatascience.com/multiclass-text-classification-using-lstm-in-pytorch-eac56baed8df

        """

        # Constructor #TODO: why is this necessary? And what about this:
        # super().__init__()
        super(TextClassificationLSTM, self).__init__()

        # initialize embedding layer
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embedding_dim,
                                      padding_idx=0)

        # TODO: Try Glove and Word2Vec (pretrained) word embeddings


        # useful for later in forward function
        self.batch_first = batch_first

        # initialize LSTM
        self.lstm = nn.LSTM(input_size = embedding_dim,
                            hidden_size = hidden_dim,
                            num_layers = num_layers,
                            bidirectional = bidirectional,
                            dropout = dropout,
                            batch_first = self.batch_first)

        # fully connected output layer (pre-activation)
        self.num_directions = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * self.num_directions, num_classes)

        self.dropout = nn.Dropout(p=dropout)

        # Activation function
        # TODO: figure out over which dimension to do this
        self.act = nn.ReLU()

    def forward(self, text, text_lengths, vidhya = False):
        """ Performs forward pass of input text through classification module.
        Args
        ----
        text (Tensor): Encoded input text.
            dim = [batch_size, sequence_length]
        text_lengths (Tensor or list(int)): pre-padding lengths of input
            sequences. dim = [batch_size]

        Returns
        -------
        output (Tensor): TODO: log probabilities.
            dim = [batch_size, num_classes]
        """
        # embed numericalized input text
        embedded = self.embedding(text)
        # embedded dims: [batch_size, sequence_length, embedding_dim]

        # pack padded sequence
        # here's why: https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch
        packed_embedded = nn.utils.rnn.pack_padded_sequence(input = embedded,
                                                            lengths = text_lengths.cpu(),
                                                            batch_first = self.batch_first,
                                                            enforce_sorted = False)


        # Do forward pass through lstm model
        # NB: output tuple (hidden, cell) is ignored
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        # hidden: [num_layers * num_directions, batch_size, hidden_dim]
        # NB: the layers can be separated using h_n.view(num_layers, num_directions, batch, hidden_size)
        # cell: [num_layers * num_directions, batch_size, hidden_dim]

        if self.num_directions == 1:
            # packed_output, (final_hidden, final_cell) = self.lstm(packed_embedded)
            # take last layer of final hidden state (i.e., shape [batch_size, hidden_dim])
            # pass through linear layer and activation
            output = self.fc(hidden[-1, :, :])
            # OR: THIS
            # unpacked_output = nn.utils.rnn.pad_packed_sequence(packed_output)
            # output = unpacked_output[0]
            # output = self.fc(output[-1, :, :])
            # out[-1, :, :] and hidden[-1, :, :] are supposed to be identical
            output = self.act(output)
        else:
            # Vidhya's method
            # concatenate the final forward and backward hidden states
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim = 1)
            # hidden dim: [batch_size, hidden_dim * num_directions]

            output = self.fc(hidden)
            output = self.dropout(output)
            output = self.act(output)

            # # inverse operation of pack_padded_sequence(). i.e., unpacks packed
            # # sequences, returns padded sequences and corresponding lengths
            # # Cheng's blog way
            # unpacked_output, _ = nn.utils.rnn.pad_packed_sequence(sequence = packed_output,
            #                                                       batch_first = True)
            # # forward direction
            # out_forward = unpacked_output[]

        return output


class TextClassificationLogit(nn.Module):
    """
    Based on this tutorial: https://medium.com/biaslyai/pytorch-linear-and-logistic-regression-models-5c5f0da2cb9
    """
    def __init__(self, num_classes):
        super(TextClassificationLogit, self).__init__()
        self.linear = nn.Linear(1, num_classes)

    def forward(self, x):
        out = self.linear(x)
        out = nn.functional.sigmoid(out)

        return out

class TextClassificationBERT(nn.Module):
    """
    For inspiration: https://towardsdatascience.com/bert-text-classification-using-pytorch-723dfb8b6b5b
    Also check out this one by the same author about LSTM text classification: https://towardsdatascience.com/lstm-text-classification-using-pytorch-2c6c657f8fc0
    """

    def __init__(self, num_classes):
        super(TextClassificationBERT, self).__init__()

        options_name = "bert-base-uncased"
        config = BertConfig.from_pretrained(options_name, output_attentions=True)
        config.num_labels = num_classes
        # config.max_position_embeddings = 1024
        self.encoder = BertForSequenceClassification.from_pretrained(options_name, config=config)

    def forward(self, text, label, bertviz=False, token_type_ids=None):

        if bertviz:
            # returns loss, text_features, attentions
            return self.encoder(text, labels=label, token_type_ids=token_type_ids)
        else:
            loss, text_fea = self.encoder(text, labels=label)[:2]
            return loss, text_fea

class FrozenBERT(nn.Module):
    """
    Based on this tutorial: https://www.analyticsvidhya.com/blog/2020/07/transfer-learning-for-nlp-fine-tuning-bert-for-text-classification/
    """

    def __init__(self, num_classes):
        super(FrozenBERT, self).__init__()

        # import BERT-base pretrained model
        self.bert = AutoModel.from_pretrained('bert-base-uncased')

        # dropout layer
        self.dropout = nn.Dropout(0.1)

        # relu activation function
        self.relu = nn.ReLU()

        # dense layer 1
        self.fc1 = nn.Linear(768, 512)

        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(512, num_classes)

        # softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    # define the forward pass
    def forward(self, sent_id, mask):
        # pass the inputs to the model
        _, cls_hs = self.bert(sent_id, attention_mask=mask)

        x = self.fc1(cls_hs)

        x = self.relu(x)

        x = self.dropout(x)

        # output layer
        x = self.fc2(x)

        # apply softmax activation
        x = self.softmax(x)

        return x

if __name__ == "__main__":

    set_trace()
