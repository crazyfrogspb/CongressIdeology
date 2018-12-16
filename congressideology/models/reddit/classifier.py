import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from congressideology.config import reddit_config


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size,
                 num_layers=1, dropout=0.0, bidirectional=True):
        super().__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(
            self.input_size, self.embedding_size, padding_idx=reddit_config.PAD_token)

        self.rnn = nn.GRU(self.embedding_size, self.hidden_size, self.num_layers,
                          batch_first=True, bidirectional=self.bidirectional,
                          dropout=self.dropout)

    def forward(self, x, lengths, hidden=None):
        embed = self.embedding(x)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            embed, lengths, batch_first=True)
        encoder_output, hidden = self.rnn(packed, hidden)
        encoder_output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            encoder_output, batch_first=True)

        if self.bidirectional:
            hidden = torch.cat(
                (hidden[0].unsqueeze(0), hidden[1].unsqueeze(0)), 2)
        else:
            hidden = hidden[-1]

        return encoder_output, hidden


class Attention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim):
        super().__init__()
        self.scale = 1. / math.sqrt(query_dim)

    def forward(self, query, keys, values):
        query = query.unsqueeze(1)
        keys = keys.transpose(1, 2)
        energy = torch.bmm(query, keys)
        energy = F.softmax(energy.mul_(self.scale), dim=2)

        linear_combination = torch.bmm(energy, values).squeeze(1)
        return energy, linear_combination


class Classifier(nn.Module):
    def __init__(self, encoder, attention, num_classes):
        super().__init__()
        self.encoder = encoder
        self.attention = attention
        self.decoder = nn.Linear(self.encoder.hidden_size, num_classes)

    def forward(self, x):
        input_seq = x['input']
        input_length = x['input_length']
        encoder_output, hidden = self.encoder(input_seq, input_length)

        energy, linear_combination = self.attention(
            hidden, encoder_output, encoder_output)
        logits = self.decoder(linear_combination)
        return logits, energy


def initialize_model(input_size, embedding_size, hidden_size, num_layers,
                     dropout, bidirectional, num_classes):
    encoder = Encoder(input_size, embedding_size, hidden_size,
                      num_layers, dropout, bidirectional)
    attention_size = 2 * hidden_size if bidirectional else hidden_size
    attention = Attention(attention_size, attention_size, attention_size)

    return Classifier(encoder, attention, num_classes).to(reddit_config.device)
