import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Attention(nn.Module):
    def __init__(self, feature_dim, bias=True, **kwargs):
        """

        :param feature_dim: enc_size
        :param bias:
        :param kwargs:
        """
        super(Attention, self).__init__(**kwargs)

        self.supports_masking = True
        self.bias = bias
        self.feature_dim = feature_dim
        # self.sequence_len = sequence_len
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)    # [enc_size, 1]
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        if bias:
            self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x, mask=None):
        """

        :param x:  [batch, len, enc_size]
        :param mask:
        :return:
        """
        batch_size, sequence_len,  feature_dim = x.size()
        assert (feature_dim == self.feature_dim)

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, sequence_len)                     # [batch, sequence_len]
        # if self.bias:
        #     eij = eij + self.b

        eij = torch.tanh(eij)

        # softmax
        # a = torch.exp(eij)
        # if mask is not None:
        #     a = a * mask
        # a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)
        weights = torch.softmax(eij, dim=-1)
        if mask is not None:
            weights = weights * mask

        weighted_input = x * torch.unsqueeze(weights, -1)  # [batch, seq_len, enc_size]
        return torch.sum(weighted_input, 1)                # [batch, enc_size]


class rnn_attn(nn.Module):
    def __init__(self, embedding_matrix, hidden_size = 60, num_penultmate=16):
        super(rnn_attn, self).__init__()
        max_features, embed_size = embedding_matrix.shape
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.embedding_dropout = nn.Dropout2d(0.1)
        self.lstm = nn.GRU(embed_size, hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)

        self.lstm_attention = Attention(hidden_size * 2)   # [batch, seq_len, 2*hidden_size]
        self.gru_attention = Attention(hidden_size * 2)

        self.linear = nn.Linear(hidden_size * 8, num_penultmate)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(num_penultmate, 1)

    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = torch.squeeze(self.embedding_dropout(torch.unsqueeze(h_embedding, 0)))

        h_lstm, _ = self.lstm(h_embedding)
        h_gru, _ = self.gru(h_lstm)

        h_lstm_atten = self.lstm_attention(h_lstm)
        h_gru_atten = self.gru_attention(h_gru)

        avg_pool = torch.mean(h_gru, 1)
        max_pool, _ = torch.max(h_gru, 1)

        conc = torch.cat((h_lstm_atten, h_gru_atten, avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)

        return out.squeeze()

if __name__ == "__main__":
    x = torch.Tensor([[1,2,3,4,5], [2,4,6,8,10]]).long()
    embed_mat = np.random.randn(100, 100)
    model = rnn_attn(embed_mat, num_penultmate=16)
    out = model(x)
    print(out.shape)



