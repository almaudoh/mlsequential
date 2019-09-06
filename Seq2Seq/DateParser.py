import torch
from torch import nn


class DateParser(nn.Module):

    def __init__(self, in_seq_len, out_seq_len, n_a, n_s, in_vocab_len, out_vocab_len):
        super(DateParser, self).__init__()
        self.hidden_size = n_s
        self.Tx = in_seq_len
        self.Ty = out_seq_len
        self.prelstm = nn.LSTM(in_vocab_len, n_a, bidirectional=True, batch_first=True)
        self.postlstm = nn.LSTM(n_s, n_s, batch_first=True)
        self.attention_dense_1 = nn.Linear(2 * n_s, 10)
        self.attention_dense_1a = nn.Tanh()
        self.attention_dense_2 = nn.Linear(10, 1)
        self.attention_dense_2a = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.output_dense_3 = nn.Linear(2 * n_a, out_vocab_len)
        self.output_dense_3a = nn.Softmax(dim=1)

    def forward(self, X):
        # Start with zero LSTM state for the first word.
        bs = X.shape[0]
        s = torch.zeros(1, bs, self.hidden_size)
        c = torch.zeros(1, bs, self.hidden_size)

        # outputs = []
        outputs = torch.zeros((X.shape[0], self.Ty, self.output_dense_3.out_features))

        pre_out, h = self.prelstm(X)
        for t in range(self.Ty):
            context = self.attention_layer(pre_out, s)
            _, (s, c) = self.postlstm(context, (s, c))
            out = self.output_dense_3a(self.output_dense_3(s))
            outputs[:, t, :] = out[0, :, :]

        return outputs

    def attention_layer(self, pre_output, state):
        # Expand state from (1, batch_size, hidden_dims) to (seq_len, batch_size, hidden_dims)
        state = torch.cat([state for _ in range(self.Tx)], dim=0)
        # Need to transpose since state shape is (n_layers, batch_size, hidden_dims)
        # while input is (batch_size, seq_len, hidden_dims)
        # Note that n_layers and seq_len are always 1 so won't cause a problem.
        # However, if these change, then transpose may not be the best approach.
        combined = torch.cat((state.transpose(0, 1), pre_output), dim=2)
        step = self.attention_dense_1(combined)
        step = self.attention_dense_1a(step)
        step = self.attention_dense_2(step)
        step = self.attention_dense_2a(step)
        alphas = self.softmax(step)
        context = torch.sum(alphas * pre_output, dim=1, keepdim=True)
        return context
