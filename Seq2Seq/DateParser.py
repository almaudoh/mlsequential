import torch
from torch import nn


class DateParser(nn.Module):

    def __init__(self, in_seq_len, out_seq_len, in_vocab_len, out_vocab_len):
        super(DateParser, self).__init__()
        self.hidden_dims = 64
        self.input_dims = 32
        self.Tx = in_seq_len
        self.Ty = out_seq_len
        self.prelstm = nn.LSTM(in_vocab_len, self.input_dims, bidirectional=True, batch_first=True)
        self.postlstm = nn.LSTM(self.hidden_dims, self.hidden_dims, batch_first=True)
        self.attention_dense_1 = nn.Linear(4 * self.input_dims, 10)
        self.tanh = nn.Tanh()
        self.attention_dense_2 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.output_dense_3 = nn.Linear(self.hidden_dims, out_vocab_len)

    def forward(self, X):
        # Start with zero LSTM state for the first word.
        device = self.attention_dense_1.weight.device
        bs = X.shape[0]
        s = torch.zeros(1, bs, self.hidden_dims).to(device=device)
        c = torch.zeros(1, bs, self.hidden_dims).to(device=device)
        X = X.to(device=device)

        outputs = torch.zeros((X.shape[0], self.Ty, self.output_dense_3.out_features)).to(device=device)

        pre_out, _ = self.prelstm(X)

        # context = self.attention_layer(pre_out, s)
        # outputs = self.tanh(self.classify_dense(pre_out))
        # outputs = self.relu(self.output_dense_3(outputs))
        # outputs = self.softmax2(self.output_dense_3(context))
        # outputs = self.softmax2(self.output_dense_3(context))
        for t in range(self.Ty):
            context = self.attention_layer(pre_out, s)
            _, (s, c) = self.postlstm(context, (s, c))
            out = self.softmax(self.output_dense_3(s))
            outputs[:, t, :] = out[0, :, :]

        return outputs

    def attention_layer(self, pre_output, state):
        # Expand state from (1, batch_size, hidden_dims) to (seq_len, batch_size, hidden_dims)
        # This assumes n_layer = 1 for state which comes from postlstm.
        state = torch.cat([state for _ in range(self.Tx)], dim=0)

        # Need to transpose since state shape is (seq_len, batch_size, hidden_dims)
        # while pre_output is (batch_size, seq_len, hidden_dims)
        combined = torch.cat((pre_output, state.transpose(0, 1)), dim=2)
        step1 = self.tanh(self.attention_dense_1(combined))
        step2 = self.relu(self.attention_dense_2(step1))
        # step2 = self.attention_dense_2(step1)
        alphas = self.softmax(step2)
        context = torch.sum(alphas * pre_output, dim=1, keepdim=True)
        return context
