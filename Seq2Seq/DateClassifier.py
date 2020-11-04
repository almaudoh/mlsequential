import torch
from torch import nn


class DateClassifier(nn.Module):

    def __init__(self, in_seq_len, in_vocab_len, output_classes):
        super(DateClassifier, self).__init__()
        self.input_dims = 32
        self.Tx = in_seq_len
        self.lstm = nn.LSTM(in_vocab_len, self.input_dims, bidirectional=True, batch_first=True)
        self.attention_dense_1 = nn.Linear(2 * self.input_dims, 10)
        self.tanh = nn.Tanh()
        self.attention_dense_2 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        self.softmax1 = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)

        self.classify_dense = nn.Linear(2 * self.input_dims, 100)
        self.output_dense_3 = nn.Linear(100, 12)

    def forward(self, X):
        # Start with zero LSTM state for the first word.
        device = self.attention_dense_1.weight.device
        X = X.to(device=device)

        pre_out, _ = self.lstm(X)

        context = self.attention_layer(pre_out)
        outputs = self.relu(self.classify_dense(context))
        outputs = self.relu(self.output_dense_3(outputs))
        outputs = self.softmax2(outputs)
        # outputs = self.softmax2(self.output_dense_3(context))
        # outputs = self.softmax2(self.output_dense_3(context))
        # for t in range(self.Ty):
        #     context = self.attention_layer(pre_out, s)
        #     _, (s, c) = self.postlstm(context, (s, c))
        #     out = self.softmax(self.output_dense_3(s))
        #     outputs[:, t, :] = out[0, :, :]

        return outputs

    def attention_layer(self, pre_output):
        step1 = self.tanh(self.attention_dense_1(pre_output))
        step2 = self.relu(self.attention_dense_2(step1))
        alphas = self.softmax1(step2)
        context = torch.sum(alphas * pre_output, dim=1, keepdim=True)
        return context
