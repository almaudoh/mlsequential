import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

import matplotlib.pyplot as plt
from datetime import date
import os
import random

from DateParser import DateParser
from torchsummary import summary
from trainer import Trainer
from nmt_utils import load_dataset, preprocess_data
from my_nmt_utils import encode_strings, decode_strings
from utils import plot_grad_flow

torch.manual_seed(1)
# np.random.seed(1)
# random.seed(1)

# For debugging.
torch.set_printoptions(profile="full")
# torch.set_printoptions(edgeitems=3)

# Start
m = 10000
in_seq_len, out_seq_len = 30, 10

dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)
model = DateParser(in_seq_len, out_seq_len, len(human_vocab), len(machine_vocab))

# summary(model, (1, in_seq_len, len(human_vocab)), batch_size=len(dataset), device='cpu')
# for parameter in model.parameters():
#     print(parameter, parameter.shape)

# Create input and labelled outputs in a single batch
X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, in_seq_len, out_seq_len)

# Xbatch, Ybatch = [], []
# for x, y in dataset:
#     Xbatch.append(x)
#     Ybatch.append(y)
#
# Xbatch = encode_strings(Xbatch, human_vocab, in_seq_len, use_onehot=True)
# Ybatch = encode_strings(Ybatch, machine_vocab, out_seq_len, use_onehot=False)

torch.set_printoptions(precision=4, sci_mode=False)

# Define training hyperparameters
n_epochs = 2000
lr = 0.15

# Define Loss, Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(.9, .999), weight_decay=0)

trainer = Trainer(optimizer=optimizer, criterion=criterion)
trainer.fit(model, torch.from_numpy(Xoh), torch.from_numpy(Y), epochs=n_epochs, batch_size=1000)


# Predict
def predict(model, sentences):
    x_pred = encode_strings(sentences, human_vocab, in_seq_len)
    output = model(x_pred)
    # print(output.detach().numpy())

    # Taking the class with the highest probability score from the output
    prob = output.data
    argmax = torch.max(prob, dim=2)[1]

    answers = {}
    for i in range(argmax.shape[0]):
        answers[sentences[i]] = decode_strings(argmax[i,:], inv_machine_vocab)

    return answers


strings = ['4/28/90', 'thursday january 26 1995']
print(predict(model, strings))

# Plot training statistics
plot_grad_flow(trainer.stats['gradient_flow'])
plt.figure(1)
plt.plot(trainer.stats['loss'])
# plt.plot(trainer.stats['epoch'], trainer.stats['loss'])
# plt.yscale('log')
plt.show()

if not os.path.isdir('training_stats'):
    os.mkdir('training_stats')
    print("Created new directory for training stats")

with open('training_stats/' + date.today().strftime('YmdHis') + '.txt', 'w+') as f:
    f.write(str(trainer.stats))

# Save weights
with open('training_stats/' + str(random.random()).replace('.', '') + '.txt', 'w+') as f:
    torch.set_printoptions(profile="full")
    for parameter in model.parameters():
        f.write(str(parameter.data.cpu().numpy()))
