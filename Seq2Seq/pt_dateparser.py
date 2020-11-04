import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

import matplotlib.pyplot as plt
from datetime import date
import os
import random

from DateClassifier import DateClassifier
from DateParser import DateParser
from torchsummary import summary
from trainer import Trainer
from nmt_utils import load_dataset, preprocess_data, string_to_int
from my_nmt_utils import encode_strings, decode_strings
from utils import plot_grad_flow

torch.manual_seed(1)
# np.random.seed(1)
# random.seed(1)

# For debugging.
torch.set_printoptions(profile="full")
# torch.set_printoptions(edgeitems=3)

# Start
batches = 10
batch_size = 1000
in_seq_len, out_seq_len = 30, 10

dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(batches)
model = DateParser(in_seq_len, out_seq_len, len(human_vocab), len(machine_vocab))
# model = DateClassifier(in_seq_len, len(human_vocab), 12)

# summary(model, (1, in_seq_len, len(human_vocab)), batch_size=len(dataset), device='cpu')
# for parameter in model.parameters():
#     print(parameter, parameter.shape)

# Create input and labelled outputs in a single batch
X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, in_seq_len, out_seq_len)

# Y = (Y[:, 5] - 1) * 10 + Y[:, 6] - 1
# Yoh = torch.zeros((Y.shape[0], 1))
# Yoh.scatter_(2, Y, 1)

torch.set_printoptions(precision=4, sci_mode=False)

# Define training hyperparameters
n_epochs = 1000
lr = 0.15
# lr_sched = [(1, .15), (1000, .05), (4000, .01)]  #, (8000, .005), (9500, .001)]
lr_sched = [(1, .005), (1000, .005), (4000, .005)]  #, (8000, .005), (9500, .001)]


def categorical_cross_entropy(y_pred, y_true):
    # Clamp the predicted value to 0 <= y_pred <= 1 to prevent failure of torch.log
    y_pred = torch.clamp(y_pred, 1e-9, 1 - 1e-9)

    # Convert truth labels to one-hot.
    y_oh = torch.zeros(y_pred.size())
    y_oh.scatter_(1, y_true.view(-1, 1), 1)
    return -(y_oh * torch.log(y_pred) + (1 - y_oh) * torch.log(1 - y_pred)).sum(dim=1).mean()


# Define Loss, Optimizer
# criterion = nn.CrossEntropyLoss()
criterion = categorical_cross_entropy
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(.9, .999), weight_decay=0)

trainer = Trainer(optimizer=optimizer, criterion=criterion)
trainer.fit(model, torch.from_numpy(Xoh), torch.from_numpy(Y), epochs=n_epochs, batch_size=batch_size,
            learning_rates=lr_sched)


# Predict
def predict(model, text):
    x_input = torch.LongTensor(string_to_int(text, in_seq_len, human_vocab))
    xoh = torch.zeros(1, x_input.shape[0], len(human_vocab))
    xoh.scatter_(2, x_input.view(1, -1, 1), 1)
    output = model(xoh)
    print(output.detach().numpy())

    plt.figure(3)
    plt.subplot(121)
    divisions = [str(i) for i in range(output.shape[2])]
    vals1 = output.detach().numpy()[0,0,:]
    plt.bar(divisions, vals1, width=.6)
    soft = F.softmax(output, 2, _stacklevel=5)
    plt.subplot(122)
    vals2 = soft.detach().numpy()[0,0,:]
    plt.bar(divisions, vals2, width=.6)

    print(output)
    print(soft)

    # Taking the class with the highest probability score from the output
    # prob = output.data
    # argmax = torch.max(prob, dim=2)[1]

    answers = {}
    # for i in range(argmax.shape[0]):
    #     answers[sentences[i]] = decode_strings(argmax[i,:], inv_machine_vocab)
    #
    return answers


text = ['4/28/90', 'thursday january 26 1995']
print(predict(model, text[0]))
print(predict(model, text[1]))

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

# with open('training_stats/' + date.today().strftime('YmdHis') + '.txt', 'w+') as f:
#     f.write(str(trainer.stats))

# Save weights
# with open('training_stats/weights-' + str(random.random()).replace('.', '') + '.txt', 'w+') as f:
#     torch.set_printoptions(profile="full")
#     for parameter in model.parameters():
#         f.write(str(parameter.data.cpu().numpy()))
