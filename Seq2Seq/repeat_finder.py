import torch
from torch import nn
from torch import optim
import random

import matplotlib.pyplot as plt

from trainer import Trainer

random.seed(1)


class TestLstm(nn.Module):

    def __init__(self):
        super(TestLstm, self).__init__()
        self.lstm = nn.LSTM(1, 32, batch_first=True)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 2)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, input):
        _, (state1, state2) = self.lstm(input)
        return self.softmax(self.fc2(self.fc1(state1)))
        # return state1


seq_len = 10
samples = int(random.random()*1000)

examples = [[int(random.random()*2) for i in range(seq_len)] for j in range(samples)]
examples = torch.FloatTensor(examples)
# input_val = torch.zeros((len(examples), len(examples[0]), 2))
examples.unsqueeze_(2)


def count_ones(input):
    # Label any one that has more than 3 1's in sequence.
    classes = torch.zeros((input.shape[0]), dtype=torch.long)
    for idx, item in enumerate(input):
        one = True
        ones = 0
        for number in item:
            one = number.item() == 1
            ones = 0 if number.item() != 1 else ones + 1
            if one is True and ones >= 3:
                classes[idx] = 1
                break
    # classes.unsqueeze_(0)
    return classes


# print(examples)
# print(labels)
# input_val[:,:,0] = examples
labels = count_ones(examples)
print(examples.size())
# print(input_val.size())
print(labels.size())

model = TestLstm()

output = model(examples)
# print(output)
# print(state1)
# print(state1.size())
# print(state2.size())
print(output.size())

print(examples)
print(labels)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.5)
trainer = Trainer(optimizer=optimizer, criterion=criterion)
lrs = [(0, 0.5), (1000, 0.1)]
trainer.fit(model, examples, labels, epochs=2000, learning_rates=lrs)

# plt.show()

# Plot training statistics
plt.figure(2)
plt.plot(trainer.stats['epoch'], trainer.stats['loss'])
plt.show()

test_batch = 10
test = [[int(random.random()*2) for i in range(seq_len)] for j in range(test_batch)]
test = torch.FloatTensor(test).unsqueeze(2)

test_output = model(test)

torch.set_printoptions(precision=4, sci_mode=False)
# print(test)
print(count_ones(test))
print(test_output)
