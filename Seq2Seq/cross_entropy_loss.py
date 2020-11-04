import torch
from torch import nn
import torch.functional as F
import random

batch_size = 100
classes = 50
X = [random.randint(0, classes - 1) for _ in range(batch_size)]
# X = [random.randint(0, classes - 1)] * batch_size
X = torch.tensor(X)

Xoh = torch.zeros((batch_size, classes))
Xoh.scatter_(1, X.view(batch_size, -1), 1)

print(X)
print(X.shape)
print(Xoh)
print(Xoh.shape)

criterion = torch.nn.CrossEntropyLoss()

loss = criterion(Xoh, X)

print(loss)

input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
loss = criterion(input, target)

print(loss)
