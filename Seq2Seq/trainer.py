import torch
import numpy as np
from utils import plot_grad_flow


class Trainer(object):

    def __init__(self, epochs=10, optimizer=None, criterion=None):
        self.epochs = epochs
        self.optimizer = optimizer
        self.criterion = criterion
        self.stats = {
            'epoch': [],
            'loss': []
        }

    def fit(self, model, X, Y, learning_rates=None, batch_size=256):
        assert self.optimizer is not None
        assert self.criterion is not None
        # Check device
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("Training on GPU")
            report_interval = 100
        else:
            device = torch.device('cpu')
            print("Training on CPU")
            report_interval = 10

        model.to(device)
        X = X.to(device)
        Y = Y.to(device)

        # Training Run
        # batchsize = X.shape[0]
        # input_seq_len = X.shape[1]
        next_lr = self.optimizer.defaults['lr']
        next_lr_epoch = self.epochs + 1
        print("LR: {:.4f}".format(next_lr))

        if learning_rates and len(learning_rates):
            next_lr_epoch, next_lr = learning_rates.pop()

        for epoch in range(1, self.epochs + 1):

            # Shuffle the input before taking batches
            shuffled = torch.randperm(X.shape[0])

            epoch_loss = []

            for i in range(0, X.shape[0], batch_size):
                # Clears existing gradients from previous epoch
                self.optimizer.zero_grad()
                # model.zero_grad()  # Clears existing gradients from previous epoch

                indices = shuffled[i:i + batch_size]
                batch_x, batch_y = X[indices], Y[indices]

                output = model(batch_x)

                # Have to convert to 2D Tensor since pytorch doesn't handle 3D properly.
                loss = self.criterion(output.view(-1, output.shape[2]), batch_y.view(-1))
                # loss = self.criterion(output, Y)

                # Does backpropagation and calculates gradients
                loss.backward()

                plot_grad_flow(model.named_parameters())

                self.optimizer.step()  # Updates the weights accordingly

                epoch_loss.append(loss.item())

            # Update training statistics.
            self.stats['loss'].append(np.average(epoch_loss))
            self.stats['epoch'].append(epoch)

            if epoch % report_interval == 0:
                print('Epoch: {}/{}.............'.format(epoch, self.epochs), end=' ')
                print("Loss: {:.4f}".format(np.average(epoch_loss)))

            # Variable LR adjustments.
            if next_lr_epoch == epoch:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = next_lr
                print("LR: {:.4f}".format(next_lr))

                if len(learning_rates):
                    next_lr_epoch, next_lr = learning_rates.pop()
