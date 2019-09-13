import torch


class Trainer(object):

    def __init__(self, optimizer=None, criterion=None):
        self.optimizer = optimizer
        self.criterion = criterion
        self.stats = {
            'epoch': [],
            'loss': [],
            'gradient_flow': [],
        }

    def fit(self, model, X, Y, epochs=10, learning_rates=None, batch_size=500):
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
        next_lr_epoch = epochs + 1
        print("LR: {:.4f}".format(next_lr))

        if learning_rates and len(learning_rates):
            next_lr_epoch, next_lr = learning_rates.pop()

        # Update training statistics.
        self.stats['loss'] = []
        self.stats['epoch'] = []

        for epoch in range(1, epochs + 1):

            if len(learning_rates):
                next_lr_epoch, next_lr = learning_rates.pop()

            # Variable LR adjustments.
            if next_lr_epoch == epoch:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = next_lr
                print("LR: {:.4f}".format(next_lr))

            # Shuffle the input before taking batches
            shuffled = torch.randperm(X.shape[0])

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

                # Does back propagation and calculates gradients
                loss.backward()
                # Updates the weights accordingly
                self.optimizer.step()

                # Update training statistics.
                self.save_gradient_flow(model.named_parameters())
                self.stats['loss'].append(loss.item())
                self.stats['epoch'].append(epoch)

            if epoch % report_interval == 0:
                print('Epoch: {}/{}.............'.format(epoch, epochs), end=' ')
                print("Loss: {:.4f}".format(loss.item()))


    def save_gradient_flow(self, named_parameters):
        ave_grads = []
        layers = []
        for n, p in named_parameters:
            if p.requires_grad and "bias" not in n:
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())

        self.stats['gradient_flow'].append({
            'layers': layers,
            'grads': ave_grads,
        })
