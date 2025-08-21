import os
from datetime import datetime

import torch
from matplotlib import pyplot as plt
from torch.nn import MSELoss
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.model import DeepLipschitzResNet

torch.set_float32_matmul_precision('high')


def train_one_epoch(training_loader, optimizer, model, loss_fn, epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        inputs = inputs.reshape((-1, 1))
        labels = labels.reshape((-1, 1))

        # Zero your gradients for every batch!
        optimizer.zero_grad(set_to_none=True)

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

        print("Current loss:", loss.item())
        if i % 10 == 9:
            last_loss = running_loss / 10  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


def train(x, y, model):
    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_folder = '../runs/sin_trainer_{}'.format(timestamp)

    writer = SummaryWriter(save_folder)

    writer.add_graph(model, torch.randn((10, model.in_features), device=model.device))

    epoch_number = 0

    EPOCHS = 100

    best_vloss = 1_000_000.

    loss_fn = MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    dataset = TensorDataset(x, y)

    training_loader = DataLoader(dataset, batch_size=1000, shuffle=True)

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(training_loader, optimizer, model, loss_fn, epoch_number, writer)

        print('LOSS train {}'.format(avg_loss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training',
                           {'Training': avg_loss},
                           epoch_number + 1)
        writer.flush()

        plt.plot(x.cpu(), y.cpu(), label='gt')
        with torch.no_grad():
            y_pred = model(x.reshape(-1, model.in_features))

            plt.plot(x.cpu(), y_pred.cpu(), label='truth')
        plt.savefig(os.path.join(save_folder, f'truth{epoch}.png'))
        plt.close()

        epoch_number += 1


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Running on device", device)

    input_features = 1
    output_features = 1

    model = DeepLipschitzResNet(input_features, output_features, (32, 256, 1024), device=device)

    print(model)

    torch.manual_seed(0)

    batch = 64

    x = torch.linspace(-10, 10, 100000, device=device)

    y = torch.sin(x + 1) * 0.5 - 0.5 * x + 2

    train(x, y, model)


if __name__ == '__main__':
    main()
