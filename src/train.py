import os
from datetime import datetime

import torch
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from torch.nn import MSELoss
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.model import DeepLipschitzLinearResNet, DeepLipschitzSequential

DPI = 600

torch.set_float32_matmul_precision('high')


# torch.autograd.set_detect_anomaly(True)


def train_one_epoch(training_loader, optimizer, model, loss_fn, epoch_index, tb_writer, logging_frequency=100):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        inputs = inputs.reshape((-1, model.in_features))
        labels = labels.reshape((-1, model.out_features))

        # Zero your gradients for every batch!
        optimizer.zero_grad(set_to_none=True)

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

        # print("Current loss:", loss.item())
        if i % logging_frequency == logging_frequency - 1:
            last_loss = running_loss / logging_frequency  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


def train(x, y, model: DeepLipschitzLinearResNet, batch_size=64, lr=1e-3, termination_error=5e-3):
    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_folder = '../runs/sin_trainer_{}'.format(timestamp)

    writer = SummaryWriter(save_folder)

    # writer.add_graph(model, torch.randn((10, model.in_features), device=model.device))

    epoch_number = 0

    EPOCHS = 100

    best_vloss = 1_000_000.

    loss_fn = MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, fused=True)

    dataset = TensorDataset(x, y)

    training_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_loss = None

    # theoretical_lower = 135.021966261
    theoretical_lower = 0

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

        plt.plot(x.cpu(), y.cpu(), label='Ground Truth')
        with torch.no_grad():
            y_pred = model(x.reshape(-1, model.in_features))

            plt.plot(x.cpu(), y_pred.cpu(), label='Prediction')
        plt.legend()
        plt.tight_layout()

        plt.savefig(os.path.join(save_folder, f'truth{epoch}.png'), dpi=DPI)
        plt.close()

        assert avg_loss >= theoretical_lower, "Problem with the lipschitz network since cannot be lower than theoretical lowest"

        if avg_loss < termination_error:
            break

        if best_loss is None or best_loss > avg_loss:
            torch.save(model.state_dict(), os.path.join(save_folder, 'best.pt'))
            best_loss = avg_loss

        epoch_number += 1


def print_num_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def main():
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Running on device", device)

    input_features = 1
    output_features = input_features

    model = DeepLipschitzLinearResNet(input_features, output_features, (32, 32, 32, 32, 32), device=device)
    # model = DeepLipschitzSequential(input_features, output_features, (256, 256, 256, 256, 256, 256, 256, 256, 256), device=device)
    # model = SLLSequential(input_features, output_features, (256, 256, 256, 256, 256, 256, 256, 256, 256), device=device)
    # model = DeepLipschitzSequential(input_features, output_features, (32, 256, 1024, 1024), device=device)

    check_gradients = False

    if check_gradients:
        for name, p in model.named_parameters():
            def function(grad):
                if grad.isnan().any():
                    print("GRADIENT IS NAN IN {}".format(name))

            p.register_post_accumulate_grad_hook(function)

    # model = torch.compile(model)
    model = torch.compile(model, mode='max-autotune')

    print(model)
    print(print_num_parameters(model))

    batch = 64

    x = torch.linspace(-10, 10, 100000 * input_features, device=device).reshape((-1, input_features))

    variation = torch.arange(input_features, device=device)

    y = torch.sin(x + variation) * 0.5  # + x + 2 + variation
    # y = torch.sin(x + variation) * 0.5  + 0.25*  x + 2 + variation

    train(x, y, model, batch_size=batch, termination_error=1e-4)


def sine_training():
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Running on device", device)

    input_features = 1
    output_features = input_features

    model = DeepLipschitzLinearResNet(input_features, output_features, (32, 32, 32, 32, 32), device=device)

    check_gradients = False

    if check_gradients:
        for name, p in model.named_parameters():
            def function(grad):
                if grad.isnan().any():
                    print("GRADIENT IS NAN IN {}".format(name))

            p.register_post_accumulate_grad_hook(function)

    model = torch.compile(model, mode='max-autotune')

    print(model)
    print(print_num_parameters(model))

    batch = 64

    x = torch.linspace(-10, 10, 100000 * input_features, device=device).reshape((-1, input_features))

    variation = torch.arange(input_features, device=device)

    y = torch.sin(x + variation) * 0.5

    train(x, y, model, batch_size=batch, termination_error=1e-4)


if __name__ == '__main__':
    # sine_training()
    main()
