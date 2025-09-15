import os
from datetime import datetime

import torch
from matplotlib import pyplot as plt
from torch.nn import MSELoss
from torch.profiler import profile, ProfilerActivity
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.model import DeepLipschitzLinearResNet

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

        plt.plot(x.cpu(), y.cpu(), label='gt')
        with torch.no_grad():
            y_pred = model(x.reshape(-1, model.in_features))

            plt.plot(x.cpu(), y_pred.cpu(), label='truth')
        plt.savefig(os.path.join(save_folder, f'truth{epoch}.png'))
        plt.close()

        assert avg_loss >= theoretical_lower, "Problem with the lipschitz network since cannot be lower than theoretical lowest"

        if avg_loss < termination_error:
            break

        if best_loss is None or best_loss > avg_loss:
            torch.save(model.state_dict(), os.path.join(save_folder, 'best.pt'))
            best_loss = avg_loss

        epoch_number += 1


def profile_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Running on device", device)

    input_features = 20
    output_features = 20

    model = DeepLipschitzLinearResNet(input_features, output_features, (32, 256, 1024), device=device)

    model = torch.compile(model, mode='max-autotune')
    # model = torch.compile(model, mode='reduce-overhead')
    # model = torch.compile(model)

    print(model)

    torch.manual_seed(0)

    batch = 64

    x = torch.linspace(-10, 10, batch * input_features, device=device).reshape(batch, -1)

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities += [ProfilerActivity.CUDA]

    for i in range(10):
        model(x)

    with profile(activities=activities, record_shapes=True) as prof:
        model(x)

    print(
        prof.key_averages(group_by_input_shape=True).table(
            sort_by="cpu_time_total", row_limit=10
        )
    )
    prof.export_chrome_trace("trace.json")


def main():
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Running on device", device)

    input_features = 4
    output_features = input_features

    model = DeepLipschitzLinearResNet(input_features, output_features, (32, 256, 1024, 1024), device=device)

    check_gradients = False

    if check_gradients:
        for name, p in model.named_parameters():
            def function(grad):
                if grad.isnan().any():
                    print("GRADIENT IS NAN IN {}".format(name))

            p.register_post_accumulate_grad_hook(function)

    # model = torch.compile(model)
    # model = torch.compile(model, mode='max-autotune')

    print(model)

    batch = 64

    x = torch.linspace(-10, 10, 100000 * input_features, device=device).reshape((-1, input_features))

    variation = torch.arange(input_features, device=device)

    y = torch.sin(x + variation) * 0.5 + 3*x + 2 + variation

    train(x, y, model, batch_size=batch)


if __name__ == '__main__':
    # profile_model()
    main()
