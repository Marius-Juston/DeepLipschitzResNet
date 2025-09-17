import torch
from torch.profiler import profile, ProfilerActivity

from models.model import DeepLipschitzSequential

def profile_model():
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Running on device", device)

    input_features = 20
    output_features = 20

    model = DeepLipschitzSequential(input_features, output_features, (256, 256, 256, 256, 256, 256, 256, 256, 256),
                                    device=device)
    # model = SLLSequential(input_features, output_features, (256, 256, 256, 256, 256, 256, 256, 256, 256), device=device)

    model = torch.compile(model, mode='max-autotune')
    # model = torch.compile(model, mode='reduce-overhead')
    # model = torch.compile(model)

    print(model)

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


if __name__ == '__main__':
    profile_model()
