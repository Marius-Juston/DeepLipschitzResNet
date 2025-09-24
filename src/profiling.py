import torch

from models.cnn_model import DeepLipschitzConvResNet

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

from torch.profiler import profile, ProfilerActivity

from models.linear_model import DeepLipschitzLinearResNet, DeepLipschitzSequential

@torch.no_grad()
def profile_model(input):
    model = input['model']

    model.eval()

    if 'compile' in input and input['compile']:
        model = torch.compile(model, mode='max-autotune')
        # model = torch.compile(model, mode='reduce-overhead')
        # model = torch.compile(model)

    print(model)

    x = torch.randn(input['data_size'], device=model.device)

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities += [ProfilerActivity.CUDA]

    # Warmup: run some iterations to stabilize caches, JIT, cuDNN
    for i in range(10):
        model(x)

    # one explicit sync to ensure warmup kernels finish
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    with profile(activities=activities, record_shapes=True) as prof:
        model(x)

    print(
        prof.key_averages(group_by_input_shape=True).table(
            sort_by="cpu_time_total", row_limit=10
        )
    )


    model_name = model._orig_mod if hasattr(model, '_orig_mod') else model

    prof.export_chrome_trace(f"trace_{model_name.__class__.__name__}.json")


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_default_device(device)

    batch_size = 64

    in_features = 16
    output_features = 16

    linear_inner_weights = (256, 256, 256, 256, 256)
    conv_weights = (256, 256, 256, 256, 256)

    image_size = (64, 64)

    inputs = [
        {
            'model': DeepLipschitzConvResNet(in_features, output_features,
                                             linear_inner_weights,
                                             padding=1,
                                             kernel_size=3,
                                             device=device),
            'data_size': (batch_size, in_features, *image_size),
            'compile': True,
        },
        {
            'model': DeepLipschitzLinearResNet(in_features, output_features, linear_inner_weights,
                                    device=device),
            'data_size': (batch_size, in_features),
        },
        {
            'model': DeepLipschitzSequential(in_features, output_features,
                                               linear_inner_weights,
                                               device=device),
            'data_size': (batch_size, in_features),
        },
    ]

    for input in inputs:
        profile_model(input)
