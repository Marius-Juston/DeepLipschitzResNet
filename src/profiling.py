import torch
from torch.autograd.profiler import record_function

from models.cnn_model import DeepLipschitzConvResNet

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

from models.linear_model import DeepLipschitzLinearResNet, DeepLipschitzSequential

import os
import torch
from torch.profiler import (
    profile,
    schedule,
    ProfilerActivity,
    tensorboard_trace_handler
)


def last_folder(base_folder):
    return max(map(int, os.listdir(base_folder)), default=0)


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)
    return d


# @torch.no_grad()
def profile_model(input_dict, *,
                  wait: int = 1,
                  warmup: int = 1,
                  active: int = 3,
                  repeat: int = 1,
                  tb_log_root: str = "./tb_trace",
                  use_cudagraphs: bool = False):
    """
    Profile a model using torch.profiler with a proper schedule and write TensorBoard traces.

    Parameters
    ----------
    input_dict : dict
        Must contain keys:
            'model' : the nn.Module instance to profile (on device already or will be moved).
            'data_size' : input shape tuple (e.g., (B, C, H, W) or (B, D) for linear).
            optional 'compile' : bool — whether to torch.compile(model)
    wait, warmup, active, repeat : ints
        Scheduler parameters (see torch.profiler.schedule). Total iterations = (wait+warmup+active)*repeat.
    tb_log_root : str
        Root folder for TensorBoard logs; per-model subfolder created.
    use_cudagraphs : bool
        If True, this run will include cudagraph marking (only enable if you want to measure cudagraph behavior).
    """
    model = input_dict['model']
    data_size = input_dict['data_size']
    do_compile = bool(input_dict.get('compile', False))

    # model.eval()

    # Optionally compile BEFORE warmup so compilation happens once and results are stable.
    if do_compile:
        model = torch.compile(model, mode='max-autotune')

    # Ensure model is on correct device
    device = next(model.parameters()).device if any(
        p.requires_grad or p.numel() for p in model.parameters()) else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print("Profiling model:", model.__class__.__name__, " device:", device, " compile:", do_compile)

    # Create input tensor (deterministic dtype/device)
    x = torch.randn(data_size, device=device)

    # profiler activities
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    # Warmup runs (to stabilize cuDNN, compilation, caching)
    print(f"Warmup: running {max(8, warmup * 2)} iterations (no profiling) ...")
    for _ in range(max(8, warmup * 2)):
        if use_cudagraphs:
            torch.compiler.cudagraph_mark_step_begin()
        model(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Scheduler and tensorboard handler
    sched = schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)
    # create per-model log dir
    model_name = model._orig_mod.__class__.__name__ if hasattr(model, "_orig_mod") else model.__class__.__name__

    base_folder = os.path.join(tb_log_root, model_name)

    ensure_dir(base_folder)

    folder_index = last_folder(base_folder)

    base_folder = os.path.join(base_folder, str(folder_index + 1))

    log_dir = ensure_dir(base_folder)
    trace_handler = tensorboard_trace_handler(log_dir)

    print("Profiling trace will be written to:", log_dir)
    total_iters = (wait + warmup + active) * repeat
    print(
        f"Profiler schedule: wait={wait}, warmup={warmup}, active={active}, repeat={repeat} -> total iters={total_iters}")

    with profile(
            activities=activities,
            schedule=sched,
            on_trace_ready=trace_handler,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,  # include flop counting if available in this build
    ) as prof:
        for itr in range(total_iters):
            if use_cudagraphs:
                # mark step for cudagraph capture behaviour if experimenting with cudagraphs
                torch.compiler.cudagraph_mark_step_begin()

            with record_function("model_inference"):
                y = model(x)

            with record_function("model_backward"):
                y.sum().backward()

            model.zero_grad(set_to_none=True)

            # step the profiler to advance the schedule
            prof.step()

    # Print the summary table (top kernels by CUDA time if available)
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=25))

    # Chrome trace already saved by tensorboard handler, but save json for convenience as well
    try:
        prof.export_chrome_trace(os.path.join(log_dir, f"trace_{model_name}.json"))
    except Exception:
        # newer versions sometimes require different APIs; ignore if export fails
        pass

    print("TensorBoard trace saved. Inspect with:\n\ttensorboard --logdir={}".format(tb_log_root))


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_default_device(device)

    batch_size = 64

    in_features = 16
    output_features = 16

    linear_inner_weights = (256, 256, 256, 256, 256)
    conv_weights = (32, 32, 32, 32)

    image_size = (64, 64)

    inputs = [
        {
            'model': DeepLipschitzConvResNet(in_features, output_features,
                                             conv_weights,
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
        profile_model(input, active=10)
