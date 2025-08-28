from typing import Tuple, Callable, Any, Optional, List

import torch


def benchmark(func: Callable[[torch.Tensor, ..., Any], Any], x, warmup: int = 3, runs: int = 10, **kwargs) -> Tuple[
    Optional[torch.Tensor], float, List[float]]:
    assert runs > 0, f"Runs needs to be greater than 0. {runs}"
    assert warmup >= 0, f"Warmpus needs to be greater than or equal 0, {warmup}"

    for _ in range(warmup):
        func(x, **kwargs)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    times = []
    result = None

    for _ in range(runs):
        torch.cuda.synchronize()
        start_event.record()
        result = func(x, **kwargs)
        end_event.record()
        torch.cuda.synchronize()

        times.append(start_event.elapsed_time(end_event))

    return result, sum(times) / runs, times
