# LDLT Lipschitz Network

This repository is the implementation of the paper https://arxiv.org/abs/2502.21279 and should be cited as [[1]](#1).

# Files

## Training

To run the specific trainings found in the paper you can use [train.py](./src/train.py),

this has multiple routines that you can enable for the specific outputs that you desire.

## Profiling

To profile the networks you can use [profiling.py](./src/profiling.py), which compiles with `torch.compile` and then finds CUDA profiles the system for performance issues. This is extremely useful for finding graph break issues with `torch.compile`

To vizualize the output `trace.json` open Chrome and load the trace file inside [chrome://tracing](chrome://tracing) as demonstrated in [here](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html#using-tracing-functionality)

## Plots

To generate the LMI plots use [model_lmi.py](./test/model_lmi.py), where you can optionally put the specific model weights and structure that you desire as a model weight path that would get loaded. This will create a [figs](./test/figs/) folder with the output graphs

To generate the numerical instability graphs we have that [numerical_instability.py](./test/numerical_instability.py) script.

# Performance

For full performance in the CNN implementation of the lipschitz network, so that `torch.compile` works, due to a bug with the inductor compiler and complex datatypes and views `.mH`, please make sure to get either the nightly build or any build that fixes PyTorch issue [https://github.com/pytorch/pytorch/issues/163243](https://github.com/pytorch/pytorch/issues/163243), where the sample code [complex_cholesky_decomposition.py](./test/complex_cholesky_decomposition.py) should run and compile the graph.

## Tensorboard

```bash
ssh -L 6006:127.0.0.1:6006 <user>@<ip address>
conda activate DeepLipschitzResNet
tensorboard --logdir=.
```

## Reference

To cite please use,

<a id="1">[1]</a> 
```
@misc{juston2025llipschitzgershgorinresnetnetwork,
      title={L-Lipschitz Gershgorin ResNet Network}, 
      author={Marius F. R. Juston and William R. Norris and Dustin Nottage and Ahmet Soylemezoglu},
      year={2025},
      eprint={2502.21279},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.21279}, 
}
```