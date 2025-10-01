# LDLT Lipschitz Network

This repository is the implementation of the paper https://arxiv.org/abs/2502.21279 and should be cited as [[1]](#1).

# Experiments

Experiments were done with PyTorch 2.8,

train & evaluation
# training
```bash
# Training
torchrun --standalone --nproc_per_node=4 src/main.py --dataset cifar10 --model-name small --train_dir small_model 2>&1 | tee output.txt

# Evaluation
torchrun --standalone --nproc_per_node=1 src/main.py --dataset cifar10 --model-name small --train_dir small_model --mode certified 2>&1 | tee output.txt
```

or running
```bash
./train.sh                 # defaults to CIFAR10
./train.sh cifar10         # train using CIFAR10
./train.sh cifar100        # train using CIFAR100
./train.sh tiny-imagenet   # train using tiny-imagenet
```

```bash
 /home/mjuston2/anaconda3/envs/DeepLipschitzResNet/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=gpu main.py --dataset cifar10 --model-name small --train_dir small_model
```

## Reproducing CIFAR10/100 results from the paper
Download the checkpoints here

```bash
python3 reproduce_tables.py
```

# Files

## Training

To run the specific trainings found in the paper you can use [linear_train.py](./src/linear_train.py) or [cnn_train.py](./src/cnn_train.py)  

this has multiple routines that you can enable for the specific outputs that you desire.

## Profiling

To profile the networks you can use [profiling.py](./src/profiling.py), which compiles with `torch.compile` and then finds CUDA profiles the system for performance issues. This is extremely useful for finding graph break issues with `torch.compile`

To vizualize the output `trace.json` open Chrome and load the trace file inside [chrome://tracing](chrome://tracing) as demonstrated in [here](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html#using-tracing-functionality)

## Plots

To generate the LMI plots use [model_lmi.py](./test/model_lmi.py), where you can optionally put the specific model weights and structure that you desire as a model weight path that would get loaded. This will create a [figs](./test/figs/) folder with the output graphs

To generate the numerical instability graphs we have that [numerical_instability.py](./test/numerical_instability.py) script.

# Performance

For full performance in the CNN implementation of the lipschitz network, so that `torch.compile` works, due to a bug with the inductor compiler and complex datatypes and views `.mH`, please make sure to get either the nightly build or any build that fixes PyTorch issue [https://github.com/pytorch/pytorch/issues/163243](https://github.com/pytorch/pytorch/issues/163243), where the sample code [complex_cholesky_decomposition.py](./test/complex_cholesky_decomposition.py) should run and compile the graph.

This issue has been fixed in the nightly build `torch-2.10.0.dev20250923` and onwards so please use this version for an optimized experience, which means that it will only fully work in `PyTorch 2.10` when that releases.

To install the prerelease:

```bash
pip3 install --pre -U torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

# Evaluation

To evaluate the network all the weights and training Tensorboard plots are available in https://huggingface.co/SuperComputer/LDLT

You can download the specific weights used for the training in the paper and validate the results yourself as well as the training run.

## Tensorboard

### For remote server
```bash
ssh -L 6006:127.0.0.1:6006 <user>@<ip address>
conda activate DeepLipschitzResNet
tensorboard --logdir=.
```
### For WSL2
```bash
tensorboard --host localhost --port 8888 --logdir=.
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