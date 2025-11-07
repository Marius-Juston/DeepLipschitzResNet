# LDLT Lipschitz Network

> Training code, evaluation scripts, and analysis tooling for **L-Lipschitz Gershgorin ResNet Network** (arXiv: [2502.21279](https://arxiv.org/abs/2502.21279)).

This repository reproduces the experiments from the paper and provides ready-made
utilities for CIFAR-10/100, Tiny-ImageNet, UCI tabular suites, and profiling of
compiled Lipschitz networks.

## Table of Contents

1. [Background](#background)
2. [Installation](#installation)
3. [Usage](#usage)
   - [Training](#training)
   - [Evaluation](#evaluation)
   - [Reproducing paper tables](#reproducing-paper-tables)
4. [Profiling & Diagnostics](#profiling--diagnostics)
5. [Plotting Utilities](#plotting-utilities)
6. [Performance Notes](#performance-notes)
7. [TensorBoard Access](#tensorboard-access)
8. [Citation](#citation)

## Background

`DeepLipschitzResNet` implements the LDLT linear and residual architectures that
enforce global \(L\)-Lipschitz constraints via LDLT Decomposition. The codebase
covers:

- convolutional and linear models for CIFAR-10/100 and Tiny-ImageNet
- UCI tabular experiments (via `src/uci_training.py`)
- certified robustness evaluation against fixed perturbation budgets
- profiling, visualization, and numerical stability diagnostics

If you use this repository, please cite the accompanying paper in the
[Citation](#citation) section.

## Installation

> Requires Python ≥ 3.10 and PyTorch ≥ 2.8 (nightly 2.10 for compiled CNN support).

```bash
git clone https://github.com/<your-org>/DeepLipschitzResNet.git
cd DeepLipschitzResNet
python -m venv .venv
source .venv/bin/activate  # or .\.venv\Scripts\activate on Windows
pip install -r requirements.txt
```

For CUDA builds unaffected by the Inductor complex-view bug (see
[Performance Notes](#performance-notes)), install the latest nightlies:

```bash
pip install --pre -U torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

Datasets are expected under `data/` (CIFAR and Tiny-ImageNet) and UCI bundles
under `data/UCI/`. See `AGENTS.md` for directory conventions.

## Usage

### Training

Single command entry point (`src/main.py`) with torchrun:

```bash
# 4-GPU CIFAR-10 (small model)
torchrun --standalone --nproc_per_node=4 \
  src/main.py --dataset cifar10 --model-name small --train_dir runs/cifar10_small

# 4-GPU CIFAR-100 (medium model, AdamW optimizer)
torchrun --standalone --nproc_per_node=4 \
  src/main.py --dataset cifar100 --model-name medium --optimizer adamw \
  --train_dir runs/cifar100_medium
```

Wrapper scripts mirror the paper’s setups:

```bash
./train_cifar10.sh           # CIFAR-10 baseline
./train_cifar100.sh          # CIFAR-100 baseline
./train.sh cifar100          # generic entry (CIFAR-100)
./train.sh tiny-imagenet     # Tiny-ImageNet variant
```

Specialized trainers for ablations:
- `src/linear_train.py`
- `src/cnn_train.py`

Both expose flags for depth, optimizer, spectral penalties, and certification
modes identical to the paper.

### Evaluation

Run certified/standard evaluation against checkpoints:

```bash
# Standard + certified accuracy
torchrun --standalone --nproc_per_node=1 \
  src/main.py --dataset cifar10 --model-name small \
  --train_dir runs/cifar10_small --mode certified
```

Pretrained weights, logs, and TensorBoard events are hosted on
[Hugging Face: SuperComputer/LDLT](https://huggingface.co/SuperComputer/LDLT).
Download a checkpoint and point `--train_dir` at the extracted run folder to
replicate the paper’s metrics.

### Reproducing paper tables

```bash
python reproduce_tables.py
```

This script downloads the published checkpoints, runs inference passes, and
emits the CIFAR-10/100 tables reported in the manuscript.

## Profiling & Diagnostics

Use the built-in profiler to trace compiled execution graphs:

```bash
python src/profiling.py --train_dir runs/cifar10_small
```

This script wraps `torch.compile`, records traces in `trace.json`, and reports
graph breaks. Visualize traces by opening Chrome and navigating to
`chrome://tracing`, then loading `trace.json`. See the
[PyTorch profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html#using-tracing-functionality)
for more details.

The `test/complex_cholesky_decomposition.py` snippet reproduces the Inductor bug
referenced in the paper; it should compile cleanly on torch nightlies ≥
`2.10.0.dev20250923`.

## Plotting Utilities

- **LMI plots**: `python test/model_lmi.py --checkpoint <path>` generates plots
  saved under `test/figs/`.
- **Numerical instability plots**: `python test/numerical_instability.py`
  reproduces the histograms from the manuscript.

Both scripts accept optional checkpoint paths to inspect specific trained
weights.

## Performance Notes

For compiled CNNs, ensure you are on a PyTorch build that includes the fix for
Inductor complex datatypes and `.mH` views (issue
[#163243](https://github.com/pytorch/pytorch/issues/163243)). Nightlies at
`torch-2.10.0.dev20250923` or later contain the patch; PyTorch 2.10 stable will
also include it.

If you encounter compilation failures, verify:

1. `torch.compile` is enabled (default in `profiling.py`).
2. You are on a patched nightly or the upcoming 2.10 release.
3. `complex_cholesky_decomposition.py` compiles locally; if not, update PyTorch.

## TensorBoard Access

TensorBoard logs live under `runs/`. Two common workflows:

### Remote server

```bash
ssh -L 6006:127.0.0.1:6006 <user>@<server-ip>
conda activate DeepLipschitzResNet
tensorboard --logdir runs
```

Point your local browser to `http://localhost:6006`.

### WSL2 / local only

```bash
tensorboard --host localhost --port 8888 --logdir runs
```

## Citation

If this repository helps your research, please cite:

<a id="1">[1]</a>

```
@misc{juston2025llipschitzgershgorinresnetnetwork,
  title={L-Lipschitz Gershgorin ResNet Network},
  author={Marius F. R. Juston and William R. Norris and Dustin Nottage and Ahmet Soylemezoglu},
  year={2025},
  eprint={2502.21279},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2502.21279}
}
```

---

Questions or issues? Please open a GitHub issue with the command you ran, full
stack trace (if any), and environment details (OS, CUDA, PyTorch version).***
