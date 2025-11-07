#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cross-validated training harness for Lipschitz-constrained models on the JKU UCI collection.

This module prepares tabular datasets, trains a variety of Lipschitz-aware models with
TensorBoard logging, evaluates both empirical and certified accuracies, and orchestrates
Wilcoxon signed-rank comparisons across algorithms. Runtime behavior strictly matches the
original implementation; only documentation and type annotations have been added.
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import traceback
from collections.abc import Sequence
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import wilcoxon, norm
from tabulate import tabulate
from torch import Tensor
from torch.multiprocessing import set_start_method
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from lbdn_models.model import Toy
from models.linear_model import (
    DeepLipschitzSequential,
    DeepLipschitzLinearResNet,
    DeepLipschitzLinearResNetLNN,
    DeepLipschitzSequentialLNN,
)
from sll_models.model import LipschitzNetworkSequential

type FloatArray = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int64]

logger = logging.getLogger(__name__)


def load_dat(path: Path) -> FloatArray:
    """
    Load a comma-delimited `.dat` file as a float64 matrix.

    Parameters
    ----------
    path : Path
        Absolute or relative path to the `.dat` file containing values arranged row-wise.

    Returns
    -------
    data : (n_rows, n_cols) float64 ndarray
        Parsed numeric matrix with `np.float64` dtype.

    Raises
    ------
    OSError
        If the file cannot be opened.
    ValueError
        If the file contents cannot be interpreted as a rectangular numeric array.

    Notes
    -----
    Internally wraps :func:`numpy.loadtxt` with `delimiter=','`, matching the format of the
    JKU UCI preprocessed artifacts.

    Mathematical sketch
    --------------------
    No transformation is applied beyond parsing; the loader simply maps ASCII tokens to
    floats via NumPy's vectorized scanner.

    Examples
    --------
    >>> from pathlib import Path
    >>> tmp = Path('example.dat')
    >>> _ = tmp.write_text('1,2\\n3,4\\n')
    >>> load_dat(tmp)
    array([[1., 2.],
           [3., 4.]])
    >>> tmp.unlink()
    """
    return np.loadtxt(path, delimiter=',')


class ArrayDataset(Dataset[tuple[Tensor, Tensor]]):
    """
    Thin dataset wrapper around dense NumPy arrays with one-hot encoded labels.

    Parameters
    ----------
    X : (n_samples, n_features) float64 ndarray
        Feature matrix that has already been standardized.
    y : (n_samples,) int64 ndarray
        Integer class labels in `[0, n_classes - 1]`.
    indices : (n_indices,) int64 ndarray
        Row indices selecting the partition (train/val/test).

    Notes
    -----
    Converts arrays into contiguous `torch.float32` tensors once during initialization to
    avoid repeated host-to-device transfers per sample.

    Returns
    -------
    None

    Raises
    ------
    None

    Mathematical sketch
    --------------------
    Each call to :meth:`__getitem__` simply indexes row `i` from the cached tensors and
    returns `(x_i, y_i)` with `y_i` already one-hot encoded.

    Examples
    --------
    >>> X = np.array([[0.0], [1.0]], dtype=np.float64)
    >>> y = np.array([0, 1], dtype=np.int64)
    >>> ArrayDataset(X, y, np.array([0, 1], dtype=np.int64))
    <...ArrayDataset object at ...>
    """

    def __init__(self, X: FloatArray, y: IntArray, indices: IntArray) -> None:
        """
        Construct the dataset and pre-compute one-hot encodings.

        Parameters
        ----------
        X : (n_samples, n_features) float64 ndarray
            Feature matrix representing the entire dataset.
        y : (n_samples,) int64 ndarray
            Ground-truth class indices with zero-based encoding.
        indices : (n_subset,) int64 ndarray
            Integer positions that choose which rows belong to the dataset split.

        Returns
        -------
        None
            The initializer mutates internal tensors in place.

        Raises
        ------
        None

        Notes
        -----
        The method converts data to contiguous CPU tensors with dtype `torch.float32`,
        ensuring compatibility with subsequent `DataLoader` usage.

        Mathematical sketch
        --------------------
        Represent each class label `c` by the canonical basis vector `e_c` in
        \\( \\mathbb{R}^{C} \\); the stored label tensor simply stacks these vectors.

        Examples
        --------
        >>> X = np.array([[0.0], [1.0]], dtype=np.float64)
        >>> y = np.array([0, 1], dtype=np.int64)
        >>> idx = np.array([0, 1], dtype=np.int64)
        >>> ds = ArrayDataset(X, y, idx)
        >>> len(ds)
        2
        """
        self.X = torch.from_numpy(X).to(torch.float32)

        n_classes = y.max() + 1

        self.y = torch.zeros((y.shape[0], n_classes), dtype=torch.float32)
        #  One hot encoding
        self.y[np.arange(y.shape[0]), y] = 1

        self.idx = indices.astype(np.int64, copy=False)

    def __len__(self) -> int:
        """
        Number of accessible samples.

        Returns
        -------
        length : int
            Cardinality of the subset represented by `self.idx`.

        Raises
        ------
        None

        Notes
        -----
        Required by PyTorch's `Dataset` protocol.

        Mathematical sketch
        --------------------
        The method simply returns the length of the cached index tensor.

        Examples
        --------
        >>> len(ArrayDataset(np.zeros((2, 1)), np.zeros(2, dtype=np.int64), np.array([0], dtype=np.int64)))
        1
        """
        return self.idx.shape[0]

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        """
        Retrieve the `i`-th sample/label pair.

        Parameters
        ----------
        i : int
            Position inside the subset index array, `0 <= i < len(self)`.

        Returns
        -------
        sample : (n_features,) float32 tensor
            Standardized tabular feature vector.
        target : (n_classes,) float32 tensor
            One-hot encoded target.

        Raises
        ------
        IndexError
            If `i` is out of range.

    Notes
    -----
    The implementation uses integer indexing into pre-materialized tensors, so the
    operation is `O(1)`.

    Mathematical sketch
    --------------------
    Retrieves row `j = idx[i]` and returns `(X_j, Y_j)` after squeezing singleton
    dimensions introduced by PyTorch's storage layout.

    Examples
    --------
    >>> X = np.array([[0.0], [1.0]], dtype=np.float64)
        >>> y = np.array([0, 1], dtype=np.int64)
        >>> ds = ArrayDataset(X, y, np.array([1], dtype=np.int64))
        >>> xb, yb = ds[0]
        >>> (xb.item(), yb.argmax().item())
        (1.0, 1)
        """
        j = self.idx[i]
        return self.X[j].squeeze(0), self.y[j].squeeze(0)


def standardize_inplace(X: FloatArray, train_idx: IntArray, eps: float = 1e-8) -> dict[str, FloatArray]:
    """
    Standardize features in place using training statistics.

    Parameters
    ----------
    X : (n_samples, n_features) float64 ndarray
        Full design matrix to standardize; mutated in place.
    train_idx : (n_train, 1) or (n_train,) int64 ndarray
        Integer indices specifying training rows used to compute moments.
    eps : float, default=1e-8
        Small positive floor to avoid division by zero when a feature is constant.

    Returns
    -------
    stats : dict[str, (n_features,) float64 ndarray]
        Dictionary with keys ``"mean"`` and ``"std"`` storing training-only moments.

    Raises
    ------
    None

    Notes
    -----
    Standardization uses \\( z = (x - \\mu) / \\sigma \\) with \\( \\sigma = \\max(\\text{std}, \\varepsilon) \\).
    All views of `X` reflect the modified values because the operation updates the array in place.

    Mathematical sketch
    --------------------
    Given training subset \\( \\mathcal{T} \\), compute
    \\( \\mu_j = \\frac{1}{|\\mathcal{T}|} \\sum_{i \\in \\mathcal{T}} X_{ij} \\)
    and \\( \\sigma_j = \\sqrt{ \\frac{1}{|\\mathcal{T}|} \\sum_{i \\in \\mathcal{T}} (X_{ij} - \\mu_j)^2 } \\).
    Each feature column is then centered and scaled by these statistics.
    Assumes `train_idx` contains at least one entry.

    Examples
    --------
    >>> X = np.array([[1., 2.], [3., 4.]], dtype=np.float64)
    >>> stats = standardize_inplace(X, np.array([0, 1], dtype=np.int64))
    >>> np.allclose(X.mean(axis=0), np.zeros(2))
    True
    >>> np.round(stats['std'], 6).tolist()
    [1.414214, 1.414214]
    """
    mu = X[train_idx].mean(axis=0)
    sigma = X[train_idx].std(axis=0)
    sigma[sigma < eps] = 1.0
    X[:] = (X - mu) / sigma
    return {"mean": mu, "std": sigma}


def class_weights_from_indices(y: IntArray, train_idx: IntArray) -> Tensor:
    """
    Compute inverse-frequency class weights from training indices.

    Parameters
    ----------
    y : (n_samples,) int64 ndarray
        Class labels for the entire dataset.
    train_idx : (n_train, 1) or (n_train,) int64 ndarray
        Indices specifying which samples belong to the training split.

    Returns
    -------
    weights : (n_classes,) float32 tensor
        Weight vector normalized so that its mean equals one.

    Raises
    ------
    None

    Notes
    -----
    Frequencies are estimated as \\( f_c = n_c / n_{\\text{train}} \\) and weights become
    \\( w_c = (\\sum_c 1 / f_c)^{-1} / f_c \\). This keeps the cross-entropy loss scale
    stable while counteracting class imbalance.

    Mathematical sketch
    --------------------
    The method sets
    \\( w_c = \\frac{C}{\\sum_{c'} 1/f_{c'}} \\cdot \\frac{1}{f_c} \\),
    where `C` is the number of observed classes. The scaling ensures \\( \\frac{1}{C} \\sum_c w_c = 1 \\).
    Assumes each class in `train_idx` appears at least once.

    Examples
    --------
    >>> y = np.array([0, 0, 1, 1, 1], dtype=np.int64)
    >>> idx = np.array([0, 2, 3, 4], dtype=np.int64)
    >>> class_weights_from_indices(y, idx).tolist()
    [1.5, 0.5]
    """
    classes, counts = np.unique(y[train_idx], return_counts=True)
    freq = counts.astype(np.float64) / counts.sum()
    w = 1.0 / np.maximum(freq, 1e-9)
    # Normalize weights to keep loss scale reasonable
    w = w * (len(w) / w.sum())
    # Ensure weight vector aligns with class indices [0..C-1]
    C = int(y.max()) + 1
    full = np.ones(C, dtype=np.float32)
    for c, val in zip(classes.tolist(), w.tolist()):
        full[c] = val
    return torch.tensor(full, dtype=torch.float32)


class MLP(nn.Module):
    """
    Simple batch-normalized multilayer perceptron baseline for tabular data.

    Parameters
    ----------
    d_in : int
        Number of input features.
    n_classes : int
        Number of mutually exclusive classes.
    width : int
        Hidden layer width.
    depth : int, default=4
        Number of hidden layers (each comprises Linear → BatchNorm → ReLU → Dropout).
    dropout : float, default=0.2
        Dropout probability applied after each activation.

    Notes
    -----
    Uses Kaiming uniform initialization appropriate for ReLU activations. No Lipschitz
    constraints are enforced; the model primarily serves as a baseline.

    Returns
    -------
    None

    Raises
    ------
    None

    Mathematical sketch
    --------------------
    The network computes
    \\( f(x) = W_{\\text{out}} h_{d}(x) + b_{\\text{out}} \\),
    where each hidden block performs
    \\( h_{k+1} = \\text{Dropout}(\\text{ReLU}(\\text{BN}(W_k h_k + b_k))) \\).

    Examples
    --------
    >>> model = MLP(d_in=8, n_classes=3, width=64)
    >>> isinstance(model, nn.Module)
    True
    """

    def __init__(
            self,
            d_in: int,
            n_classes: int,
            width: int,
            depth: int = 4,
            dropout: float = 0.2,
    ) -> None:
        """
        Initialize the multilayer perceptron backbone.

        Parameters
        ----------
        d_in : int
            Dimensionality of the input features.
        n_classes : int
            Number of logits produced by the head.
        width : int
            Width of each hidden block.
        depth : int, default=4
            Number of hidden blocks to stack.
        dropout : float, default=0.2
            Probability of zeroing activations during training.

        Returns
        -------
        None

        Raises
        ------
        None

        Notes
        -----
        Each hidden block allocates fresh Linear, BatchNorm1d, ReLU, and Dropout layers.
        Weight initialization follows `torch.nn.init.kaiming_uniform_`.

        Mathematical sketch
        --------------------
        Initializes parameters so that incoming variance is preserved, i.e.,
        \\( \\text{Var}(W x) = \\text{Var}(x) \\) under ReLU assumptions as per Kaiming initialization.

        Examples
        --------
        >>> model = MLP(d_in=4, n_classes=3, width=16)
        >>> sum(p.numel() for p in model.parameters()) > 0
        True
        """
        super().__init__()
        layers = []
        prev = d_in
        for _ in range(depth):
            layers += [nn.Linear(prev, width), nn.BatchNorm1d(width), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = width
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, n_classes)

        # Kaiming init for stability on tabular
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute logits for the provided batch.

        Parameters
        ----------
        x : (batch, d_in) float32 tensor
            Standardized feature batch on CPU or GPU.

        Returns
        -------
        logits : (batch, n_classes) float32 tensor
            Unnormalized logit scores prior to softmax.

        Raises
        ------
        RuntimeError
            If the input dimensionality mismatches `d_in`.

        Notes
        -----
        The method simply applies the sequential backbone followed by the linear head.

        Mathematical sketch
        --------------------
        Equivalent to applying the composed affine/ReLU mapping described in the class
        docstring.

        Examples
        --------
        >>> model = MLP(d_in=2, n_classes=2, width=8)
        >>> out = model(torch.zeros(4, 2))
        >>> out.shape
        torch.Size([4, 2])
        """
        x = self.backbone(x)
        return self.head(x)


def suggested_width(d_in: int, n_classes: int) -> int:
    """
    Heuristic for choosing MLP width on small-to-mid-sized tabular datasets.

    Parameters
    ----------
    d_in : int
        Number of input features.
    n_classes : int
        Number of output classes; larger values bias towards broader hidden layers.

    Returns
    -------
    width : int
        Selected width constrained to the discrete set ``{32, 64, 128, 256, 384, 512}``.

    Notes
    -----
    The heuristic scales as `min(512, max(32, 4 * d_in))` with an additional 25% boost
    when `n_classes > 10`, then rounds to the nearest bucket for reproducibility.

    Mathematical sketch
    --------------------
    The rule encodes the intuition that each feature contributes roughly four hidden units,
    capping capacity at 512 to avoid overfitting. Assumes datasets roughly follow the UCI
    statistics seen in Lipschitz model benchmarks.

    Examples
    --------
    >>> suggested_width(10, 3)
    64
    >>> suggested_width(200, 20)
    512

    Raises
    ------
    None
    """
    base = max(32, min(512, 4 * d_in))
    if n_classes > 10:
        base = int(base * 1.25)
    buckets = [32, 64, 128, 256, 384, 512]
    return min(buckets, key=lambda b: abs(b - base))


# ------------- Training ------------- #

def set_seed(seed: int):
    """
    Synchronize Python, NumPy, and PyTorch RNG states.

    Parameters
    ----------
    seed : int
        Non-negative seed applied to `random`, `numpy`, and `torch`.

    Returns
    -------
    None

    Raises
    ------
    None

    Notes
    -----
    Deterministic seeding aids reproducibility of cross-validation splits and model
    initialization.

    Mathematical sketch
    --------------------
    Calls `random.seed`, `np.random.seed`, and `torch.manual_seed` with the same integer.

    Examples
    --------
    >>> set_seed(0)
    >>> random.randint(0, 10)
    6
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def eval_loop(model: nn.Module, loader: DataLoader[tuple[Tensor, Tensor]], device: torch.device) -> float:
    """
    Evaluate categorical accuracy over a dataloader.

    Parameters
    ----------
    model : torch.nn.Module
        Network that outputs raw logits.
    loader : DataLoader[((n_features,), (n_classes,))]
        Iterable emitting `(inputs, one_hot_labels)` tuples already on CPU.
    device : torch.device
        Target device for inference (CPU or CUDA).

    Returns
    -------
    accuracy : float
        Fraction of correctly classified samples in `[0, 1]`.

    Raises
    ------
    RuntimeError
        Propagated if model forward pass fails.

    Notes
    -----
    Uses `argmax` over logits and one-hot labels, so both tensors must share consistent
    class ordering.

    Mathematical sketch
    --------------------
    Computes \\( \\frac{1}{N} \\sum_{i=1}^{N} \\mathbb{1}\\{\\arg\\max_k \\hat{y}_{ik} = \\arg\\max_k y_{ik}\\} \\).

    Examples
    --------
    >>> class Constant(nn.Module):
    ...     def forward(self, x):
    ...         return torch.tensor([[1., 0.]], dtype=x.dtype).expand(x.size(0), -1)
    >>> loader = DataLoader([(torch.zeros(1, 2), torch.tensor([[1., 0.]]))])
    >>> eval_loop(Constant(), loader, torch.device('cpu'))
    1.0
    """
    model.eval()
    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        pred = logits.argmax(dim=1)
        correct += (pred == yb.argmax(dim=1)).sum().item()
        total += yb.size(0)
    return correct / max(total, 1)


def convert_cert_vals(cert_names: Sequence[float], cert_vals: Tensor) -> dict[str, float]:
    """
    Zip epsilon radii with their empirical certified accuracies.

    Parameters
    ----------
    cert_names : sequence of float
        Certified radii expressed in pixel-space (0–255 scaling).
    cert_vals : (n_eps,) float tensor
        Certified accuracy per epsilon as returned by :func:`eval_certified`.

    Returns
    -------
    mapping : dict[str, float]
        Dictionary keyed by integer epsilon string (e.g., ``\"36\"``) with scalar accuracies.

    Notes
    -----
    Converts tensor entries to Python floats to ease JSON serialization.

    Mathematical sketch
    --------------------
    The function performs a bijective mapping from `(ε, metric)` pairs to dictionary items
    by stringifying the epsilon budget.

    Examples
    --------
    >>> convert_cert_vals([36, 72], torch.tensor([0.8, 0.5]))
    {'36': 0.8, '72': 0.5}

    Raises
    ------
    None
    """
    return {str(int(n)): v.item() for n, v in zip(cert_names, cert_vals)}


@torch.no_grad()
def eval_certified(
        model: nn.Module,
        loader: DataLoader[tuple[Tensor, Tensor]],
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
) -> tuple[list[float], float, Tensor]:
    """
    Compute certified accuracies for fixed perturbation radii.

    Parameters
    ----------
    model : torch.nn.Module
        Lipschitz-constrained classifier exposing `last_weight()` or a final linear layer.
    loader : DataLoader[((n_features,), (n_classes,))]
        Validation dataloader emitting normalized features and one-hot labels.
    dtype : torch.dtype, optional
        Desired tensor dtype for computations; defaults to model parameters' dtype.
    device : torch.device, optional
        Device hosting inference; defaults to the active CUDA device if available.

    Returns
    -------
    cert_values : list of float
        Pixel-space epsilon radii `[36, 72, 108, 255]`.
    accuracy : float
        Standard accuracy under point predictions.
    certified : (len(cert_values),) float tensor
        Fraction of correctly certified examples for each epsilon.

    Raises
    ------
    ValueError
        If the model type is unsupported for certified evaluation.

    Notes
    -----
    The certification radius per sample equals
    \\( r(x) = \\min_{k \\neq y} \\frac{(f_y(x) - f_k(x))}{\\|w_y - w_k\\|_2} \\),
    where `w_y` are normalized final-layer weights and `f` are logits. Samples whose
    radius exceeds `ε / 255` contribute to the certified accuracy at that epsilon.

    Mathematical sketch
    --------------------
    For margin vector `m = f_y - f_k` and normalized weights `\\hat{w}`, the Lipschitz
    bound implies robustness ball radius `m / \\|\\hat{w}_y - \\hat{w}_k\\|_2`. The function
    evaluates this for every competing class, aggregates minima, and compares against
    scaled epsilon budgets. Assumes `model` exposes the last linear weight either via
    `.last_weight()` or `.model[-1].weight`.

    Examples
    --------
    >>> dummy = Toy(in_channels=4, num_classes=2, depth=1, layer='Aol', padding=8)  # doctest: +SKIP
    >>> loader = DataLoader([ (torch.zeros(1, 4), torch.tensor([[1., 0.]])) ])      # doctest: +SKIP
    >>> eval_certified(dummy, loader, dtype=torch.float32, device=torch.device('cpu'))  # doctest: +SKIP
    ([36.0, 72.0, 108.0, 255.0], 0.0, tensor([0., 0., 0., 0.]))
    """
    cert_values = [36., 72., 108., 255.]

    eps_float = torch.tensor(cert_values, dtype=dtype, device=device) / 255.
    model.eval()
    running_accuracy = 0

    running_certified = torch.zeros_like(eps_float)

    running_inputs = 0
    lip_cst = 1.

    if isinstance(model, DeepLipschitzLinearResNetLNN) or isinstance(model, DeepLipschitzSequentialLNN):
        last_weight = model.last.weight
        normalized_weight = F.normalize(last_weight, p=2, dim=1)
    elif isinstance(model, DeepLipschitzSequential):
        normalized_weight = model.last_weight()
    elif isinstance(model, DeepLipschitzLinearResNet):
        normalized_weight = model.last_weight()
    elif isinstance(model, LipschitzNetworkSequential) or isinstance(model, Toy):
        last_weight = model.model[-1].weight
        normalized_weight = F.normalize(last_weight, p=2, dim=1)
    else:
        raise ValueError("Using unsupported model type for certified evaluation")

    for batch_n, data in enumerate(loader):
        inputs, labels = data
        inputs, labels = inputs.to(device=device, dtype=dtype), labels.to(device=device, dtype=dtype)
        outputs = model(inputs)
        predicted = outputs.argmax(axis=1)
        correct = predicted == labels.argmax(axis=1)
        margins, indices = torch.sort(outputs, 1)

        top_idx = indices[:, -1]
        other_idx = indices[:, :-1]

        num = margins[:, -1][:, None] - margins[:, :-1]

        w_top = normalized_weight[top_idx]
        w_oth = normalized_weight[other_idx]

        denom = torch.norm(w_top[:, None, :] - w_oth, dim=-1, p=2).clamp_min(1e-12)

        radii = (num / denom).min(dim=1).values

        certified = radii[:, None] > eps_float * lip_cst
        running_accuracy += correct.sum()
        running_certified += torch.sum(torch.unsqueeze(correct, -1) & certified, dim=0)
        running_inputs += inputs.size(0)

    accuracy = running_accuracy / running_inputs
    certified = running_certified / running_inputs
    return cert_values, accuracy, certified


def train_one_fold(
        X: FloatArray,
        y: IntArray,
        train_idx: IntArray,
        test_idx: IntArray,
        val_idx: IntArray,
        algorithm: str,
        seed: int,
        log_dir: Path,
        hparams: dict[str, Any],
        max_epochs: int = 300,
        patience: int = 30,
        batch_size: int = 128,
        num_workers: int = 8,
        depth: int = 4,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
) -> dict[str, Any]:
    """
    Train, validate, and test a single fold of cross-validation.

    Parameters
    ----------
    X : (n_samples, n_features) float64 ndarray
        Full dataset features; mutated in place during standardization.
    y : (n_samples,) int64 ndarray
        Class labels aligned with `X`.
    train_idx, test_idx, val_idx : (n_subset, 1) int64 ndarray
        Boolean-fold-derived indices for training, testing, and validation splits.
    algorithm : {'mlp', 'sdp', 'ldlt', 'ldlt-resnet', 'aol', 'ortho', 'sandwich'}
        Model family identifier selecting the appropriate architecture.
    seed : int
        Deterministic seed for data shuffling and initialization.
    log_dir : Path
        Directory where TensorBoard events and checkpoints are persisted.
    hparams : dict
        Hyperparameter metadata recorded to TensorBoard.
    max_epochs : int, default=300
        Maximum optimization epochs.
    patience : int, default=30
        Early-stopping patience measured in epochs.
    batch_size : int, default=128
        Training mini-batch size; validation/test use 512.
    num_workers : int, default=8
        DataLoader worker processes.
    depth : int, default=4
        Network depth where applicable.
    lr : float, default=1e-3
        AdamW learning rate.
    weight_decay : float, default=1e-4
        AdamW weight decay coefficient.
    dtype : torch.dtype, optional
        Computation dtype for model parameters and tensors.
    device : torch.device, optional
        Execution device; CUDA is preferred when available.

    Returns
    -------
    metrics : dict
        Dictionary containing best epoch, accuracies, architectural metadata, and
        certified evaluations (when applicable).

    Raises
    ------
    ValueError
        If `algorithm` is unrecognized.

    Notes
    -----
    Implements standard training with TensorBoard logging, ReduceLROnPlateau scheduling,
    and gradient clipping (`\\|g\\|_2 <= 5`). Statistics for standardization are measured
    on `train_idx` only and serialized with the results.

    Mathematical sketch
    --------------------
    The optimization objective is the weighted cross-entropy
    \\( \\mathcal{L} = -\\sum_i w_{y_i} \\log p_{y_i}(x_i) \\),
    minimized via AdamW with cosine-like LR reductions triggered by validation accuracy
    plateaus. Early stopping selects the checkpoint with maximal validation accuracy.

    Examples
    --------
    >>> X = np.random.randn(20, 4)
    >>> y = np.random.randint(0, 2, size=20).astype(np.int64)
    >>> idx = np.arange(20, dtype=np.int64)
    >>> res = train_one_fold(  # doctest: +SKIP
    ...     X, y, idx[:10], idx[10:], idx[10:], 'mlp', 0, Path('tmp'), {}, max_epochs=1
    ... )
    >>> sorted(res.keys())  # doctest: +SKIP
    ['best_epoch', 'best_val_acc', 'cert_evals', 'depth', 'in_features', 'out_features', 'padding', 'parameters', 'stats', 'test_acc', 'test_size', 'train_size', 'val_size', 'width']
    """
    set_seed(seed)

    # Standardize
    stats = standardize_inplace(X, train_idx)

    # Datasets & loaders
    ds_train = ArrayDataset(X, y, train_idx)
    ds_val = ArrayDataset(X, y, val_idx)
    ds_test = ArrayDataset(X, y, test_idx)

    train_loader = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True
    )
    val_loader = DataLoader(
        ds_val, batch_size=512, shuffle=False, drop_last=False, pin_memory=True
    )
    test_loader = DataLoader(
        ds_test, batch_size=512, shuffle=False, drop_last=False, pin_memory=True
    )

    n_classes = int(y.max()) + 1
    d_in = X.shape[1]
    width = suggested_width(d_in, n_classes)

    padding = max(d_in * 2, 10)

    if algorithm == 'mlp':
        model = MLP(d_in, n_classes, width=width, depth=depth, dropout=0.2).to(device=device, dtype=dtype)
    elif algorithm == 'sdp':
        model = LipschitzNetworkSequential(cin=d_in, n_dense=depth, padding=padding, dense_inner_dim=width,
                                           n_classes=n_classes).to(device=device, dtype=dtype)
    elif algorithm == 'ldlt':
        model = DeepLipschitzSequentialLNN(d_in, n_classes, layers_widths=[width] * (depth - 1), device=device,
                                           dtype=dtype)
    elif algorithm == 'ldlt-resnet':
        model = DeepLipschitzLinearResNetLNN(d_in, n_classes, layers_widths=[width] * (depth - 1), device=device,
                                             dtype=dtype)
    elif algorithm == 'aol':
        model = Toy(in_channels=d_in, num_classes=n_classes, depth=depth, layer='Aol', padding=padding).to(
            device=device, dtype=dtype)
    elif algorithm == 'ortho':
        model = Toy(in_channels=d_in, num_classes=n_classes, depth=depth, layer='Orthogon', padding=padding).to(
            device=device, dtype=dtype)
    elif algorithm == 'sandwich':
        model = Toy(in_channels=d_in, num_classes=n_classes, depth=depth, layer='Sandwich', padding=padding).to(
            device=device, dtype=dtype)
    else:
        raise ValueError("Please provide a valid algorithm name 'mlp'")

    total_trainable_parameters = sum([p.numel() for p in model.parameters() if p.requires_grad])

    class_w = class_weights_from_indices(y, train_idx).to(device=device, dtype=dtype)
    criterion = nn.CrossEntropyLoss(weight=class_w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8)

    # Limit BLAS threads per process to avoid oversubscription (assumes 4 GPUs, 256 cores)
    torch.set_num_threads(max(1, os.cpu_count() // torch.cuda.device_count()))

    writer = SummaryWriter(str(log_dir))
    writer.add_hparams(hparams, {"hparam/acc/val": 0.0, 'hparam/acc/train': 0.0}, global_step=0)

    best_val = -1.0
    best_state = None
    best_epoch = -1
    epochs_no_improve = 0

    global_step = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for xb, yb in train_loader:
            xb = xb.to(device=device, dtype=dtype, non_blocking=True)
            yb = yb.to(device=device, dtype=dtype, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if algorithm == 'mlp' and xb.shape[0] == 1:
                # Skip entry due to BatchNorm having problems otherwise
                continue

            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item() * yb.size(0)
            pred = logits.argmax(dim=1)
            epoch_correct += (pred == yb.argmax(dim=1)).sum().item()
            epoch_total += yb.size(0)
            global_step += 1

        train_loss = epoch_loss / max(epoch_total, 1)
        train_acc = epoch_correct / max(epoch_total, 1)
        val_acc = eval_loop(model, val_loader, device)

        if algorithm != 'mlp':
            cert_names, ac, cert_vals = eval_certified(model, val_loader, device=device, dtype=dtype)
            cert_dict_ = convert_cert_vals(cert_names, cert_vals)

            for n, v in cert_dict_.items():
                writer.add_scalar(f'eval_cert/{n}', v, epoch)

        scheduler.step(val_acc)

        # TensorBoard logging
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("acc/train", train_acc, epoch)
        writer.add_scalar("acc/val", val_acc, epoch)

        writer.add_hparams(hparams, {"hparam/acc/val": val_acc, 'hparam/acc/train': train_acc}, global_step=epoch)

        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        # Early stopping (maximize val_acc)
        improved = val_acc > (best_val + 1e-6)
        if improved:
            best_val = val_acc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() if torch.is_tensor(v) else v
                          for k, v in model.state_dict().items()}

            logger.info(f"New best state [{epoch}]={best_val}")

            torch.save(best_state, log_dir / 'best_state.pth')

            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logger.warning(f"Early stopping due to no improvements to the validation accuracy for {patience} epochs")
            break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    test_acc = eval_loop(model, test_loader, device)

    if algorithm != 'mlp':
        cert_names, ac, cert_vals = eval_certified(model, val_loader, device=device, dtype=dtype)

        cert_dict_ = convert_cert_vals(cert_names, cert_vals)

        for n, v in cert_dict_.items():
            writer.add_scalar(f'best_eval_cert/{n}', v, best_epoch if best_epoch > 0 else 0)
    else:
        cert_dict_ = None

    writer.add_scalar("acc/test_final", test_acc, best_epoch if best_epoch > 0 else 0)
    writer.add_text("early_stopping/best_epoch", str(best_epoch), 0)
    writer.add_text("early_stopping/best_val_acc", f"{best_val:.6f}", 0)

    writer.flush()
    writer.close()

    return {
        "best_epoch": int(best_epoch),
        "best_val_acc": float(best_val),
        "test_acc": float(test_acc),
        "train_size": train_idx.shape[0],
        "val_size": val_idx.shape[0],
        "test_size": test_idx.shape[0],
        "stats": {"mean": stats["mean"].tolist(), "std": stats["std"].tolist()},
        "width": int(width),
        'in_features': d_in,
        'out_features': n_classes,
        'depth': depth,
        'padding': padding,
        'parameters': total_trainable_parameters,
        'cert_evals': cert_dict_
    }


# ------------- Worker: trains one dataset for one repeat (10 folds) ------------- #

def load_dataset_bundle(dataset_dir: Path) -> tuple[FloatArray, IntArray, IntArray, IntArray]:
    """
    Load features, labels, folds, and validation folds for a single dataset.

    Parameters
    ----------
    dataset_dir : Path
        Directory containing ``*_py.dat`` files as produced by the preprocessing scripts.

    Returns
    -------
    X : (n_samples, n_features) float64 ndarray
        Feature matrix.
    y : (n_samples,) int64 ndarray
        Class labels.
    folds : (n_samples, n_folds) int64 ndarray
        Binary fold assignment matrix where `1` denotes training membership.
    val_map_folds : (n_samples, n_folds) int64 ndarray
        Binary matrix marking validation splits.

    Raises
    ------
    AssertionError
        If fold matrices contain invalid markers or mismatched shapes.
    OSError
        If any required file is missing or unreadable.

    Notes
    -----
    The function enforces that each row belongs to at most one of train/val/test for every
    fold by verifying `folds + val_map_folds` is binary.

    Mathematical sketch
    --------------------
    Treat folds as indicator matrices `F` and validation map as `V`; constraints ensure
    `F_ik + V_ik ∈ {0, 1}` so the implicit test set is their complement.

    Examples
    --------
    >>> load_dataset_bundle(Path('data/uci/abalone'))  # doctest: +SKIP
    """
    features_path = dataset_dir / f"{dataset_dir.name}_py.dat"
    folds_path = dataset_dir / "folds_py.dat"
    labels_path = dataset_dir / "labels_py.dat"
    val_map_path = dataset_dir / "validation_folds_py.dat"

    X = load_dat(features_path)
    y = load_dat(labels_path).astype(np.int64).reshape(-1)
    folds = load_dat(folds_path).astype(np.int64)

    assert folds.max() == 1 and folds.min() == 0, 'Invalid fold assignment'

    val_map_folds = load_dat(val_map_path).astype(np.int64)
    assert val_map_folds.max() == 1 and val_map_folds.min() == 0, 'Invalid fold assignment'

    assert val_map_folds.shape[1] == folds.shape[1], "Folds needs to be the same shape"

    total_ = folds + val_map_folds
    assert total_.max() == 1 and total_.min() == 0, 'Invalid fold assignment'

    total_zeros = (total_.sum(axis=1) == 0).sum()

    if total_zeros > 0:
        logger.warning(f"Found {total_zeros} zeros ({total_zeros / total_.shape[0] * 100:.1f}%) in {dataset_dir.name}")

    assert X.shape[0] == y.shape[0] == folds.shape[0] == val_map_folds.shape[0], \
        f"Size mismatch in {dataset_dir}: X={X.shape}, y={y.shape}, folds={folds.shape}"

    return X, y, folds, val_map_folds


def _run_task(pack: tuple[Path, str, str, int, str, int, argparse.Namespace]) -> tuple[str, dict[str, Any]]:
    """
    Train every fold for one dataset/algorithm pair on a dedicated GPU.

    Parameters
    ----------
    pack : tuple
        `(dataset_dir, dataset_name, algorithm_name, gpu_id, out_dir, base_seed, args)`
        as prepared by :func:`schedule_and_run`.

    Returns
    -------
    status : {'ok', 'error'}
        Execution outcome flag.
    payload : dict
        Fold-aggregated metrics when status is ``'ok'``; otherwise includes traceback info.

    Raises
    ------
    None

    Notes
    -----
    Sets `CUDA_VISIBLE_DEVICES` per task to avoid GPU contention and caches per-fold
    results under `out_dir/algorithm_name/runs/<dataset>/fold_k/result.json`.

    Mathematical sketch
    --------------------
    No additional math is performed beyond delegating to :func:`train_one_fold`; the
    routine orchestrates cross-validation loops, ensuring seeds increment by fold index.

    Examples
    --------
    >>> # _run_task is invoked internally by multiprocessing; no stand-alone example.
    """
    dataset_dir, dataset_name, algorithm_name, gpu_id, out_dir, base_seed, args = pack
    try:
        # Pin this task to a specific GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        X, y, folds, val_map_folds = load_dataset_bundle(dataset_dir)
        K = folds.shape[1]

        # Copy master features once; per fold we will standardize an in-place copy
        X_master = X.copy()

        n_classes = int(y.max()) + 1
        d_in = X.shape[1]

        run_root = Path(out_dir) / "runs" / dataset_name
        run_root.mkdir(parents=True, exist_ok=True)

        # Prepare fold-wise results
        fold_results = []

        # Determine per-fold indices under both representations
        for k in range(K):
            X = X_master.copy()

            train_idx = np.argwhere(folds[:, k] == 1)
            val_idx = np.argwhere(val_map_folds[:, k] == 1)
            test_idx = np.argwhere(folds[:, k] + val_map_folds[:, k] == 0)

            # Build hparams log
            fold_seed = int(base_seed + k)
            hparams = dict(
                dataset=dataset_name,
                algorithm=algorithm_name,
                fold=k,
                seed=fold_seed,
                d_in=int(d_in),
                n_classes=int(n_classes),
                batch_size=args.batch_size,
                lr=args.lr,
                max_epochs=args.max_epochs,
                patience=args.patience,
                depth=args.depth,
                gpu_id=gpu_id
            )

            log_dir = run_root / f"fold_{k}"

            results_json = log_dir / f"result.json"

            if os.path.exists(results_json):
                logger.info(
                    f"Skipping the training for {algorithm_name}:{dataset_name}:{k} and loading from {str(results_json)}")
                with open(results_json, "r") as f:
                    res = json.load(f)
            else:
                res = train_one_fold(
                    X, y,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    val_idx=val_idx,
                    algorithm=algorithm_name,
                    seed=fold_seed,
                    device=device,
                    log_dir=log_dir,
                    hparams=hparams,
                    max_epochs=args.max_epochs,
                    patience=args.patience,
                    batch_size=args.batch_size,
                    depth=args.depth,
                    num_workers=min(16, max(4, (os.cpu_count() // max(1, args.gpus)) // 4)),
                    lr=args.lr,
                )

                with open(results_json, "w") as f:
                    json.dump(res, f, indent=4)

            # Attach identifiers for aggregation
            res.update({'fold': k})
            fold_results.append(res)

        certs = ['36', '72', '108', '255']

        test_accs = [fr["test_acc"] for fr in fold_results]

        result = {
            "dataset": dataset_name,
            "algorithm": algorithm_name,
            "n_folds": K,
            "fold_results": fold_results,
            "mean_test_acc": float(np.mean(test_accs)) if len(test_accs) else float("nan"),
            "std_test_acc": float(np.std(test_accs, ddof=1) if len(test_accs) > 1 else 0.0),
            "n_samples": int(X_master.shape[0]),
            "n_features": int(X_master.shape[1]),
            "n_classes": int(n_classes),
        }

        if algorithm_name != 'mlp':
            for cert in certs:
                values = [fr["cert_evals"][cert] for fr in fold_results]

                result.update({
                    f'mean_cert_acc_{cert}': float(np.mean(values)) if len(values) else float("nan"),
                    f'std_cert_acc_{cert}': float(np.std(values, ddof=1) if len(values) > 1 else 0.0)
                })

        return "ok", result
    except Exception as e:
        tb = traceback.format_exc()
        err = {
            "dataset": dataset_name,
            "algorithm": algorithm_name,
            "traceback": tb,
        }
        return "error", err


# ------------- Orchestrator ------------- #

def wilcoxon_prep_all_metrics(out_dir: Path, algorithms: list[str]) -> pd.DataFrame:
    """
    Assemble a long-format table of per-dataset mean metrics for Wilcoxon tests.

    Parameters
    ----------
    out_dir : Path
        Directory containing ``<algorithm>/summary.json`` files.
    algorithms : list of str
        Algorithm identifiers to include; missing summaries are skipped.

    Returns
    -------
    df : pandas.DataFrame
        Table with columns ``[dataset, algorithm, metric, value]`` sorted by metric.

    Raises
    ------
    None

    Notes
    -----
    Only keys prefixed with ``\"mean_\"`` are extracted, mirroring the certified accuracy
    metrics produced earlier in the pipeline.

    Mathematical sketch
    --------------------
    The routine simply flattens nested dictionaries; no statistical transformation occurs.

    Examples
    --------
    >>> wilcoxon_prep_all_metrics(Path('./runs'), ['sdp'])  # doctest: +SKIP
    """
    rows = []
    for alg in algorithms:
        summary_path = out_dir / alg / "summary.json"
        if not summary_path.exists():
            logging.warning(f"[wilcoxon_prep_all] Missing {summary_path}; skipping '{alg}'.")
            continue
        try:
            with open(summary_path, "r") as f:
                summary = json.load(f)
        except Exception as e:
            logging.error(f"[wilcoxon_prep_all] Could not read {summary_path}: {e}")
            continue

        datasets = summary.get("datasets", {})
        for ds_name, ds_entry in datasets.items():
            # collect all mean_* numeric metrics
            for k, v in ds_entry.items():
                if k.startswith("mean_"):
                    try:
                        val = float(v)
                    except Exception:
                        continue
                    if math.isfinite(val):
                        rows.append({"dataset": ds_name, "algorithm": alg, "metric": k, "value": val})

    df = pd.DataFrame(rows, columns=["dataset", "algorithm", "metric", "value"]).sort_values(
        ["metric", "dataset", "algorithm"]
    )
    save_path = out_dir / "wilcoxon_prep_all.json"
    df.to_json(save_path, orient="records", indent=4)
    logging.info(f"[wilcoxon_prep_all] Wrote {save_path} with {len(df)} rows.")
    return df


def _holm_adjust(pvals: Sequence[float]) -> list[float]:
    """
    Apply the Holm–Bonferroni correction to a list of p-values.

    Parameters
    ----------
    pvals : sequence of float
        Raw p-values from dependent hypothesis tests.

    Returns
    -------
    adjusted : list of float
        Holm-adjusted p-values preserving the input order.

    Notes
    -----
    Implements the step-down procedure where `p_(i)` is scaled by `(m - i + 1)` with
    cumulative monotonic enforcement.

    Mathematical sketch
    --------------------
    Let `p_(1) ≤ ... ≤ p_(m)` be sorted values. The adjusted statistic is
    \\( q_{(i)} = \\max(q_{(i-1)}, (m - i + 1) p_{(i)}) \\), clipped to `[0, 1]`.

    Examples
    --------
    >>> _holm_adjust([0.01, 0.04, 0.02])
    [0.03, 0.04, 0.04]
    """
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adjusted = [0.0] * m
    prev = 0.0
    for k, i in enumerate(order, start=1):
        adj = (m - k + 1) * pvals[i]
        adj = max(adj, prev)
        adjusted[i] = min(1.0, adj)
        prev = adjusted[i]
    return adjusted


def run_wilcoxon_tests_all_metrics(out_dir: Path, algorithms: list[str]) -> pd.DataFrame:
    """
    Perform pairwise Wilcoxon signed-rank tests across all metrics and algorithms.

    Parameters
    ----------
    out_dir : Path
        Directory containing `wilcoxon_prep_all.json` or per-algorithm summaries.
    algorithms : list of str
        Algorithms to compare; others are ignored.

    Returns
    -------
    df : pandas.DataFrame
        Table of pairwise statistics including Holm-adjusted p-values.

    Raises
    ------
    None

    Notes
    -----
    Outputs CSV/JSON artifacts and prints a Markdown-formatted summary to stdout. Metrics
    are ordered so that test accuracy and certified accuracies appear first.

    Mathematical sketch
    --------------------
    For each pair `(a, b)` the Wilcoxon signed-rank test compares metric differences on
    datasets where both algorithms report results. The resulting statistic approximates a
    paired `t`-test for non-Gaussian data; Holm corrections control family-wise error.

    Examples
    --------
    >>> run_wilcoxon_tests_all_metrics(Path('./runs'), ['sdp', 'ldlt'])  # doctest: +SKIP
    """
    prep_all_path = out_dir / "wilcoxon_prep_all.json"
    if prep_all_path.exists():
        df = pd.read_json(prep_all_path)
    else:
        df = wilcoxon_prep_all_metrics(out_dir, algorithms)

    if df.empty:
        logging.warning("[run_wilcoxon_tests_all_metrics] No data to compare.")
        return df

    algs_present = sorted(set(df["algorithm"]) & set(algorithms))
    if len(algs_present) < 2:
        logging.warning("[run_wilcoxon_tests_all_metrics] Need ≥2 algorithms; nothing to compare.")
        return pd.DataFrame()

    # Order metrics nicely: test acc first, then cert accs in numeric order, then any others alphabetical
    def _metric_order_key(m: str):
        if m == "mean_test_acc":
            return (0, 0)
        if m.startswith("mean_cert_acc_"):
            try:
                return (1, int(m.split("_")[-1]))
            except Exception:
                return (1, 10 ** 9)
        return (2, m)

    metrics = sorted(df["metric"].unique().tolist(), key=_metric_order_key)

    results = []
    for metric in metrics:
        subm = df[df["metric"] == metric]
        if subm.empty:
            continue
        pivot = subm.pivot_table(index="dataset", columns="algorithm", values="value")
        # compare all pairs on common datasets
        for i, a in enumerate(algs_present):
            for b in algs_present[i + 1:]:
                if a not in pivot.columns or b not in pivot.columns:
                    continue
                mat = pivot[[a, b]].dropna()
                if mat.empty:
                    logging.warning(f"[{metric}] No common datasets for {a} vs {b}.")
                    continue

                diffs = (mat[a] - mat[b]).values
                n_total = len(diffs)
                n_pos = int((diffs > 0).sum())
                n_neg = int((diffs < 0).sum())
                n_tie = int((diffs == 0).sum())
                n_nonzero = n_total - n_tie

                if n_nonzero == 0:
                    stat, pval = 0.0, 1.0
                else:
                    stat, pval = wilcoxon(
                        diffs, zero_method="wilcox", alternative="two-sided", correction=True, method="auto"
                    )

                p_for_z = max(float(pval), 1e-300)
                z = float(norm.isf(p_for_z / 2.0))  # ≥0
                z *= float(np.sign(diffs.sum() if diffs.sum() != 0 else (n_pos - n_neg)))  # give direction
                r = float(abs(z) / np.sqrt(max(1, n_nonzero)))

                results.append({
                    "metric": metric,
                    "alg_a": a,
                    "alg_b": b,
                    "n_common": n_total,
                    "n_nonzero": n_nonzero,
                    "wins_a": n_pos,
                    "wins_b": n_neg,
                    "ties": n_tie,
                    "win_rate_a_over_b": (n_pos + 0.5 * n_tie) / n_total,
                    "mean_diff_a_minus_b": float(np.mean(diffs)),
                    "median_diff_a_minus_b": float(np.median(diffs)),
                    "W_stat": float(stat),
                    "p_two_sided": float(pval),
                    "z_equiv": float(z),
                    "effect_size_r": r,
                })

    res_df = pd.DataFrame(results).sort_values(["metric", "p_two_sided", "alg_a", "alg_b"]).reset_index(drop=True)
    if res_df.empty:
        logging.warning("[run_wilcoxon_tests_all_metrics] No pairwise results produced.")
        return res_df

    # Holm corrections: global (across *all* tests) and within each metric
    res_df["p_holm_global"] = _holm_adjust(res_df["p_two_sided"].tolist())
    res_df["p_holm_within_metric"] = np.nan
    for m, idx in res_df.groupby("metric").groups.items():
        p_adj = _holm_adjust(res_df.loc[idx, "p_two_sided"].tolist())
        res_df.loc[idx, "p_holm_within_metric"] = p_adj

    # Save artifacts
    csv_path = out_dir / "wilcoxon_pairwise_all.csv"
    json_path = out_dir / "wilcoxon_pairwise_all.json"
    res_df.to_csv(csv_path, index=False)
    res_df.to_json(json_path, orient="records", indent=4)
    logging.info(f"[run_wilcoxon_tests_all_metrics] Wrote {csv_path} and {json_path}.")

    # Pretty table (metric included)
    pretty = res_df[[
        "metric", "alg_a", "alg_b", "n_common", "wins_a", "wins_b", "ties",
        "win_rate_a_over_b", "mean_diff_a_minus_b", "median_diff_a_minus_b",
        "W_stat", "p_two_sided", "p_holm_within_metric", "p_holm_global", "effect_size_r"
    ]]
    print("\n=== Wilcoxon signed-rank (two-sided), across ALL metrics (mean_*), per dataset ===")
    print(tabulate(pretty, headers="keys", tablefmt="github", floatfmt=".4f"))
    return res_df


def schedule_and_run(datasets: list[Path], args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    """
    Launch multiprocessing workers to process all dataset/algorithm combinations.

    Parameters
    ----------
    datasets : list of Path
        Resolved dataset directories discovered at runtime.
    args : argparse.Namespace
        Parsed CLI arguments containing training hyperparameters.
    out_dir : Path
        Root directory for algorithm-specific subfolders and summaries.

    Returns
    -------
    alg_summary : dict
        Mapping from algorithm name to the progressively updated summary dictionary.

    Raises
    ------
    RuntimeError
        Propagated if pool execution fails.

    Notes
    -----
    Workers equal the number of GPUs requested; each task is assigned a GPU id in
    round-robin order. Intermediate summaries are written after every task completion.

    Mathematical sketch
    --------------------
    No statistical computation is performed here; the scheduler simply partitions the
    Cartesian product of datasets and algorithms into `ngpus` ordered queues.

    Examples
    --------
    >>> schedule_and_run([Path('data/sample')], argparse.Namespace(gpus=1, algorithms=['mlp']), Path('./runs'))  # doctest: +SKIP
    """
    ngpus = max(1, args.gpus)
    base_seed = args.seed

    # Build task list with round-robin GPU assignment
    tasks = []
    for alg in args.algorithms:
        for d in sorted(datasets, key=lambda p: p.name.lower()):
            name = d.name
            gid = (len(tasks)) % ngpus
            tasks.append((d, name, alg, gid, str(out_dir / alg), base_seed, args))

    total = len(tasks)
    logger.info(f"Launching Pool with {ngpus} workers over {total} tasks...")

    results: list[dict[str, Any]] = []
    ctx = get_context("spawn")

    # Aggregate summary JSON

    json_save_paths = dict()
    alg_summary = dict()
    for alg in args.algorithms:
        json_save_paths[alg] = out_dir / alg / f"summary.json"

        summary = {
            "metadata": args.__dict__,
            "datasets": {},
            "overall": {}
        }

        summary['metadata'].update({'gpus': ngpus, 'n_tasks': total, 'run_dir': os.getcwd()})

        alg_summary[alg] = summary.copy()

    del summary

    with ctx.Pool(processes=ngpus) as pool:
        # Iterate as results come in
        for i, (status, payload) in enumerate(pool.imap_unordered(_run_task, tasks, chunksize=1), start=1):
            if status == "ok":
                results.append(payload)

                alg = payload['algorithm']

                alg_summary[alg]['datasets'].update({payload['dataset']: payload})

                with open(json_save_paths[alg], 'w') as f:
                    json.dump(alg_summary[alg], f, indent=4)

            else:
                logger.error(
                    f"[{status}] for dataset {payload['dataset']} using {payload['algorithm']}: {payload.get('traceback')}")

            if i % 5 == 0 or i == total:
                logger.info(f"[Progress] {i}/{total} tasks complete.")

    logger.info(f"\nWrote summary JSONs to: {json_save_paths}")

    return alg_summary


def parse_args() -> argparse.Namespace:
    """
    Build the command-line interface and parse user arguments.

    Returns
    -------
    args : argparse.Namespace
        Parsed configuration with attributes documented in the help strings.

    Raises
    ------
    SystemExit
        If parsing fails or help text is requested.

    Notes
    -----
    Defaults mirror the experimental setup described in the project README.

    Mathematical sketch
    --------------------
    Not applicable.

    Examples
    --------
    >>> _ = parse_args()  # doctest: +SKIP
    """
    p = argparse.ArgumentParser(description="Train MLP on JKU UCI collection with 10-fold CV and TensorBoard logging.")
    p.add_argument("--data-root", type=str, default='../data/UCI',
                   help="Root directory with data/<dataset_name>/ subfolders.")
    p.add_argument("--out", type=str, default='../runs/UCI', help="Output directory for TensorBoard runs and JSON.")
    p.add_argument("--gpus", type=int, default=torch.cuda.device_count(),
                   help="Number of GPUs to use concurrently (one process per GPU).")
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--verbosity", type=str, choices=['debug', 'error', 'fatal', 'info', 'warn'], default='info',
                   help="Logging verbosity.")
    p.add_argument("--datasets", nargs="*", default=None,
                   help="Optional explicit list of dataset names to run (match folder names under data_root).")

    p.add_argument("--algorithms", type=str, nargs='+',
                   default=['sdp', 'ldlt', 'ldlt-resnet', 'aol', 'ortho', 'sandwich'],
                   choices=['sdp', 'ldlt', 'ldlt-resnet', 'aol', 'ortho', 'sandwich'])

    p.add_argument("--depth", type=int, default=4, help="The depth of the networks")

    return p.parse_args()


def setup_logging(config: argparse.Namespace, log_folder: Path) -> None:
    """
    Configure module-level logging to stream and file handlers.

    Parameters
    ----------
    config : argparse.Namespace
        Parsed arguments whose `verbosity` attribute selects the log level.
    log_folder : Path
        Directory where `logs.txt` will be written.

    Returns
    -------
    None

    Raises
    ------
    OSError
        If the log file cannot be created.

    Notes
    -----
    Formats log records with timestamps and process/thread identifiers for debugging
    multi-processing runs.

    Mathematical sketch
    --------------------
    Not applicable.

    Examples
    --------
    >>> setup_logging(argparse.Namespace(verbosity='info'), Path('.'))  # doctest: +SKIP
    """
    log_file = log_folder / 'logs.txt'

    level = {'DEBUG': 10, 'ERROR': 40, 'FATAL': 50,
             'INFO': 20, 'WARN': 30}[config.verbosity.upper()]
    format_ = "[%(asctime)s %(process)d:%(thread)d %(filename)s:%(lineno)s:%(module)s:%(funcName)s] %(message)s"
    logging.basicConfig(level=level, format=format_, datefmt='%H:%M:%S',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)])


def main() -> None:
    """
    Entry point that discovers datasets, launches training, and runs statistics.

    Returns
    -------
    None

    Raises
    ------
    SystemExit
        If no valid datasets are found.

    Notes
    -----
    Sets the multiprocessing start method to ``"spawn"`` for CUDA compatibility.

    Mathematical sketch
    --------------------
    Not applicable.

    Examples
    --------
    >>> main()  # doctest: +SKIP
    """
    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(args, out_dir)

    # Discover dataset folders
    all_ds = []
    root = data_root
    if (root / "data").exists():
        root = root / "data"
    else:
        raise FileNotFoundError(f"{root} does not exist")

    for p in root.iterdir():
        if p.is_dir() and not p.name.startswith("."):
            # Consider dataset folder only if required files exist
            folds_ok = (p / "folds_py.dat").exists()
            labels_ok = (p / "labels_py.dat").exists()
            feat_ok = (p / f"{p.name}_py.dat").exists()

            if folds_ok and labels_ok and feat_ok:
                all_ds.append(p)
            else:
                logger.warning(
                    f'Skipping {p} as it is missing data folds: {folds_ok}, labels: {labels_ok}, and feats: {feat_ok}')

    # Filter by user-provided list, if any
    if args.datasets:
        wanted = set(args.datasets)
        all_ds = [p for p in all_ds if p.name in wanted]
        missing = wanted - {p.name for p in all_ds}
        if missing:
            logger.warning(f"Requested datasets not found: {sorted(missing)}")

    if not all_ds:
        logger.error(f"No valid datasets found under {root}.")
        sys.exit(1)

    logger.info(f"Discovered {len(all_ds)} datasets.")

    summary = schedule_and_run(all_ds, args, out_dir)

    wilcoxon_prep_all_metrics(out_dir, args.algorithms)
    run_wilcoxon_tests_all_metrics(out_dir, args.algorithms)

    logger.info(f"TensorBoard logs under: {out_dir / 'runs'}")
    logger.info(f"To view: tensorboard --logdir {str(out_dir / 'runs')}")


if __name__ == "__main__":
    main()
