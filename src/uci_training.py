#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import json
import logging
import math
import os
import random
import sys
import traceback
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.multiprocessing import set_start_method
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.linear_model import DeepLipschitzSequential, DeepLipschitzLinearResNet
from sll_models.model import LipschitzNetworkSequential

logger = logging.getLogger(__name__)


def load_dat(path: Path):
    return np.loadtxt(path, delimiter=',')


class ArrayDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, indices: np.ndarray):
        self.X = torch.from_numpy(X).to(torch.float32)

        n_classes = y.max() + 1

        self.y = torch.zeros((y.shape[0], n_classes), dtype=torch.float32)
        #  One hot encoding
        self.y[np.arange(y.shape[0]), y] = 1

        self.idx = indices.astype(np.int64, copy=False)

    def __len__(self):
        return self.idx.shape[0]

    def __getitem__(self, i):
        j = self.idx[i]
        return self.X[j].squeeze(0), self.y[j].squeeze(0)


def standardize_inplace(X: np.ndarray, train_idx: np.ndarray, eps: float = 1e-8) -> Dict[str, np.ndarray]:
    """
    Standardize features using mean/std computed on training examples only.
    Returns the stats for reproducibility.
    """
    mu = X[train_idx].mean(axis=0)
    sigma = X[train_idx].std(axis=0)
    sigma[sigma < eps] = 1.0
    X[:] = (X - mu) / sigma
    return {"mean": mu, "std": sigma}


def class_weights_from_indices(y: np.ndarray, train_idx: np.ndarray) -> torch.Tensor:
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
    def __init__(self, d_in: int, n_classes: int, width: int, depth: int = 4, dropout: float = 0.2):
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

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)


def suggested_width(d_in: int, n_classes: int) -> int:
    # heuristic tailored for small-to-mid tabular datasets
    base = max(32, min(512, 4 * d_in))
    if n_classes > 10:
        base = int(base * 1.25)
    # round to nearest power of two-ish bucket
    buckets = [32, 64, 128, 256, 384, 512]
    return min(buckets, key=lambda b: abs(b - base))


# ------------- Training ------------- #

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def eval_loop(model, loader, device):
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


def convert_cert_vals(cert_names: List[float], cert_vals: Tensor):
    return {
        str(int(n)): v.item() for n, v in zip(cert_names, cert_vals)
    }


@torch.no_grad()
def eval_certified(model, loader, dtype=None, device=None):
    cert_values = [36., 72., 108., 255.]

    eps_float = torch.tensor(cert_values, dtype=dtype, device=device) / 255.
    model.eval()
    running_accuracy = 0

    running_certified = torch.zeros_like(eps_float)

    running_inputs = 0
    lip_cst = 1.

    if isinstance(model, DeepLipschitzSequential):
        normalized_weight = model.last_weight()
    elif isinstance(model, DeepLipschitzLinearResNet):
        normalized_weight = model.last_weight()
    elif isinstance(model, LipschitzNetworkSequential):
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
        margins = torch.diff(margins)

        margins /= torch.norm(normalized_weight[indices[:, None, -1]] - normalized_weight[indices[:, 0:-1]], dim=-1,
                              p=2)

        # for idx in range(margins.shape[0]):
        #     margins[idx] /= torch.norm(
        #         normalized_weight[indices[idx, -1]] - normalized_weight[indices[idx, 0:-1]], dim=1, p=2)
        margins, _ = torch.sort(margins, 1)
        certified = margins[:, 0, None] > eps_float * lip_cst
        running_accuracy += correct.sum()
        running_certified += torch.sum(torch.unsqueeze(correct, -1) & certified, dim=0)
        running_inputs += inputs.size(0)

    accuracy = running_accuracy / running_inputs
    certified = running_certified / running_inputs
    return cert_values, accuracy, certified


def train_one_fold(
        X: np.ndarray,
        y: np.ndarray,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        val_idx: np.ndarray,
        algorithm: str,
        seed: int,
        log_dir: Path,
        hparams: Dict[str, Any],
        max_epochs: int = 300,
        patience: int = 30,
        batch_size: int = 128,
        num_workers: int = 8,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        dtype=None,
        device: torch.device = None,
) -> Dict[str, Any]:
    """
    Train on folds != {test_fold, val_fold}, validate on val_fold, test on test_fold.
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

    depth = 4

    if algorithm == 'mlp':
        model = MLP(d_in, n_classes, width=width, depth=depth, dropout=0.2).to(device=device, dtype=dtype)
    elif algorithm == 'sdp':
        padding = max(d_in * 2, 10)

        model = LipschitzNetworkSequential(cin=d_in, n_dense=depth, padding=padding, dense_inner_dim=width,
                                           n_classes=n_classes).to(device=device, dtype=dtype)
    else:
        raise ValueError("Please provide a valid algorithm name 'mlp'")

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
        'cert_evals': cert_dict_
    }


# ------------- Worker: trains one dataset for one repeat (10 folds) ------------- #

def load_dataset_bundle(dataset_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def _run_task(pack) -> Tuple[str, Dict[str, Any]]:
    """
    pack = (dataset_dir, dataset_name, repeat_id, gpu_id, out_dir, base_seed, args_dict)
    Returns ("ok", result) or ("error", errdict).
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

def wilcoxon_prep(alg_summary: Dict[str, Dict[str, Any]], out_dir: Path):
    # Wilcoxon-friendly flattened table
    wilcoxon_ready = []
    for alg_name, alg in alg_summary.items():
        for ds_name, ds_entry in alg["datasets"].items():
            mean_val = ds_entry["mean_test_acc"]
            if mean_val is not None:
                wilcoxon_ready.append({"dataset": ds_name, "algorithm": alg_name, "mean_accuracy": mean_val})

    with open(out_dir / "wilcoxon_prep.json", "w") as f:
        json.dump(wilcoxon_ready, f, indent=4)


def schedule_and_run(datasets: List[Path], args: argparse.Namespace, out_dir: Path) -> Dict[str, Any]:
    """
    Use a Pool with 'spawn' context. Pool size == args.gpus (one concurrent task per GPU).
    Each task (dataset, repeat) is pre-assigned a GPU id in round-robin.
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

    results: List[Dict[str, Any]] = []
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


def parse_args():
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

    p.add_argument("--algorithms", type=str, nargs='+', default=['mlp'], choices=['mlp', 'sdp'])

    return p.parse_args()


def setup_logging(config, log_folder):
    log_file = log_folder / 'logs.txt'

    level = {'DEBUG': 10, 'ERROR': 40, 'FATAL': 50,
             'INFO': 20, 'WARN': 30}[config.verbosity.upper()]
    format_ = "[%(asctime)s %(process)d:%(thread)d %(filename)s:%(lineno)s:%(module)s:%(funcName)s] %(message)s"
    logging.basicConfig(level=level, format=format_, datefmt='%H:%M:%S',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)])


def main():
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

    wilcoxon_prep(summary, out_dir)

    logger.info(f"TensorBoard logs under: {out_dir / 'runs'}")
    logger.info(f"To view: tensorboard --logdir {str(out_dir / "runs")}")


if __name__ == "__main__":
    main()
