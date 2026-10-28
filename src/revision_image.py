"""Matched small CIFAR comparison; raw-pixel l2 certificates, no online services."""
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision import transforms

from certification import certified_radii, clear_weight_caches, audit_dense_certificate
from models.certified_conv import CertifiedCircularResidual
from models.certified_ldl import contraction, assemble_lmi
from models.linear_model import DeepLipschitzLinearResNet
from lbdn_models.layers import SandwichConv, SandwichFc, LinearNormalized
from sll_models.layers import SDPBasedLipschitzConvLayer, SDPBasedLipschitzLinearLayer


class Projection(nn.Linear):
    def forward(self, x):
        # Factor the smaller Gram matrix when reducing dimensions.
        return F.linear(x, contraction(self.weight.T).T, self.bias)


class ImageModel(nn.Module):
    def __init__(self, algorithm, channels=16, features=64):
        super().__init__()
        self.channels = channels
        if algorithm == 'ldlt-resnet':
            self.conv = nn.Sequential(*[CertifiedCircularResidual(channels, depth=2) for _ in range(2)])
            self.dense = DeepLipschitzLinearResNet(features, features, [features, features], device='cpu')
        elif algorithm == 'sandwich':
            self.conv = nn.Sequential(*[SandwichConv(channels, channels, 3, padding=1) for _ in range(4)])
            self.dense = nn.Sequential(*[SandwichFc(features, features) for _ in range(2)])
        elif algorithm == 'sdp':
            self.conv = nn.Sequential(*[SDPBasedLipschitzConvLayer(channels, channels) for _ in range(4)])
            self.dense = nn.Sequential(*[SDPBasedLipschitzLinearLayer(features, features) for _ in range(2)])
        else:
            raise ValueError(algorithm)
        self.projection = Projection(channels * 8 * 8, features)
        self.head = LinearNormalized(features, 10)

    def forward(self, x):
        # Translation, channel zero-padding, and normalized disjoint pooling
        # each have Lipschitz constant one in raw pixel Euclidean coordinates.
        x = F.pad(x - .5, (0, 0, 0, 0, 0, self.channels-3))
        x = self.conv(x)
        x = F.avg_pool2d(x, 4, divisor_override=4).flatten(1)
        return self.head(self.dense(self.projection(x)))

    def head_weight(self):
        return F.normalize(self.head.weight, dim=1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    clear_weight_caches(model)
    correct, total, radii_all = 0, 0, []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
        radii_all.append(certified_radii(logits, y, model.head_weight()).cpu())
    radii = torch.cat(radii_all)
    return dict(clean=correct/total, certified={str(e):float((radii > e/255).float().mean()) for e in (36,72,108)},
                radius_quantiles={str(q):float(torch.quantile(radii, q)) for q in (.1,.5,.9)}, n=total)


def run_image(job, output: Path, data_root: Path, device):
    """Train with a fixed stratified validation split; select on validation only."""
    seed = job['seed']
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    basic = transforms.ToTensor()
    augmented = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), basic])
    train = CIFAR10(data_root, train=True, transform=augmented, download=False)
    val = CIFAR10(data_root, train=True, transform=basic, download=False)
    test = CIFAR10(data_root, train=False, transform=basic, download=False)
    # Same split for every architecture and seed, fixed before any training.
    rng = np.random.default_rng(2026)
    labels = np.asarray(train.targets)
    val_idx, train_idx = [], []
    for cls in range(10):
        idx = rng.permutation(np.flatnonzero(labels == cls))
        val_idx.extend(idx[:500]); train_idx.extend(idx[500:])
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output/'split.npz', train=np.asarray(train_idx), validation=np.asarray(val_idx))
    batch = job.get('batch_size', 128)
    training = DataLoader(Subset(train, train_idx), batch_size=batch, shuffle=True, num_workers=0)
    validation = DataLoader(Subset(val, val_idx), batch_size=batch, num_workers=0)
    testing = DataLoader(test, batch_size=batch, num_workers=0)
    model = ImageModel(job['algorithm']).to(device)
    # Some published Fourier baseline layers lazily register their scale.
    model(torch.zeros(1, 3, 32, 32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = job['epochs']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    first, best_score, best_state, best_epoch = 0, -1., None, -1
    resume = output/'resume.pth'
    if resume.exists():
        state = torch.load(resume, map_location=device, weights_only=False)
        if state['job'] != job:
            raise ValueError('Checkpoint job mismatch')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        first, best_score, best_state, best_epoch = state['epoch']+1, state['score'], state['best'], state['best_epoch']
        torch.set_rng_state(state['rng'].cpu()); np.random.set_state(state['np_rng']); random.setstate(state['py_rng'])
        if device.type == 'cuda': torch.cuda.set_rng_state(state['cuda_rng'].cpu(), device)
    started = time.monotonic()
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(first, epochs):
        model.train()
        for x, y in training:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            temperature = 1.
            if job['loss'] == 'margin':
                head = model.head_weight()
                distances = torch.linalg.vector_norm(head[y,None,:]-head[None,:,:], dim=-1)
                epsilon = (72/255) * min(1., (epoch+1)/20)
                logits = logits + epsilon * distances
                temperature = .25
            loss = temperature * F.cross_entropy(logits / temperature, y)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite image loss')
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
        scheduler.step()
        metrics = evaluate(model, validation, device)
        score = metrics['clean'] if job['loss'] == 'ce' else metrics['certified']['72']
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            torch.save(best_state, output/'best.pth')
        snapshot = dict(job=job, model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                        epoch=epoch, score=best_score, best=best_state, best_epoch=best_epoch,
                        rng=torch.get_rng_state(), np_rng=np.random.get_state(), py_rng=random.getstate(),
                        cuda_rng=torch.cuda.get_rng_state(device) if device.type=='cuda' else None)
        torch.save(snapshot, output/'resume.tmp'); (output/'resume.tmp').replace(resume)
        (output/'progress.json').write_text(json.dumps(dict(epoch=epoch+1, epochs=epochs, validation=metrics)))
    model.load_state_dict(best_state)
    audits = []
    if job['algorithm'] == 'ldlt-resnet':
        model.eval()
        with torch.no_grad():
            for block in model.conv:
                w = block.normalized_weights((32,32))
                # Validate the effective symbols, not newly normalized double parameters.
                matrix = assemble_lmi(w.a.to(torch.complex128), [c.to(torch.complex128) for c in w.cs],
                                      w.b.to(torch.complex128))
                minimum = float(torch.linalg.eigvalsh(matrix).min())
                if minimum <= 0: raise RuntimeError(f'Nonpositive convolution certificate margin: {minimum}')
                audits.append(dict(min_eigenvalue=minimum, frequencies=1024))
            audits.append(audit_dense_certificate(model.dense))
    metrics = evaluate(model, testing, device)
    return dict(protocol_version=2, dataset='cifar10', algorithm=job['algorithm'], loss=job['loss'], seed=seed,
                parameters=sum(p.numel() for p in model.parameters()), best_epoch=best_epoch,
                elapsed_seconds_this_session=time.monotonic()-started,
                peak_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
                input_coordinates='raw [0,1] pixels', certificate_audits=audits, metrics=metrics)
