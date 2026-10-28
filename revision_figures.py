#!/usr/bin/env python3
"""Regenerate source-based vector figures; never wrap or trace raster images."""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'test')]
import model_lmi as plots
from models.linear_model import DeepLipschitzLinearResNet
from models.certified_conv import CertifiedCircularResidual
from models.certified_ldl import assemble_lmi
from numerical_instability import main as stability


def main():
    torch.set_num_threads(4)
    torch.manual_seed(0)
    figures = ROOT.parent/'LDL-Lipschitz-ResNet-LMI'/'Figures'
    out = figures/'regenerated'
    linear = out/'Linear'; linear.mkdir(parents=True, exist_ok=True)
    conv = out/'CNN'; conv.mkdir(parents=True, exist_ok=True)
    model = DeepLipschitzLinearResNet(1, 1, (64,)*5, dtype=torch.float64, device='cpu')
    with torch.no_grad():
        w = model.normalized_weights()
        # Retain the original plotting convention M <= 0.
        matrix = -assemble_lmi(w.a, w.cs, w.b)
        plots.plot_information(matrix, w.a, w.cs, w.b, [1]+[64]*5, str(linear))
        plots.model_artifacting(DeepLipschitzLinearResNet, (64,)*5,
                               fig_name=str(linear/'b_artifacting_grid'),
                               factory_kwargs={'dtype':torch.float64, 'device':'cpu'})
        np.savez_compressed(linear/'source_data.npz', a=w.a.cpu().numpy(), b=w.b.cpu().numpy(),
                            matrix=matrix.cpu().numpy(), **{f'c{i}':c.cpu().numpy() for i,c in enumerate(w.cs)})
        cnn = CertifiedCircularResidual(1, 32, depth=5, dtype=torch.float64, device='cpu')
        wc = cnn.normalized_weights((16,16))
        mc = -assemble_lmi(wc.a, wc.cs, wc.b)
        plots.plot_histogram(torch.linalg.eigvalsh(mc).flatten(), str(conv/'cnn_eigenvalues_lmi'))
        # Batched singular extremes are summarized across all frequencies.
        matrices = [wc.a, *wc.cs, wc.b]
        sv = [torch.linalg.svdvals(m) for m in matrices]
        fig, ax = plt.subplots()
        ax.plot([v.min().item() for v in sv], 'o-', label='Minimum across frequencies')
        ax.plot([v.max().item() for v in sv], 's-', label='Maximum across frequencies')
        ax.set(xlabel='Weight index (A, C1,...,C5, B)', ylabel='Singular value')
        ax.legend(); fig.tight_layout(); fig.savefig(conv/'cnn_singular_weights.pdf'); plt.close(fig)
        mf = mc[0,0].real
        plots.plot_zoomed_in_system(mf-torch.diag(torch.diag(mf)),
                                   (wc.b[0,0].mH@wc.b[0,0]).real,
                                   str(conv/'inplace_zoomed_in_b_0_real'))
        np.savez_compressed(conv/'source_data.npz', a=wc.a.cpu().numpy(), b=wc.b.cpu().numpy(),
                            matrix=mc.cpu().numpy(), **{f'c{i}':c.cpu().numpy() for i,c in enumerate(wc.cs)})
    # This original figure is formula-based and can be regenerated directly.
    oldcwd = Path.cwd()
    os.chdir(figures/'LDL')
    try: stability()
    finally: os.chdir(oldcwd)
    produced = sorted([*out.rglob('*.pdf'), *(figures/'LDL').glob('numerical_stability_n_*.pdf'),
                       *(figures/'revision').glob('*.pdf')])
    for path in produced:
        result = subprocess.run(['pdfimages','-list',str(path)], check=True, capture_output=True, text=True)
        if len(result.stdout.strip().splitlines()) > 2:
            raise RuntimeError(f'Embedded raster image in {path}: {result.stdout}')
    manifest = {
        'vector_verified': [str(p.relative_to(figures)) for p in produced],
        'method': 'Native PDF paths/text; matrix cells use non-rasterized pcolormesh. Verified with pdfimages.',
        'original_formula_regenerated': 'LDL/numerical_stability_n_5.pdf (historical formula, not a validated theorem)',
        'corrected_model_replacements': 'regenerated/ contains new version-2 model plots, not exact reproductions of historical PNG data; seed 0, float64, saved numeric arrays.',
        'pending_original_sources': ['Six Gershgorin figures: generator/data not located',
                                     'trained_network and sine_* figures: matching training checkpoint/plot data not located'],
        'publication_policy': 'Only Figures/revision is currently used in the JMLR manuscripts; historical diagnostic plots are not empirical validation.'}
    (figures/'vector_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))

if __name__ == '__main__': main()
