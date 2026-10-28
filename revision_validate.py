#!/usr/bin/env python3
"""Generate reproducible numerical diagnostics and genuine vector figures."""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from models.certified_ldl import residual_weights, assemble_lmi


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cpu')
    p.add_argument('--out',default='revision_artifacts')
    p.add_argument('--figures',default='../LDL-Lipschitz-ResNet-LMI/Figures/revision')
    args=p.parse_args()
    out,figures=Path(args.out),Path(args.figures)
    out.mkdir(parents=True,exist_ok=True); figures.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    records=[]
    for dtype in [torch.float32,torch.float64]:
        for depth in [1,2,4,8]:
            for scale in [.1,1.,10.,100.]:
                torch.manual_seed(42)
                width=8
                raw_a=torch.randn(4,4,dtype=dtype,device=args.device)*scale
                raw_cs=[torch.randn(width,4 if i==0 else width,dtype=dtype,device=args.device)*scale for i in range(depth)]
                raw_b=torch.randn(4,width,dtype=dtype,device=args.device)*scale
                row=dict(dtype=str(dtype),depth=depth,raw_scale=scale,width=width)
                try:
                    if args.device.startswith('cuda'): torch.cuda.synchronize()
                    start=time.perf_counter()
                    w=residual_weights(raw_a,raw_cs,raw_b)
                    if args.device.startswith('cuda'): torch.cuda.synchronize()
                    row['construction_seconds']=time.perf_counter()-start
                    m=assemble_lmi(w.a.cpu().double(),[c.cpu().double() for c in w.cs],w.b.cpu().double())
                    eig=torch.linalg.eigvalsh(m)
                    row.update(min_eigenvalue=float(eig.min()),max_eigenvalue=float(eig.max()),
                               max_pivot_condition=max(float(torch.linalg.cond(d.double())) for d in w.leading_pivots),
                               success=bool(eig.min()>0))
                except torch.linalg.LinAlgError as error:
                    row.update(success=False,error=str(error).splitlines()[0])
                records.append(row)
    payload=dict(seed=42,device=args.device,torch=torch.__version__,python=platform.python_version(),
                 cpu=platform.processor(),threads=4,records=records,
                 note='Stress diagnostics, not benchmark accuracy or interval certification; first-call timing includes warmup.')
    (out/'validation.json').write_text(json.dumps(payload,indent=2,allow_nan=False))
    plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.size':10})
    fig,ax=plt.subplots(figsize=(5.2,3.6),layout='constrained')
    c=np.linspace(-np.sqrt(2),np.sqrt(2),1001); a=.8
    center=-a*c; half=np.sqrt((1-a*a)*np.maximum(0,2-c*c))
    ax.fill_between(c,center-half,center+half,color='#2677b8',alpha=.2,label='Exact LMI-feasible set')
    ax.plot(c,center-half,color='#2677b8'); ax.plot(c,center+half,color='#2677b8')
    ax.plot(c,center,'--',color='#2677b8',label=r'Center $B_0=-AC_1$')
    for k,sign in enumerate([-1,1]):
        ax.axvline(sign*np.sqrt(2*(1-a*a)),color='#c44928',linestyle=':',
                   label=r'Additional restriction $C_1^2<2(1-A^2)$' if k==0 else None)
    ax.set(xlabel=r'$C_1$',ylabel=r'$B$',title=r'Scalar residual certificate: $A=0.8$, $\mathcal{L}=1$')
    ax.legend(fontsize=8,loc='upper right'); ax.grid(alpha=.15)
    fig.savefig(figures/'scalar_feasible_set.pdf'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(5.2,3.6),layout='constrained')
    for dtype,marker in [('torch.float32','o'),('torch.float64','s')]:
        rows=[r for r in records if r['dtype']==dtype and r['raw_scale']==10. and 'min_eigenvalue' in r]
        ax.plot([r['depth'] for r in rows],[r['min_eigenvalue'] for r in rows],marker=marker,label=dtype)
    ax.axhline(0,color='black',linewidth=.6)
    ax.set(xlabel='Hidden depth',ylabel='Minimum eigenvalue of original LMI',
           title='Effective-weight certificate under a fixed stress scale')
    ax.set_yscale('symlog',linthresh=1e-7); ax.legend();ax.grid(alpha=.15)
    fig.savefig(figures/'conditioning_diagnostic.pdf');plt.close(fig)
    print(json.dumps(dict(cases=len(records),positive_certificate_cases=sum(r['success'] for r in records),
                          report=str(out/'validation.json'),figures=str(figures)),indent=2))


if __name__=='__main__':main()
