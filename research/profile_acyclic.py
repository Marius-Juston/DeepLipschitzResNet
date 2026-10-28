"""Bounded CPU implementation ablation on IDENTICAL functions/free parameters.

This compares direct factor evaluation with materializing physical weights from
the SAME parameterization. It is not a benchmark against an optimized LBEN,
Sandwich, or Wang training implementation, and does not establish such a speedup.
Run: uv run --locked python research/profile_acyclic.py
"""
import argparse
import json
from pathlib import Path
import platform
import statistics
import time

import torch
from acyclic import parameterize


def explicit(x, cs, output):
    states=x
    for c in cs:
        states=torch.cat([states,torch.tanh(states@c.T)],dim=-1)
    return states@output.T


def measure(fn, repeats):
    for _ in range(3):fn()
    times=[]
    for _ in range(repeats):
        before=time.perf_counter()
        fn()
        times.append(1000*(time.perf_counter()-before))
    return dict(median_ms=statistics.median(times),min_ms=min(times),max_ms=max(times))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',default='runs/revision_acyclic/profile_cpu.json')
    parser.add_argument('--repeats',type=int,default=20)
    args=parser.parse_args()
    torch.set_num_threads(1)
    results=[]
    for width in (16,32):
        for depth in (2,4,8,16):
            torch.manual_seed(523)
            raw=[(torch.randn(width,i*width,dtype=torch.float64)*.1/(i*width)**.5).requires_grad_()
                 for i in range(1,depth+1)]
            ro=(torch.randn(4,(depth+1)*width,dtype=torch.float64)*.1/((depth+1)*width)**.5).requires_grad_()
            x=torch.randn(64,width,dtype=torch.float64)
            factors=parameterize(raw,ro)
            cs,o=factors.weights()
            error=float((factors.forward(x)-explicit(x,cs,o)).detach().abs().max())
            if error>1e-11:raise RuntimeError('Function mismatch')
            def train(materialize):
                factors=parameterize(raw,ro)
                y=explicit(x,*factors.weights()) if materialize else factors.forward(x)
                torch.autograd.grad(y.square().mean(),[*raw,ro])
            row=dict(width=width,depth=depth,batch=64,outputs=4,
                     parameters=sum(v.numel() for v in [*raw,ro]),max_output_error=error)
            row['factor_train']=measure(lambda:train(False),args.repeats)
            row['materialized_train']=measure(lambda:train(True),args.repeats)
            with torch.no_grad():
                factors=parameterize(raw,ro);cs,o=factors.weights()
                row['factor_frozen_inference']=measure(lambda:factors.forward(x),args.repeats)
                row['physical_frozen_inference']=measure(lambda:explicit(x,cs,o),args.repeats)
            results.append(row)
            print(f"width={width} depth={depth}: training materialized/factor "
                  f"{row['materialized_train']['median_ms']/row['factor_train']['median_ms']:.2f}x",flush=True)
    artifact=dict(purpose=__doc__,torch_version=torch.__version__,platform=platform.platform(),
                  threads=torch.get_num_threads(),dtype='float64',device='cpu',repeats=args.repeats,
                  results=results)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(artifact,indent=2)+'\n')


if __name__=='__main__':main()
