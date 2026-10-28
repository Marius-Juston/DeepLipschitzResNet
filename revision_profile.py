#!/usr/bin/env python3
"""Measure complete forward/backward steps, including normalization, without training data."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from models.linear_model import DeepLipschitzLinearResNet
from revision_image import ImageModel


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cuda:0');p.add_argument('--steps',type=int,default=20)
    p.add_argument('--out',default='revision_artifacts/profile.json')
    p.add_argument('--quick',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    def sync():
        if device.type=='cuda': torch.cuda.synchronize(device)
    rows=[]
    cases=[]
    for width in ([8] if args.quick else [32,64,128,256]):
        for depth in ([2] if args.quick else [1,2,4,8]):
            cases.append((f'dense-w{width}-d{depth}',
                          lambda w=width,d=depth:DeepLipschitzLinearResNet(w,w,(w,)*d), (64,width)))
    for alg in ['ldlt-resnet','sandwich','sdp']:
        cases.append((f'image-{alg}',lambda a=alg:ImageModel(a), (2 if args.quick else 64,3,32,32)))
    for name,make,shape in cases:
        try:
            model=make().to(device);x=torch.randn(shape,device=device)
            times=[];gradient_norms=[]
            for i in range(3+args.steps):
                model.train();model.zero_grad(set_to_none=True)
                if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
                sync();start=time.perf_counter()
                prediction=model(x); prediction.square().mean().backward()
                sync();elapsed=time.perf_counter()-start
                if i>=3:
                    times.append(elapsed)
                    gradient_norms.append(sum(float(p.grad.detach().double().square().sum()) for p in model.parameters() if p.grad is not None)**.5)
            row=dict(name=name,shape=shape,parameters=sum(p.numel() for p in model.parameters()),
                     median_forward_backward_seconds=statistics.median(times),
                     min_forward_backward_seconds=min(times),max_forward_backward_seconds=max(times),
                     gradient_norms=gradient_norms,
                     peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None)
        except (RuntimeError,ValueError) as e: row=dict(name=name,error=str(e))
        rows.append(row); print(json.dumps(row),flush=True)
        path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(dict(device=str(device),torch=torch.__version__,steps=args.steps,
                                       note='Training-mode forward/backward including factorization; no optimizer step. Warmup 3 iterations. CPU memory not measured.',rows=rows),indent=2))

if __name__=='__main__': main()
