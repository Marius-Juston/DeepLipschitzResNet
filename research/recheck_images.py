"""Fresh checkpoint evaluation with an independent all-class certificate formula.

Reads all 18 completed CIFAR checkpoints, writes separate audit/radius artifacts,
and never trains or changes the frozen campaign. Numerical LMI checks remain
diagnostics, not interval-arithmetic guarantees.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'src'))
from revision_image import ImageModel
from certification import clear_weight_caches


def original_matrix(weights,bound=1.):
    """Assemble the original quadratic form, without the production assembler."""
    a,cs,b=weights.a,weights.cs,weights.b
    dtype=torch.complex128 if a.is_complex() else torch.float64
    a,b=a.to(dtype),b.to(dtype);cs=[c.to(dtype) for c in cs]
    dims=[a.shape[-1],*[c.shape[-2] for c in cs]]
    offsets=np.cumsum([0,*dims]).tolist();n=offsets[-1]
    mat=a.new_zeros((*a.shape[:-2],n,n))
    mat[...,:dims[0],:dims[0]]=bound**2*torch.eye(dims[0],device=a.device,dtype=dtype)
    for i,c in enumerate(cs,1):
        sl=slice(offsets[i],offsets[i+1]);prior=slice(offsets[i-1],offsets[i])
        mat[...,sl,sl]=2*torch.eye(dims[i],device=a.device,dtype=dtype)
        mat[...,sl,prior]=-c;mat[...,prior,sl]=-c.mH
    out=a.new_zeros((*a.shape[:-2],a.shape[-2],n))
    out[...,:,:dims[0]]=a;out[...,offsets[-2]:]=b
    return mat-out.mH@out


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--out',default='revision_artifacts/checkpoint_reevaluation')
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    device=torch.device(args.device);run=ROOT/'runs/revision_campaign'
    manifest=json.loads((run/'manifest.json').read_text())
    from revision_experiments import code_digest
    assert code_digest()==manifest['code_sha256']
    output=ROOT/args.out;output.mkdir(parents=True,exist_ok=True)
    data=CIFAR10(manifest['image_data'],train=False,transform=ToTensor(),download=False)
    loader=DataLoader(data,batch_size=128,shuffle=False,num_workers=0)
    started=time.monotonic();rows=[]
    summary=dict(source_sha256=manifest['code_sha256'],device=str(device),torch=torch.__version__,
                 complete=False,rows=rows,method='Fresh state_dict loads; explicit competing-class radii in float64; independent original-QC block assembly on effective weights.')
    for job in manifest['jobs']:
        if job['kind']!='image':continue
        folder=run/'jobs'/job['id'];saved=json.loads((folder/'result.json').read_text())
        assert saved['job']==job and saved['code_sha256']==manifest['code_sha256']
        model=ImageModel(job['algorithm']).to(device)
        model(torch.zeros(1,3,32,32,device=device))  # register published lazy scale
        model.load_state_dict(torch.load(folder/'best.pth',map_location=device,weights_only=True))
        clear_weight_caches(model);model.eval()
        audits=[]
        if job['algorithm']=='ldlt-resnet':
            for block in model.conv:
                w=block.normalized_weights((32,32));matrix=original_matrix(w)
                minimum=float(torch.linalg.eigvalsh(matrix).min())
                assert minimum>0
                audits.append(minimum)
            minimum=float(torch.linalg.eigvalsh(original_matrix(model.dense.normalized_weights())).min())
            assert minimum>0;audits.append(minimum)
        head=model.head_weight().double()
        # All pairs are small (ten classes). Biases are already in the logits.
        distances=torch.cdist(head,head)
        radii=[];correct=[];predictions=[];targets=[]
        for x,y in loader:
            logits=model(x.to(device)).double();y=y.to(device)
            prediction=logits.argmax(-1)
            margin=logits.gather(1,y[:,None])-logits
            denom=distances[y]
            ratios=torch.full_like(margin,torch.inf)
            positive=denom>0
            ratios[positive]=margin[positive]/denom[positive]
            ratios[(~positive)&(margin<=0)]=0
            ratios.scatter_(1,y[:,None],torch.inf)
            radius=ratios.amin(1).clamp_min(0)
            radius[prediction!=y]=0
            radii.append(radius.cpu());correct.append((prediction==y).cpu())
            predictions.append(prediction.cpu());targets.append(y.cpu())
        radius=torch.cat(radii);accuracy=float(torch.cat(correct).double().mean())
        cert={str(e):float((radius>e/255).double().mean()) for e in (36,72,108)}
        errors={'clean':abs(accuracy-saved['metrics']['clean']),
                **{e:abs(cert[e]-saved['metrics']['certified'][e]) for e in cert}}
        # Counts must agree; saved float32 means introduce < 1e-7 rounding.
        assert max(errors.values())<1e-7,(job['id'],errors)
        np.savez_compressed(output/(job['id']+'.npz'),radius=radius.numpy(),
                            prediction=torch.cat(predictions).numpy(),target=torch.cat(targets).numpy())
        rows.append(dict(job=job['id'],clean=accuracy,certified=cert,
                         max_metric_difference=max(errors.values()),independent_lmi_minima=audits,
                         checkpoint_sha256=hashlib.sha256((folder/'best.pth').read_bytes()).hexdigest()))
        summary['elapsed_seconds']=time.monotonic()-started
        (output/'image_audit.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(f"{len(rows)}/18 {job['id']}: metrics reproduced; certificate minima {audits}",flush=True)
        del model
    summary['complete']=len(rows)==18
    (output/'image_audit.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
