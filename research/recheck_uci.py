"""Fresh UCI checkpoint checks, with independent split/statistic/radius arithmetic.

Default datasets are a declared diagnostic subset spanning 4 to 256 features;
this is not a claim to reevaluate all 3,096 UCI fits. Use --datasets to expand.
No training or campaign writes occur.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
import uci_training as training
from certification import clear_weight_caches
from revision_experiments import code_digest
from recheck_images import original_matrix


def build(alg,r,device):
    d,c,w,depth,pad=[r[k] for k in ['in_features','out_features','width','depth','padding']]
    if alg=='ldlt-resnet':
        return training.DeepLipschitzLinearResNetLNN(d,c,[w]*(depth-1),device=device,dtype=torch.float32)
    if alg=='ldlt':
        return training.DeepLipschitzSequentialLNN(d,c,[w]*(depth-1),device=device,dtype=torch.float32)
    if alg=='sdp':
        return training.LipschitzNetworkSequential(cin=d,n_dense=depth,padding=pad,dense_inner_dim=w,n_classes=c).to(device)
    return training.Toy(in_channels=d,num_classes=c,depth=depth,
                        layer={'aol':'Aol','ortho':'Orthogon','sandwich':'Sandwich'}[alg],padding=pad).to(device)


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets',nargs='+',default=['iris','abalone','letter','semeion'])
    parser.add_argument('--all',action='store_true',help='Reevaluate every UCI fit in the manifest')
    parser.add_argument('--device',default='cuda:1')
    args=parser.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device=torch.device(args.device);run=ROOT/'runs/revision_campaign'
    manifest=json.loads((run/'manifest.json').read_text());assert code_digest()==manifest['code_sha256']
    if args.all:
        args.datasets=sorted({j['dataset'] for j in manifest['jobs'] if j['kind']=='uci'})
    output=ROOT/'revision_artifacts/checkpoint_reevaluation';output.mkdir(exist_ok=True,parents=True)
    audit_file=output/('uci_all_audit.json' if args.all else 'uci_audit.json')
    jobs=[j for j in manifest['jobs'] if j['kind']=='uci' and j['dataset'] in args.datasets]
    assert set(j['dataset'] for j in jobs)==set(args.datasets)
    rows=[];cache={};started=time.monotonic()
    summary=dict(datasets=args.datasets,planned_jobs=len(jobs),device=str(device),source_sha256=manifest['code_sha256'],
                 complete=False,rows=rows,scope='Fresh loads for all UCI checkpoints.' if args.all else
                 'Fresh loads for selected datasets, all available methods/losses and four splits; not all UCI checkpoints.')
    for number,job in enumerate(jobs,1):
        directory=run/'jobs'/job['id'];result=json.loads((directory/'result.json').read_text())
        assert result['job']==job and result['code_sha256']==manifest['code_sha256']
        ds=job['dataset'];alg=job['algorithm']
        if ds not in cache:
            folder=Path(manifest['data_root'])/'data'/ds
            for name,expected in manifest['dataset_sha256'][ds].items():
                assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==expected
            cache[ds]=training.load_dataset_bundle(folder)
        x,y,test,validation=cache[ds]
        for k,r in enumerate(result['fold_results']):
            assert r['fold']==k
            train_idx=(test[:,k]!=1)&(validation[:,k]!=1)
            test_idx=test[:,k]==1
            assert not np.any(test_idx&(validation[:,k]==1))
            mean=x[train_idx].mean(0);std=x[train_idx].std(0);std[std<1e-8]=1
            np.testing.assert_allclose(mean,r['stats']['mean'],rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(std,r['stats']['std'],rtol=1e-12,atol=1e-12)
            inputs=torch.tensor((x[test_idx]-mean)/std,dtype=torch.float32,device=device)
            labels=torch.tensor(y[test_idx],dtype=torch.long,device=device)
            model=build(alg,r,device)
            path=directory/'runs'/ds/f'fold_{k}'/'best_state.pth'
            model.load_state_dict(torch.load(path,map_location=device,weights_only=True))
            clear_weight_caches(model);model.eval()
            minimum=None;bound=1.
            if alg.startswith('ldlt'):
                bound=model.lipschitz_constant
                head=model.last_weight()
                minimum=float(torch.linalg.eigvalsh(original_matrix(model.normalized_weights(),bound)).min())
                assert minimum>0
            else:
                layer=model.model[-1]
                head=F.normalize(layer.weight,dim=1)*getattr(layer,'scale',1.)
            head=head.double()
            distances=torch.linalg.vector_norm(head[:,None]-head[None,:],dim=-1)*bound
            radii=[];hits=[]
            for start in range(0,len(inputs),512):
                logits=model(inputs[start:start+512]).double();target=labels[start:start+512]
                margins=logits.gather(1,target[:,None])-logits
                denom=distances[target]
                ratios=torch.where(denom>0,margins/denom.clamp_min(1e-300),
                                   torch.where(margins>0,torch.inf,0.))
                ratios.scatter_(1,target[:,None],torch.inf)
                hit=logits.argmax(1)==target
                radius=ratios.amin(1).clamp_min(0);radius[~hit]=0
                hits.append(hit.cpu());radii.append(radius.cpu())
            radius=torch.cat(radii);accuracy=float(torch.cat(hits).double().mean())
            cert={str(e):float((radius>e/255).double().mean()) for e in [36,72,108,255]}
            error=max(abs(accuracy-r['test_acc']),*[abs(cert[e]-r['cert_evals'][e]) for e in cert])
            assert error<1e-7,(job['id'],k,error,cert,r['cert_evals'])
            rows.append(dict(job=job['id'],fold=k,max_metric_difference=error,independent_lmi_minimum=minimum,
                             checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            np.savez_compressed(output/(job['id']+f'-fold{k}.npz'),radius=radius.numpy())
            del model
        summary['elapsed_seconds']=time.monotonic()-started
        audit_file.write_text(json.dumps(summary,indent=2)+'\n')
        print(f'{number}/{len(jobs)} {job["id"]}: all four split metrics reproduced',flush=True)
    summary['complete']=len(rows)==4*len(jobs)
    audit_file.write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
