"""Cold-cache versus warm evaluation of all 18 trained image checkpoints.

This profiles the existing model implementations without changing training code.
Warm evaluation uses their existing caches; it is not a fully materialized or
compiled deployment benchmark. Host/device transfers and data loading are excluded.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from revision_image import ImageModel
from certification import clear_weight_caches
from revision_experiments import code_digest


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    torch.set_num_threads(4);torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    device=torch.device(args.device)
    run=ROOT/'runs/revision_campaign';manifest=json.loads((run/'manifest.json').read_text())
    assert code_digest()==manifest['code_sha256']
    output=ROOT/'revision_artifacts/inference_profile';output.mkdir(parents=True,exist_ok=True)
    rows=[]
    report=dict(device=torch.cuda.get_device_name(device),torch=torch.__version__,
                source_sha256=manifest['code_sha256'],cold_repetitions=30,warm_repetitions=100,
                warmup=10,complete=False,rows=rows,
                scope=__doc__,timing='Synchronized wall time, eval mode, no gradients. Cold includes cache reset and normalization; warm retains existing caches.')
    def sync():torch.cuda.synchronize(device)
    for job in manifest['jobs']:
        if job['kind']!='image':continue
        path=run/'jobs'/job['id']/'best.pth'
        model=ImageModel(job['algorithm']).to(device)
        model(torch.zeros(1,3,32,32,device=device))
        model.load_state_dict(torch.load(path,map_location=device,weights_only=True))
        for batch in [1,64]:
            x=torch.rand(batch,3,32,32,device=device)
            def cold():
                model.eval()  # also clears LDL convolution caches
                clear_weight_caches(model)
                return model(x)
            reference=cold()
            for _ in range(10):model(x)
            warm=model(x)
            torch.testing.assert_close(warm,reference,atol=1e-6,rtol=1e-6)
            entry=dict(job=job['id'],algorithm=job['algorithm'],batch=batch,
                       checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       max_output_difference=float((warm-reference).abs().max()))
            for mode,fn,count in [('warm',lambda:model(x),100),('cold',cold,30)]:
                times=[]
                for _ in range(count):
                    sync();start=time.perf_counter();fn();sync()
                    times.append(1000*(time.perf_counter()-start))
                entry[mode]=dict(median_ms=statistics.median(times),min_ms=min(times),max_ms=max(times),samples_ms=times)
            rows.append(entry)
            (output/'profile.json').write_text(json.dumps(report,indent=2)+'\n')
        print(f'{job["id"]}: batch 1/64 cold and warm completed',flush=True)
        del model
    report['complete']=len(rows)==36
    (output/'profile.json').write_text(json.dumps(report,indent=2)+'\n')
    names={'ldlt-resnet':'LDL residual','sandwich':'Sandwich','sdp':'SLL'}
    lines=[r'\begin{tabular}{lrrr}',r'\toprule',r'Method & Batch & Cold (ms) & Warm (ms) \\',r'\midrule']
    for alg in names:
        for batch in [1,64]:
            rs=[r for r in rows if r['algorithm']==alg and r['batch']==batch]
            cold=statistics.median(r['cold']['median_ms'] for r in rs)
            warm=statistics.median(r['warm']['median_ms'] for r in rs)
            lines.append(f'{names[alg]} & {batch} & {cold:.2f} & {warm:.2f} '+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (ROOT.parent/'LDL-Lipschitz-ResNet-LMI/result_tables/inference.tex').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
