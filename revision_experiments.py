#!/usr/bin/env python3
"""Resumable local experiment queue with optional persistent wall/GPU-time budgets.

Run `uv run python revision_experiments.py --help`. No messages or artifacts
are sent to external services. Workers write atomic results and epoch snapshots.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'src'))


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def code_digest():
    digest = hashlib.sha256()
    for path in sorted([*ROOT.glob('src/**/*.py'), ROOT/'revision_experiments.py', ROOT/'uv.lock']):
        digest.update(str(path.relative_to(ROOT)).encode()); digest.update(path.read_bytes())
    return digest.hexdigest()


def budget_exhausted(state, manifest):
    """JSON null disables a limit; elapsed resource usage is still recorded."""
    return any(manifest[key] is not None and state[counter] >= 3600*manifest[key]
               for counter, key in [('wall_seconds', 'wall_budget_hours'),
                                    ('gpu_seconds', 'gpu_budget_hours')])


def prepare(args):
    output = Path(args.out).resolve()
    if (output/'manifest.json').exists():
        raise ValueError('Manifest already exists; use run to resume, or a new output directory')
    data = Path(args.data_root).resolve()
    folders = sorted(p for p in (data/'data').iterdir() if (p/'folds_py.dat').exists())
    if args.datasets:
        folders = [p for p in folders if p.name in args.datasets]
        if set(args.datasets) != {p.name for p in folders}: raise ValueError('Requested dataset is missing')
    if not folders: raise ValueError('No datasets found')
    jobs = []
    hashes = {}
    for folder in folders:
        hashes[folder.name] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.glob('*_py.dat'))}
        for algorithm in ['ldlt-resnet','ldlt','sandwich','sdp','aol','ortho']:
            jobs.append(dict(id=f'uci-{folder.name}-{algorithm}-ce', kind='uci', dataset=folder.name,
                             algorithm=algorithm, loss='ce', epochs=args.epochs, seed=1337))
    # A fixed, data-independent alphabetic subset; never select using outcomes.
    ablation = folders[:min(12,len(folders))]
    for folder in ablation:
        for algorithm in ['ldlt-resnet','ldlt','sandwich','sdp']:
            jobs.append(dict(id=f'uci-{folder.name}-{algorithm}-margin', kind='uci', dataset=folder.name,
                             algorithm=algorithm, loss='margin', epochs=args.epochs, seed=1337))
    if args.image_data:
        for seed in [42,43,44]:
            for loss in ['margin','ce']:
                for algorithm in ['ldlt-resnet','sandwich','sdp']:
                    jobs.append(dict(id=f'cifar10-{algorithm}-{loss}-{seed}', kind='image', dataset='cifar10',
                                     algorithm=algorithm, loss=loss, epochs=args.image_epochs, seed=seed))
    manifest = dict(protocol_version=2, code_sha256=code_digest(), data_root=str(data), dataset_sha256=hashes,
                    image_data=str(Path(args.image_data).resolve()) if args.image_data else None,
                    jobs=jobs, wall_budget_hours=args.hours or None, gpu_budget_hours=args.gpu_hours or None,
                    created=time.time(), uci_ablation_datasets=[p.name for p in ablation],
                    note='Benchmark evidence remains pending until complete comparisons are validated')
    atomic_json(output/'manifest.json', manifest)
    print(f'Prepared {len(jobs)} jobs in {output}; no training started.')


def worker(args):
    import torch
    from importlib.metadata import version
    output = Path(args.out).resolve()
    manifest = json.loads((output/'manifest.json').read_text())
    if manifest['code_sha256'] != code_digest():
        raise ValueError('Code changed since preparation; refusing to mislabel worker results.')
    job = next(j for j in manifest['jobs'] if j['id'] == args.job)
    directory = output/'jobs'/job['id']; directory.mkdir(parents=True, exist_ok=True)
    try:
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if job['kind'] == 'uci':
            import uci_training
            config = argparse.Namespace(gpus=int(torch.cuda.is_available()), depth=4, batch_size=64, lr=1e-3,
                                        max_epochs=job['epochs'], patience=30,
                                        offset=.5 if job['loss']=='margin' else 0.,
                                        temperature=.25 if job['loss']=='margin' else 1.)
            dataset = Path(manifest['data_root'])/'data'/job['dataset']
            # Detect changed data before starting or resuming an experiment.
            for name, expected in manifest['dataset_sha256'][job['dataset']].items():
                if hashlib.sha256((dataset/name).read_bytes()).hexdigest() != expected:
                    raise ValueError('Dataset hash differs from manifest')
            status, result = uci_training._run_task((dataset, job['dataset'], job['algorithm'], 0,
                                                   str(directory), job['seed'], config))
            if status != 'ok': raise RuntimeError(result['traceback'])
        else:
            from revision_image import run_image
            result = run_image(job, directory, Path(manifest['image_data']),
                               torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        result['environment'] = {p:version(p) for p in ['torch','numpy','scipy','torchvision']}
        result['execution'] = dict(device='cuda:0' if torch.cuda.is_available() else 'cpu',
                                   device_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
                                   torch_threads=torch.get_num_threads())
        result['job'] = job
        result['code_sha256'] = manifest['code_sha256']
        atomic_json(directory/'result.json', result)
    except Exception:
        atomic_json(directory/'error.json', dict(traceback=traceback.format_exc(), time=time.time()))
        raise


def run(args):
    output = Path(args.out).resolve()
    manifest = json.loads((output/'manifest.json').read_text())
    if manifest['code_sha256'] != code_digest():
        raise ValueError('Code changed since preparation. Prepare a new manifest; do not mix experiments.')
    # Advisory lock prevents two controllers from allocating the same jobs.
    import fcntl
    lock = (output/'controller.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = output/'status.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(wall_seconds=0.,gpu_seconds=0.,jobs={})
    cpu_mode = args.gpus == 'cpu'
    if cpu_mode:
        if not 1 <= args.cpu_workers <= 16:
            raise ValueError('Choose between one and sixteen CPU workers')
        gpus = [f'cpu:{i}' for i in range(args.cpu_workers)]
    else:
        gpus = args.gpus.split(',')
        import torch
        available = torch.cuda.device_count()
        if any(not g.isdigit() or int(g)>=available for g in gpus):
            raise RuntimeError(f'Requested GPUs are unavailable; visible device count is {available}')
    if len(set(gpus)) != len(gpus): raise ValueError('Worker devices must be distinct')
    pending = [j for j in manifest['jobs'] if not (output/'jobs'/j['id']/'result.json').exists()
               and (not cpu_mode or j['kind']=='uci')
               and (args.retry_failed or state['jobs'].get(j['id'],{}).get('status') != 'failed')]
    # CPU workers advance tabular evidence; image jobs remain pending for a GPU run.
    state['worker_mode'] = 'cpu-uci' if cpu_mode else 'gpu'
    running = {}
    previous = time.monotonic()
    stop = False
    def request_stop(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM,request_stop); signal.signal(signal.SIGINT,request_stop)
    try:
        while pending or running:
            now = time.monotonic(); elapsed = now-previous; previous=now
            state['wall_seconds'] += elapsed
            state['gpu_seconds'] += elapsed * sum(not g.startswith('cpu:') for g in running)
            for gpu, (process,job,log) in list(running.items()):
                if process.poll() is not None:
                    ok = process.returncode==0 and (output/'jobs'/job['id']/'result.json').exists()
                    state['jobs'][job['id']].update(status='complete' if ok else 'failed', ended=time.time(), exit_code=process.returncode)
                    log.close(); del running[gpu]
            exhausted = budget_exhausted(state, manifest)
            if stop or exhausted: break
            for gpu in gpus:
                if gpu in running or not pending: continue
                job = pending.pop(0)
                directory=output/'jobs'/job['id']; directory.mkdir(parents=True,exist_ok=True)
                log=(directory/'worker.log').open('a')
                env=dict(os.environ, CUDA_VISIBLE_DEVICES='' if gpu.startswith('cpu:') else gpu,
                         CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
                process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'worker','--out',str(output),'--job',job['id']],
                                         cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                running[gpu]=(process,job,log)
                state['jobs'][job['id']]=dict(status='running',gpu=gpu,pid=process.pid,started=time.time())
            state.update(updated=time.time(), pending=len(pending), controller_pid=os.getpid())
            atomic_json(state_path,state)
            time.sleep(2)
    finally:
        for process,job,log in running.values():
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL); process.wait()
                state['jobs'][job['id']]['status']='paused'
            else:
                ok = process.returncode==0 and (output/'jobs'/job['id']/'result.json').exists()
                state['jobs'][job['id']].update(status='complete' if ok else 'failed',
                                               ended=time.time(), exit_code=process.returncode)
            log.close()
        state.update(updated=time.time(), controller_pid=None)
        atomic_json(state_path,state)
    status(args)


def status(args):
    output=Path(args.out)
    manifest=json.loads((output/'manifest.json').read_text())
    state=json.loads((output/'status.json').read_text()) if (output/'status.json').exists() else {}
    counts={}
    for job in manifest['jobs']:
        phase='complete' if (output/'jobs'/job['id']/'result.json').exists() else state.get('jobs',{}).get(job['id'],{}).get('status','pending')
        counts[phase]=counts.get(phase,0)+1
    completed=[v['ended']-v['started'] for v in state.get('jobs',{}).values() if v['status']=='complete' and 'ended' in v]
    eta=(sum(completed)/len(completed)*(len(manifest['jobs'])-counts.get('complete',0))/max(1,len({v.get('gpu') for v in state.get('jobs',{}).values()}))/3600) if completed else None
    print(json.dumps(dict(counts=counts,wall_hours=state.get('wall_seconds',0)/3600,
                          gpu_hours=state.get('gpu_seconds',0)/3600,last_update=state.get('updated'),
                          rough_remaining_hours=eta, eta_caveat='Mean completed job time; dataset sizes and image jobs differ substantially',
                          running={k:v for k,v in state.get('jobs',{}).items() if v['status']=='running'}),indent=2))


def report(args):
    from scipy.stats import wilcoxon
    import numpy as np
    from itertools import combinations
    output=Path(args.out)
    manifest=json.loads((output/'manifest.json').read_text())
    results={}
    images=[]
    missing=[]
    for job in manifest['jobs']:
        path=output/'jobs'/job['id']/'result.json'
        if not path.exists(): missing.append(job['id']); continue
        result=json.loads(path.read_text())
        if result['code_sha256'] != manifest['code_sha256']: raise ValueError('Mixed code revisions')
        if job['kind']=='uci': results[(job['algorithm'],job['loss'],job['dataset'])]=result
        else: images.append(result)
    algorithms=['ldlt-resnet','ldlt','sandwich','sdp','aol','ortho']
    datasets=sorted(set.intersection(*[{d for a,l,d in results if a==alg and l=='ce'} for alg in algorithms]))
    comparisons=[]; means={}
    for metric in ['mean_test_acc',*[f'mean_cert_acc_{e}' for e in (36,72,108,255)]]:
        vals={a:np.array([results[a,'ce',d][metric] for d in datasets]) for a in algorithms}
        means[metric]={a:float(v.mean()) if len(v) else None for a,v in vals.items()}
        rows=[]
        for a,b in combinations(algorithms,2):
            diff=vals[a]-vals[b]
            p=1. if not len(diff) or np.all(diff==0) else float(wilcoxon(diff,zero_method='wilcox').pvalue)
            rows.append(dict(metric=metric,a=a,b=b,n=len(diff),p=p,
                             mean_difference=float(diff.mean()) if len(diff) else None))
        order=sorted(range(len(rows)),key=lambda i:rows[i]['p']); previous=0.
        for rank,i in enumerate(order):
            previous=max(previous,min(1.,rows[i]['p']*(len(rows)-rank))); rows[i]['p_holm']=previous
        comparisons.extend(rows)
    summary=dict(complete=not missing,missing_jobs=missing,common_uci_datasets=datasets,
                 means=means,paired_tests=comparisons,image_results=images,
                 note='Only common complete datasets are compared. Partial results are not the 121-dataset benchmark.')
    atomic_json(output/'report.json',summary)
    print(f'Report: {output / "report.json"}; common UCI datasets: {len(datasets)}; missing jobs: {len(missing)}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for command in ['prepare','run','worker','status','report']:
        p=sub.add_parser(command); p.add_argument('--out',default='runs/revision_v2')
        if command=='prepare':
            p.add_argument('--data-root',required=True);p.add_argument('--image-data')
            p.add_argument('--datasets',nargs='+');p.add_argument('--epochs',type=int,default=100)
            p.add_argument('--image-epochs',type=int,default=100)
            p.add_argument('--hours',type=float,default=48.,help='Wall-hour limit; 0 disables it')
            p.add_argument('--gpu-hours',type=float,default=192.,help='Aggregate GPU-hour limit; 0 disables it')
        if command=='run':
            p.add_argument('--gpus',default='0,1,2,3');p.add_argument('--retry-failed',action='store_true')
            p.add_argument('--cpu-workers',type=int,default=1,help='Concurrent UCI workers when --gpus cpu; image jobs stay pending')
        if command=='worker': p.add_argument('--job',required=True)
    args=parser.parse_args();globals()[args.command](args)


if __name__=='__main__': main()
