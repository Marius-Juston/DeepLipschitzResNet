"""Read-only validation of the published completed-campaign result snapshot."""
import hashlib
import json
from pathlib import Path
import statistics
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def verify():
    digest=hashlib.sha256()
    for p in sorted([*ROOT.glob('src/**/*.py'),ROOT/'revision_experiments.py',ROOT/'uv.lock']):
        digest.update(str(p.relative_to(ROOT)).encode()); digest.update(p.read_bytes())
    summary=json.loads((ROOT/'revision_artifacts/publication_results/summary.json').read_text())
    metrics=['mean_test_acc',*[f'mean_cert_acc_{r}' for r in (36,72,108,255)]]
    splits=0; slacks=[]
    with zipfile.ZipFile(ROOT/'revision_artifacts/completed_campaign_results.zip') as z:
        manifest=json.loads(z.read('manifest.json'))
        assert digest.hexdigest()==manifest['code_sha256']==summary['source_sha256']
        assert len(manifest['jobs'])==len(summary['result_sha256'])==792
        expected={'manifest.json'}|{f"jobs/{j['id']}/result.json" for j in manifest['jobs']}
        assert set(z.namelist())==expected and len(z.namelist())==len(expected)
        for job in manifest['jobs']:
            raw=z.read(f"jobs/{job['id']}/result.json"); r=json.loads(raw)
            assert hashlib.sha256(raw).hexdigest()==summary['result_sha256'][job['id']]
            assert r['job']==job and r['code_sha256']==manifest['code_sha256']
            if job['kind']=='uci':
                assert r['n_folds']==len(r['fold_results'])==4
                splits+=4
                for f in r['fold_results']:
                    assert f['split_policy']=='test_mask_one'
                    if job['algorithm'].startswith('ldlt'):
                        slacks.append(f['certificate_audit']['min_eigenvalue'])
                for metric in metrics:
                    actual=statistics.mean(f['test_acc'] if metric=='mean_test_acc' else
                        f['cert_evals'][metric.rsplit('_',1)[1]] for f in r['fold_results'])
                    assert abs(actual-r[metric])<1e-12
            else:
                slacks.extend(a['min_eigenvalue'] for a in r['certificate_audits'])
    assert splits==3096 and len(slacks)==summary['recorded_positive_ldl_audits']
    assert min(slacks)>0 and min(slacks)==summary['min_recorded_eigenvalue']
    return dict(jobs=792,uci_splits=splits,source_sha256=digest.hexdigest(),
        original_result_hashes_verified=792,positive_recorded_certificates=len(slacks),
        minimum_recorded_eigenvalue=min(slacks),
        scope='Saved-result provenance and split means, not a new checkpoint reevaluation or interval proof')


if __name__=='__main__':
    print(json.dumps(verify(),indent=2))
