#!/usr/bin/env python3
"""Summarize paired CE/margin results without treating partial coverage as full evidence."""
import argparse
import json
from pathlib import Path
import statistics

METRICS=['mean_test_acc',*[f'mean_cert_acc_{e}' for e in (36,72,108,255)]]
ALGORITHMS=['ldlt-resnet','ldlt','sandwich','sdp']


def analyze(output):
    manifest=json.loads((output/'manifest.json').read_text())
    expected=manifest['code_sha256']
    results={}
    for job in manifest['jobs']:
        if job['kind']!='uci': continue
        path=output/'jobs'/job['id']/'result.json'
        if not path.exists(): continue
        result=json.loads(path.read_text())
        if result['code_sha256']!=expected or result['job']!=job:
            raise ValueError(f'Result provenance mismatch: {path}')
        for fold in result['fold_results']:
            if fold['split_policy']!='test_mask_one': raise ValueError('Unsupported split policy')
        results[job['dataset'],job['algorithm'],job['loss']]=result
    planned=manifest['uci_ablation_datasets']
    common=[d for d in planned if all((d,a,l) in results for a in ALGORITHMS for l in ['ce','margin'])]
    rows=[]
    for algorithm in ALGORITHMS:
        row=dict(algorithm=algorithm,datasets=common,n=len(common),metrics={})
        for metric in METRICS:
            ce=[results[d,algorithm,'ce'][metric] for d in common]
            margin=[results[d,algorithm,'margin'][metric] for d in common]
            differences=[m-c for c,m in zip(ce,margin)]
            row['metrics'][metric]=dict(ce=statistics.mean(ce) if ce else None,
                                       margin=statistics.mean(margin) if margin else None,
                                       paired_mean_difference=statistics.mean(differences) if differences else None,
                                       per_dataset_difference=dict(zip(common,differences)))
        rows.append(row)
    summary=dict(code_sha256=expected,planned_datasets=planned,common_complete_datasets=common,
                 complete_ablation=len(common)==len(planned),rows=rows,
                 interpretation='Dataset means are paired before aggregation. This descriptive ablation does not establish significance, causality across architectures, or full benchmark coverage.',
                 input_coordinates='train-standardized feature Euclidean norm')
    (output/'margin_report.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    lines=['% Generated descriptive CE/margin ablation; inspect coverage before using.',
           r'\begin{tabular}{lrrr}',r'Method & CE clean (\%) & Margin clean (\%) & Difference (pp) \\',r'\hline']
    for row in rows:
        v=row['metrics']['mean_test_acc']
        if v['ce'] is not None:
            lines.append(f"{row['algorithm']} & {100*v['ce']:.2f} & {100*v['margin']:.2f} & {100*v['paired_mean_difference']:+.2f} "+r'\\')
    lines.extend([r'\end{tabular}',f'% Coverage: {len(common)} of {len(planned)} planned ablation datasets.'])
    (output/'margin_clean_table.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:summary[k] for k in ['complete_ablation','common_complete_datasets']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',default='runs/revision_verified')
    analyze(Path(parser.parse_args().out))
