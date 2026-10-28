"""A direct worker must not attest a different source revision."""
import argparse
import importlib.util
import json
from pathlib import Path
import pytest


@pytest.mark.parametrize('wall,gpu,expected', [
    (None,None,False), (1.,None,True), (None,1.,True), (3.,4.,False),
])
def test_optional_campaign_budgets(wall,gpu,expected):
    path = Path(__file__).resolve().parents[2] / 'revision_experiments.py'
    spec = importlib.util.spec_from_file_location('revision_queue_budget', path)
    queue = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(queue)
    assert queue.budget_exhausted(
        {'wall_seconds':7200.,'gpu_seconds':10800.},
        {'wall_budget_hours':wall,'gpu_budget_hours':gpu}) is expected


def test_direct_worker_rejects_stale_manifest(tmp_path):
    path = Path(__file__).resolve().parents[2] / 'revision_experiments.py'
    spec = importlib.util.spec_from_file_location('revision_queue', path)
    queue = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(queue)
    (tmp_path / 'manifest.json').write_text(json.dumps({'code_sha256': 'stale', 'jobs': []}))
    with pytest.raises(ValueError, match='mislabel'):
        queue.worker(argparse.Namespace(out=str(tmp_path), job='anything'))
    assert not (tmp_path / 'jobs').exists()
