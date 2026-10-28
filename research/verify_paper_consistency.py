"""Verify the mathematical source snapshot associated with the final paper.

No data, GPU, network, or paper checkout is needed for the default hash check.
--paper-root also verifies the working files and immutable paper Git objects.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def verify(paper_root=None):
    spec = json.loads((ROOT/'research/paper_consistency.json').read_text())
    for entry in spec['files']:
        raw = (ROOT/entry['code_path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry['sha256']:
            raise ValueError(f"Code differs from the audited paper snapshot: {entry['code_path']}")
        if paper_root is not None:
            paper_root = Path(paper_root).resolve()
            if raw != (paper_root/entry['paper_path']).read_bytes():
                raise ValueError(f"Paper working file differs: {entry['paper_path']}")
            committed = subprocess.check_output(
                ['git','show',spec['paper_commit']+':'+entry['paper_path']],cwd=paper_root)
            if raw != committed:
                raise ValueError(f"Pinned paper source differs: {entry['paper_path']}")
    return dict(paper_commit=spec['paper_commit'],matching_files=len(spec['files']),
                paper_checkout_verified=paper_root is not None)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper-root',type=Path)
    print(json.dumps(verify(parser.parse_args().paper_root),indent=2))
