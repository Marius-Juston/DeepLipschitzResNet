#!/usr/bin/env bash
# User-started campaign: one GPU profile, then the resumable queue without time caps.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
RUN_DIR="runs/revision_campaign"
uv sync --locked
uv run --locked python -c 'import json; from pathlib import Path; from revision_experiments import code_digest; m=json.loads(Path("runs/revision_campaign/manifest.json").read_text()); assert m["code_sha256"] == code_digest(), "Source changed: prepare a new campaign before training"'
# Do not remove the completion marker on resume: profiling must not be repeated.
if [[ ! -f "$RUN_DIR/profile.complete" ]]; then
    uv run --locked python revision_profile.py --device cuda:0 --out "$RUN_DIR/profile_gpu.json"
    uv run --locked python -c 'import json; from pathlib import Path; p=Path("runs/revision_campaign/profile_gpu.json"); r=json.loads(p.read_text()); assert len(r["rows"]) == 19 and all("error" not in x for x in r["rows"]), "Inspect profile errors before starting the campaign"'
    touch "$RUN_DIR/profile.complete"
fi
uv run --locked python revision_experiments.py run --gpus 0,1,2,3 --out "$RUN_DIR"
uv run --locked python revision_experiments.py report --out "$RUN_DIR"
uv run --locked python revision_analyze.py --out "$RUN_DIR"
