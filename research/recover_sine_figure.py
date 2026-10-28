"""Export a supplied historical sine checkpoint as native vector graphics.

This is an archival plot, not validation of the corrected residual certificate.
Run with the locked DeepLipschitzResNet uv environment; no training occurs.
"""
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    workspace = Path(__file__).resolve().parents[2]
    legacy = Path("/home/mjuston2/Documents/LDLTFlossing")
    run = legacy / "runs/grid_search_flossing_non_adaptive/grid_L3_h16_gpu1_20251028_005705_495545"
    metadata = json.loads((run / "metadata.json").read_text())
    sys.path.insert(0, str(legacy / "src"))
    from models.linear_model import DeepLipschitzSequential

    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    model = DeepLipschitzSequential(
        1, 1, (metadata["hidden_size"],) * metadata["L_actual"], device="cpu"
    )
    checkpoint = run / "logs/best.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    x = torch.linspace(-10, 10, 100000, dtype=torch.float32).reshape(-1, 1)
    with torch.no_grad():
        prediction = torch.cat([model(batch) for batch in x.split(2048)])
    assert torch.isfinite(prediction).all()
    truth = torch.sin(x)
    output = workspace / "LDL-Lipschitz-ResNet-LMI/Figures/LDL/Linear"
    fig, ax = plt.subplots()
    ax.plot(x.numpy(), truth.numpy(), label="Ground Truth")
    ax.plot(x.numpy(), prediction.numpy(), label="Prediction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "trained_network.pdf")
    plt.close(fig)
    arrays = output / "trained_network_data.npz"
    np.savez_compressed(arrays, x=x.numpy(), truth=truth.numpy(), prediction=prediction.numpy())
    record = {
        "source_run": str(run), "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha(checkpoint), "saved_best_epoch": metadata["best_epoch"],
        "user_reference_image": str(run / "logs/truth13.png"),
        "exact_original_reproduction": False,
        "note": "Reevaluated saved best checkpoint (epoch 19), not the unavailable epoch-13 checkpoint or original paper curve. Historical sequential/flossing model; no corrected-residual certificate claim.",
        "model": "LDLTFlossing DeepLipschitzSequential",
        "hidden_size": metadata["hidden_size"], "hidden_layers": metadata["L_actual"],
        "points": len(x), "mse": float(torch.mean((prediction - truth) ** 2)),
        "data_sha256": sha(arrays),
        "source_hashes": {str(p.relative_to(legacy)): sha(p) for p in (
            legacy / "src/models/linear_model.py", legacy / "src/models/linear_layers.py",
            legacy / "src/utils.py", run / "metadata.json",
        )},
    }
    (output / "trained_network_provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("saved_best_epoch", "mse", "exact_original_reproduction")}, indent=2))


if __name__ == "__main__":
    main()
