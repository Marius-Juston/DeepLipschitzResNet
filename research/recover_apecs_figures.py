"""Archive recovered native-vector variants without claiming exact PNG reproduction.

Run from DeepLipschitzResNet with ``uv run --locked python research/recover_apecs_figures.py``.
No training is performed and the supplied APECS artifacts are never overwritten.
"""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[2]
    source = root / "apecs_library/tests"
    figures = root / "LDL-Lipschitz-ResNet-LMI/Figures"
    output = figures / "recovered/apecs"
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "source_repository_commit": subprocess.check_output(
            ["git", "-C", str(source.parent), "rev-parse", "HEAD"], text=True
        ).strip(),
        "scope": "Recovered variants, not exact reproductions of the manuscript PNGs; excluded from current manuscript claims.",
        "pdfs": [],
        "exact_png_matches": [],
        "remaining": ["Original optimizer trajectories/checkpoints", "Original circle matrix/seed", "Original histogram data/configuration", "Both timing plots", "Three DeepLipschitzResNet trained/sine plots"],
    }
    for name in ("circles", "eigenvalue_distribution", "loss_optimizers", "optimizers"):
        original = source / f"{name}.pdf"
        target = output / original.name
        shutil.copy2(original, target)
        listing = subprocess.check_output(["pdfimages", "-list", str(target)], text=True)
        if len(listing.strip().splitlines()) != 2:
            raise RuntimeError(f"Unexpected raster image in {target}")
        report["pdfs"].append({
            "source": str(original.relative_to(root)),
            "target": str(target.relative_to(root)),
            "sha256": digest(target), "raster_images": 0,
            "comparison": "Visual comparison found different curves or axis ranges from the paper PNG.",
        })
    for original in (figures / "Gershgorin").glob("*.png"):
        for candidate in source.glob("*.png"):
            if digest(original) == digest(candidate):
                report["exact_png_matches"].append({"paper": original.name, "apecs": candidate.name})

    # Reproduce the recovered NPZ with the historical script's filtering rule.
    # This is a distinct saved dataset; it does not match the original paper plot.
    data_path = source / "eigen_distribution.npz"
    values = np.load(data_path, allow_pickle=False)["eigen_values"]
    q1, q3 = np.quantile(values, [0.25, 0.75])
    cutoff = float(q1 - 15 * (q3 - q1))
    filtered = values[values >= cutoff]
    density, edges = np.histogram(filtered, bins=10000, density=True)
    fig, ax = plt.subplots()
    ax.stairs(density, edges, fill=True)
    ax.set(yscale="log", xlabel="Eigenvalue", ylabel="Probability density (log scale)")
    fig.tight_layout()
    target = output / "eigen_distribution_from_saved_npz.pdf"
    fig.savefig(target)
    plt.close(fig)
    np.savez_compressed(output / "eigen_distribution_histogram.npz", density=density, edges=edges, cutoff=cutoff)
    report["regenerated_histogram"] = {
        "source": str(data_path.relative_to(root)), "source_sha256": digest(data_path),
        "target": str(target.relative_to(root)), "sha256": digest(target),
        "sample_count": len(values), "retained_count": len(filtered),
        "cutoff": cutoff, "bins": 10000,
        "exact_original_reproduction": False,
    }
    listing = subprocess.check_output(["pdfimages", "-list", str(target)], text=True)
    assert len(listing.strip().splitlines()) == 2
    (output / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
