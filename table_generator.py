#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------- Helpers -------------------------------- #

ALG_PRETTY = {
    "mlp": "MLP",
    "sdp": "SDP",
    "ldlt": "LDLT",
    "ldlt-resnet": "LDLT-ResNet",
    "aol": "AOL",
    "ortho": "Orthogonal",
    "sandwich": "Sandwich",
}

METRICS_PRETTY = {
    'mean_cert_acc_108': "Mean Certified Accuracy (108/255)",
    'mean_cert_acc_255': "Mean Certified Accuracy (255/255)",
    'mean_cert_acc_36': "Mean Certified Accuracy (36/255)",
    'mean_cert_acc_72': "Mean Certified Accuracy (72/255)",
    'mean_test_acc': "Mean Accuracy"
}

STAR_THRESH = [(0.001, r"^{***}"), (0.01, r"^{**}"), (0.05, r"^{*}")]


def alg_pretty(a):
    return ALG_PRETTY.get(a, a)


def fmt_mean_std(x):
    if pd.isna(x["mean"]) or pd.isna(x["std"]):
        return r"\textemdash"
    return rf"{x['mean']:.4f}\,\tiny$\pm${x['std']:.4f}"


def p_to_stars(p):
    if pd.isna(p):
        return ""
    for th, tag in STAR_THRESH:
        if p <= th:
            return tag
    return ""


def bestbold(series):
    # bold (max) numerical entries; ignore em-dash
    vals = pd.to_numeric(series.str.extract(r"([0-9]+\.?[0-9]*)")[0], errors="coerce")
    if vals.isna().all():
        return series
    best = vals.max(skipna=True)
    mask = vals.eq(best)
    out = series.copy()
    out[mask] = out[mask].apply(lambda s: rf"\textbf{{{s}}}")
    return out


def latex_escape(s):
    return (s.replace("_", r"\_")
            .replace("%", r"\%")
            .replace("&", r"\&")
            .replace("#", r"\#")
            .replace("$", r"\$")
            .replace("{", r"\{")
            .replace("}", r"\}")
            .replace("~", r"\textasciitilde{}")
            .replace("^", r"\textasciicircum{}")
            .replace("\\", r"\textbackslash{}"))


def metric_to_name(s: str):
    return METRICS_PRETTY.get(s, latex_escape(s))


def ensure_algos(path):
    # discover algorithms by presence of */summary.json under out_dir
    algs = []
    for p in path.iterdir():
        if p.is_dir() and (p / "summary.json").exists():
            algs.append(p.name)
    return sorted(algs)


# -------------------------- Load experiment data ------------------------- #

def load_all(out_dir: Path, algorithms=None):
    # summary.json per algorithm (produced during schedule_and_run)
    if algorithms is None:
        algorithms = ensure_algos(out_dir)

    per_alg = {}
    for alg in algorithms:
        f = out_dir / alg / "summary.json"
        if not f.exists():
            continue
        with open(f, "r") as fh:
            j = json.load(fh)
        per_alg[alg] = j

    # Wilcoxon pairwise (produced by run_wilcoxon_tests_all_metrics)
    wilc = out_dir / "wilcoxon_pairwise_all.json"
    wilc_df = pd.read_json(wilc) if wilc.exists() else pd.DataFrame()
    return per_alg, wilc_df, algorithms


# --------------------- Build metric summary across DS -------------------- #

def sorting_key(m: str):
    if m == 'mean_test_acc':
        return (0, 0)
    # sort certs numerically by epsilon value at the end
    try:
        return (1, int(m.split('_')[-1]))
    except Exception:
        return (2, m)


def build_metric_summary(per_alg):
    """
    Returns:
      metrics -> DataFrame with rows=Algorithm, cols=['N_datasets', 'Value'] where Value is mean±std string.
    Also returns raw numeric table for potential ranking (mean across datasets).
    """
    # collect all (dataset, algorithm) rows with all mean_* metrics
    rows = []
    for alg, j in per_alg.items():
        for ds, payload in j.get("datasets", {}).items():
            for k, v in payload.items():
                if k.startswith("mean_") and (isinstance(v, (int, float)) and math.isfinite(v)):
                    rows.append({"dataset": ds, "algorithm": alg, "metric": k, "value": float(v)})
    df = pd.DataFrame(rows)
    if df.empty:
        return {}, {}

    metrics = sorted(df["metric"].unique().tolist(), key=sorting_key)
    result_tables = {}
    numeric_tables = {}

    for m in metrics:
        sub = df[df["metric"] == m]
        # mean±std across datasets per algorithm
        grp = sub.groupby("algorithm")["value"].agg(["mean", "std", "count"]).reset_index()
        grp["Algorithm"] = grp["algorithm"].map(alg_pretty)
        grp = grp.sort_values("Algorithm")

        show = grp[["Algorithm", "count", "mean", "std"]].copy()
        show.rename(columns={"count": "$N$ datasets"}, inplace=True)
        show["Value"] = show.apply(fmt_mean_std, axis=1)
        show = show[["Algorithm", "$N$ datasets", "Value"]]
        show["Value"] = bestbold(show["Value"])

        result_tables[m] = show
        numeric_tables[m] = grp[["Algorithm", "mean"]].set_index("Algorithm")["mean"]
    return result_tables, numeric_tables


# ------------------------ Build Wilcoxon tables -------------------------- #

def build_wilcoxon_tables(wilc_df):
    if wilc_df.empty:
        return {}

    # Make prettier labels and stars on Holm-within-metric
    tdict = {}
    for m, sub in wilc_df.groupby("metric"):
        t = sub.copy()
        t["Alg A"] = t["alg_a"].map(alg_pretty)
        t["Alg B"] = t["alg_b"].map(alg_pretty)
        t["WinRate A"] = (t["win_rate_a_over_b"]).round(4)
        t["Median AB"] = t["median_diff_a_minus_b"].round(4)  # Median Δ; we'll label with LaTeX in header
        t["W"] = t["W_stat"].round(4)
        t["p"] = t["p_two_sided"].apply(lambda p: f"{p:.3g}")
        t["p_within"] = t["p_holm_within_metric"].apply(lambda p: f"${p:.3g}{p_to_stars(p)}$")
        t["p_global"] = t["p_holm_global"].apply(lambda p: f"${p:.3g}{p_to_stars(p)}$")
        t["r"] = t["effect_size_r"].round(4)
        t["n"] = t["n_common"].astype(int)
        t["winsA"] = t["wins_a"].astype(int)
        t["winsB"] = t["wins_b"].astype(int)
        t["ties"] = t["ties"].astype(int)

        cols = ["Alg A", "Alg B", "n", "winsA", "winsB", "ties",
                "WinRate A", "Median AB", "W", "p", "p_within", "p_global", "r",
                "p_holm_within_metric"]
        tdict[m] = t[cols].sort_values(["p_holm_within_metric", "Alg A", "Alg B"]).drop(
            columns=["p_holm_within_metric"])
    return tdict


# ------------------------ Model size ranges table ------------------------ #

def build_model_size_ranges(per_alg):
    """
    Inspect fold_results to compute min–max across all datasets/folds for:
      width, depth, parameters, padding, in_features, out_features
    Return DataFrame with one row per algorithm.
    """
    recs = []
    for alg, j in per_alg.items():
        ws, ds, ps, pad, din, dout = [], [], [], [], [], []
        for ds_name, payload in j.get("datasets", {}).items():
            for fr in payload.get("fold_results", []):
                for key, bag in [("width", ws), ("depth", ds), ("parameters", ps),
                                 ("padding", pad), ("in_features", din), ("out_features", dout)]:
                    if key in fr and fr[key] is not None:
                        bag.append(fr[key])

        def rng(a):
            if not a:
                return r"\textemdash"
            return rf"{int(np.min(a))}--{int(np.max(a))}"  # use TeX en-dash

        recs.append({
            "Algorithm": alg_pretty(alg),
            "Width": rng(ws),
            "Depth": rng(ds),
            "Parameters": rng(ps),
            "Padding": rng(pad),
            "Input dim": rng(din),
            "Output dim": rng(dout),
        })
    df = pd.DataFrame(recs).sort_values("Algorithm")
    return df


# ---------------------------- LaTeX emitters ---------------------------- #

HEADER = r"""
% JMLR-ready tables exported from UCI tabular experiments
% Requires: \usepackage{booktabs}, \usepackage{threeparttable}, \usepackage{threeparttablex}, \usepackage{longtable}
% NOTE: siunitx is NOT required.
"""


def tex_table_metric_summary(metric_name, df):
    cap = f"Mean$\\pm$std across datasets for each algorithm on {metric_to_name(metric_name)}."
    lab = f"tab:{metric_name}"
    # Only basic column types to avoid siunitx
    colspec = "l r l"
    lines = [r"\begin{table}[t]",
             r"\centering",
             r"\begin{threeparttable}",
             rf"\caption{{{cap}}}",
             rf"\label{{{lab}}}",
             rf"\begin{{tabular}}{{{colspec}}}",
             r"\toprule",
             r"Algorithm & $N$ datasets & Value \\",
             r"\midrule"]
    for _, r in df.iterrows():
        lines.append(rf"{r['Algorithm']} & {int(r['$N$ datasets'])} & {r['Value']} \\")
    lines += [r"\bottomrule",
              r"\end{tabular}",
              r"\end{threeparttable}",
              r"\end{table}"]
    return "\n".join(lines)


def tex_longtable_wilcoxon(metric_name, df):
    cap = (r"Wilcoxon signed-rank tests (two-sided) for " + metric_to_name(metric_name) +
           r"; $p$-values with Holm FWER corrections within-metric and global. "
           r"Stars mark significance ($^*\,p\!\le\!0.05$, $^{**}\,p\!\le\!0.01$, $^{***}\,p\!\le\!0.001$).")
    lab = f"tab:wilcoxon:{metric_name}"
    # Plain columns; no siunitx
    colspec = "l l r r r r r r r r r r r"
    head = (r"Alg A & Alg B & $n$ & wins$_A$ & wins$_B$ & ties & "
            r"WinRate A & Median $\Delta$ (A--B) & $W$ & $p$ & $p_{\text{Holm,within}}$ & $p_{\text{Holm,global}}$ & $r$ \\")

    lines = [r"\begin{ThreePartTable}",
             rf"\begin{{TableNotes}}\item {cap}\end{{TableNotes}}",
             r"\begin{longtable}{" + colspec + "}",
             rf"\caption[{metric_to_name(metric_name)}]{{{cap}}}\label{{{lab}}}\\",
             r"\toprule",
             head,
             r"\midrule",
             r"\endfirsthead",
             r"\toprule",
             head,
             r"\midrule",
             r"\endhead",
             r"\bottomrule",
             r"\insertTableNotes",
             r"\endfoot"]
    for _, r in df.iterrows():
        lines.append(
            rf"{r['Alg A']} & {r['Alg B']} & {r['n']} & {r['winsA']} & {r['winsB']} & {r['ties']} & "
            rf"{r['WinRate A']:.4f} & {r['Median AB']:.4f} & {r['W']:.4f} & {r['p']} & "
            rf"{r['p_within']} & {r['p_global']} & {r['r']:.4f} \\")
    lines += [r"\end{longtable}",
              r"\end{ThreePartTable}"]
    return "\n".join(lines)


def tex_model_ranges(df):
    cap = ("Model dimension ranges (min--max across all datasets and folds). "
           "Input/Output dimensions follow dataset label spaces.")
    lab = "tab:model-ranges"
    colspec = "l r r r r r r"
    lines = [r"\begin{table}[t]",
             r"\centering",
             r"\begin{threeparttable}",
             rf"\caption{{{cap}}}",
             rf"\label{{{lab}}}",
             rf"\begin{{tabular}}{{{colspec}}}",
             r"\toprule",
             r"Algorithm & Width & Depth & Parameters & Padding & Input dim & Output dim \\",
             r"\midrule"]
    for _, r in df.iterrows():
        lines.append(rf"{r['Algorithm']} & {r['Width']} & {r['Depth']} & {r['Parameters']} & "
                     rf"{r['Padding']} & {r['Input dim']} & {r['Output dim']} \\")
    lines += [r"\bottomrule",
              r"\end{tabular}",
              r"\end{threeparttable}",
              r"\end{table}"]
    return "\n".join(lines)


# ------------------------------ Main CLI -------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default='runs/UCI', type=Path,
                    help="The SAME directory you passed as --out to the trainer (contains per-alg subdirs and wilcoxon files).")
    ap.add_argument("--algorithms", nargs="*", default=None,
                    help="Optional explicit list; default is to detect subfolders containing summary.json")
    ap.add_argument("--tex", default="jmlr_tables.tex", help="Output LaTeX file")
    args = ap.parse_args()

    per_alg, wilc_df, algs = load_all(args.out_dir, args.algorithms)
    if not per_alg:
        raise SystemExit(f"No summary.json files found under {args.out_dir}")

    metric_tables, numeric = build_metric_summary(per_alg)
    wilc_tables = build_wilcoxon_tables(wilc_df)
    ranges_df = build_model_size_ranges(per_alg)

    tex_chunks = [HEADER.strip(), ""]

    # Model ranges first (small and universally useful)
    tex_chunks.append(tex_model_ranges(ranges_df))
    tex_chunks.append("")

    # Per-metric performance summaries and their Wilcoxon longtables
    for m in sorted(metric_tables.keys(), key=sorting_key):
        tex_chunks.append(tex_table_metric_summary(m, metric_tables[m]))
        tex_chunks.append("")
        if m in wilc_tables:
            tex_chunks.append(tex_longtable_wilcoxon(m, wilc_tables[m]))
            tex_chunks.append("")

    out = "\n".join(tex_chunks).strip() + "\n"
    Path(args.tex).write_text(out, encoding="utf-8")
    print(f"Wrote LaTeX to {args.tex}")


if __name__ == "__main__":
    main()
