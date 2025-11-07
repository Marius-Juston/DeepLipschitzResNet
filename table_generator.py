#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import rankdata, f, studentized_range

# ------------------------------- Helpers -------------------------------- #

ALG_PRETTY = {
    "mlp": "MLP",
    "sdp": "SLL",
    "ldlt": "LDLT-L",
    "ldlt-resnet": "LDLT-R",
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
        numeric_tables[m] = grp[["Algorithm", "mean", "std"]].set_index("Algorithm")[["mean", 'std']]
    return result_tables, numeric_tables


# ------------------------ Build Wilcoxon tables -------------------------- #

def format_small_p(p):
    if p == 0:
        return f"$0{p_to_stars(p)}$"

    return f"${p:.1e}{p_to_stars(p)}$"


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
        t["p"] = t["p_two_sided"].apply(format_small_p)
        t["p_within"] = t["p_holm_within_metric"].apply(format_small_p)
        t["p_global"] = t["p_holm_global"].apply(format_small_p)
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
            return rf"{int(np.min(a))}--{int(np.max(a))}"  # TeX en-dash

        recs.append({
            "Algorithm": alg_pretty(alg),
            "Width": rng(ws),
            "Depth": rng(ds),
            "Parameters": rng(ps),
            "Padding": rng(pad),
            "Input dim": rng(din),
            "Output dim": rng(dout),
        })

    df = pd.DataFrame(recs)

    df = df.sort_values(
        by="Algorithm",
        key=lambda s: list(zip(s.ne("MLP"), s.str.contains("LDLT", na=False), s.str.casefold()))
    ).reset_index(drop=True)

    return df


# ---- NEW: per-dataset matrix (rows=datasets, cols=algorithms) ----
def _dataset_metric_matrix(per_alg: dict, metric: str) -> pd.DataFrame:
    rows = []
    for alg, js in per_alg.items():
        for ds, payload in js.get("datasets", {}).items():
            v = payload.get(metric, None)
            if isinstance(v, (int, float)) and np.isfinite(v):
                rows.append({"dataset": ds, "algorithm": alg_pretty(alg), "value": float(v)})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.pivot_table(index="dataset", columns="algorithm", values="value", aggfunc="mean")


# ---- NEW: Friedman / Iman–Davenport / Nemenyi CD ----
def _friedman_iman_nemenyi(pivot: pd.DataFrame, alpha: float = 0.05):
    if pivot.empty or pivot.shape[1] < 2:
        return None
    X = pivot.to_numpy(float)
    N, k = X.shape
    ranks = np.vstack([rankdata(-row, method="average") for row in X])  # 1=best
    avg_ranks = pd.Series(ranks.mean(axis=0), index=pivot.columns).sort_values()

    chi2_F = (12 * N) / (k * (k + 1)) * (np.sum(avg_ranks.values ** 2) - k * (k + 1) ** 2 / 4.0)
    denom = N * (k - 1) - chi2_F
    F_F = ((N - 1) * chi2_F) / denom if denom > 1e-12 else np.inf
    p_F = 1 - f.cdf(F_F, k - 1, (k - 1) * (N - 1))

    q_alpha = studentized_range.isf(alpha, k, np.inf)
    CD = float(q_alpha * np.sqrt(k * (k + 1) / (6.0 * N)))

    return {"N": N, "k": k, "avg_ranks": avg_ranks, "F_F": float(F_F), "p_F": float(p_F), "CD": CD}


# ---- NEW: summarize Wilcoxon wins/losses for each algorithm (Holm within-metric) ----
def _wilcoxon_summary_for_metric(wilc_df: pd.DataFrame, metric: str, alpha: float = 0.05) -> pd.DataFrame:
    if wilc_df is None or wilc_df.empty:
        return pd.DataFrame()
    sub = wilc_df[wilc_df["metric"] == metric].copy()
    if sub.empty:
        return pd.DataFrame()

    sub["win_A"] = (sub["median_diff_a_minus_b"] > 0) & (sub["p_holm_within_metric"] <= alpha)
    sub["win_B"] = (sub["median_diff_a_minus_b"] < 0) & (sub["p_holm_within_metric"] <= alpha)

    algs = sorted(set(sub["alg_a"]).union(set(sub["alg_b"])))
    rows = []
    for a in algs:
        sig_wins = int(sub[(sub["alg_a"] == a) & sub["win_A"]].shape[0] +
                       sub[(sub["alg_b"] == a) & sub["win_B"]].shape[0])
        sig_losses = int(sub[(sub["alg_a"] == a) & sub["win_B"]].shape[0] +
                         sub[(sub["alg_b"] == a) & sub["win_A"]].shape[0])

        r_wins = []
        r_wins += sub[(sub["alg_a"] == a) & sub["win_A"]]["effect_size_r"].tolist()
        r_wins += sub[(sub["alg_b"] == a) & sub["win_B"]]["effect_size_r"].tolist()
        mean_r_win = float(np.mean(r_wins)) if len(r_wins) else 0.0

        total_pairs = len(algs) - 1
        rows.append({
            "Algorithm": alg_pretty(a),
            "sig_wins": sig_wins,
            "sig_losses": sig_losses,
            "net_sig_wins": sig_wins - sig_losses,
            "sig_win_share": sig_wins / total_pairs if total_pairs > 0 else np.nan,
            "mean_r_win": mean_r_win,
        })

    return pd.DataFrame(rows).set_index("Algorithm").sort_values(
        ["net_sig_wins", "sig_wins", "mean_r_win"], ascending=False
    )


# ---- NEW: overall metric stats (avg ranks + Wilcoxon summaries) ----
def build_overall_metric_stats(per_alg: dict, wilc_df: pd.DataFrame, alpha: float = 0.05):
    # collect metrics that appear in summaries
    metrics = set()
    for _, js in per_alg.items():
        for _, payload in js.get("datasets", {}).items():
            for k, v in payload.items():
                if k.startswith("mean_") and isinstance(v, (int, float)) and np.isfinite(v):
                    metrics.add(k)
    metrics = sorted(metrics, key=sorting_key)

    out_tables, caps = {}, {}
    for m in metrics:
        pivot = _dataset_metric_matrix(per_alg, m)
        fried = _friedman_iman_nemenyi(pivot, alpha=alpha)
        if fried is None:
            continue
        avg = fried["avg_ranks"].rename("Avg rank ↓").to_frame()
        wsum = _wilcoxon_summary_for_metric(wilc_df, m, alpha=alpha)
        tbl = avg.join(wsum, how="left").fillna(
            {"sig_wins": 0, "sig_losses": 0, "net_sig_wins": 0, "sig_win_share": 0.0, "mean_r_win": 0.0}
        )

        upper = tbl.drop(index=[i for i in tbl.index if i in ("LDLT-L", "LDLT-R")], errors="ignore")
        lower = tbl.loc[[i for i in ("LDLT-L", "LDLT-R") if i in tbl.index]]
        out_tables[m] = pd.concat([upper, lower], axis=0)

        caps[m] = {"N": fried["N"], "k": fried["k"], "F_F": fried["F_F"], "p_F": fried["p_F"], "CD": fried["CD"]}
    return out_tables, caps


# ---------------------------- LaTeX emitters ---------------------------- #

HEADER = r"""
% JMLR-ready tables for two-column papers
% Required packages (no siunitx):
% \usepackage{booktabs}
% \usepackage{threeparttable}
% \usepackage{threeparttablex} % for TableNotes + longtable
% \usepackage{longtable}
% Optional for landscape: \usepackage{pdflscape}
"""


def tex_table_metric_summary(metric_name, df, wide=False):
    cap = f"Sorted mean$\\pm$std across $N$ datasets for each algorithm on {metric_to_name(metric_name)}."
    lab = f"tab:{metric_name}"
    env = "table*" if wide else "table"
    colspec = "l r l"
    lines = [fr"\begin{{{env}}}[t]",
             r"\centering",
             r"\begin{threeparttable}",
             rf"\caption{{{cap}}}",
             rf"\label{{{lab}}}",
             rf"\begin{{tabular}}{{{colspec}}}",
             r"\toprule",
             r"Algorithm & $N$ & Value \\",
             r"\midrule"]

    df = df.sort_values(by='Value', ascending=False)

    for _, r in df.iterrows():
        lines.append(rf"{r['Algorithm']} & {int(r['$N$ datasets'])} & {r['Value']} \\")
    lines += [r"\bottomrule",
              r"\end{tabular}",
              r"\end{threeparttable}",
              fr"\end{{{env}}}"]
    return "\n".join(lines)


def get_data_metrics(lines: List[str], algorithms: List[str], metrics: List[str], metric_summary: pd.DataFrame):
    for alg in algorithms:
        data = [alg]

        for col in metrics:
            if not metric_summary[col]['Algorithm'].isin([alg]).any():
                data.append(r"\textemdash")
                continue

            metric_values = metric_summary[col]

            entries = metric_values.loc[metric_values["Algorithm"] == alg]

            if len(data) == 1:
                data.append(str(entries["$N$ datasets"].iat[0]))

            data.append(entries["Value"].iat[0])

        lines.append(" & ".join(data) + r" \\")


def tex_table_metric_summary_combined(metric_summary, wide=True):
    cap = f"Sorted mean$\\pm$std across $N$ datasets for each algorithm."
    lab = f"tab:metric_summary"
    env = "table*" if wide else "table"

    columns = sorted(metric_summary.keys(), key=sorting_key)

    colspec = f"l r {'l' * len(columns)}"
    lines = [fr"\begin{{{env}}}[t]",
             r"\centering",
             r"\begin{threeparttable}",
             rf"\caption{{{cap}}}",
             rf"\label{{{lab}}}",
             rf"\begin{{tabular}}{{{colspec}}}",
             r"\toprule",
             r' &  &  & \multicolumn{4}{c}{Certified Accuracy} \\',
             r'\cmidrule(lr){4-7}',
             r"Algorithm & $N$ & Accuracy & 36/255 & 72/255 & 108/255 & 255/255 \\",
             r"\midrule"]

    seperate = "LDLT"

    methods = set(list(metric_summary.values())[0]['Algorithm'].values)

    separate_methods = set(m for m in methods if seperate in m)
    grouped_methods = methods.difference(separate_methods)

    grouped_methods = list(sorted(grouped_methods, key=lambda x: (x != 'MLP', 'LDLT' in x, str.casefold(x))))
    separate_methods = list(sorted(separate_methods))

    if len(grouped_methods) > 0:
        get_data_metrics(lines, grouped_methods, columns, metric_summary)

        lines.append(r"\midrule")

    if len(separate_methods) > 0:
        get_data_metrics(lines, separate_methods, columns, metric_summary)

    lines += [r"\bottomrule",
              r"\end{tabular}",
              r"\end{threeparttable}",
              fr"\end{{{env}}}"]
    return "\n".join(lines)


def tex_longtable_wilcoxon(metric_name, df, span_mode="table*"):
    r"""
    span_mode:
      - 'longtable'  : emits ThreePartTable + longtable (requires single-column or \onecolumn ... \twocolumn page breaks)
      - 'table*'     : emits wide float using tabular (no page breaks; good for two-column spanning)
      - 'landscape'  : wraps table* inside landscape environment
    """
    cap_base = (r"Wilcoxon signed-rank tests (two-sided) for " + metric_to_name(metric_name) +
                r"; $p$-values with Holm FWER corrections within-metric and global.")
    lab = f"tab:wilcoxon:{metric_name}"
    head_groups = (
        r'\multicolumn{2}{c}{Algorithms} & \multicolumn{6}{c}{Run Statistics} & \multicolumn{5}{c}{Wilcoxon pairwise Statistics} \\'
        r'\cmidrule(lr){1-2} \cmidrule(lr){3-8} \cmidrule(lr){9-13}')
    head = (r"Alg A & Alg B & $n$ & wins$_A$ & wins$_B$ & ties & "
            r"WinRate A & \shortstack{Median \\ $\Delta$ (A--B)} & $W$ & $p$ & $p_{\text{Holm,within}}$ & $p_{\text{Holm,global}}$ & $r$ \\")

    df = df.sort_values(by='W')

    if span_mode == "longtable":
        colspec = "ll r r r r r r r r r r r"
        lines = [r"\onecolumn",
                 r"\begin{ThreePartTable}",
                 rf"\begin{{TableNotes}}[flushleft]\footnotesize {cap_base}\end{{TableNotes}}",
                 r"\begin{longtable}{" + colspec + "}",
                 rf"\caption[{metric_to_name(metric_name)}]{{{cap_base}}}\label{{{lab}}}\\",
                 r"\toprule",
                 head_groups,
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
                rf"{r['WinRate A']:.4f} & {r['Median AB']:.4f} & {r['W']:.0f} & {r['p']} & "
                rf"{r['p_within']} & {r['p_global']} & {r['r']:.4f} \\")
        lines += [r"\end{longtable}",
                  r'\endgroup'
                  r"\end{ThreePartTable}",
                  r"\twocolumn"]
        return "\n".join(lines)

    # For two-column spanning without page breaks: table* + tabular
    colspec = "ll r r r r r r r r r r r"
    maybe_landscape_begin = r"\begin{landscape}" if span_mode == "landscape" else ""
    maybe_landscape_end = r"\end{landscape}" if span_mode == "landscape" else ""
    lines = [maybe_landscape_begin,
             r"\begin{table*}[t]",
             r"\centering",
             r"\begin{threeparttable}",
             rf"\caption[{metric_to_name(metric_name)}]{{{cap_base}}}",
             rf"\label{{{lab}}}",
             r'\begingroup',
             r'\setlength{\tabcolsep}{4pt}',
             rf"\begin{{tabular}}{{{colspec}}}",
             r"\toprule",
             head_groups,
             head,
             r"\midrule"]
    for _, r in df.iterrows():
        lines.append(
            rf"{r['Alg A']} & {r['Alg B']} & {r['n']} & {r['winsA']} & {r['winsB']} & {r['ties']} & "
            rf"{r['WinRate A']:.4f} & {r['Median AB']:.4f} & {r['W']:.0f} & {r['p']} & "
            rf"{r['p_within']} & {r['p_global']} & {r['r']:.4f} \\")
    lines += [r"\bottomrule",
              r"\end{tabular}",
              r'\begin{tablenotes}',
              r'\item Stars mark significance ($^*\,p\!\le\!0.05$, $^{**}\,p\!\le\!0.01$, $^{***}\,p\!\le\!0.001$).',
              r'\end{tablenotes}',
              r'\endgroup',
              r"\end{threeparttable}",
              r"\end{table*}",
              maybe_landscape_end]
    return "\n".join(lines)


def tex_model_ranges(df, wide=False):
    cap = ("Model dimension ranges (min--max across all datasets and folds). "
           "Input/Output dimensions follow dataset label spaces.")
    lab = "tab:model-ranges"
    env = "table*" if wide else "table"
    colspec = "l r r r r r r"
    lines = [fr"\begin{{{env}}}[t]",
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
              fr"\end{{{env}}}"]
    return "\n".join(lines)


# ---- NEW: overall per-metric summary table (two-column ready) ----
def tex_table_overall_summary(metric_name: str, df: pd.DataFrame, caps: dict, wide: bool = True) -> str:
    env = "table*" if wide else "table"
    cap = (f"Overall comparison on {metric_to_name(metric_name)}: average rank (lower is better) "
           f"with Iman--Davenport $F={caps['F_F']:.2f}$ (df={int(caps['k'] - 1)},{int((caps['k'] - 1) * (caps['N'] - 1))}), "
           f"$p={caps['p_F']:.3g}$; Nemenyi CD$={caps['CD']:.3f}$. "
           r"Counts are significant wins/losses after Holm within-metric at $\alpha=0.05$.")
    lab = f"tab:overall:{metric_name}"
    colspec = "@{}l r r r r r r@{}"
    header = r"Algorithm & \shortstack{Avg \\ rank} $\downarrow$ & \shortstack{sig \\ wins} & \shortstack{sig \\ losses} & \shortstack{net \\ wins} & \shortstack{win \\ share} & mean $r$ \\"

    body = []
    # upper = df.drop(index=[i for i in df.index if i in ("LDLT-L", "LDLT-R")], errors="ignore")
    upper = df.sort_values(by='Avg rank ↓')

    # lower = df.loc[[i for i in ("LDLT-L", "LDLT-R") if i in df.index]]
    for idx, row in upper.iterrows():
        body.append(rf"{idx} & {row['Avg rank ↓']:.3f} & {int(row['sig_wins'])} & {int(row['sig_losses'])} & "
                    rf"{int(row['net_sig_wins'])} & {row['sig_win_share']:.3f} & {row['mean_r_win']:.3f} \\")
    # if not lower.empty:
    #     body.append(r"\midrule")
    #     for idx, row in lower.iterrows():
    #         body.append(rf"{idx} & {row['Avg rank ↓']:.3f} & {int(row['sig_wins'])} & {int(row['sig_losses'])} & "
    #                     rf"{int(row['net_sig_wins'])} & {row['sig_win_share']:.3f} & {row['mean_r_win']:.3f} \\")

    lines = [
        fr"\begin{{{env}}}[t]", r"\centering", r"\begin{threeparttable}", r"{\small",
        rf"\caption{{{cap}}}", rf"\label{{{lab}}}", r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", header, r"\midrule",
        *body, r"\bottomrule", r"\end{tabular}", r"}", r"\end{threeparttable}", fr"\end{{{env}}}",
    ]
    return "\n".join(lines)


# ---- NEW: critical difference (CD) diagram (TikZ) ----
def tex_cd_diagram(metric_name: str, avg_ranks: pd.Series, CD: float, width: float = 12.0) -> str:
    algs = list(avg_ranks.index)
    ranks = avg_ranks.values
    k = len(algs)
    xmin, xmax = 1.0, float(k)

    def sx(r):
        return (r - xmin) / (xmax - xmin) * width

    # simple greedy cliques for non-significant groups (range <= CD)
    blocks, i = [], 0
    while i < k:
        j, rmin, rmax = i, ranks[i], ranks[i]
        while j + 1 < k:
            rmin, rmax = min(rmin, ranks[j + 1]), max(rmax, ranks[j + 1])
            if (rmax - rmin) <= CD + 1e-12:
                j += 1
            else:
                break
        blocks.append((i, j))
        i = j + 1

    cap = (rf"Critical Difference diagram for {metric_to_name(metric_name)} "
           rf"(CD = {CD:.2f}; lower rank = better).")
    lab = f"fig:cd:{metric_name}"
    lines = [r"\begin{figure*}[t]", r"\centering", r"\begin{threeparttable}",
             r"{\small",
             rf"\label{{{lab}}}",
             r"\begin{tikzpicture}[x=1cm,y=1cm]",
             rf"\draw[thick] (0,0) -- ({width:.3f},0);"]
    for r in range(1, k + 1):
        x = sx(float(r))
        lines += [rf"\draw ({x:.3f},0.1) -- ({x:.3f},-0.1) node[below]{{\scriptsize {r}}};"]
    cd_left, cd_right = sx(xmin), sx(xmin + CD)
    lines += [rf"\draw[thick] ({cd_left:.3f},1.0) -- ({cd_right:.3f},1.0);",
              rf"\draw[thick] ({cd_left:.3f},0.9) -- ({cd_left:.3f},1.1);",
              rf"\draw[thick] ({cd_right:.3f},0.9) -- ({cd_right:.3f},1.1);",
              rf"\node[above] at ({(cd_left + cd_right) / 2:.3f},1.1) {{\scriptsize CD}};"]
    for idx, (alg, r) in enumerate(zip(algs, ranks)):
        x = sx(r)
        y = 0.35 if (idx % 2 == 0) else -0.35
        lines += [rf"\draw[thick] ({x:.3f},0) -- ({x:.3f},{0.25 if y > 0 else -0.25});",
                  rf"\node[align=center] at ({x:.3f},{y:.3f}) {{\scriptsize {alg}}};"]
    y0, dy = 0.65, 0.18
    for b, (i0, i1) in enumerate(blocks):
        if i1 <= i0: continue
        xL, xR = sx(ranks[i0]), sx(ranks[i1])
        y = y0 + b * dy
        lines += [rf"\draw[very thick] ({xL:.3f},{y:.3f}) -- ({xR:.3f},{y:.3f});"]
    lines += [r"\end{tikzpicture}", r"}",
              r"\end{threeparttable}",
              rf"\caption{{{cap}}}",
              r"\end{figure*}"]
    return "\n".join(lines)


# ---- NEW: Wilcoxon significance matrix (▲/▼/·) ----
def tex_significance_matrix(metric_name: str, wilc_df: pd.DataFrame, alpha: float = 0.05, wide=False) -> str:
    sub = wilc_df[wilc_df["metric"] == metric_name].copy()
    if sub.empty: return "% No Wilcoxon data for this metric."
    algs = sorted(set(sub["alg_a"]).union(set(sub["alg_b"])))
    idx = {a: i for i, a in enumerate(algs)}

    tie = r"$\cdot$"
    win = r"\textcolor{green}{$\blacktriangle$}" # ▲
    loss = r"\textcolor{red}{$\blacktriangledown$}" # ▼

    sym = [[tie for _ in algs] for _ in algs]

    for _, r in sub.iterrows():
        a, b = r["alg_a"], r["alg_b"]
        pa, med = r["p_holm_within_metric"], r["median_diff_a_minus_b"]
        i, j = idx[a], idx[b]
        if pa <= alpha and med > 0:
            sym[i][j], sym[j][i] = win, loss  # ▲, ▼
        elif pa <= alpha and med < 0:
            sym[i][j], sym[j][i] = loss, win  # ▼, ▲

    env = "table*" if wide else "table"
    colspec = "@{}l " + " ".join(["c"] * len(algs)) + " @{}"
    cap = (r"Pairwise Wilcoxon outcomes for " + metric_to_name(metric_name) +
           fr" (Holm within-metric at $\alpha=0.05$): row vs. column ({win} win, {loss} loss, {tie} none).")
    lab = f"tab:signif:{metric_name}"
    header = " & ".join(alg_pretty(a) for a in algs)

    lines = [fr"\begin{{{env}}}[t]", r"\centering", r"\begin{threeparttable}", r"{",
             rf"\caption{{{cap}}}", rf"\label{{{lab}}}", r"\setlength{\tabcolsep}{3pt}",
             rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", rf" & {header} \\", r"\midrule"]
    for i, a in enumerate(algs):
        row = " & ".join(sym[i])
        lines.append(rf"{alg_pretty(a)} & {row} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{threeparttable}", fr"\end{{{env}}}"]
    return "\n".join(lines)


# ------------------------------ Main CLI -------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default='runs/UCI', type=Path,
                    help="The SAME directory you passed as --out to the trainer (contains per-alg subdirs and wilcoxon files).")
    ap.add_argument("--algorithms", nargs="*", default=None,
                    help="Optional explicit list; default is to detect subfolders containing summary.json")
    ap.add_argument("--tex", default="jmlr_tables.tex", help="Output LaTeX file")
    ap.add_argument("--wide-summaries", type=bool, default=True, action=argparse.BooleanOptionalAction,
                    help="Emit model ranges as two-column floats (table*).")
    ap.add_argument("--wide-metrics-summary", type=bool, default=False, action=argparse.BooleanOptionalAction,
                    help="Emit model metric summaries as two-column floats (table*).")
    ap.add_argument("--use-per-metrics-summary", type=bool, default=False, action=argparse.BooleanOptionalAction,
                    help="Generate the metric summaries for each metric separately.")
    ap.add_argument("--wilcoxon-span", choices=["table*", "longtable", "landscape"], default="table*",
                    help="How to span the Wilcoxon tables in two-column docs: 'table*' (default), 'longtable' (forces page breaks via \\onecolumn/\\twocolumn), or 'landscape' (table* inside pdflscape).")
    # NEW flags in argparse:
    ap.add_argument("--emit-overall-summary", action=argparse.BooleanOptionalAction, default=True,
                    help="Emit overall per-metric summary table (avg ranks + Wilcoxon counts).")
    ap.add_argument("--emit-cd-diagram", action=argparse.BooleanOptionalAction, default=False,
                    help="Emit TikZ critical-difference diagrams per metric (requires \\usepackage{tikz}).")
    ap.add_argument("--emit-signif-matrix", action=argparse.BooleanOptionalAction, default=True,
                    help="Emit Wilcoxon significance matrix (▲/▼/·) per metric.")

    args = ap.parse_args()

    per_alg, wilc_df, algs = load_all(args.out_dir, args.algorithms)
    if not per_alg:
        raise SystemExit(f"No summary.json files found under {args.out_dir}")

    metric_tables, numeric = build_metric_summary(per_alg)
    wilc_tables = build_wilcoxon_tables(wilc_df)
    ranges_df = build_model_size_ranges(per_alg)

    overall_tables, overall_caps = build_overall_metric_stats(per_alg, wilc_df, alpha=0.05)

    tex_chunks = [HEADER.strip(), ""]

    # Model ranges first (universally useful)
    tex_chunks.append(tex_model_ranges(ranges_df, wide=args.wide_summaries))
    tex_chunks.append("")

    tex_chunks.append(tex_table_metric_summary_combined(metric_tables))
    tex_chunks.append("")

    # Per-metric performance summaries and their Wilcoxon tables
    for m in sorted(metric_tables.keys(), key=sorting_key):
        if args.use_per_metrics_summary:
            tex_chunks.append(tex_table_metric_summary(m, metric_tables[m], wide=args.wide_metrics_summary))
            tex_chunks.append("")

    for m in sorted(metric_tables.keys(), key=sorting_key):
        if args.emit_overall_summary and m in overall_tables:
            tex_chunks.append(tex_table_overall_summary(m, overall_tables[m], overall_caps[m], wide=False))
            tex_chunks.append("")


    for m in sorted(metric_tables.keys(), key=sorting_key):
        if args.emit_signif_matrix and m in wilc_tables:
            tex_chunks.append(tex_significance_matrix(m, wilc_df, alpha=0.05))
            tex_chunks.append("")

    for m in sorted(metric_tables.keys(), key=sorting_key):
        if args.emit_cd_diagram and m in overall_tables:
            avg_ranks = overall_tables[m]["Avg rank ↓"]
            tex_chunks.append(tex_cd_diagram(m, avg_ranks, overall_caps[m]["CD"], width=12.0))
            tex_chunks.append("")

    for m in sorted(metric_tables.keys(), key=sorting_key):
        if m in wilc_tables:
            tex_chunks.append(tex_longtable_wilcoxon(m, wilc_tables[m], span_mode=args.wilcoxon_span))
            tex_chunks.append("")

    out = "\n".join(tex_chunks).strip() + "\n"
    (args.out_dir / Path(args.tex)).write_text(out, encoding="utf-8")
    print(f"Wrote LaTeX to {args.tex}")


if __name__ == "__main__":
    main()
