#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate LaTeX-ready performance tables, Wilcoxon significance summaries, and
critical-difference diagrams for the UCI Lipschitz benchmark.

The script consumes the JSON artifacts emitted by ``uci_training.py`` and produces
publication-quality tables compliant with JMLR two-column formatting constraints.
Runtime behavior matches the original version; only type hints and documentation were
added.
"""

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata, f, studentized_range

type SummaryDict = dict[str, Any]
type PerAlgorithmSummaries = dict[str, SummaryDict]
type MetricTableMap = dict[str, pd.DataFrame]
type NumericTableMap = dict[str, pd.DataFrame]

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


def alg_pretty(a: str) -> str:
    """
    Canonicalize internal algorithm keys into human-readable labels.

    Parameters
    ----------
    a : str
        Algorithm identifier as used in training summaries (e.g., ``"ldlt"``).

    Returns
    -------
    label : str
        Pretty-printed label; falls back to the original key if no mapping exists.

    Raises
    ------
    None

    Notes
    -----
    The mapping is defined in the module-level ``ALG_PRETTY`` dictionary to keep table
    captions consistent across outputs.

    Mathematical sketch
    --------------------
    Not applicable.

    Examples
    --------
    >>> alg_pretty("ldlt")
    'LDLT-L'
    >>> alg_pretty("unknown")
    'unknown'
    """
    return ALG_PRETTY.get(a, a)


def fmt_mean_std(x: pd.Series) -> str:
    """
    Format mean and standard deviation into a LaTeX-friendly string.

    Parameters
    ----------
    x : pandas.Series
        Row containing ``"mean"`` and ``"std"`` entries derived from dataset aggregations.

    Returns
    -------
    formatted : str
        Expression such as ``"0.9123\\,\\tiny$\\pm$0.0123"`` or an em-dash when values
        are missing.

    Raises
    ------
    None

    Notes
    -----
    Uses four decimal places to match prior tables and inserts a ``\\tiny`` marker so
    uncertainty occupies minimal horizontal space.

    Mathematical sketch
    --------------------
    No additional computation beyond simple rounding.

    Examples
    --------
    >>> row = pd.Series({"mean": 0.9, "std": 0.05})
    >>> fmt_mean_std(row)
    '0.9000\\,\\tiny$\\pm$0.0500'
    """
    if pd.isna(x["mean"]) or pd.isna(x["std"]):
        return r"\textemdash"
    return rf"{x['mean']:.4f}\,\tiny$\pm${x['std']:.4f}"


def p_to_stars(p: float | np.floating[Any]) -> str:
    """
    Map a p-value to LaTeX superscript stars following significance thresholds.

    Parameters
    ----------
    p : float
        Two-sided p-value in `[0, 1]`; NaNs produce an empty string.

    Returns
    -------
    stars : str
        One of ``"^{***}"``, ``"^{**}"``, ``"^{*}"``, or ``""``.

    Raises
    ------
    None

    Notes
    -----
    Thresholds are defined in ``STAR_THRESH`` with increasingly lenient cutoffs.

    Mathematical sketch
    --------------------
    Not applicable.

    Examples
    --------
    >>> p_to_stars(0.004)
    '^{**}'
    """
    if pd.isna(p):
        return ""
    for th, tag in STAR_THRESH:
        if p <= th:
            return tag
    return ""


def bestbold(series: pd.Series) -> pd.Series:
    """
    Boldface the maximum numeric entry in a LaTeX-formatted series.

    Parameters
    ----------
    series : pandas.Series
        Column of formatted strings such as ``"0.9000\\,\\tiny$\\pm$0.0100"`` or
        ``"\\textemdash"`` placeholders.

    Returns
    -------
    highlighted : pandas.Series
        Copy of the input where the maximal numeric entry is wrapped in ``\\textbf{...}``.

    Raises
    ------
    None

    Notes
    -----
    Extracts the first floating number using a regex, so the behavior assumes the value
    precedes any ``±`` annotations.

    Mathematical sketch
    --------------------
    Equivalent to computing ``argmax`` over extracted floats and boldfacing the matching
    entries.

    Examples
    --------
    >>> s = pd.Series(["0.80", "0.85", "0.83"])
    >>> bestbold(s).tolist()
    ['0.80', '\\\\textbf{0.85}', '0.83']
    """
    vals = pd.to_numeric(series.str.extract(r"([0-9]+\.?[0-9]*)")[0], errors="coerce")
    if vals.isna().all():
        return series
    best = vals.max(skipna=True)
    mask = vals.eq(best)
    out = series.copy()
    out[mask] = out[mask].apply(lambda s: rf"\textbf{{{s}}}")
    return out


def latex_escape(s: str) -> str:
    """
    Escape LaTeX-special characters in a string.

    Parameters
    ----------
    s : str
        Input text potentially containing characters interpreted specially by TeX.

    Returns
    -------
    escaped : str
        String safe for insertion into tabular environments.

    Raises
    ------
    None

    Notes
    -----
    Covers underscores, percent signs, ampersands, hash marks, braces, tildes, carets,
    dollar signs, and backslashes.

    Mathematical sketch
    --------------------
    Not applicable.

    Examples
    --------
    >>> latex_escape("A_1%")
    'A\\_1\\%'
    """
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


def metric_to_name(s: str) -> str:
    """
    Map internal metric identifiers to descriptive names.

    Parameters
    ----------
    s : str
        Metric key (e.g., ``"mean_cert_acc_36"``).

    Returns
    -------
    label : str
        Human-friendly description with LaTeX escapes applied when necessary.

    Raises
    ------
    None

    Notes
    -----
    Falls back to escaping the original key when the identifier is not present in
    ``METRICS_PRETTY``.

    Mathematical sketch
    --------------------
    Not applicable.

    Examples
    --------
    >>> metric_to_name("mean_test_acc")
    'Mean Accuracy'
    """
    return METRICS_PRETTY.get(s, latex_escape(s))


def ensure_algos(path: Path) -> list[str]:
    """
    Detect algorithm subdirectories containing ``summary.json`` artifacts.

    Parameters
    ----------
    path : Path
        Output directory passed to the training script (contains per-algorithm folders).

    Returns
    -------
    algorithms : list of str
        Sorted list of algorithm identifiers derived from subfolder names.

    Raises
    ------
    OSError
        If the directory cannot be listed.

    Notes
    -----
    This helper mirrors the discovery logic from the training orchestrator to avoid
    re-specifying algorithm names during table generation.

    Mathematical sketch
    --------------------
    Not applicable.

    Examples
    --------
    >>> ensure_algos(Path('runs/UCI'))  # doctest: +SKIP
    ['ldlt', 'sdp', ...]
    """
    algs = []
    for p in path.iterdir():
        if p.is_dir() and (p / "summary.json").exists():
            algs.append(p.name)
    return sorted(algs)


# -------------------------- Load experiment data ------------------------- #

def load_all(out_dir: Path, algorithms: Sequence[str] | None = None) -> tuple[
    PerAlgorithmSummaries, pd.DataFrame, list[str]]:
    """
    Load per-algorithm summaries and the Wilcoxon comparison table from disk.

    Parameters
    ----------
    out_dir : Path
        Directory containing subfolders for each algorithm plus shared JSON artifacts.
    algorithms : sequence of str, optional
        Explicit subset of algorithms to load; defaults to auto-discovery via
        :func:`ensure_algos`.

    Returns
    -------
    per_alg : dict
        Mapping from algorithm key to its full ``summary.json`` dictionary.
    wilc_df : pandas.DataFrame
        Dataframe representation of ``wilcoxon_pairwise_all.json`` (empty if missing).
    loaded_algs : list of str
        Actual algorithms inspected, preserving the requested/auto-discovered order.

    Raises
    ------
    None

    Notes
    -----
    Each ``summary.json`` file originates from ``schedule_and_run`` and includes
    dataset-level fold aggregates under ``"datasets"``.

    Mathematical sketch
    --------------------
    Not applicable.

    Examples
    --------
    >>> summaries, wilc, algs = load_all(Path('runs/UCI'))  # doctest: +SKIP
    """
    if algorithms is None:
        algorithms = ensure_algos(out_dir)

    per_alg: PerAlgorithmSummaries = {}
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
    return per_alg, wilc_df, list(algorithms)


# --------------------- Build metric summary across DS -------------------- #

def sorting_key(m: str) -> tuple[int, int | str]:
    """
    Provide a deterministic ordering for metric identifiers.

    Parameters
    ----------
    m : str
        Metric key such as ``"mean_test_acc"`` or ``"mean_cert_acc_36"``.

    Returns
    -------
    key : tuple
        Tuple used for sorting: accuracy first, then certified metrics ordered by
        epsilon value, followed by any remaining metrics alphabetically.

    Raises
    ------
    None

    Notes
    -----
    Ensures tables list accuracy metrics before robustness metrics to align with
    reporting conventions.

    Mathematical sketch
    --------------------
    Parsing splits the identifier on underscores and attempts to convert the last token
    to an integer epsilon value.

    Examples
    --------
    >>> sorting_key('mean_cert_acc_36')
    (1, 36)
    """
    if m == 'mean_test_acc':
        return (0, 0)
    try:
        return (1, int(m.split('_')[-1]))
    except Exception:
        return (2, m)


def build_metric_summary(per_alg: PerAlgorithmSummaries) -> tuple[MetricTableMap, NumericTableMap]:
    """
    Aggregate mean metrics across datasets for each algorithm.

    Parameters
    ----------
    per_alg : dict
        Mapping returned by :func:`load_all` containing dataset-level ``mean_*`` entries.

    Returns
    -------
    pretty_tables : dict[str, pandas.DataFrame]
        Table per metric with columns ``["Algorithm", "$N$ datasets", "Value"]`` where
        ``Value`` is a ``mean±std`` string and the maximum entry is bolded.
    numeric_tables : dict[str, pandas.DataFrame]
        Companion numeric tables containing raw means/stds indexed by algorithm for
        downstream ranking.

    Raises
    ------
    None

    Notes
    -----
    Only keys starting with ``"mean_"`` are considered, mirroring the training output.
    Missing metrics yield empty dictionaries.

    Mathematical sketch
    --------------------
    For each algorithm and metric, the method computes
    \( \mu = \frac{1}{|D|}\sum_{d \in D} v_{d} \) and the sample standard deviation over
    datasets \( D \), where \( v_d \) is the dataset-level metric.

    Examples
    --------
    >>> build_metric_summary({"mlp": {"datasets": {"foo": {"mean_test_acc": 0.9}}}})[0].keys()
    dict_keys(['mean_test_acc'])
    """
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
    result_tables: MetricTableMap = {}
    numeric_tables: NumericTableMap = {}

    for m in metrics:
        sub = df[df["metric"] == m]
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

def format_small_p(p: float | np.floating[Any]) -> str:
    """
    Render a p-value in scientific notation with optional significance stars.

    Parameters
    ----------
    p : float
        P-value in `[0, 1]`; a value of zero is displayed explicitly.

    Returns
    -------
    formatted : str
        LaTeX math-mode string (e.g., ``"$1.2e-03^{**}$"``).

    Raises
    ------
    None

    Notes
    -----
    Calls :func:`p_to_stars` for star annotation.

    Mathematical sketch
    --------------------
    Not applicable.

    Examples
    --------
    >>> format_small_p(0.0)
    '$0^{***}$'
    """
    if p == 0:
        return f"$0{p_to_stars(p)}$"
    return f"${p:.1e}{p_to_stars(p)}$"


def build_wilcoxon_tables(wilc_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Post-process Wilcoxon pairwise statistics for LaTeX rendering.

    Parameters
    ----------
    wilc_df : pandas.DataFrame
        Output of ``wilcoxon_pairwise_all.json`` containing pairwise comparisons.

    Returns
    -------
    tables : dict[str, pandas.DataFrame]
        Mapping from metric key to a formatted dataframe with pretty algorithm names,
        integer counts, and scientific-notation p-values.

    Raises
    ------
    None

    Notes
    -----
    The function retains Holm-adjusted p-values for sorting but drops them from the
    displayed output to avoid redundant columns.

    Mathematical sketch
    --------------------
    Not applicable.

    Examples
    --------
    >>> build_wilcoxon_tables(pd.DataFrame())  # empty input
    {}
    """
    if wilc_df.empty:
        return {}

    tdict: dict[str, pd.DataFrame] = {}
    for m, sub in wilc_df.groupby("metric"):
        t = sub.copy()
        t["Alg A"] = t["alg_a"].map(alg_pretty)
        t["Alg B"] = t["alg_b"].map(alg_pretty)
        t["WinRate A"] = (t["win_rate_a_over_b"]).round(4)
        t["Median AB"] = t["median_diff_a_minus_b"].round(4)
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

def build_model_size_ranges(per_alg: PerAlgorithmSummaries) -> pd.DataFrame:
    """
    Compute min–max architecture statistics across all datasets and folds.

    Parameters
    ----------
    per_alg : dict
        Summary dictionary whose ``"datasets"`` entries include ``fold_results``.

    Returns
    -------
    ranges : pandas.DataFrame
        Table with columns ``["Algorithm", "Width", "Depth", "Parameters", "Padding",
        "Input dim", "Output dim"]`` containing string ranges (e.g., ``"64--256"``) or
        em-dashes when unavailable.

    Raises
    ------
    None

    Notes
    -----
    Iterates through every recorded fold to capture architecture variation introduced
    by hyperparameter sweeps.

    Mathematical sketch
    --------------------
    Each column reports ``min(value) -- max(value)`` over the collected list per
    algorithm.

    Examples
    --------
    >>> build_model_size_ranges({"mlp": {"datasets": {}}}).empty
    False
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


def _dataset_metric_matrix(per_alg: PerAlgorithmSummaries, metric: str) -> pd.DataFrame:
    """
    Construct a dataset-by-algorithm pivot table for a specific metric.

    Parameters
    ----------
    per_alg : dict
        Loaded summaries keyed by algorithm.
    metric : str
        Metric identifier (must exist inside the per-dataset payloads).

    Returns
    -------
    pivot : pandas.DataFrame
        Wide table with datasets as rows and pretty algorithm labels as columns.

    Raises
    ------
    None

    Notes
    -----
    Filters out missing or non-finite entries to keep subsequent statistical tests well
    defined.

    Mathematical sketch
    --------------------
    Equivalent to pivoting the long-format table on columns `(dataset, algorithm)`.

    Examples
    --------
    >>> _dataset_metric_matrix({"mlp": {"datasets": {"foo": {"mean_test_acc": 0.9}}}}, "mean_test_acc")
    algorithm  MLP
    dataset
    foo        0.9
    """
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


def _friedman_iman_nemenyi(pivot: pd.DataFrame, alpha: float = 0.05) -> dict[str, Any] | None:
    """
    Compute Friedman/Iman–Davenport statistics and the Nemenyi critical difference.

    Parameters
    ----------
    pivot : pandas.DataFrame
        Dataset-by-algorithm matrix with higher values indicating better performance.
    alpha : float, default=0.05
        Family-wise error rate for the Nemenyi test.

    Returns
    -------
    stats : dict or None
        Dictionary containing ``N`` datasets, ``k`` algorithms, average ranks (Series),
        F-statistic, p-value, and the critical difference. Returns ``None`` if less than
        two algorithms are present.

    Raises
    ------
    None

    Notes
    -----
    Ranks are assigned per row with 1 denoting the best algorithm. The Iman–Davenport
    F-statistic refines the Friedman test for finite samples, and the CD is derived from
    the Studentized range distribution.

    Mathematical sketch
    --------------------
    Computes:

    * Friedman statistic \( \chi_F^2 = \frac{12N}{k(k+1)} \sum_j \bar{r}_j^2 - 3N(k+1) \).
    * Iman–Davenport \( F_F = \frac{(N-1)\chi_F^2}{N(k-1) - \chi_F^2} \).
    * Critical difference \( \text{CD} = q_\alpha \sqrt{\frac{k(k+1)}{6N}} \).

    Examples
    --------
    >>> pivot = pd.DataFrame({'A': [0.9, 0.8], 'B': [0.85, 0.82]})
    >>> _friedman_iman_nemenyi(pivot) is not None
    True
    """
    if pivot.empty or pivot.shape[1] < 2:
        return None
    X = pivot.to_numpy(float)
    N, k = X.shape
    ranks = np.vstack([rankdata(-row, method="average") for row in X])
    avg_ranks = pd.Series(ranks.mean(axis=0), index=pivot.columns).sort_values()

    chi2_F = (12 * N) / (k * (k + 1)) * (np.sum(avg_ranks.values ** 2) - k * (k + 1) ** 2 / 4.0)
    denom = N * (k - 1) - chi2_F
    F_F = ((N - 1) * chi2_F) / denom if denom > 1e-12 else np.inf
    p_F = 1 - f.cdf(F_F, k - 1, (k - 1) * (N - 1))

    q_alpha = studentized_range.isf(alpha, k, np.inf)
    CD = float(q_alpha * np.sqrt(k * (k + 1) / (6.0 * N)))

    return {"N": N, "k": k, "avg_ranks": avg_ranks, "F_F": float(F_F), "p_F": float(p_F), "CD": CD}


def _wilcoxon_summary_for_metric(wilc_df: pd.DataFrame, metric: str, alpha: float = 0.05) -> pd.DataFrame:
    """
    Summarize significant Wilcoxon wins/losses per algorithm for a metric.

    Parameters
    ----------
    wilc_df : pandas.DataFrame
        Full Wilcoxon comparison table.
    metric : str
        Metric identifier to filter.
    alpha : float, default=0.05
        Holm-adjusted significance level.

    Returns
    -------
    summary : pandas.DataFrame
        Index by pretty algorithm name with columns reporting significant wins, losses,
        net wins, share of significant wins, and average effect size `r`.

    Raises
    ------
    None

    Notes
    -----
    Counts both roles (Alg A/B) to ensure symmetry. When no comparisons exist, returns
    an empty dataframe.

    Mathematical sketch
    --------------------
    For each pair (a, b) the summary increments wins when the median difference is
    positive and the Holm-adjusted p-value does not exceed ``alpha``.

    Examples
    --------
    >>> _wilcoxon_summary_for_metric(pd.DataFrame(), "mean_test_acc").empty
    True
    """
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


def build_overall_metric_stats(
        per_alg: PerAlgorithmSummaries,
        wilc_df: pd.DataFrame,
        alpha: float = 0.05,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, float]]]:
    """
    Combine average ranks, Wilcoxon outcomes, and CD statistics per metric.

    Parameters
    ----------
    per_alg : dict
        Summary dictionaries by algorithm.
    wilc_df : pandas.DataFrame
        Wilcoxon pairwise results.
    alpha : float, default=0.05
        Significance level for Wilcoxon and CD computations.

    Returns
    -------
    tables : dict[str, pandas.DataFrame]
        Table per metric with columns: average rank, significant wins/losses, net wins,
        win share, and mean effect size.
    captions : dict[str, dict[str, float]]
        Metadata containing ``N``, ``k``, ``F_F``, ``p_F``, and ``CD`` for caption text.

    Raises
    ------
    None

    Notes
    -----
    LDLT variants are kept at the bottom of the table to emphasize other methods first,
    mirroring paper layout preferences.

    Mathematical sketch
    --------------------
    Relies on :func:`_friedman_iman_nemenyi` for rank statistics and merges them with
    :func:`_wilcoxon_summary_for_metric` outputs.

    Examples
    --------
    >>> build_overall_metric_stats({"mlp": {"datasets": {}}}, pd.DataFrame())
    ({}, {})
    """
    metrics = set()
    for _, js in per_alg.items():
        for _, payload in js.get("datasets", {}).items():
            for k, v in payload.items():
                if k.startswith("mean_") and isinstance(v, (int, float)) and np.isfinite(v):
                    metrics.add(k)
    metrics = sorted(metrics, key=sorting_key)

    out_tables: dict[str, pd.DataFrame] = {}
    caps: dict[str, dict[str, float]] = {}
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


def tex_table_metric_summary(metric_name: str, df: pd.DataFrame, wide: bool = False) -> str:
    """
    Render a per-metric mean±std table into LaTeX.

    Parameters
    ----------
    metric_name : str
        Metric key used for caption and label.
    df : pandas.DataFrame
        Summary table produced by :func:`build_metric_summary`.
    wide : bool, default=False
        If ``True``, emit a ``table*`` environment for two-column layouts.

    Returns
    -------
    latex : str
        LaTeX source string.

    Raises
    ------
    None

    Notes
    -----
    Sorts rows by descending value to highlight top performers.

    Mathematical sketch
    --------------------
    Not applicable.

    Examples
    --------
    >>> tex_table_metric_summary("mean_test_acc", pd.DataFrame(columns=["Algorithm","$N$ datasets","Value"]))
    '\\\\begin{table}[t]\\n\\\\centering\\n...'
    """
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


def get_data_metrics(
        lines: list[str],
        algorithms: Sequence[str],
        metrics: Sequence[str],
        metric_summary: Mapping[str, pd.DataFrame],
) -> None:
    """
    Append LaTeX row entries for each algorithm across multiple metrics.

    Parameters
    ----------
    lines : list of str
        Accumulated LaTeX lines mutated in place.
    algorithms : sequence of str
        Ordered list of pretty algorithm names to include.
    metrics : sequence of str
        Metric keys whose tables exist in ``metric_summary``.
    metric_summary : mapping
        Dictionary from metric key to dataframe returned by
        :func:`build_metric_summary`.

    Returns
    -------
    None

    Raises
    ------
    None

    Notes
    -----
    Adds ``$N$`` once per algorithm (taken from the first metric) and inserts em-dashes
    for missing entries.

    Mathematical sketch
    --------------------
    Not applicable.

    Examples
    --------
    >>> lines = []
    >>> get_data_metrics(lines, ["MLP"], ["mean_test_acc"], {"mean_test_acc": pd.DataFrame({
    ...     "Algorithm": ["MLP"], "$N$ datasets": [1], "Value": ["0.9"]
    ... })})
    >>> lines[0]
    'MLP & 1 & 0.9 \\\\'
    """
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


def tex_table_metric_summary_combined(metric_summary: Mapping[str, pd.DataFrame], wide: bool = True) -> str:
    """
    Emit a consolidated LaTeX table spanning accuracy and certified metrics.

    Parameters
    ----------
    metric_summary : mapping
        Dict from metric key to dataframe built by :func:`build_metric_summary`.
    wide : bool, default=True
        Use ``table*`` when ``True`` to span both columns.

    Returns
    -------
    latex : str
        Generated table string.

    Raises
    ------
    None

    Notes
    -----
    Separates LDLT variants with a mid-rule to match stylistic conventions.

    Mathematical sketch
    --------------------
    Not applicable.

    Examples
    --------
    >>> tex_table_metric_summary_combined({})
    '\\\\begin{table*}[t]\\n\\\\centering\\n...'
    """
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


def tex_longtable_wilcoxon(metric_name: str, df: pd.DataFrame, span_mode: str = "table*") -> str:
    r"""
    Render Wilcoxon pairwise statistics into LaTeX with configurable span modes.

    Parameters
    ----------
    metric_name : str
        Key identifying the metric under comparison.
    df : pandas.DataFrame
        Formatted table returned by :func:`build_wilcoxon_tables`.
    span_mode : {"table*", "longtable", "landscape"}, default="table*"
        Choice of LaTeX environment:
        ``"table*"`` for wide floats, ``"longtable"`` for multi-page output, or
        ``"landscape"`` to wrap ``table*`` inside ``pdflscape``.

    Returns
    -------
    latex : str
        Complete LaTeX environment string.

    Raises
    ------
    None

    Notes
    -----
    Includes explanatory captions and table notes describing the significance stars.

    Mathematical sketch
    --------------------
    Not applicable.

    Examples
    --------
    >>> tex_longtable_wilcoxon("mean_test_acc", pd.DataFrame(), span_mode="table*")
    '\\\\begin{table*}[t]\\n\\\\centering\\n...'
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


def tex_model_ranges(df: pd.DataFrame, wide: bool = False) -> str:
    """
    Produce LaTeX table summarizing architectural ranges per algorithm.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :func:`build_model_size_ranges`.
    wide : bool, default=False
        Emit ``table*`` when ``True``.

    Returns
    -------
    latex : str
        LaTeX snippet ready for inclusion in a document.

    Raises
    ------
    None

    Notes
    -----
    Reuses column order to maintain readability between metrics.

    Mathematical sketch
    --------------------
    Not applicable.

    Examples
    --------
    >>> tex_model_ranges(pd.DataFrame(columns=['Algorithm','Width','Depth','Parameters','Padding','Input dim','Output dim']))
    '\\\\begin{table}[t]\\n\\\\centering\\n...'
    """
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
def tex_table_overall_summary(metric_name: str, df: pd.DataFrame, caps: dict[str, float], wide: bool = True) -> str:
    """
    Render the combined average-rank and Wilcoxon summary table.

    Parameters
    ----------
    metric_name : str
        Metric identifier.
    df : pandas.DataFrame
        Table produced by :func:`build_overall_metric_stats`.
    caps : dict[str, float]
        Dictionary containing ``N``, ``k``, ``F_F``, ``p_F``, and ``CD`` values for captions.
    wide : bool, default=True
        Emit ``table*`` when ``True``.

    Returns
    -------
    latex : str
        LaTeX source describing the table.

    Raises
    ------
    None

    Notes
    -----
    Emphasizes lower average ranks and includes explanatory caption text about the
    Iman–Davenport test.

    Mathematical sketch
    --------------------
    Not applicable.

    Examples
    --------
    >>> tex_table_overall_summary("mean_test_acc", pd.DataFrame(), {"N":0,"k":0,"F_F":0,"p_F":1,"CD":0})
    '\\\\begin{table*}[t]\\n\\\\centering\\n...'
    """
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
    """
    Generate TikZ code for a critical-difference diagram.

    Parameters
    ----------
    metric_name : str
        Metric identifier for captions/labels.
    avg_ranks : pandas.Series
        Series indexed by pretty algorithm names with average ranks.
    CD : float
        Critical difference threshold from the Nemenyi test.
    width : float, default=12.0
        Horizontal axis width in centimeters.

    Returns
    -------
    latex : str
        TikZ figure environment depicting ranks and non-significant cliques.

    Raises
    ------
    None

    Notes
    -----
    Alternates label positions above/below the axis to reduce overlap and draws thick
    segments for algorithm groups within the critical difference.

    Mathematical sketch
    --------------------
    Groups consecutive algorithms when ``max(rank) - min(rank) <= CD`` to form cliques.

    Examples
    --------
    >>> tex_cd_diagram("mean_test_acc", pd.Series([1.0, 2.0], index=["A","B"]), 1.0)
    '\\\\begin{figure*}[t]\\n\\\\centering\\n...'
    """
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
def tex_significance_matrix(metric_name: str, wilc_df: pd.DataFrame, alpha: float = 0.05, wide: bool = False) -> str:
    """
    Produce a triangular matrix of Wilcoxon outcomes (win/loss/none).

    Parameters
    ----------
    metric_name : str
        Metric key to filter.
    wilc_df : pandas.DataFrame
        Wilcoxon results dataframe.
    alpha : float, default=0.05
        Holm-adjusted significance level.
    wide : bool, default=False
        Emit ``table*`` when ``True``.

    Returns
    -------
    latex : str
        LaTeX table or a comment when data is missing.

    Raises
    ------
    None

    Notes
    -----
    Uses colored triangles to indicate whether the row algorithm significantly beats
    (green) or loses to (red) the column algorithm.

    Mathematical sketch
    --------------------
    Not applicable.

    Examples
    --------
    >>> tex_significance_matrix("mean_test_acc", pd.DataFrame(), 0.05)
    '% No Wilcoxon data for this metric.'
    """
    sub = wilc_df[wilc_df["metric"] == metric_name].copy()
    if sub.empty:
        return "% No Wilcoxon data for this metric."
    algs = sorted(set(sub["alg_a"]).union(set(sub["alg_b"])))
    idx = {a: i for i, a in enumerate(algs)}

    tie = r"$\cdot$"
    win = r"\textcolor{green}{$\blacktriangle$}"  # ▲
    loss = r"\textcolor{red}{$\blacktriangledown$}"  # ▼

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

def main() -> None:
    """
    Command-line interface for generating all LaTeX artifacts.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    SystemExit
        If no ``summary.json`` files are discovered under ``--out-dir``.

    Notes
    -----
    Parses CLI arguments, loads artifacts via :func:`load_all`, generates every
    requested table/figure, and writes the concatenated LaTeX file to ``--tex``.

    Mathematical sketch
    --------------------
    Not applicable.

    Examples
    --------
    >>> main()  # doctest: +SKIP
    """
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
