"""
Shared helpers for the PCM reproducibility package.

Provenance: consolidates small pieces of logic that were duplicated across
several original scripts in analyses/figuremaking/ and
analyses/revised_analysis/code/ (io_utils.py, stats_utils.py), rewritten to
use only relative paths and the AMN network label.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

# ---------------------------------------------------------------------------
# Paths (everything relative to the package root, one level above code/)
# ---------------------------------------------------------------------------
PKG_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PKG_ROOT / "data"
COMBINED_CSV = DATA_DIR / "combined_metrics" / "combined_metrics_used_in_paper.csv"
FIG5_DIR = DATA_DIR / "figure5_conditions"
OUT_DIR = PKG_ROOT / "outputs"

SUBJECTS = ["PFM3T7T01", "PFM3T7T02", "PFM3T7T03", "PFM3T7T04"]
NETWORK = "AMN"  # renamed from "CO" (cingulo-opercular) throughout this package
CANONICAL_15_NETWORKS = [
    "AMN", "Aud", "DAN", "DMN", "FP", "MTL", "PMN", "PON",
    "SCAN", "SMd", "SMl", "Sal", "Tpole", "VAN", "Vis",
]
CLINICAL_4_NETWORKS = ["DMN", "FP", "SCAN", "Sal"]  # Supp Fig 1 / Supp Table 1

EXP_MINS_USED = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
THRESHOLDS_USED = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
REF_MIN_USED = 70

# The precision-recall figures (Fig 4C, Supp Fig 1 right column) only ever
# plot these three data lengths -- confirmed from the actual cluster command
# used to generate the published panels.
PR_CURVE_EXP_MINS = [5, 25, 70]


def ensure_outdir(sub: str = "") -> Path:
    d = OUT_DIR / sub if sub else OUT_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_combined(metrics=None, network=None, thresholds=None, exp_mins=None,
                   ref_min: int = REF_MIN_USED) -> pd.DataFrame:
    """Load + filter the trimmed combined_metrics_used_in_paper.csv."""
    df = pd.read_csv(COMBINED_CSV)
    if metrics is not None:
        df = df[df["Metric"].isin(metrics)]
    if ref_min is not None:
        df = df[df["RefMin"] == ref_min]
    if exp_mins is not None:
        df = df[df["ExpMin"].isin(exp_mins)]
    if thresholds is not None:
        df = df[df["Threshold"].isin(thresholds)]
    if network is not None:
        # whole-brain metrics (MIn_cort) aren't restricted to one network
        wb_mask = df["Metric"].str.contains(r"\bMIn_", regex=True)
        df = df[wb_mask | (~wb_mask & df["Network"].eq(network))]
    return df.copy()


def load_figure5_condition(name: str) -> pd.DataFrame:
    """
    name in {"TM_5min", "TM_70min", "PCM_5min_thr0.99", "PCM_70min_thr0.99"}.
    Columns: Subject, Threshold, Metric, Network, SubjIndex_Exp, SubjIndex_Ref,
    SubjectID_Exp, SubjectID_Ref, Value.
    """
    path = FIG5_DIR / f"{name}.csv"
    df = pd.read_csv(path)
    return df


def within_between_by_subject(df: pd.DataFrame, metric: str, network: str = NETWORK) -> pd.DataFrame:
    """
    From a Figure-5-style condition table, return one row per subject i with
    within = value(i,i) and between_mean = mean over j != i of value(i,j).
    """
    sub = df[(df["Metric"] == metric) & (df["Network"] == network)].copy()
    if sub.empty:
        raise ValueError(f"No rows for Metric={metric} Network={network}")
    rows = []
    for i in range(1, 5):
        within = sub.loc[(sub["SubjIndex_Exp"] == i) & (sub["SubjIndex_Ref"] == i), "Value"]
        within_val = float(within.iloc[0]) if len(within) else np.nan
        between_vals = sub.loc[(sub["SubjIndex_Exp"] == i) & (sub["SubjIndex_Ref"] != i), "Value"].to_numpy(float)
        between_mean = float(np.mean(between_vals)) if between_vals.size else np.nan
        rows.append({
            "i": i, "subject": SUBJECTS[i - 1],
            "within": within_val, "between_mean": between_mean,
            "between_n": int(between_vals.size),
        })
    return pd.DataFrame(rows)


def hedges_g_paired(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    n = diff.size
    sd = diff.std(ddof=1)
    if sd == 0 or n < 2:
        return float("nan")
    d = diff.mean() / sd
    J = 1 - (3 / (4 * n - 1))
    return float(d * J)


def paired_t(x, y) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    t, p = ttest_rel(x, y)
    diff = x - y
    return dict(N=len(x), df=len(x) - 1, t=float(t), p=float(p),
                mean_diff=float(diff.mean()),
                sd_diff=float(diff.std(ddof=1)) if len(diff) > 1 else 0.0)


def holm_adjust(pvals) -> np.ndarray:
    """Holm-Bonferroni step-down correction across the given family of p-values."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    p_sorted = p[order]
    adj_sorted = np.array([(m - k) * p_sorted[k] for k in range(m)], dtype=float)
    adj_sorted = np.maximum.accumulate(adj_sorted)
    adj = np.empty(m, dtype=float)
    adj[order] = adj_sorted
    return np.clip(adj, 0, 1)


def bootstrap_ci(diff, n_boot=5000, alpha=0.05, random_state=42):
    diff = np.asarray(diff, dtype=float)
    rng = np.random.default_rng(random_state)
    boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)]
    return float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2)))


# Consistent color scheme reused by every figure script
COLOR_TM = "#6B7280"    # neutral gray - Template Matching
COLOR_PCM = "#2563EB"   # blue - PCM (confidence-thresholded); generic fallback
# Figure 5 specifically uses two teal shades for PCM (matching the published
# figure's 5-min/70-min distinction) instead of a single blue -- see
# outputs_reference/PUBLISHED_Figure5_native.jpeg panels C/D.
COLOR_TM_LIGHT = "#C9CDD3"   # TM, lighter fill (used for "between" boxes)
COLOR_PCM_5MIN = "#3ED8C3"   # PCM 5-min, light teal
COLOR_PCM_70MIN = "#0E8C7A"  # PCM 70-min, dark teal
CMAP_HEATMAP = "magma"

def sig_stars(p):
    """Standard significance-star mapping used for Figure 5 annotations."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def metric_mean_matrix(df: pd.DataFrame, metric: str, network=None,
                        exp_mins=EXP_MINS_USED, thresholds=THRESHOLDS_USED) -> pd.DataFrame:
    """
    Build a (Threshold x ExpMin) matrix of the across-subject mean of `metric`.
    If `network` is a single string, filters to it. If a list, averages across
    those networks first (per subject/threshold/ExpMin) -- used for the
    "averaged across all 15 canonical networks" panels.
    Whole-brain metrics (MIn_cort) ignore `network` entirely.
    Returns a DataFrame indexed by Threshold, columns ExpMin, values = mean.
    """
    sub = df[df["Metric"] == metric].copy()
    is_whole_brain = metric.startswith("MIn_")
    if not is_whole_brain and network is not None:
        nets = [network] if isinstance(network, str) else list(network)
        sub = sub[sub["Network"].isin(nets)]
        if len(nets) > 1:
            sub = (sub.groupby(["Subject", "Threshold", "ExpMin"])["Value"]
                      .mean().reset_index())
    sub = sub[sub["ExpMin"].isin(exp_mins) & sub["Threshold"].isin(thresholds)]
    mat = (sub.groupby(["Threshold", "ExpMin"])["Value"].mean()
              .unstack("ExpMin")
              .reindex(index=sorted(thresholds), columns=sorted(exp_mins)))
    return mat


def plot_heatmap_matrix(mat: pd.DataFrame, title: str, outpath, vmin=None, vmax=None,
                         cbar_label="Value", cmap=CMAP_HEATMAP):
    """
    vmin/vmax default to None, which lets matplotlib auto-scale the colorbar
    to this panel's own data min/max -- confirmed against the actual
    published figure that each heatmap panel is scaled independently to its
    own value range, not to a shared fixed 0-1 (or any other shared) range.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat.values, origin="lower", cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index)
    ax.set_xlabel("Exploratory data length (minutes)")
    ax.set_ylabel("Confidence threshold")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"  saved {outpath}")
