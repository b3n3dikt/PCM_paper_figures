#!/usr/bin/env python3
"""
Table 1 -- AMN (formerly "CO"/cingulo-opercular) targeting metrics at 5/25/70
minutes, threshold 0 vs 0.99: mean +/- SD, paired t-test, Wilcoxon, Hedges' g,
bootstrap CI on the difference.

Provenance: adapted from analyses/figuremaking/sumarize_results_and_stats_CSV.py
(Bene Ramirez, June 2025), with the AMN rename and a fix to the Network label
in the output (the original script mislabeled every row "WB" whenever
MIn_cort was included in the same --metrics run; here each row is labeled by
its own metric type).

Usage:  python3 table1_stats.py
Output: outputs/table1/summary_AMN.csv, outputs/table1/stats_AMN.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_combined, hedges_g_paired, bootstrap_ci, ensure_outdir, NETWORK

METRICS = ["PPV_cort", "FPR_cort", "TNR_cort", "FNR_cort", "TPR_cort"]
EXP_MINS = [5, 25, 70]
THRESHOLDS = [0.0, 0.99]


def is_whole_brain_metric(metric: str) -> bool:
    return metric.startswith("MIn_")


def run(network: str = NETWORK, metrics=METRICS, exp_mins=EXP_MINS, out_subdir: str = "table1"):
    df = load_combined(metrics=metrics, network=network, thresholds=THRESHOLDS, exp_mins=exp_mins)

    summary = (
        df.groupby(["Metric", "Network", "ExpMin", "Threshold"], dropna=False)
          .agg(Mean=("Value", "mean"), SD=("Value", "std"), N=("Subject", "nunique"))
          .reset_index()
          .sort_values(["Metric", "ExpMin", "Threshold"])
    )

    stat_rows = []
    for metric in metrics:
        label = "WholeBrain" if is_whole_brain_metric(metric) else network
        for exp in exp_mins:
            subset = df[(df["Metric"] == metric) & (df["ExpMin"] == exp)]
            wide = subset.pivot_table(index="Subject", columns="Threshold", values="Value")
            if not {0.0, 0.99}.issubset(wide.columns):
                continue
            # Descriptive Thr=0 / Thr=0.99 stats use every subject with a
            # non-missing value at that threshold, independently per column
            # (matches the published tables). A subject can be missing at
            # Thr=0.99 specifically (e.g. Sal network, 5min: 2/4 subjects
            # have zero surviving voxels at that strict a threshold, so PPV
            # is undefined) without being dropped from the Thr=0 column.
            col0 = wide[0.0].dropna()
            col99 = wide[0.99].dropna()
            # The paired significance test (diff/t/Wilcoxon/Hedges g/CI) can
            # only use subjects present at BOTH thresholds.
            paired = wide[[0.0, 0.99]].dropna()
            n_paired = len(paired)
            if n_paired < 2:
                continue
            diffs = paired[0.99] - paired[0.0]
            tstat, p_t = ttest_rel(paired[0.99], paired[0.0])
            try:
                wstat, p_w = wilcoxon(paired[0.99], paired[0.0], zero_method="wilcox")
            except ValueError:
                wstat, p_w = np.nan, np.nan
            g = hedges_g_paired(paired[0.99].values, paired[0.0].values)
            ci_low, ci_high = bootstrap_ci(diffs.values)
            stat_rows.append(dict(
                Metric=metric, Network=label, ExpMin=exp,
                N_Thr0=len(col0), N_Thr099=len(col99), N_paired=n_paired,
                Mean_Thr0=col0.mean(), SD_Thr0=col0.std(ddof=1),
                Mean_Thr099=col99.mean(), SD_Thr099=col99.std(ddof=1),
                Mean_diff=diffs.mean(), SD_diff=diffs.std(ddof=1),
                t=tstat, df=n_paired - 1, p_t=p_t, W=wstat, p_w=p_w,
                Hedges_g=g, CI_low=ci_low, CI_high=ci_high,
            ))
    stats = pd.DataFrame(stat_rows)

    outdir = ensure_outdir(out_subdir)
    summary_path = outdir / f"summary_{network}.csv"
    stats_path = outdir / f"stats_{network}.csv"
    summary.to_csv(summary_path, index=False)
    stats.to_csv(stats_path, index=False)
    print(f"[table1_stats] wrote {summary_path}")
    print(f"[table1_stats] wrote {stats_path}")
    return summary, stats


if __name__ == "__main__":
    run()
