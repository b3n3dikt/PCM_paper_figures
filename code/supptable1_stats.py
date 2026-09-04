#!/usr/bin/env python3
"""
Supplementary Table 1 -- same design as Table 1 (see table1_stats.py) but for
the four clinically-relevant networks (DMN, FP/FPN, SCAN, Sal/SAL) plus a
"mean across all 15 canonical networks" row, PPV and TPR only, at 5/25/70min.

Usage:  python3 supptable1_stats.py
Output: outputs/supptable1/stats_<network>.csv for each of DMN, FP, SCAN, Sal,
        outputs/supptable1/stats_All15Networks_mean.csv
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLINICAL_4_NETWORKS, CANONICAL_15_NETWORKS, load_combined, ensure_outdir
from table1_stats import run as run_table1_style

METRICS = ["PPV_cort", "TPR_cort"]
EXP_MINS = [5, 25, 70]


def run_all15_mean():
    """Average PPV/TPR across all 15 networks per subject, then run the same
    paired-stats machinery as table1_stats.run() on the averaged values."""
    df = load_combined(metrics=METRICS, thresholds=[0.0, 0.99], exp_mins=EXP_MINS)
    df = df[df["Network"].isin(CANONICAL_15_NETWORKS)]
    avg = (df.groupby(["Subject", "Metric", "ExpMin", "Threshold"])["Value"]
             .mean().reset_index())
    avg["Network"] = "All15Networks_mean"

    outdir = ensure_outdir("supptable1")
    tmp_csv = outdir / "_tmp_all15_avg.csv"
    # give it the columns load_combined()/pivot logic expects (RefMin not needed downstream)
    avg["RefMin"] = 70
    avg.to_csv(tmp_csv, index=False)

    # reuse the pivot/stat logic directly rather than round-tripping through common.load_combined
    import numpy as np
    from scipy.stats import ttest_rel, wilcoxon
    from common import hedges_g_paired, bootstrap_ci

    stat_rows = []
    for metric in METRICS:
        for exp in EXP_MINS:
            subset = avg[(avg["Metric"] == metric) & (avg["ExpMin"] == exp)]
            wide = subset.pivot_table(index="Subject", columns="Threshold", values="Value")
            wide = wide[[0.0, 0.99]].dropna()
            n = len(wide)
            diffs = wide[0.99] - wide[0.0]
            tstat, p_t = ttest_rel(wide[0.99], wide[0.0])
            try:
                wstat, p_w = wilcoxon(wide[0.99], wide[0.0], zero_method="wilcox")
            except ValueError:
                wstat, p_w = np.nan, np.nan
            g = hedges_g_paired(wide[0.99].values, wide[0.0].values)
            ci_low, ci_high = bootstrap_ci(diffs.values)
            stat_rows.append(dict(
                Metric=metric, Network="All15Networks_mean", ExpMin=exp, N=n,
                Mean_Thr0=wide[0.0].mean(), SD_Thr0=wide[0.0].std(ddof=1),
                Mean_Thr099=wide[0.99].mean(), SD_Thr099=wide[0.99].std(ddof=1),
                Mean_diff=diffs.mean(), SD_diff=diffs.std(ddof=1),
                t=tstat, df=n - 1, p_t=p_t, W=wstat, p_w=p_w,
                Hedges_g=g, CI_low=ci_low, CI_high=ci_high,
            ))
    stats = pd.DataFrame(stat_rows)
    stats_path = outdir / "stats_All15Networks_mean.csv"
    stats.to_csv(stats_path, index=False)
    tmp_csv.unlink()
    print(f"[supptable1_stats] wrote {stats_path}")
    return stats


def run():
    for net in CLINICAL_4_NETWORKS:
        run_table1_style(network=net, metrics=METRICS, exp_mins=EXP_MINS, out_subdir="supptable1")
    run_all15_mean()


if __name__ == "__main__":
    run()
