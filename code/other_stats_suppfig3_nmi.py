#!/usr/bin/env python3
"""
Supplementary Figure 3 -- whole-map NMI within/between-subject reliability
(TM vs PCM@0.99, 5 vs 70 min).

Unlike Figure 4/5 and Supp Figs 1-2 (which compare exploratory data against
a FIXED 70-minute reference half), Supp Fig 3 compares two INDEPENDENT
halves of the SAME duration (Half 1 vs Half 2, both 5min or both 70min).

PCM side: computed here from data/suppfig3_nmi/pcm_thr0.99_matched_duration_pairs.csv
(all 4x4 subject-pair MIn_cort values at matched duration, threshold 0.99).
Within-subject = diagonal (same subject on both sides); between-subject =
each subject's mean agreement with the other three, then paired against
within across the 4 subjects.

TM side: Standard Template Matching has no PCM confidence threshold, so it
isn't part of the ExpMin/RefMin threshold sweep used everywhere else in this
package -- it comes from a separately-run pipeline stage. The per-subject
raw comparisons for that stage weren't available, only its finished
within/between summary statistics
(data/suppfig3_nmi/standardTM_within_between_stats.csv); those are used
directly here rather than recomputed.

Usage:  python3 other_stats_suppfig3_nmi.py
Output: outputs/other_stats/suppfig3_nmi_within_between.csv
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, ensure_outdir, paired_t, hedges_g_paired


def pcm_condition_stats(exp_minutes: int) -> dict:
    df = pd.read_csv(DATA_DIR / "suppfig3_nmi" / "pcm_thr0.99_matched_duration_pairs.csv")
    df = df[df.ExpDuration_min == exp_minutes]

    within = (df[df.SubjIndex_Exp == df.SubjIndex_Ref]
              .set_index("SubjIndex_Exp")["MIn_cort"].sort_index())
    between = (df[df.SubjIndex_Exp != df.SubjIndex_Ref]
               .groupby("SubjIndex_Exp")["MIn_cort"].mean().sort_index())

    stats = paired_t(within, between)
    g = hedges_g_paired(within, between)
    return dict(
        condition=f"PCM_{exp_minutes}min_thr0.99",
        N=stats["N"],
        Mean_within=within.mean(), SD_within=within.std(ddof=1),
        Mean_between=between.mean(), SD_between=between.std(ddof=1),
        Mean_diff=stats["mean_diff"], SD_diff=stats["sd_diff"],
        t=stats["t"], p=stats["p"], Hedges_g=g,
    )


def tm_condition_stats(exp_minutes: int) -> dict:
    df = pd.read_csv(DATA_DIR / "suppfig3_nmi" / "standardTM_within_between_stats.csv")
    row = df[df.ExpMin == exp_minutes].iloc[0]
    return dict(
        condition=f"TM_{exp_minutes}min",
        N=int(row.N),
        Mean_within=row.Mean_within, SD_within=row.SD_within,
        Mean_between=row.Mean_between, SD_between=row.SD_between,
        Mean_diff=row.Mean_diff, SD_diff=row.SD_diff,
        t=row.t, p=row.p, Hedges_g=row.Hedges_g,
    )


def run():
    outdir = ensure_outdir("other_stats")
    rows = [
        tm_condition_stats(5),
        pcm_condition_stats(5),
        tm_condition_stats(70),
        pcm_condition_stats(70),
    ]
    out = pd.DataFrame(rows)
    out_path = outdir / "suppfig3_nmi_within_between.csv"
    out.to_csv(out_path, index=False)
    print(f"[other_stats_suppfig3_nmi] -> {out_path}")
    print(out.to_string(index=False))
    return out


if __name__ == "__main__":
    run()
