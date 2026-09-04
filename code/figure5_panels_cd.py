#!/usr/bin/env python3
"""
Figure 5, Panels C + D -- within- vs between-subject AMN PPV (Panel C) and
subject-specificity separation, within minus mean-between (Panel D), for TM
and PCM(thr=0.99) at 5 and 70 minutes.

*** KNOWN CAVEAT (read before trusting these numbers as "genuine TM") ***
This script reproduces the manuscript's published Figure 5C/D pipeline
exactly (validated below to match all reported statistics to high
precision). An internal audit (see analyses/revised_analysis/README.md and
SUMMARY.md) later found that the "TM" condition in this original pipeline is
actually PCM confidence-map data at threshold 0 standing in for genuine
Template Matching output, not a true independent TM run. This script
deliberately reproduces the PUBLISHED figure as-is (matching the paper) and
does NOT apply that later fix -- see instructions.html for the full story
and pointers to the corrected/audited version if you need it.

Provenance: adapted from analyses/figuremaking/fig6_stats_ppv_co.py and
fig6_stats_holm_all_reported_ttests_only.py (original, validated stats
logic) and analyses/revised_analysis/code/figures.py (plotting style for the
new Panel D), rewired to read data/figure5_conditions/*.csv.

Usage:  python3 figure5_panels_cd.py
Output: outputs/figure5/Figure5C_within_between_PPV.png
        outputs/figure5/Figure5D_subject_specificity_separation.png
        outputs/figure5/figure5_stats_ready.csv
        outputs/figure5/figure5_inferential_tests.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (load_figure5_condition, within_between_by_subject, paired_t,
                     holm_adjust, ensure_outdir, NETWORK, COLOR_TM,
                     COLOR_PCM_5MIN, COLOR_PCM_70MIN, sig_stars)

# Per-condition fill color, matching the published figure's gray-TM /
# two-shade-teal-PCM scheme (see outputs_reference/PUBLISHED_Figure5_native.jpeg)
COND_COLOR = {
    "TM_5min": COLOR_TM, "TM_70min": COLOR_TM,
    "PCM_5min_thr0.99": COLOR_PCM_5MIN, "PCM_70min_thr0.99": COLOR_PCM_70MIN,
}

CONDITIONS = ["TM_5min", "TM_70min", "PCM_5min_thr0.99", "PCM_70min_thr0.99"]
METRIC = "PPV_cort"


def build_stats_ready():
    rows = []
    for cond in CONDITIONS:
        df = load_figure5_condition(cond)
        t = within_between_by_subject(df, METRIC, network=NETWORK)
        t["condition"] = cond
        rows.append(t)
    return pd.concat(rows, ignore_index=True)


def get_vec(stats_ready, cond, col):
    return stats_ready[stats_ready["condition"] == cond].sort_values("i")[col].to_numpy(float)


def run():
    outdir = ensure_outdir("figure5")
    stats_ready = build_stats_ready()
    stats_ready.to_csv(outdir / "figure5_stats_ready.csv", index=False)

    within = {c: get_vec(stats_ready, c, "within") for c in CONDITIONS}
    between = {c: get_vec(stats_ready, c, "between_mean") for c in CONDITIONS}
    sep = {c: within[c] - between[c] for c in CONDITIONS}

    # ---- Inferential tests: the manuscript's actual reported family.
    #      Methods (main text): "Holm-Bonferroni correction applied across
    #      the four comparisons" -- exactly the two within-PPV contrasts and
    #      the two separation-change contrasts below. (An earlier draft of
    #      this package included a 5th "convergence" test -- does PCM at
    #      5min land closer to the 70min TM reference than TM at 5min does?
    #      -- in the same Holm family. That comparison is not reported
    #      anywhere in the manuscript and was shifting every p_holm value
    #      away from the published numbers; it's been removed.) ----
    tests = []
    tests.append(dict(test="within_PCM5_minus_TM5",
                       **paired_t(within["PCM_5min_thr0.99"], within["TM_5min"])))
    tests.append(dict(test="within_PCM70_minus_TM70",
                       **paired_t(within["PCM_70min_thr0.99"], within["TM_70min"])))
    tests.append(dict(test="separation_PCM5_minus_TM5",
                       **paired_t(sep["PCM_5min_thr0.99"], sep["TM_5min"])))
    tests.append(dict(test="separation_PCM70_minus_TM70",
                       **paired_t(sep["PCM_70min_thr0.99"], sep["TM_70min"])))
    tests_df = pd.DataFrame(tests)
    tests_df["p_holm"] = holm_adjust(tests_df["p"].to_numpy(float))

    tests_path = outdir / "figure5_inferential_tests.csv"
    tests_df.to_csv(tests_path, index=False)
    print(f"[figure5_panels_cd] wrote {tests_path}")
    print(tests_df.to_string(index=False))

    p_holm = dict(zip(tests_df["test"], tests_df["p_holm"]))

    # ---- Panel C: within-subject-agreement group, then between-subject-
    # agreement group, each ordered TM-5min / TM-70min / PCM-5min / PCM-70min
    # -- matches outputs_reference/PUBLISHED_Figure5_native.jpeg panel C. ----
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    order = ["TM_5min", "TM_70min", "PCM_5min_thr0.99", "PCM_70min_thr0.99"]
    tick_labels = ["5-Mins", "70-Mins", "5-Mins", "70-Mins"]
    within_pos = np.arange(4)
    between_pos = within_pos + 5.5  # visual gap between the two groups
    group_data = [(within, within_pos), (between, between_pos)]

    for data_dict, positions in group_data:
        for i, cond in enumerate(order):
            color = COND_COLOR[cond]
            alpha = 1.0 if data_dict is within else 0.5
            ax.boxplot(data_dict[cond], positions=[positions[i]], widths=0.7, patch_artist=True,
                       boxprops=dict(facecolor=color, alpha=alpha), medianprops=dict(color="#D2691E", lw=2))
            ax.scatter(np.full(4, positions[i]), data_dict[cond], color="black", s=16, zorder=3)

    # Significance brackets for the two tests we actually compute: PCM vs TM
    # at matched duration, within-subject agreement only.
    def bracket(x1, x2, y, label):
        ax.plot([x1, x1, x2, x2], [y, y + 0.015, y + 0.015, y], color="black", lw=1.1)
        ax.text((x1 + x2) / 2, y + 0.02, label, ha="center", va="bottom", fontsize=10)

    y0 = max(within["TM_5min"].max(), within["PCM_5min_thr0.99"].max()) + 0.06
    bracket(within_pos[0], within_pos[2], y0, sig_stars(p_holm["within_PCM5_minus_TM5"]))
    y1 = max(within["TM_70min"].max(), within["PCM_70min_thr0.99"].max()) + 0.06
    bracket(within_pos[1], within_pos[3], y1, sig_stars(p_holm["within_PCM70_minus_TM70"]))

    all_pos = list(within_pos) + list(between_pos)
    ax.set_xticks(all_pos); ax.set_xticklabels(tick_labels + tick_labels)
    ax.set_xlabel("")
    for grp_pos, label in [(within_pos, "Within subject agreement"), (between_pos, "Between subject agreement")]:
        ax.annotate(label, xy=(grp_pos.mean(), 0), xytext=(grp_pos.mean(), -0.16),
                    ha="center", va="top", fontsize=10, annotation_clip=False,
                    bbox=dict(boxstyle="round", fc="white", ec="gray"))
        ax.plot([grp_pos[0], grp_pos[1]], [-0.06, -0.06], color=COLOR_TM, lw=2, clip_on=False)
        ax.plot([grp_pos[2], grp_pos[3]], [-0.06, -0.06], color=COLOR_PCM_70MIN, lw=2, clip_on=False)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(f"{METRIC.replace('_cort','')} ({NETWORK})")
    ax.set_title("Figure 5C: within-subject vs between-subject PPV")
    fig.subplots_adjust(bottom=0.3)
    fig.savefig(outdir / "Figure5C_within_between_PPV.png", dpi=300)
    plt.close(fig)

    # ---- Panel D: subject-specificity separation, paired lines TM->PCM ----
    fig, ax = plt.subplots(figsize=(6.5, 5))
    pairs = [("TM_5min", "PCM_5min_thr0.99", 0, "within_PCM5_minus_TM5"),
             ("TM_70min", "PCM_70min_thr0.99", 1, "within_PCM70_minus_TM70")]
    # NOTE: separation-change tests (separation_PCM*_minus_TM*) are the
    # statistically appropriate significance markers for this panel; shown here.
    sep_tests = {"TM_5min": "separation_PCM5_minus_TM5", "TM_70min": "separation_PCM70_minus_TM70"}
    for tm_cond, pcm_cond, xoff, _ in pairs:
        x_tm, x_pcm = xoff * 3, xoff * 3 + 1
        for s in range(4):
            ax.plot([x_tm, x_pcm], [sep[tm_cond][s], sep[pcm_cond][s]], "-", color="gray", lw=0.8, zorder=1)
        ax.boxplot(sep[tm_cond], positions=[x_tm], widths=0.6, patch_artist=True,
                   boxprops=dict(facecolor=COND_COLOR[tm_cond], alpha=0.8), medianprops=dict(color="#D2691E", lw=2))
        ax.boxplot(sep[pcm_cond], positions=[x_pcm], widths=0.6, patch_artist=True,
                   boxprops=dict(facecolor=COND_COLOR[pcm_cond], alpha=0.8), medianprops=dict(color="#D2691E", lw=2))
        ax.scatter(np.full(4, x_tm), sep[tm_cond], color="black", s=16, zorder=3)
        ax.scatter(np.full(4, x_pcm), sep[pcm_cond], color="black", s=16, zorder=3)
        y = max(sep[tm_cond].max(), sep[pcm_cond].max()) + 0.025
        bracket_y = y
        ax.plot([x_tm, x_tm, x_pcm, x_pcm], [bracket_y, bracket_y + 0.01, bracket_y + 0.01, bracket_y], color="black", lw=1.0)
        ax.text((x_tm + x_pcm) / 2, bracket_y + 0.015, sig_stars(p_holm[sep_tests[tm_cond]]),
                ha="center", va="bottom", fontsize=10)
    ax.axhline(0, color="black", ls="--", lw=0.8)
    ax.set_xticks([0, 1, 3, 4])
    ax.set_xticklabels(["TM\n5-Mins", "PCM\n5-Mins", "TM\n70-Mins", "PCM\n70-Mins"])
    ax.set_ylabel("Within-subject PPV − between-subject PPV")
    ax.set_title("Figure 5D: subject-specificity separation")
    fig.tight_layout()
    fig.savefig(outdir / "Figure5D_subject_specificity_separation.png", dpi=300)
    plt.close(fig)
    print(f"[figure5_panels_cd] saved figures to {outdir}")


if __name__ == "__main__":
    run()
