#!/usr/bin/env python3
"""
Supplementary Figure 2 -- same design as Figure 4 (see figure4_heatmaps.py)
but averaged across all 15 canonical networks instead of AMN alone, plus a
Panel C: whole-map Normalized Mutual Information (NMI, metric MIn_cort) and
Dice coefficient (DC_cort, averaged across the 15 networks) heatmaps.

Usage:  python3 suppfig2_heatmaps.py
Output: outputs/suppfig2/SuppFig2A_PPV_heatmap_All15Networks.png
        outputs/suppfig2/SuppFig2B_<metric>_heatmap_All15Networks.png
        outputs/suppfig2/SuppFig2C_NMI_heatmap.png
        outputs/suppfig2/SuppFig2C_Dice_heatmap_All15Networks.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_combined, metric_mean_matrix, plot_heatmap_matrix, ensure_outdir, CANONICAL_15_NETWORKS

PANEL_B_METRICS = ["TNR_cort", "FPR_cort", "FNR_cort", "TPR_cort"]


def run():
    outdir = ensure_outdir("suppfig2")
    df = load_combined(metrics=["PPV_cort", "DC_cort", "MIn_cort"] + PANEL_B_METRICS)

    mat = metric_mean_matrix(df, "PPV_cort", network=CANONICAL_15_NETWORKS)
    plot_heatmap_matrix(mat, "Supp Fig 2A: PPV (mean of 15 networks)",
                         outdir / "SuppFig2A_PPV_heatmap_All15Networks.png", cbar_label="Mean PPV (N=4, 15 networks)")

    for metric in PANEL_B_METRICS:
        mat = metric_mean_matrix(df, metric, network=CANONICAL_15_NETWORKS)
        plot_heatmap_matrix(mat, f"Supp Fig 2B: {metric.replace('_cort','')} (mean of 15 networks)",
                             outdir / f"SuppFig2B_{metric}_heatmap_All15Networks.png", cbar_label=f"Mean {metric.replace('_cort','')}")

    mat = metric_mean_matrix(df, "MIn_cort")  # whole-brain metric, network ignored
    plot_heatmap_matrix(mat, "Supp Fig 2C: Normalized Mutual Information (whole map)",
                         outdir / "SuppFig2C_NMI_heatmap.png", cbar_label="Mean NMI")

    mat = metric_mean_matrix(df, "DC_cort", network=CANONICAL_15_NETWORKS)
    plot_heatmap_matrix(mat, "Supp Fig 2C: Dice coefficient (mean of 15 networks)",
                         outdir / "SuppFig2C_Dice_heatmap_All15Networks.png", cbar_label="Mean Dice coefficient")


if __name__ == "__main__":
    run()
