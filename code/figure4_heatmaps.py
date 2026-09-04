#!/usr/bin/env python3
"""
Figure 4, Panels A+B -- AMN (formerly "CO") targeting heatmaps as a function
of exploratory data length (x) and confidence threshold (y): PPV (A), and
TNR/FPR/FNR/TPR (B), averaged across the 4 subjects.

Provenance: built directly from the trimmed combined_metrics_used_in_paper.csv
(Threshold x ExpMin x Metric x Network x Subject). An earlier candidate,
analyses/figuremaking/plot_heatmaps_allComparisons_fewer_mins.py, turned out
on inspection to plot a *different* thing (an ExpMinutes x RefMinutes
reliability grid at a fixed threshold, built from a separate raw .txt file
tree) rather than the Threshold x ExpMin design Figure 4 actually uses -- so
this script reimplements the heatmap directly from the tidy CSV instead of
adapting that file. See instructions.html for the full explanation.

Colorbar ranges below are fixed to match the published Figure 4 exactly
(read directly off the native embedded figure image extracted from the
manuscript PDF -- see outputs_reference/PUBLISHED_Figure4_native.jpeg) --
unlike Supp Fig 1/2, which auto-scale per panel, Figure 4 uses manually-set
round-number ranges in the original.

Usage:  python3 figure4_heatmaps.py
Output: outputs/figure4/Figure4A_PPV_heatmap_AMN.png
        outputs/figure4/Figure4B_<metric>_heatmap_AMN.png (TNR, FPR, FNR, TPR)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_combined, metric_mean_matrix, plot_heatmap_matrix, ensure_outdir, NETWORK

PANEL_B_METRICS = ["TNR_cort", "FPR_cort", "FNR_cort", "TPR_cort"]

# Fixed colorbar ranges matching the published Figure 4 exactly (panel: (vmin, vmax))
CBAR_RANGES = {
    "PPV_cort": (0.40, 0.80),
    "FPR_cort": (0.04, 0.16),
    "FNR_cort": (0.10, 0.45),
    "TNR_cort": (0.84, 0.96),
    "TPR_cort": (0.55, 0.90),
}


def run():
    outdir = ensure_outdir("figure4")
    df = load_combined(metrics=["PPV_cort"] + PANEL_B_METRICS, network=NETWORK)

    mat = metric_mean_matrix(df, "PPV_cort", network=NETWORK)
    vmin, vmax = CBAR_RANGES["PPV_cort"]
    plot_heatmap_matrix(mat, f"Figure 4A: PPV ({NETWORK})", outdir / "Figure4A_PPV_heatmap_AMN.png",
                         vmin=vmin, vmax=vmax, cbar_label="Mean PPV (N=4)")

    for metric in PANEL_B_METRICS:
        mat = metric_mean_matrix(df, metric, network=NETWORK)
        vmin, vmax = CBAR_RANGES[metric]
        plot_heatmap_matrix(mat, f"Figure 4B: {metric.replace('_cort','')} ({NETWORK})",
                             outdir / f"Figure4B_{metric}_heatmap_AMN.png",
                             vmin=vmin, vmax=vmax, cbar_label=f"Mean {metric.replace('_cort','')} (N=4)")


if __name__ == "__main__":
    run()
