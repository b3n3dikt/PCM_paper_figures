#!/usr/bin/env python3
"""
Supplementary Figure 1 -- for each of the four clinically-relevant networks
(FP/FPN, DMN, SCAN, Sal/Salience): a PPV heatmap (Threshold x ExpMin, same
design as Figure 4A) plus the Precision-vs-Recall analysis from Figure 4C.
(The published figure also shows representative thresholded/unthresholded
brain maps per network -- not reproducible here, no surface-mapping tools in
this package; see outputs_reference/PUBLISHED_SuppFig1_p2.png for those.)

Usage:  python3 suppfig1_pr_recall.py
Output: outputs/suppfig1/SuppFig1_PPV_heatmap_<network>.png (x4)
        outputs/suppfig1/SuppFig1_PR_recall_<network>.png (x4)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLINICAL_4_NETWORKS, load_combined, metric_mean_matrix, plot_heatmap_matrix, ensure_outdir
from figure4_pr_recall import run as run_pr_recall

NETWORK_DISPLAY = {"FP": "FPN", "DMN": "DMN", "SCAN": "SCAN", "Sal": "Salience"}


def run():
    outdir = ensure_outdir("suppfig1")
    for net in CLINICAL_4_NETWORKS:
        disp = NETWORK_DISPLAY[net]
        df = load_combined(metrics=["PPV_cort"], network=net)
        mat = metric_mean_matrix(df, "PPV_cort", network=net)
        plot_heatmap_matrix(mat, f"Supp Fig 1: PPV ({disp})",
                             outdir / f"SuppFig1_PPV_heatmap_{net}.png", cbar_label=f"Mean PPV (N=4, {disp})")
        run_pr_recall(network=net, title=f"Precision-Recall ({disp})", xlabel="Recall (TPR)",
                       out_name=f"SuppFig1_PR_recall_{net}.png", outdir_name="suppfig1")


if __name__ == "__main__":
    run()
