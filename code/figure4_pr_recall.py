#!/usr/bin/env python3
"""
Figure 4, Panel C -- Precision (PPV) vs Recall (TPR) for AMN: one line per
exploratory data length (5, 25, 70 min -- these are the only three lengths
the published panel plots), points along each line = confidence threshold
(0 to 0.99), point color = threshold.

Provenance: ports the exact plotting logic of
analyses/figuremaking/plot_TPR-FPR_PRrecall_curves_PaperFigures.py's
--plot_precision_recall code path, matching the actual invocation used to
generate the published panels (--exp_min 5 25 70, --tp/fp/fn_metric
TP_cort/FP_cort/FN_cort, --pr_horizontal_line 0.6, --line_width 3,
--marker_size 300, --marker_alpha 0.9, --pr_xrange/yrange 0 1). Precision and
recall are computed from pooled TP/FP/FN counts (summed across the 4
subjects at each ExpMin x Threshold cell), exactly as the original script
does -- NOT from averaging each subject's own PPV_cort/TPR_cort value.

Usage:  python3 figure4_pr_recall.py
Output: outputs/figure4/Figure4C_PR_recall_AMN.png
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap, LinearSegmentedColormap
from matplotlib import cm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_combined, ensure_outdir, NETWORK, PR_CURVE_EXP_MINS, THRESHOLDS_USED

# Same endpoints/colormaps as the original PaperFigures script.
CMAP_THRESHOLD = LinearSegmentedColormap.from_list("cyan_to_blue", ["#23f3fa", "#0225b3"])
LOW_FRAC, HIGH_FRAC = 0.1, 0.9
PR_HORIZONTAL_LINE = 0.6
LINE_WIDTH = 3.0
MARKER_SIZE = 300.0
MARKER_ALPHA = 0.9


def _buffered_colormap(colormap_name, num_colors):
    base = plt.get_cmap(colormap_name)
    if num_colors == 1:
        return ListedColormap([base(0.5 * (LOW_FRAC + HIGH_FRAC))])
    colors = [base(LOW_FRAC + i * (HIGH_FRAC - LOW_FRAC) / (num_colors - 1)) for i in range(num_colors)]
    return ListedColormap(colors)


def run(network=NETWORK, title_suffix="", out_name="Figure4C_PR_recall_AMN.png", outdir_name="figure4",
        title="Precision Coverage Trade Off", xlabel="Coverage (TPR)"):
    df = load_combined(metrics=["TP_cort", "FP_cort", "FN_cort"], network=network,
                        exp_mins=PR_CURVE_EXP_MINS)

    exp_mins = sorted(df["ExpMin"].unique())
    thr_norm = Normalize(vmin=min(THRESHOLDS_USED), vmax=max(THRESHOLDS_USED))
    cmap_expmin = _buffered_colormap("BuPu", len(exp_mins))

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    for i, exp in enumerate(exp_mins):
        slice_exp = df[df["ExpMin"] == exp]
        thresholds = np.sort(slice_exp["Threshold"].unique())
        prec_list, rec_list = [], []
        for thr in thresholds:
            slice_thr = slice_exp[slice_exp["Threshold"] == thr]
            tp_sum = slice_thr.loc[slice_thr["Metric"] == "TP_cort", "Value"].sum()
            fp_sum = slice_thr.loc[slice_thr["Metric"] == "FP_cort", "Value"].sum()
            fn_sum = slice_thr.loc[slice_thr["Metric"] == "FN_cort", "Value"].sum()
            prec_list.append(tp_sum / (tp_sum + fp_sum + 1e-12))
            rec_list.append(tp_sum / (tp_sum + fn_sum + 1e-12))

        line_color = cmap_expmin.colors[i]
        ax.plot(rec_list, prec_list, color=line_color, lw=LINE_WIDTH, zorder=3, label=f"{exp} min")
        for thr, rr, pp in zip(thresholds, rec_list, prec_list):
            c = cm.ScalarMappable(norm=thr_norm, cmap=CMAP_THRESHOLD).to_rgba(thr)
            ax.scatter(rr, pp, color=c, s=MARKER_SIZE, edgecolor="none", alpha=MARKER_ALPHA, zorder=2)

    ax.axhline(y=PR_HORIZONTAL_LINE, color="black", linestyle=":", lw=1.5, zorder=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Precision (PPV)")
    ax.set_title(f"{title}{title_suffix}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    sm = cm.ScalarMappable(cmap=CMAP_THRESHOLD, norm=thr_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label("Confidence threshold")

    ax.legend(title="ExpMin", bbox_to_anchor=(1.55, 1), loc="upper left", fontsize=9)

    fig.subplots_adjust(right=0.72)
    outdir = ensure_outdir(outdir_name)
    outpath = outdir / out_name
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure4_pr_recall] saved {outpath}")


if __name__ == "__main__":
    run()
