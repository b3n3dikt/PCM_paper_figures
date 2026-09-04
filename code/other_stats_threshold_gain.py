#!/usr/bin/env python3
"""
Other reported stat (main text, not in any table): the PPV gain from
confidence thresholding (Thr=0.99 minus Thr=0) is significantly larger for
5-minute scans than for 70-minute scans of AMN.

gain(subject, duration) = PPV(thr=0.99) - PPV(thr=0) for that subject/duration
Then a paired t-test compares gain(5min) vs gain(70min) across the 4 subjects.

Manuscript-reported value: mean paired diff = 0.103 +/- 0.013, t(3) = 16.35,
p = 4.98e-4.

Usage:  python3 other_stats_threshold_gain.py
Output: outputs/other_stats/threshold_gain_by_duration.csv
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_combined, paired_t, ensure_outdir, NETWORK


def run():
    df = load_combined(metrics=["PPV_cort"], network=NETWORK,
                        thresholds=[0.0, 0.99], exp_mins=[5, 70])
    wide = df.pivot_table(index=["Subject", "ExpMin"], columns="Threshold", values="Value").reset_index()
    wide["gain"] = wide[0.99] - wide[0.0]
    gain_5 = wide[wide["ExpMin"] == 5].sort_values("Subject")["gain"].to_numpy()
    gain_70 = wide[wide["ExpMin"] == 70].sort_values("Subject")["gain"].to_numpy()

    stats = paired_t(gain_5, gain_70)
    out = pd.DataFrame([{
        "test": "PPV_thresholding_gain_5min_minus_70min",
        "network": NETWORK, **stats,
    }])

    outdir = ensure_outdir("other_stats")
    out_path = outdir / "threshold_gain_by_duration.csv"
    out.to_csv(out_path, index=False)
    print(f"[other_stats_threshold_gain] wrote {out_path}")
    print(out.to_string(index=False))
    return out


if __name__ == "__main__":
    run()
