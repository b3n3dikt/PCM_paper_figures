#!/usr/bin/env python3
"""
Runs every figure/table/stats script in this package, in order, writing
everything into ../outputs/. Data prep (prep_data.py) is NOT run here --
data/ ships pre-built in this package; re-run prep_data.py yourself only if
you need to regenerate it from the original project files.

Usage:  python3 run_all.py
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

SCRIPTS = [
    "table1_stats.py",
    "supptable1_stats.py",
    "figure4_heatmaps.py",
    "figure4_pr_recall.py",
    "suppfig1_pr_recall.py",
    "suppfig2_heatmaps.py",
    "figure5_panels_cd.py",
    "other_stats_threshold_gain.py",
    "other_stats_suppfig3_nmi.py",
]


def main():
    for script in SCRIPTS:
        print(f"\n{'='*70}\nRunning {script}\n{'='*70}")
        result = subprocess.run([sys.executable, str(HERE / script)])
        if result.returncode != 0:
            print(f"\n*** {script} FAILED (exit {result.returncode}) -- stopping. ***")
            sys.exit(result.returncode)
    print("\nAll scripts completed. See ../outputs/ for results.")


if __name__ == "__main__":
    main()
