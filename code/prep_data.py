#!/usr/bin/env python3
"""
One-time data-preparation step: builds this package's data/ folder from the
original project files. Not part of run_all.py (data/ is already checked in
once this has been run) -- kept here for provenance/re-runnability if the
source files ever change.

Source of truth (outside this package, read-only):
  ../../analyses/all_subjects_all_thresholds_combined_cortical_brain_bayes_5p_outputs.csv
  ../../analyses/Exp_mins_with_GroundTruthRef-70m_FIXEDRefThr-0/.../outputs/All-thresh-*/*.csv

What it does:
  1. Combined metrics CSV: renames Network "CO" -> "AMN", trims to the
     RefMin/ExpMin/Threshold values and metrics actually used in the paper.
  2. Figure 5 condition CSVs: renames Network "CO" -> "AMN", renames the
     confusing ExpMin/RefMin-as-subject-index columns to explicit
     SubjIndex_Exp/SubjIndex_Ref + SubjectID_Exp/SubjectID_Ref.
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SUBJECTS, EXP_MINS_USED, THRESHOLDS_USED, REF_MIN_USED, DATA_DIR

SRC_ROOT = Path(__file__).resolve().parents[2] / "analyses"
SRC_COMBINED = SRC_ROOT / "all_subjects_all_thresholds_combined_cortical_brain_bayes_5p_outputs.csv"
SRC_FIG5_ROOT = SRC_ROOT / "Exp_mins_with_GroundTruthRef-70m_FIXEDRefThr-0"

UNUSED_METRICS = {"PPVcorr_Jeffreys_cort", "PPVcorr_bayes_cort", "PPVcorr_d1d2_cort"}

FIG5_CONDITIONS = {
    # Genuine Standard Template Matching (ExpMethod-TM), NOT the
    # PCM-confidence-map-at-threshold-0 proxy used in an earlier draft of
    # this package. The proxy numerically matched an OLDER version of the
    # manuscript; the current manuscript text (verified by direct read of
    # the PDF, e.g. "within-subject PPV increased from 0.48 +/- 0.14 with TM
    # to 0.66 +/- 0.14 with PCM (Delta = 0.177 +/- 0.018, paired t(3) =
    # 19.76 ...)") matches ONLY the genuine-TM data below, exactly, to 3+
    # significant figures across all 16+ reported Figure 5 quantities.
    "TM_5min": SRC_FIG5_ROOT / "ExpMin-5_ExpMethod-TM__RefMin-70_RefThr-0_cortical/outputs/All-thresh-0/GrounTruthRef70m_FIXEDRefThr-0_Exp5m_MethodstandardTM_outputs_thresh_0.csv",
    "TM_70min": SRC_FIG5_ROOT / "ExpMin-70_ExpMethod-TM__RefMin-70_RefThr-0_cortical/outputs/All-thresh-0/GrounTruthRef70m_FIXEDRefThr-0_Exp70m_MethodstandardTM_outputs_thresh_0.csv",
    "PCM_5min_thr0.99": SRC_FIG5_ROOT / "ExpMin-5_ExpThr-0.99__RefMin-70_RefThr-0_cortical/outputs/All-thresh-0.99/GrounTruthRef70m_FIXEDRefThr-0_Exp5m_ExpThr-0.99_outputs_thresh_0.99.csv",
    "PCM_70min_thr0.99": SRC_FIG5_ROOT / "ExpMin-70_ExpThr-0.99__RefMin-70_RefThr-0_cortical/outputs/All-thresh-0.99/GrounTruthRef70m_FIXEDRefThr-0_Exp70m_ExpThr-0.99_outputs_thresh_0.99.csv",
}


def rename_network_col(df: pd.DataFrame) -> pd.DataFrame:
    if "Network" in df.columns:
        df["Network"] = df["Network"].replace({"CO": "AMN"})
    return df


def build_combined():
    print(f"Reading {SRC_COMBINED} ...")
    df = pd.read_csv(SRC_COMBINED)
    n0 = len(df)

    df = df[~df["Metric"].isin(UNUSED_METRICS)]
    df = df[df["RefMin"] == REF_MIN_USED]
    df = df[df["ExpMin"].isin(EXP_MINS_USED)]
    df = df[df["Threshold"].isin(THRESHOLDS_USED)]
    df = rename_network_col(df)

    out = DATA_DIR / "combined_metrics" / "combined_metrics_used_in_paper.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"  {n0:,} rows -> {len(df):,} rows  ->  {out}")


def build_figure5():
    outdir = DATA_DIR / "figure5_conditions"
    outdir.mkdir(parents=True, exist_ok=True)
    subj_map = {i + 1: SUBJECTS[i] for i in range(4)}
    for name, path in FIG5_CONDITIONS.items():
        df = pd.read_csv(path)
        df = rename_network_col(df)
        df = df.rename(columns={"ExpMin": "SubjIndex_Exp", "RefMin": "SubjIndex_Ref"})
        df["SubjectID_Exp"] = df["SubjIndex_Exp"].map(subj_map)
        df["SubjectID_Ref"] = df["SubjIndex_Ref"].map(subj_map)
        cols = ["Subject", "Threshold", "Metric", "Network",
                "SubjIndex_Exp", "SubjIndex_Ref", "SubjectID_Exp", "SubjectID_Ref", "Value"]
        df = df[cols]
        out = outdir / f"{name}.csv"
        df.to_csv(out, index=False)
        print(f"  {path.name} -> {out}  ({len(df)} rows)")


if __name__ == "__main__":
    build_combined()
    build_figure5()
    print("Data prep complete.")
