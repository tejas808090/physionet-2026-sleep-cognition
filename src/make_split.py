"""Make a stratified train/holdout split laid out like Challenge data folders (no data copied: folder junctions).

Usage: python make_split.py <data_folder> <split_dir> [holdout_frac] [seed] [records_csv]
records_csv (optional): only use the records whose BidsFolder is listed in this file.
Creates <split_dir>/train and <split_dir>/holdout (holdout demographics has the label columns removed,
as in the hidden sets) plus <split_dir>/holdout_labels.csv for scoring.
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

data, split = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
frac = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
demo = pd.read_csv(data / "demographics.csv")
if len(sys.argv) > 5:
    demo = demo[demo["BidsFolder"].isin(pd.read_csv(sys.argv[5])["BidsFolder"])]
strat = demo["Cognitive_Impairment"].astype(str) + "_" + demo["SiteID"].astype(str)
tr, ho = train_test_split(demo, test_size=frac, random_state=seed, stratify=strat)
LABEL_COLS = ["Time_to_Event", "Cognitive_Impairment", "Last_Known_Visit_Date", "Time_to_Last_Visit"]
for name, df in (("train", tr), ("holdout", ho)):
    d = split / name
    d.mkdir(parents=True, exist_ok=True)
    (df if name == "train" else df.drop(columns=LABEL_COLS)).to_csv(d / "demographics.csv", index=False)
    for sub in ("physiological_data", "algorithmic_annotations") + (("human_annotations",) if name == "train" else ()):
        if (data / sub).exists() and not (d / sub).exists():
            subprocess.run(["cmd", "/c", "mklink", "/J", str(d / sub), str(data / sub)], check=True, capture_output=True)
ho.to_csv(split / "holdout_labels.csv", index=False)
print(f"train {len(tr)} ({int((tr.Cognitive_Impairment == True).sum())} positive), holdout {len(ho)} ({int((ho.Cognitive_Impairment == True).sum())} positive)")
