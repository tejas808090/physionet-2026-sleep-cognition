"""Repeated stratified CV of the entry's model, scored with the official Challenge metrics.

Usage: python cv.py <data_folder> <out_dir> [tag] [repeats] [all|subset|matched|pairs] [first_repeat]
  all    : every record, features that exist for everyone (demographics + CAISR annotations)
  subset : only records that have cached raw-signal features (case-control design)
  matched: the age/site-matched case-control records in data/subset.csv, features that exist for all of them
  pairs  : matched pairs in data/subset.csv where both the positive and its matched negative have raw-signal features
In the matched designs age is balanced by construction, so the models are compared without demographics.
Features are rebuilt each run from the annotation files and the signal cache (fast).
Writes OOF predictions in the official output format and prints the official scores.
"""
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "entry"))
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")  # harmless threading notice
os.environ.setdefault("PSG_SIGNAL_CACHE", str(ROOT / "out" / "signal_cache"))
from evaluate_model import compute_auroc, compute_auroc_age, compute_auroc_weighted, compute_auprc  # noqa: E402
from features import ALGO_KEYS, feature_table  # noqa: E402
from model import Ensemble  # noqa: E402
from signals import ECG_KEYS, EEG_KEYS, SPO2_KEYS  # noqa: E402

data, out = Path(sys.argv[1]), Path(sys.argv[2])
tag = sys.argv[3] if len(sys.argv) > 3 else "cv"
REPEATS = int(sys.argv[4]) if len(sys.argv) > 4 else 5
MODE = sys.argv[5] if len(sys.argv) > 5 else "all"
START = int(sys.argv[6]) if len(sys.argv) > 6 else 0  # lets long runs be split into chunks of repeats
out.mkdir(parents=True, exist_ok=True)

demo = pd.read_csv(data / "demographics.csv")
t0 = time.time()
X = feature_table(str(data), demo)
have_algo, have_sig = X["tst_h"].notna(), X["psg_hours"].notna()
print(f"features {X.shape} in {time.time() - t0:.0f}s | with annotations {have_algo.sum()} | with raw-signal features {have_sig.sum()}")
if MODE == "subset":
    keep = (have_sig & have_algo).values
elif MODE == "matched":
    keep = (demo["BidsFolder"].isin(pd.read_csv(ROOT / "data" / "subset.csv")["BidsFolder"]) & have_algo).values
elif MODE == "pairs":
    sub = pd.read_csv(ROOT / "data" / "subset.csv")
    pos = sub[sub["Cognitive_Impairment"]].sort_values("Age", ascending=False, kind="stable")["BidsFolder"].values
    neg = sub[~sub["Cognitive_Impairment"]]["BidsFolder"].values  # stored in the order of the positives they match
    ok = set(demo.loc[(have_sig & have_algo).values, "BidsFolder"])
    both = [b for a, c in zip(pos, neg) if a in ok and c in ok for b in (a, c)]
    keep = demo["BidsFolder"].isin(both).values
else:
    keep = have_algo.values
demo, X = demo[keep].reset_index(drop=True), X[keep].reset_index(drop=True)
y = demo["Cognitive_Impairment"].astype(str).str.lower().eq("true").astype(int).values
ages = demo["Age"].values.astype(float)
sites = demo["SiteID"].values
print(f"mode {MODE}: {len(X)} records, {y.sum()} positive, sites {pd.Series(sites).value_counts().to_dict()}")

DEMO = ["age", "sex_male", "race_white", "race_black", "race_asian", "eth_hispanic"]
SLEEP = [k for k in ALGO_KEYS if k != "invalid_frac"]
variants = {"demographics": DEMO, "demo + sleep staging/events": DEMO + SLEEP}
if MODE in ("matched", "pairs"):
    variants = {"sleep staging/events": SLEEP}
if MODE == "pairs":
    variants.update({"EEG": EEG_KEYS, "ECG": ECG_KEYS, "SpO2": SPO2_KEYS, "sleep + EEG": SLEEP + EEG_KEYS,
                     "sleep + EEG + SpO2": SLEEP + EEG_KEYS + SPO2_KEYS, "sleep + all signals": SLEEP + EEG_KEYS + ECG_KEYS + SPO2_KEYS})
if MODE == "subset":
    variants.update({
        "demo + EEG": DEMO + EEG_KEYS, "demo + ECG": DEMO + ECG_KEYS, "demo + SpO2": DEMO + SPO2_KEYS,
        "demo + sleep + EEG": DEMO + SLEEP + EEG_KEYS,
        "demo + sleep + all signals": DEMO + SLEEP + EEG_KEYS + ECG_KEYS + SPO2_KEYS,
    })


def scores(p, m=None):
    m = np.ones(len(y), bool) if m is None else m
    return (compute_auroc_age(y[m], p[m], ages[m], gap=2), compute_auroc_weighted(y[m], p[m], ages[m], gap=2),
            compute_auroc(y[m], p[m]), compute_auprc(y[m], p[m]))


rows, oofs, reps = [], {}, []
for name, cols in variants.items():
    per_rep, oof = [], np.zeros(len(X))
    for r in range(START, START + REPEATS):
        p = np.zeros(len(X))
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=r).split(X, y):
            p[va] = Ensemble(seed=r).fit(X.iloc[tr][cols], y[tr]).predict_proba(X.iloc[va][cols])
        per_rep.append(scores(p)[0])
        reps.append((name, r, per_rep[-1]))
        oof += p / REPEATS
    s = scores(oof)
    oofs[name] = oof
    rows.append((name, len(cols), np.mean(per_rep), np.std(per_rep), *s[1:]))
    print(f"{name:30s} n_feat {len(cols):3d} | age-cond AUROC {np.mean(per_rep):.3f} +/- {np.std(per_rep):.3f} over repeats"
          f" | age-weighted {s[1]:.3f} | AUROC {s[2]:.3f} | AUPRC {s[3]:.3f}", flush=True)

res = pd.DataFrame(rows, columns=["features", "n_features", "age_cond_auroc", "sd_over_repeats", "age_weighted_auroc", "auroc", "auprc"])
res.to_csv(out / f"cv_{tag}.csv", index=False)
pd.DataFrame(reps, columns=["features", "repeat", "age_cond_auroc"]).to_csv(out / f"reps_{tag}.csv", index=False)
best = list(variants)[-1]
o = demo.copy()
o["Cognitive_Impairment_Probability"] = oofs[best]
o["Cognitive_Impairment"] = oofs[best] >= np.quantile(oofs[best], 1 - min(2 * y.mean(), 0.5))
o.to_csv(out / f"oof_{tag}.csv", index=False)
print("wrote", out / f"cv_{tag}.csv")
