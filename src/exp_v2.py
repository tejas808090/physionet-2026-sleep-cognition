"""Candidate model changes for the next version of the entry, scored on every record with annotations
(natural prevalence) with the official age-conditioned AUROC, repeated stratified 5-fold CV.

Usage: python exp_v2.py <data_folder> <repeats> [variant,variant,...]
"""
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "entry"))
warnings.filterwarnings("ignore")
os.environ["PSG_SIGNAL_CACHE"] = str(ROOT / "out" / "no_cache")  # annotation features only
from evaluate_model import compute_auroc_age  # noqa: E402
from features import ALGO_KEYS, feature_table  # noqa: E402
from model import Ensemble, _prep  # noqa: E402

data = Path(sys.argv[1])
REPEATS = int(sys.argv[2])
demo = pd.read_csv(data / "demographics.csv")
cache = ROOT / "out" / "cv" / "features_annot.pkl"
if cache.exists():
    X = pd.read_pickle(cache)
else:
    X = feature_table(str(data), demo)
    X.to_pickle(cache)
keep = X["tst_h"].notna().values
demo, X = demo[keep].reset_index(drop=True), X[keep].reset_index(drop=True)
y = demo["Cognitive_Impairment"].astype(str).str.lower().eq("true").astype(int).values
ages = demo["Age"].values.astype(float)
print(f"{len(y)} records, {y.sum()} positive")

DEMO = ["sex_male", "race_white", "race_black", "race_asian", "eth_hispanic"]
SLEEP = [k for k in ALGO_KEYS if k != "invalid_frac"]
COLS = DEMO + SLEEP


def lr(C=0.05):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=C, class_weight="balanced", max_iter=2000))


def gb(seed):
    return HistGradientBoostingClassifier(max_depth=2, learning_rate=0.03, max_iter=150, min_samples_leaf=20,
                                          l2_regularization=5.0, class_weight="balanced", random_state=seed)


def rank(p, ref):
    return np.searchsorted(np.sort(ref), p) / len(ref)


class AgeResid:
    """Replace each sleep feature by its residual from a linear fit on age (fitted on the training rows)."""

    def fit(self, Xa, age):
        self.med = Xa.median()
        Xf = Xa.fillna(self.med)
        self.reg = {c: LinearRegression().fit(age[:, None], Xf[c].values) for c in SLEEP}
        return self

    def transform(self, Xb, age):
        Xb = Xb.copy()
        for c in SLEEP:
            Xb[c] = Xb[c].fillna(self.med[c]) - self.reg[c].predict(age[:, None])
        return Xb


def variant_scores(name, Xa, ya, aa, Xb, ab, seed):
    Pa, Pb = _prep(Xa[COLS]), _prep(Xb[COLS])
    if name == "base":
        return Ensemble(seed).fit(Xa[COLS], ya).predict_proba(Xb[COLS])
    if name.startswith("lr"):
        C = float(name.split("@")[1]) if "@" in name else 0.05
        return lr(C).fit(Pa, ya).predict_proba(Pb)[:, 1]
    if name == "gb":
        return np.mean([gb(seed + s).fit(Pa, ya).predict_proba(Pb)[:, 1] for s in range(5)], axis=0)
    if name in ("resid", "resid_lr"):
        r = AgeResid().fit(Pa, aa)
        Ra, Rb = r.transform(Pa, aa), r.transform(Pb, ab)
        m = lr().fit(Ra, ya)
        p_lr = rank(m.predict_proba(Rb)[:, 1], m.predict_proba(Ra)[:, 1])
        if name == "resid_lr":
            return p_lr
        g = [gb(seed + s).fit(Ra, ya) for s in range(5)]
        pa = np.mean([x.predict_proba(Ra)[:, 1] for x in g], axis=0)
        pb = np.mean([x.predict_proba(Rb)[:, 1] for x in g], axis=0)
        return 0.5 * (p_lr + rank(pb, pa))
    if name == "with_age":
        Pa2, Pb2 = Pa.assign(age_=aa), Pb.assign(age_=ab)
        m = lr().fit(Pa2, ya)
        g = [gb(seed + s).fit(Pa2, ya) for s in range(5)]
        pa = np.mean([x.predict_proba(Pa2)[:, 1] for x in g], axis=0)
        pb = np.mean([x.predict_proba(Pb2)[:, 1] for x in g], axis=0)
        return 0.5 * (rank(m.predict_proba(Pb2)[:, 1], m.predict_proba(Pa2)[:, 1]) + rank(pb, pa))
    raise ValueError(name)


variants = sys.argv[3].split(",") if len(sys.argv) > 3 else ["base", "lr", "gb", "resid", "resid_lr", "with_age"]
for name in variants:
    reps = []
    for r in range(REPEATS):
        p = np.zeros(len(y))
        for a, b in StratifiedKFold(5, shuffle=True, random_state=r).split(X, y):
            p[b] = variant_scores(name, X.iloc[a], y[a], ages[a], X.iloc[b], ages[b], r)
        reps.append(compute_auroc_age(y, p, ages, gap=2))
    print(f"{name:10s} age-cond AUROC {np.mean(reps):.4f} +/- {np.std(reps):.4f}  ({' '.join(f'{v:.3f}' for v in reps)})",
          flush=True)
