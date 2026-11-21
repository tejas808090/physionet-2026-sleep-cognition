"""Model for the PhysioNet Challenge 2026 entry: a small rank-averaged ensemble on tabular sleep features.

The Challenge metric compares positives and negatives of similar age (within 2 years), so the model has to
add information beyond age. Positives are rare (about 8%), so the models are kept small and regularised.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LOG_COLS = ("index", "_min", "dur_s", "frac")
# Left out on purpose: these describe how and where a study was recorded or documented (equipment, montage, units,
# record keeping), not the patient's sleep, so they would not transfer to hospitals outside the training data.
# Age is also left out: the Challenge metric only compares patients of similar age, so age itself cannot help, and
# on age-matched training data it is actively misleading.
EXCLUDE_EXACT = {"age", "bmi", "psg_hours", "psg_n_channels", "invalid_frac", "ecg_valid_frac", "spo2_valid_frac", "eeg_c_artifact_frac"}
EXCLUDE_SUFFIX = ("_logpow",)


def model_columns(columns):
    return [c for c in columns if c not in EXCLUDE_EXACT and not c.endswith(EXCLUDE_SUFFIX)]


def _prep(X):
    """Log-compress the heavy-tailed event and duration features."""
    X = X.copy()
    for c in X.columns:
        if c.endswith(LOG_COLS) or "_index" in c:
            X[c] = np.log1p(X[c].clip(lower=0))
    return X


class LinearModel:
    """Version 2: one strongly regularised logistic regression (C=0.005).

    On all 1,090 small-set records with annotations (natural 8% prevalence) it scored 0.644 age-conditioned AUROC in
    5x repeated 5-fold CV, against 0.629 for the version-1 ensemble below; the boosted trees alone scored 0.597.
    signal_features=False drops the raw-signal features (they did not help on the 168 records that have them).
    """

    def __init__(self, C=0.005, signal_features=True):
        self.C, self.signal_features = C, signal_features

    def fit(self, X, y):
        cols = model_columns(X.columns)
        if not self.signal_features:
            from signals import SIGNAL_KEYS
            cols = [c for c in cols if c not in SIGNAL_KEYS]
        self.columns = cols
        Xp = _prep(X[self.columns])
        self.lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                LogisticRegression(C=self.C, class_weight="balanced", max_iter=5000))
        self.lr.fit(Xp, y)
        self.ref = np.sort(self.lr.predict_proba(Xp)[:, 1])
        self.prevalence = float(np.mean(y))
        return self

    def predict_proba(self, X):
        """Score in [0, 1]: training-set percentile of the logistic score. Works for a single record."""
        Xp = _prep(X.reindex(columns=self.columns))
        return np.searchsorted(self.ref, self.lr.predict_proba(Xp)[:, 1]) / len(self.ref)

    def predict(self, X, top_frac=None):
        top_frac = top_frac if top_frac is not None else min(2 * self.prevalence, 0.5)
        return self.predict_proba(X) >= 1 - top_frac


class Ensemble:
    """Version 1 (kept for reference): rank average of a logistic regression and shallow boosted trees."""

    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, X, y):
        self.columns = model_columns(X.columns)
        Xp = _prep(X[self.columns])
        self.lr = make_pipeline(SimpleImputer(strategy="median", add_indicator=False), StandardScaler(),
                                LogisticRegression(C=0.05, class_weight="balanced", max_iter=2000))
        self.lr.fit(Xp, y)
        self.gbs = []
        for s in range(5):
            gb = HistGradientBoostingClassifier(max_depth=2, learning_rate=0.03, max_iter=150, min_samples_leaf=20,
                                                l2_regularization=5.0, class_weight="balanced", random_state=self.seed + s)
            self.gbs.append(gb.fit(Xp, y))
        # reference scores for turning new scores into training-set percentiles, and the alert threshold
        self.ref_lr = np.sort(self.lr.predict_proba(Xp)[:, 1])
        self.ref_gb = np.sort(self._gb(Xp))
        self.prevalence = float(np.mean(y))
        return self

    def _gb(self, Xp):
        return np.mean([g.predict_proba(Xp)[:, 1] for g in self.gbs], axis=0)

    def predict_proba(self, X):
        """Score in [0, 1]: mean training-set percentile of the two models. Works for a single record."""
        Xp = _prep(X.reindex(columns=self.columns))
        a = np.searchsorted(self.ref_lr, self.lr.predict_proba(Xp)[:, 1]) / len(self.ref_lr)
        b = np.searchsorted(self.ref_gb, self._gb(Xp)) / len(self.ref_gb)
        return 0.5 * (a + b)

    def predict(self, X, top_frac=None):
        """Flag the records scoring above the (1 - top_frac) training percentile; default 2x prevalence."""
        top_frac = top_frac if top_frac is not None else min(2 * self.prevalence, 0.5)
        return self.predict_proba(X) >= 1 - top_frac
