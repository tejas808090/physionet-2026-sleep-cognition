"""Feature extraction for the PhysioNet Challenge 2026 entry (cognitive impairment from sleep studies).

One record -> one flat dict of named features. Missing inputs give NaN features, never an exception,
so the model can still score a record when a file is absent or unreadable.
"""
import os

import numpy as np
import pandas as pd

from signals import features_from_intermediates, load_stages, signal_features

ALGO_DIR = "algorithmic_annotations"
PHYS_DIR = "physiological_data"
EPOCH_S = 30.0
# CAISR stage codes
N3, N2, N1, REM, WAKE = 1, 2, 3, 4, 5


def demographic_features(row):
    """Age, sex, BMI and race/ethnicity indicators from one demographics.csv row (dict)."""
    def num(k):
        try:
            return float(row.get(k))
        except (TypeError, ValueError):
            return np.nan
    sex = str(row.get("Sex", "")).strip().lower()
    race = str(row.get("Race", "")).strip().lower()
    eth = str(row.get("Ethnicity", "")).strip().lower()
    f = {
        "age": num("Age"),
        "bmi": num("BMI"),
        "sex_male": 1.0 if sex.startswith("m") else (0.0 if sex.startswith("f") else np.nan),
        "race_white": float(race == "white"),
        "race_black": float(race == "black"),
        "race_asian": float(race == "asian"),
        "eth_hispanic": float(eth == "hispanic"),
    }
    return f


def _read_edf(path):
    import edfio
    edf = edfio.read_edf(path, lazy_load_data=False)
    sig = {s.label.lower().strip(): np.asarray(s.data, dtype=np.float64) for s in edf.signals}
    fs = {s.label.lower().strip(): float(s.sampling_frequency) for s in edf.signals}
    return sig, fs


def _runs(mask):
    """Start indices and lengths of the True runs in a boolean array."""
    m = np.asarray(mask, dtype=np.int8)
    if m.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    d = np.diff(np.concatenate([[0], m, [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    return starts, ends - starts


ALGO_KEYS = (
    ["tib_h", "tst_h", "sleep_eff", "sol_min", "rem_lat_min", "waso_min", "pct_n1", "pct_n2", "pct_n3", "pct_rem",
     "awakenings_ph", "transitions_ph", "frag_index", "bout_rem_min", "bout_n3_min", "bout_wake_min", "rem_bouts",
     "rem_half_ratio", "n3_half_ratio", "prob_n1", "prob_n2", "prob_n3", "prob_rem", "prob_wake", "stage_entropy",
     "stage_maxprob", "stage_entropy_sleep", "arousal_index", "arousal_frac", "arousal_prob", "arousal_dur_s",
     "resp_index", "resp_frac", "resp_dur_s", "resp_index_rem", "resp_index_nrem"]
    + [f"resp_t{i}_index" for i in range(1, 6)] + ["limb_index", "limb_frac", "limb_t2_index", "invalid_frac"]
)


def algorithmic_features(path):
    """Sleep architecture, staging-uncertainty and event features from one CAISR annotation EDF."""
    f = dict.fromkeys(ALGO_KEYS, np.nan)
    if not path or not os.path.exists(path):
        return f
    try:
        sig, fs = _read_edf(path)
    except Exception:
        return f

    st = sig.get("stage_caisr")
    sleep_h = np.nan
    stage_at = None  # function: seconds -> stage code
    if st is not None and len(st):
        st = np.round(st).astype(int)
        valid = (st >= 1) & (st <= 5)
        f["invalid_frac"] = 1.0 - valid.mean()
        if valid.any():
            first, last = np.where(valid)[0][[0, -1]]
            s = st[first:last + 1]
            v = (s >= 1) & (s <= 5)
            asleep = v & (s != WAKE)
            f["tib_h"] = v.sum() * EPOCH_S / 3600
            f["tst_h"] = sleep_h = asleep.sum() * EPOCH_S / 3600
            f["sleep_eff"] = asleep.sum() / max(v.sum(), 1)
            if asleep.any():
                onset = np.argmax(asleep)
                end = len(s) - np.argmax(asleep[::-1])
                f["sol_min"] = onset * EPOCH_S / 60
                rem = s == REM
                f["rem_lat_min"] = (np.argmax(rem) - onset) * EPOCH_S / 60 if rem.any() else np.nan
                per = s[onset:end]
                f["waso_min"] = (per == WAKE).sum() * EPOCH_S / 60
                n = asleep.sum()
                for name, code in (("pct_n1", N1), ("pct_n2", N2), ("pct_n3", N3), ("pct_rem", REM)):
                    f[name] = (s == code).sum() / n
                wake_starts, wake_len = _runs(per == WAKE)
                f["awakenings_ph"] = len(wake_starts) / sleep_h
                pv = per[(per >= 1) & (per <= 5)]
                f["transitions_ph"] = (np.diff(pv) != 0).sum() / sleep_h
                # fragmentation: shifts from deeper sleep to wake or N1, per hour of sleep
                light = np.isin(pv, [WAKE, N1])
                f["frag_index"] = ((~light[:-1]) & light[1:]).sum() / sleep_h
                for name, code in (("bout_rem_min", REM), ("bout_n3_min", N3), ("bout_wake_min", WAKE)):
                    _, ln = _runs(per == code)
                    f[name] = ln.mean() * EPOCH_S / 60 if len(ln) else 0.0
                f["rem_bouts"] = len(_runs(per == REM)[0])
                h = len(per) // 2
                for name, code in (("rem_half_ratio", REM), ("n3_half_ratio", N3)):
                    a, b = (per[:h] == code).sum(), (per[h:] == code).sum()
                    f[name] = (b - a) / (a + b) if a + b else np.nan
            probs = [sig.get(k) for k in ("caisr_prob_n1", "caisr_prob_n2", "caisr_prob_n3", "caisr_prob_r", "caisr_prob_w")]
            if all(p is not None and len(p) == len(st) for p in probs):
                P = np.stack(probs, 1)[first:last + 1][v]
                P = np.clip(P, 0, 1)
                for name, col in zip(("prob_n1", "prob_n2", "prob_n3", "prob_rem", "prob_wake"), P.T):
                    f[name] = col.mean()
                Pn = P / np.clip(P.sum(1, keepdims=True), 1e-6, None)
                ent = -(Pn * np.log(Pn + 1e-9)).sum(1)
                f["stage_entropy"] = ent.mean()
                f["stage_maxprob"] = Pn.max(1).mean()
                sl = asleep[v]
                f["stage_entropy_sleep"] = ent[sl].mean() if sl.any() else np.nan

            def stage_at(t_s, _st=st):
                idx = np.clip((np.asarray(t_s) // EPOCH_S).astype(int), 0, len(_st) - 1)
                return _st[idx]

    hours = sleep_h if np.isfinite(sleep_h) and sleep_h > 0.5 else np.nan

    ar = sig.get("arousal_caisr")
    if ar is not None and len(ar):
        starts, ln = _runs(ar > 0.5)
        f["arousal_index"] = len(starts) / hours
        f["arousal_frac"] = (ar > 0.5).mean()
        f["arousal_dur_s"] = ln.mean() / fs["arousal_caisr"] if len(ln) else 0.0
    pa = sig.get("caisr_prob_arous")
    if pa is not None and len(pa):
        f["arousal_prob"] = np.clip(pa, 0, 1).mean()

    rs = sig.get("resp_caisr")
    if rs is not None and len(rs):
        rs = np.round(rs).astype(int)
        ev = (rs >= 1) & (rs <= 5)
        starts, ln = _runs(ev)
        rfs = fs["resp_caisr"]
        f["resp_index"] = len(starts) / hours
        f["resp_frac"] = ev.mean()
        f["resp_dur_s"] = ln.mean() / rfs if len(ln) else 0.0
        for i in range(1, 6):
            f[f"resp_t{i}_index"] = len(_runs(rs == i)[0]) / hours
        if stage_at is not None and len(starts) and np.isfinite(hours):
            es = stage_at(starts / rfs)
            rem_h = f["pct_rem"] * sleep_h
            nrem_h = sleep_h - rem_h
            f["resp_index_rem"] = (es == REM).sum() / rem_h if rem_h > 0.25 else np.nan
            f["resp_index_nrem"] = np.isin(es, [N1, N2, N3]).sum() / nrem_h if nrem_h > 0.25 else np.nan
        elif np.isfinite(hours):
            f["resp_index_rem"] = f["resp_index_nrem"] = 0.0

    lb = sig.get("limb_caisr")
    if lb is not None and len(lb):
        lb = np.round(lb).astype(int)
        f["limb_index"] = len(_runs(lb > 0)[0]) / hours
        f["limb_frac"] = (lb > 0).mean()
        f["limb_t2_index"] = len(_runs(lb == 2)[0]) / hours
    return f


def record_paths(data_folder, row):
    site, pid, ses = row["SiteID"], row["BidsFolder"], row["SessionID"]
    return (os.path.join(data_folder, ALGO_DIR, str(site), f"{pid}_ses-{ses}_caisr_annotations.edf"),
            os.path.join(data_folder, PHYS_DIR, str(site), f"{pid}_ses-{ses}.edf"))


def record_features(data_folder, row, signals=True):
    """All features for one demographics row (dict with SiteID, BidsFolder, SessionID, Age, ...).
    signals=False skips the raw recording (the version-2 model does not use those features)."""
    algo_path, phys_path = record_paths(data_folder, row)
    f = demographic_features(row)
    f.update(algorithmic_features(algo_path))
    if signals:
        f.update(_signal_features(phys_path, algo_path, row))
    return f


def _signal_features(phys_path, algo_path, row):
    """Signal features from the raw recording. Development only: if the recording is not on disk and the
    PSG_SIGNAL_CACHE folder holds the compact summaries saved earlier for this record (src/fetch.py), use those."""
    cache = os.environ.get("PSG_SIGNAL_CACHE")
    if cache and not os.path.exists(phys_path):
        p = os.path.join(cache, f"{row['BidsFolder']}_ses-{row['SessionID']}.npz")
        if os.path.exists(p):
            with np.load(p) as z:
                return features_from_intermediates({k: z[k] for k in z.files}, load_stages(algo_path))
    return signal_features(phys_path, algo_path)


def feature_table(data_folder, demographics, verbose=False, signals=True):
    """Feature DataFrame for every row of a demographics DataFrame, in the same order."""
    rows = []
    for i, row in enumerate(demographics.to_dict("records")):
        rows.append(record_features(data_folder, row, signals=signals))
        if verbose and (i + 1) % 100 == 0:
            print(f"  features {i + 1}/{len(demographics)}", flush=True)
    return pd.DataFrame(rows, index=demographics.index)
