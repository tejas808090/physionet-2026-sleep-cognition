"""Raw-signal features from one polysomnogram EDF: sleep EEG spectra by stage, heart rate variability, oxygen saturation.

Only the few channels needed are read (lazily), one at a time, so memory stays near one channel's size.
Everything is label-free and mostly relative (ratios, slopes, peak frequencies) so it transfers across sites
with different amplifiers and units. Any failure yields NaN features for that block, never an exception.
"""
import os

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt, spectrogram

from helper_code import load_rename_rules, standardize_channel_names_rename_only

EPOCH_S = 30.0
N3, N2, N1, REM, WAKE = 1, 2, 3, 4, 5
GROUPS = {"nrem": (N2, N3), "rem": (REM,), "wake": (WAKE,)}
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12), "sigma": (11, 16), "beta": (16, 30)}
EEG_REGIONS = {
    "c": [("c3-m2", "c3", "m2"), ("c4-m1", "c4", "m1")],
    "f": [("f3-m2", "f3", "m2"), ("f4-m1", "f4", "m1")],
    "o": [("o1-m2", "o1", "m2"), ("o2-m1", "o2", "m1")],
}
_trapz = getattr(np, "trapezoid", None) or np.trapz  # name changed across numpy versions
_DIR = os.path.dirname(os.path.abspath(__file__))
_RULES = None


def _eeg_keys(prefix):
    k = []
    for g in GROUPS:
        k += [f"{prefix}_{g}_rel_{b}" for b in BANDS]
        k += [f"{prefix}_{g}_slowing", f"{prefix}_{g}_theta_alpha", f"{prefix}_{g}_slope", f"{prefix}_{g}_peak_hz",
              f"{prefix}_{g}_peak_amp", f"{prefix}_{g}_logpow"]
    return k + [f"{prefix}_delta_decline", f"{prefix}_so_frac", f"{prefix}_artifact_frac"]


EEG_KEYS = _eeg_keys("eeg_c") + [f"eeg_{r}_{g}_{m}" for r in ("f", "o") for g in GROUPS for m in ("slowing", "slope", "peak_hz")]
ECG_KEYS = [f"hr_{g}" for g in GROUPS] + [f"rmssd_{g}" for g in GROUPS] + [f"sdnn_{g}" for g in GROUPS] + \
           ["hr_dip", "hr_rem_nrem", "hr_min", "hr_cv_nrem", "ecg_valid_frac"]
SPO2_KEYS = ["spo2_mean", "spo2_p01", "spo2_std", "spo2_lt90", "spo2_lt88", "spo2_odi3", "spo2_burden",
             "spo2_sleep_wake", "spo2_valid_frac"]
SIGNAL_KEYS = EEG_KEYS + ECG_KEYS + SPO2_KEYS + ["psg_hours", "psg_n_channels"]


class _Record:
    """Lazy view of a PSG EDF with the Challenge's standard channel names."""

    def __init__(self, path):
        import edfio
        global _RULES
        if _RULES is None:
            _RULES = load_rename_rules(os.path.join(_DIR, "channel_table.csv"))
        self.edf = edfio.read_edf(path, lazy_load_data=True)
        raw = {}
        for s in self.edf.signals:
            raw.setdefault(s.label.lower().strip(), s)
        rename, drop = standardize_channel_names_rename_only(list(raw), _RULES)
        self.sig = {}
        for label, s in raw.items():
            if label not in drop:
                self.sig.setdefault(rename.get(label, label), s)
        self.duration = float(self.edf.duration)

    def get(self, name):
        s = self.sig.get(name)
        if s is None:
            return None, None
        return np.asarray(s.data, dtype=np.float64), float(s.sampling_frequency)

    def derived(self, target, pos, neg):
        """A bipolar channel, taken directly if recorded, otherwise derived from its two electrodes."""
        x, fs = self.get(target)
        if x is not None:
            return x, fs
        a, fa = self.get(pos)
        b, fb = self.get(neg)
        if a is None or b is None or fa != fb or len(a) != len(b):
            return None, None
        return a - b, fa


def _epoch_psd(x, fs):
    """Median PSD per 30-s epoch from 4-s windows. Returns freqs (<=35 Hz) and array (n_epochs, n_freqs)."""
    nper = int(round(4 * fs))
    f, t, S = spectrogram(x, fs=fs, nperseg=nper, noverlap=nper // 2, detrend="constant", scaling="density")
    keep = f <= 35
    f, S = f[keep], S[keep]
    ep = (t // EPOCH_S).astype(int)
    n_ep = int(len(x) / fs // EPOCH_S)
    P = np.full((n_ep, len(f)), np.nan)
    order = np.argsort(ep, kind="stable")
    ep_s, S = ep[order], S[:, order]
    bounds = np.searchsorted(ep_s, np.arange(n_ep + 1))
    for i in range(n_ep):
        if bounds[i + 1] - bounds[i] >= 6:
            P[i] = np.median(S[:, bounds[i]:bounds[i + 1]], axis=1)
    return f, P


def _band(f, p, lo, hi):
    m = (f >= lo) & (f < hi)
    return _trapz(p[..., m], f[m], axis=-1)


PEAK_RANGE = {"wake": (7.0, 13.0), "rem": (6.0, 13.0), "nrem": (10.5, 16.0)}  # alpha, alpha/theta, sleep spindles


def _spectral_summary(f, p, peak_range=(6.0, 16.0)):
    """Relative band powers, slowing ratios, aperiodic slope and oscillatory peak of one mean spectrum."""
    out = {}
    total = _band(f, p, 0.5, 30)
    bp = {b: _band(f, p, lo, hi) for b, (lo, hi) in BANDS.items()}
    for b in BANDS:
        out[f"rel_{b}"] = bp[b] / total
    out["slowing"] = np.log((bp["delta"] + bp["theta"]) / (bp["alpha"] + bp["beta"]))
    out["theta_alpha"] = np.log(bp["theta"] / bp["alpha"])
    out["logpow"] = np.log10(total)
    # aperiodic slope: fit log-log outside the 6-16 Hz oscillatory range, then find the peak above the fit
    pos = f > 0
    fp, lf, lp = f[pos], np.log10(f[pos]), np.log10(p[pos])
    mf = ((fp >= 2) & (fp < 6)) | ((fp >= 18) & (fp <= 30))
    slope, icpt = np.polyfit(lf[mf], lp[mf], 1)
    out["slope"] = slope
    m_pk = (fp >= peak_range[0]) & (fp <= peak_range[1])
    resid = lp[m_pk] - (slope * lf[m_pk] + icpt)
    j = int(np.argmax(resid))
    out["peak_hz"] = fp[m_pk][j]
    out["peak_amp"] = resid[j]
    return out


def _clean_epochs(f, P):
    """Mask of usable epochs: finite, not flat, and total power not an extreme outlier (movement, electrode pop)."""
    tot = _band(f, P, 0.5, 30)
    ok = np.isfinite(tot) & (tot > 0)
    if ok.sum() < 20:
        return ok
    lt = np.log10(tot[ok])
    med = np.median(lt)
    mad = np.median(np.abs(lt - med)) * 1.4826 + 1e-6
    ok[ok] = (lt < med + 4 * mad) & (lt > med - 6 * mad)
    return ok


def eeg_spectra(rec):
    """Per-epoch spectra for each scalp region: {region: (freqs, array (n_epochs, n_freqs))}, mean over its channels."""
    spectra = {}
    for region, chans in EEG_REGIONS.items():
        specs, f = [], None
        for target, pos, neg in chans:
            try:
                x, fs = rec.derived(target, pos, neg)
                if x is None or fs < 64 or len(x) < fs * 600:
                    continue
                f, P = _epoch_psd(x, fs)
                specs.append(P)
            except Exception:
                continue
            finally:
                x = None
        if specs:
            n = min(len(P) for P in specs)
            with np.errstate(all="ignore"):
                spectra[region] = (f, np.nanmean(np.stack([p[:n] for p in specs]), axis=0))
    return spectra


def eeg_features(spectra, stages):
    out = dict.fromkeys(EEG_KEYS, np.nan)
    for region, (f, P) in spectra.items():
        n = len(P) if stages is None else min(len(P), len(stages))
        P = P[:n]
        ok = _clean_epochs(f, P)
        pre = f"eeg_{region}"
        if region == "c":
            out[f"{pre}_artifact_frac"] = 1.0 - ok.mean()
        if stages is None:
            continue
        st = stages[:n]
        for g, codes in GROUPS.items():
            m = ok & np.isin(st, codes)
            if m.sum() < 10:
                continue
            try:
                s = _spectral_summary(f, np.mean(P[m], axis=0), PEAK_RANGE[g])
            except Exception:
                continue
            for k, v in s.items():
                key = f"{pre}_{g}_{k}"
                if key in out:
                    out[key] = float(v)
        if region == "c":
            m = ok & np.isin(st, GROUPS["nrem"])
            if m.sum() >= 40:
                idx = np.where(m)[0]
                ld = np.log10(_band(f, P[idx], 0.5, 4))
                # slow-wave activity decline across the night (per hour): a marker of sleep homeostasis
                out[f"{pre}_delta_decline"] = float(np.polyfit(idx * EPOCH_S / 3600, ld, 1)[0])
                out[f"{pre}_so_frac"] = float(np.mean(_band(f, P[idx], 0.5, 1.25) / _band(f, P[idx], 0.5, 4)))
    return out


def _r_peaks(x, fs):
    sos = butter(2, [5, 20], btype="bandpass", fs=fs, output="sos")
    y = sosfiltfilt(sos, x)
    y = np.diff(y, prepend=y[0]) ** 2
    w = max(int(0.12 * fs), 1)
    y = np.convolve(y, np.ones(w) / w, mode="same")
    # local scale: per-10-s block maximum, smoothed over neighbouring blocks, so the threshold tracks amplitude changes
    blk = int(10 * fs)
    nb = len(y) // blk
    if nb < 3:
        return np.array([], dtype=int)
    bm = y[:nb * blk].reshape(nb, blk).max(1)
    pad = np.pad(bm, 2, mode="edge")
    sm = np.median(np.stack([pad[i:i + nb] for i in range(5)]), axis=0)
    scale = np.repeat(sm, blk)
    scale = np.concatenate([scale, np.full(len(y) - len(scale), sm[-1])])
    pk, _ = find_peaks(y, distance=int(0.3 * fs))
    return pk[y[pk] > 0.3 * scale[pk]]


def ecg_epochs(rec):
    """Heart rate (bpm), RMSSD and SDNN (ms) per 30-s epoch as an array (n_epochs, 3), NaN where unreliable."""
    x, fs = rec.get("ecg")
    if x is None or fs < 50 or len(x) < fs * 600:
        return None
    pk = _r_peaks(x, fs)
    x = None
    if len(pk) < 500:
        return None
    t = pk / fs
    rr = np.diff(t)
    tb = t[1:]
    good = (rr > 0.33) & (rr < 2.0)
    n_ep = int(tb[-1] // EPOCH_S) + 1
    E = np.full((n_ep, 3), np.nan)
    ep = (tb // EPOCH_S).astype(int)
    bounds = np.searchsorted(ep, np.arange(n_ep + 1))
    for i in range(n_ep):
        r = rr[bounds[i]:bounds[i + 1]]
        g = good[bounds[i]:bounds[i + 1]]
        if len(r) < 12 or g.mean() < 0.9:
            continue
        med = np.median(r[g])
        g = g & (np.abs(r - med) < 0.25 * med)  # drop ectopic and missed beats
        if g.sum() < 12:
            continue
        d = np.diff(r)[g[1:] & g[:-1]]
        E[i] = (60.0 / r[g].mean(), np.sqrt(np.mean(d ** 2)) * 1000 if len(d) >= 8 else np.nan, r[g].std() * 1000)
    return E


def ecg_features(E, stages):
    out = dict.fromkeys(ECG_KEYS, np.nan)
    if E is None or len(E) == 0:
        return out
    hr, rmssd, sdnn = E[:, 0], E[:, 1], E[:, 2]
    out["ecg_valid_frac"] = float(np.isfinite(hr).mean())
    if np.isfinite(hr).sum() < 20:
        return out
    out["hr_min"] = float(np.nanpercentile(hr, 2))
    if stages is not None:
        n = min(len(hr), len(stages))
        st = stages[:n]
        for g, codes in GROUPS.items():
            m = np.isin(st, codes) & np.isfinite(hr[:n])
            if m.sum() >= 10:
                out[f"hr_{g}"] = float(np.median(hr[:n][m]))
                with np.errstate(all="ignore"):
                    out[f"sdnn_{g}"] = float(np.nanmedian(sdnn[:n][m]))
                    out[f"rmssd_{g}"] = float(np.nanmedian(rmssd[:n][m]))
                if g == "nrem":
                    out["hr_cv_nrem"] = float(np.std(hr[:n][m]) / np.mean(hr[:n][m]))
        out["hr_dip"] = out["hr_wake"] - out["hr_nrem"]
        out["hr_rem_nrem"] = out["hr_rem"] - out["hr_nrem"]
    return out


def spo2_series(rec):
    """Oxygen saturation at 1 Hz in percent (raw values, invalid ones included), or None."""
    x, fs = rec.get("spo2")
    if x is None or len(x) < fs * 600:
        return None
    step = max(int(round(fs)), 1)
    n = len(x) // step
    v = np.median(x[:n * step].reshape(n, step), axis=1) if step > 1 else x[:n].copy()
    if np.nanpercentile(v, 95) <= 1.5:  # stored as a fraction
        v = v * 100
    return v


def spo2_features(v, stages):
    out = dict.fromkeys(SPO2_KEYS, np.nan)
    if v is None or len(v) == 0:
        return out
    v = np.asarray(v, dtype=np.float64)
    ok = (v >= 50) & (v <= 100)
    out["spo2_valid_frac"] = float(ok.mean())
    if ok.sum() < 1800:
        return out
    vv = v[ok]
    out["spo2_mean"] = float(vv.mean())
    out["spo2_p01"] = float(np.percentile(vv, 1))
    out["spo2_std"] = float(vv.std())
    out["spo2_lt90"] = float((vv < 90).mean())
    out["spo2_lt88"] = float((vv < 88).mean())
    # desaturations: at least 3 points below the highest value of the preceding 2 minutes, for 10 s or longer
    vf = np.where(ok, v, -np.inf)
    W = 120
    pad = np.concatenate([np.full(W, -np.inf), vf])
    base = np.lib.stride_tricks.sliding_window_view(pad, W)[:len(vf)].max(axis=1)  # max of the previous W seconds
    base[~np.isfinite(base)] = np.nan
    drop = np.where(ok, base - v, np.nan)
    d = np.nan_to_num(drop, nan=0.0) >= 3
    dd = np.diff(np.concatenate([[0], d.astype(np.int8), [0]]))
    starts, ends = np.where(dd == 1)[0], np.where(dd == -1)[0]
    out["spo2_odi3"] = float(((ends - starts) >= 10).sum() / (ok.sum() / 3600))
    out["spo2_burden"] = float(np.nansum(np.clip(drop, 0, None)) / ok.sum())
    if stages is not None:
        st = np.repeat(stages, int(EPOCH_S))[:len(v)]
        vs, oks = v[:len(st)], ok[:len(st)]
        sl, wk = oks & np.isin(st, (N1, N2, N3, REM)), oks & (st == WAKE)
        if sl.sum() > 600 and wk.sum() > 300:
            out["spo2_sleep_wake"] = float(vs[sl].mean() - vs[wk].mean())
    return out


def load_stages(algo_path):
    """CAISR sleep stage per 30-s epoch (int array, 9 = unscored), or None if unavailable."""
    if not algo_path or not os.path.exists(algo_path):
        return None
    try:
        import edfio
        edf = edfio.read_edf(algo_path, lazy_load_data=True)
        for s in edf.signals:
            if s.label.lower().strip() == "stage_caisr":
                return np.round(np.asarray(s.data, dtype=np.float64)).astype(int)
    except Exception:
        pass
    return None


def signal_intermediates(phys_path):
    """Compact per-epoch summaries of one recording (a few MB): enough to recompute every feature without the raw file.
    Keys: duration_h, n_channels, eeg_freqs, eeg_<region> (n_epochs, n_freqs), ecg (n_epochs, 3), spo2 (1 Hz)."""
    rec = _Record(phys_path)
    inter = {"duration_h": np.float32(rec.duration / 3600), "n_channels": np.float32(len(rec.sig))}
    try:
        for region, (f, P) in eeg_spectra(rec).items():
            inter["eeg_freqs"] = f.astype(np.float32)
            inter[f"eeg_{region}"] = P.astype(np.float32)
    except Exception:
        pass
    for key, fn in (("ecg", ecg_epochs), ("spo2", spo2_series)):
        try:
            v = fn(rec)
            if v is not None:
                inter[key] = v.astype(np.float32)
        except Exception:
            pass
    return inter


def features_from_intermediates(inter, stages):
    out = dict.fromkeys(SIGNAL_KEYS, np.nan)
    out["psg_hours"] = float(inter["duration_h"])
    out["psg_n_channels"] = float(inter["n_channels"])
    spectra = {r: (np.asarray(inter["eeg_freqs"], dtype=np.float64), np.asarray(inter[f"eeg_{r}"], dtype=np.float64))
               for r in EEG_REGIONS if f"eeg_{r}" in inter}
    blocks = ((eeg_features, spectra), (ecg_features, inter["ecg"] if "ecg" in inter else None),
              (spo2_features, inter["spo2"] if "spo2" in inter else None))
    for fn, arg in blocks:
        try:
            out.update(fn(arg, stages))
        except Exception:
            pass
    return out


def signal_features(phys_path, algo_path):
    """All raw-signal features for one recording; NaN for anything that cannot be computed."""
    out = dict.fromkeys(SIGNAL_KEYS, np.nan)
    if not phys_path or not os.path.exists(phys_path):
        return out
    try:
        inter = signal_intermediates(phys_path)
    except Exception:
        return out
    return features_from_intermediates(inter, load_stages(algo_path))
