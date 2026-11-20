"""Limited-data fetch for a PC that cannot hold the 229 GB training set.

Phase A: CAISR annotation files for a case-control subset (all positives + age/site-matched negatives).
Phase B: raw recordings for that subset, one at a time: download -> compact per-epoch summaries (.npz, a few MB)
         -> delete. Features are recomputed from the summaries, so feature changes never need a re-download.
Phase C: CAISR annotation files for everyone else (small files).
Never keeps more than one raw recording on disk. Resumable; backs off when Kaggle rate-limits.

Usage: python fetch.py [negatives_per_positive=1] [phases=ABC]
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "entry"))
from signals import features_from_intermediates, load_stages, signal_intermediates  # noqa: E402

DATASET = ("physionet", "physionetchallenge2026data")
DATA = ROOT / "data" / "small"
CACHE = ROOT / "out" / "signal_cache"
TMP = ROOT / "data" / "tmp"
RATIO = int(sys.argv[1]) if len(sys.argv) > 1 else 1
PHASES = sys.argv[2] if len(sys.argv) > 2 else "ABC"


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def pick_subset(demo, ratio):
    """All positives plus, for each, the closest-age unused negatives from the same site."""
    pos = demo[demo["Cognitive_Impairment"]]
    neg = demo[~demo["Cognitive_Impairment"]].copy()
    rng = np.random.default_rng(0)
    neg["tie"] = rng.random(len(neg))
    chosen = []
    for _ in range(ratio):
        for _, p in pos.sort_values("Age", ascending=False).iterrows():
            cand = neg[~neg.index.isin(chosen)]
            same = cand[cand["SiteID"] == p["SiteID"]]
            cand = same if len(same) and (same["Age"] - p["Age"]).abs().min() <= 2 else cand
            d = (cand["Age"] - p["Age"]).abs() + cand["tie"] * 0.5
            chosen.append(d.idxmin())
    return pd.concat([pos, demo.loc[chosen]])


_api = None


def download(file_name, dest_root):
    """One file via one API request (skips the CLI's extra metadata call). Returns the local path."""
    global _api
    from kaggle.api import kaggle_api_extended as kx
    from kagglesdk.datasets.types.dataset_api_service import ApiDownloadDatasetRequest
    if _api is None:
        _api = kx.KaggleApi()
        _api.authenticate()
    wait = 180
    while True:
        try:
            with _api.build_kaggle_client() as kaggle:
                req = ApiDownloadDatasetRequest()
                req.owner_slug, req.dataset_slug, req.file_name = DATASET[0], DATASET[1], file_name
                resp = kaggle.datasets.dataset_api_client.download_dataset(req)
                outfile = kx._resolve_download_path(str(dest_root), file_name, resp.request.url)
                os.makedirs(os.path.dirname(outfile), exist_ok=True)
                _api.download_file(resp, outfile, kaggle.http_client(), True, False)
            if kx._is_auto_compressed(outfile, file_name):
                kx._extract_and_remove_zip(outfile, os.path.dirname(outfile), unwrap_to=outfile[:-len(".zip")])
                outfile = outfile[:-len(".zip")]
            return Path(outfile)
        except Exception as e:  # rate limit or network: wait and retry
            msg = str(e)
            if "404" in msg:
                raise
            log(f"  waiting {wait}s after: {msg[:90]}")
            time.sleep(wait)
            wait = min(wait * 2, 900)


def main():
    demo = pd.read_csv(DATA / "demographics.csv")
    listed = set(l.split(" ")[0] for l in (ROOT / "data" / "small_files.txt").read_text().splitlines())
    sub = pick_subset(demo, RATIO)
    sub.to_csv(ROOT / "data" / "subset.csv", index=False)
    log(f"subset: {len(sub)} records ({int(sub.Cognitive_Impairment.sum())} positive), mean age pos "
        f"{sub[sub.Cognitive_Impairment].Age.mean():.1f} / neg {sub[~sub.Cognitive_Impairment].Age.mean():.1f}")
    CACHE.mkdir(parents=True, exist_ok=True)

    def algo(r):
        return f"algorithmic_annotations/{r.SiteID}/{r.BidsFolder}_ses-{r.SessionID}_caisr_annotations.edf"

    def get_algo(rows, name):
        todo = [algo(r) for r in rows.itertuples() if algo(r) in listed and not (DATA / algo(r)).exists()]
        log(f"phase {name}: {len(todo)} annotation files to fetch")
        for i, f in enumerate(todo):
            download(f, DATA)
            if (i + 1) % 25 == 0:
                log(f"  {name}: {i + 1}/{len(todo)}")

    if "A" in PHASES:
        get_algo(sub, "A")
    if "B" in PHASES:
        rows = [r for r in sub.itertuples() if not (CACHE / f"{r.BidsFolder}_ses-{r.SessionID}.npz").exists()]
        log(f"phase B: {len(rows)} raw recordings to stream")
        for i, r in enumerate(rows):
            f = f"physiological_data/{r.SiteID}/{r.BidsFolder}_ses-{r.SessionID}.edf"
            if f not in listed:
                continue
            t0 = time.time()
            p = download(f, TMP)
            t1 = time.time()
            try:
                inter = signal_intermediates(str(p))
            except Exception as e:
                log(f"  B: {r.BidsFolder} unreadable: {str(e)[:80]}")
                inter = {"duration_h": np.float32(np.nan), "n_channels": np.float32(np.nan)}
            size = p.stat().st_size / 1e6
            p.unlink()
            np.savez_compressed(CACHE / f"{r.BidsFolder}_ses-{r.SessionID}.npz", **inter)
            feats = features_from_intermediates(inter, load_stages(str(DATA / algo(r))))
            n_ok = sum(1 for v in feats.values() if v == v)
            log(f"  B: {i + 1}/{len(rows)} {r.BidsFolder} {size:.0f} MB, download {t1 - t0:.0f}s, summarise {time.time() - t1:.0f}s, "
                f"{n_ok}/{len(feats)} features, kept {sorted(k for k in inter if k[0] == 'e' or k == 'spo2')}")
    if "C" in PHASES:
        get_algo(demo, "C")
    log("done")


if __name__ == "__main__":
    main()
