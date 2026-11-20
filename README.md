# PhysioNet Challenge 2026 — post-challenge entry

**Task.** [George B. Moody PhysioNet Challenge 2026](https://moody-challenge.physionet.org/2026/): from one overnight
sleep study (polysomnogram) and basic demographics, predict whether the patient will be diagnosed with cognitive
impairment 1 to 6 years later. The official metric is the **age-conditioned AUROC**: positives and negatives are only
compared when their ages are within 2 years, so a model has to add information beyond age.

**Status.** The Challenge closed on 20 August 2026, so this is a post-challenge entry: it follows the official
submission format and is scored locally with the organisers' unedited scripts. It has no official ranking, and it has
not been scored on the hidden validation or test sets.

## Approach

`entry/` is a complete Challenge entry (official template, with `train_model.py`, `run_model.py`, `evaluate_model.py`
and `helper_code.py` unedited).

| File | What it does |
|---|---|
| `entry/features.py` | Demographics, plus 43 sleep-architecture and event features from the CAISR annotations: sleep efficiency, stage shares, REM latency, fragmentation, staging uncertainty, arousal, breathing-event and limb-movement indices. |
| `entry/signals.py` | 79 features from the raw recording, read one channel at a time: EEG spectra by sleep stage (relative band power, slowing ratios, aperiodic slope, alpha and spindle peak), heart-rate variability by stage, oxygen saturation (desaturation index, hypoxic burden). |
| `entry/model.py` | Rank-averaged ensemble of a regularised logistic regression and shallow gradient-boosted trees. Inputs that describe the recording set-up, not the patient (channel count, absolute amplitude, BMI availability), and age itself, are left out. |
| `entry/team_code.py` | The three functions the Challenge calls: `train_model`, `load_model`, `run_model`. |

## Results (local, limited data)

The small training set is 229 GB (1,103 recordings), too large for the development PC. Raw recordings were therefore
streamed one at a time (download, reduce to a per-epoch summary of about 1.5 MB, delete) for a case-control subset:
all 84 positive patients and 84 negatives matched on site and age (every pair within 2 years).
All 168 recordings have been processed; 22 of them give only 58 of the 79 raw-signal features.

Repeated 5-fold cross-validation (10 repeats) on the 168 matched records, official metric:

| Features | Age-conditioned AUROC |
|---|---|
| Sleep staging and events | 0.625 ± 0.026 |
| EEG spectra | 0.585 ± 0.016 |
| Heart-rate variability | 0.440 ± 0.025 |
| Oxygen saturation | 0.582 ± 0.031 |
| Staging + EEG | 0.590 ± 0.024 |
| Staging + EEG + SpO2 | 0.604 ± 0.028 |
| Staging + all signals | 0.595 ± 0.027 |

(± is the spread over CV repeats, not a confidence interval; with 84 positives the sampling error is roughly ±0.04.
The raw-signal features do not improve on the staging and event features here. An earlier run on 42 of the pairs
gave EEG 0.593 and staging 0.559, the opposite order, which shows how unstable the ranking is at this sample size.)

Official loop (`src/run_loop.sh`: `train_model.py` → `run_model.py` on a held-out 25% with labels removed →
`evaluate_model.py`), run in a Python 3.10 environment with the entry's pinned packages. Both rows use the same five
splits of the 168 matched records (126 for training, 42 held out):

| Features used | Age-conditioned AUROC per split | Mean |
|---|---|---|
| Demographics + sleep staging and events | 0.694, 0.738, 0.635, 0.681, 0.621 | 0.674 |
| Everything (the entry as shipped) | 0.569, 0.762, 0.676, 0.556, 0.667 | 0.646 |

Adding the raw-signal features raised the score on three splits and lowered it on two, by 0.03 on average, which is
within noise for 42-record holdouts.

**At the natural positive rate.** Annotation files (no raw recordings needed) were fetched for 1,054 of the 1,103
small-set patients, including all 84 positives (8.0%). On these, with demographics and sleep staging/event features
only (no raw-signal features, which exist for just the 168):

| Evaluation | Age-conditioned AUROC |
|---|---|
| 5-fold CV, 5 repeats: sex, race and ethnicity only | 0.499 ± 0.019 |
| 5-fold CV, 5 repeats: plus sleep staging and events | 0.623 ± 0.014 |
| Official loop, five splits (790 train / 264 held out, 21 positives each) | 0.691, 0.642, 0.475, 0.561, 0.614 (mean 0.597) |

For reference, on the hidden test set the winning team scored 0.749 and the median of the 42 ranked teams 0.629
(`reference/challenge_2026_results.tsv`). Those teams trained on the full data, so the numbers are not directly
comparable with the ones above.

## Limits

- **Small, selected sample.** Raw-signal results come from 168 of 1,103 small-set records, chosen as matched cases and
  controls. Differences between feature groups are within noise.
- **No hidden-set score.** Everything is cross-validation or a local holdout on training data.
- **No Docker run.** Docker is not installed on the development PC. The entry was run in a clean Python 3.10
  environment with the pinned `requirements.txt`; the Docker image itself has not been built.
  `src/docker_loop.sh` builds it and repeats the official loop inside the container (untested so far).
- **Raw files not kept.** Locally the entry reads cached per-epoch summaries (`PSG_SIGNAL_CACHE`) in place of the raw
  recordings. With raw recordings present it reads them directly; that path was checked on real recordings during the
  streaming step and on a synthetic recording.
- **Dataset shortcuts.** In the small set the positive rate rises with recording year (20% for 2016–2019 against 4–5%
  before 2014) and is 19.5% where BMI is recorded against 3.8% where it is missing. These reflect how the data were
  assembled. The model does not use recording date or BMI.

## Reproduce

```bash
# data: annotations for a matched subset + streamed raw-signal summaries (resumable, never stores more than one recording)
.venv/Scripts/python.exe -I src/fetch.py 1 ABC
# cross-validation with the official metric
.venv/Scripts/python.exe -I src/cv.py data/small out/cv matched168 10 matched
.venv/Scripts/python.exe -I src/cv.py data/small out/cv pairs 10 pairs
.venv/Scripts/python.exe -I src/cv.py data/small out/cv all 5 all
# official train -> predict -> score loop in the pinned environment
bash src/run_loop.sh full_s0 0 data/subset.csv
bash src/run_loop.sh matched_s0 0 data/subset.csv nosignals
bash src/run_loop.sh all_s0 0 data/with_annotations.csv nosignals
# the same loop inside the entry's Docker image (run in WSL/Linux with Docker installed)
sudo bash src/docker_loop.sh full_s0
```

`submission/physionet2026_entry.zip` is the entry packaged as the repository root the organisers ask for.
