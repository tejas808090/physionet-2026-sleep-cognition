# PhysioNet Challenge 2026 — post-challenge entry

**Task.** [George B. Moody PhysioNet Challenge 2026](https://moody-challenge.physionet.org/2026/): from one overnight
sleep study (polysomnogram) and basic demographics, predict whether the patient will be diagnosed with cognitive
impairment 1 to 6 years later. The official metric is the **age-conditioned AUROC**: positives and negatives are only
compared when their ages are within 2 years, so a model has to add information beyond age.

**Status.** The Challenge closed on 20 August 2026, so this is a post-challenge entry: it follows the official
submission format and is scored locally with the organisers' unedited scripts. It has no official ranking, and it has
not been scored on the hidden validation or test sets.

## Approach (version 2)

`entry/` is a complete Challenge entry (official template, with `train_model.py`, `run_model.py`, `evaluate_model.py`
and `helper_code.py` unedited).

| File | What it does |
|---|---|
| `entry/features.py` | Demographics, plus 43 sleep-architecture and event features from the CAISR annotations: sleep efficiency, stage shares, REM latency, fragmentation, staging uncertainty, arousal, breathing-event and limb-movement indices. |
| `entry/model.py` | `LinearModel` (version 2): one strongly regularised logistic regression (C = 0.005) on log-compressed, standardised features. Inputs that describe the recording set-up rather than the patient (channel count, absolute amplitude, BMI availability), and age itself, are left out. `Ensemble` (version 1) is kept for reference. |
| `entry/team_code.py` | The three functions the Challenge calls: `train_model`, `load_model`, `run_model`. |
| `entry/signals.py` | 79 features from the raw recording (EEG spectra by sleep stage, heart-rate variability, oxygen saturation). Built and tested in version 1; version 2 does not use them (see below), so the entry no longer reads the raw recordings. |

**What changed from version 1, and why.** Version 1 rank-averaged a logistic regression with shallow boosted trees
and used the raw-signal features. Version 2 followed three findings:

1. On all 1,090 small-set patients with annotations, the boosted trees alone scored 0.597 and the logistic model alone
   0.634, so the trees were diluting the ensemble (0.629).
2. Much stronger regularisation helped: 5× repeated 5-fold CV gave 0.615 at C = 1, 0.630 at C = 0.05 (version 1) and
   0.644 at C = 0.005, flat between 0.003 and 0.005.
3. On the 168 records with raw-signal summaries, the same logistic model scored 0.631 without the raw-signal features
   and 0.600 with them (10× repeated CV). Dropping them also makes the entry much faster.

Residualising each feature on age, adding age back as an input, and boosted trees alone were also tried; none helped
(`src/exp_v2.py`).

## Results (local, limited data)

The small training set is 229 GB (1,103 recordings), too large for the development PC. The CAISR annotation files
(0.5 GB) were downloaded for every patient that has one (1,090, including all 84 positives, 7.7%). Raw recordings
were streamed one at a time (download, reduce to a per-epoch summary of about 1.5 MB, delete) for a case-control subset
of 168: all 84 positive patients and 84 negatives matched on site and age (every pair within 2 years).

**Official loop** (`src/run_loop.sh`: `train_model.py` → `run_model.py` on a held-out 25% with labels removed →
`evaluate_model.py`), run in a Python 3.10 environment with the entry's pinned packages. Each row compares the two
versions on the same five splits:

| Records (holdout size) | Version 1 per split | v1 mean | Version 2 per split | v2 mean |
|---|---|---|---|---|
| 1,054 patients, natural prevalence (264) | 0.691, 0.642, 0.475, 0.561, 0.614 | 0.597 | 0.707, 0.705, 0.603, 0.569, 0.633 | **0.643** |
| 168 matched cases and controls (42) | 0.694, 0.738, 0.635, 0.681, 0.621 ¹ | 0.674 | 0.681, 0.794, 0.676, 0.674, 0.667 | **0.698** |
| 1,090 patients, natural prevalence (273) | | | 0.553, 0.753, 0.696, 0.588, 0.583 | 0.635 |

¹ Version 1 without its raw-signal features; with them (the version 1 entry as shipped) the mean was 0.646.

Version 2 is ahead on four of five splits in both comparisons. Holdouts with 21 positives swing by ±0.1 from split to
split, so the per-split numbers are noisy; the cross-validated numbers above are the steadier comparison.

For reference, on the hidden test set the winning team scored 0.749 and the median of the 42 ranked teams 0.629
(`reference/challenge_2026_results.tsv`). Those teams trained on the full data and were scored on other hospitals, so
the numbers are not directly comparable with the ones here.

### What was tried with the raw recordings (version 1)

Repeated 5-fold cross-validation (10 repeats) on the 168 matched records, version 1 model, official metric:

| Features | Age-conditioned AUROC |
|---|---|
| Sleep staging and events | 0.625 ± 0.026 |
| EEG spectra | 0.585 ± 0.016 |
| Heart-rate variability | 0.440 ± 0.025 |
| Oxygen saturation | 0.582 ± 0.031 |
| Staging + EEG | 0.590 ± 0.024 |
| Staging + EEG + SpO2 | 0.604 ± 0.028 |
| Staging + all signals | 0.595 ± 0.027 |

EEG and oxygen-saturation features carry some signal on their own, but in this sample they add nothing on top of the
staging and event features. With 84 positives the sampling error is roughly ±0.04, and an earlier run on 42 of the
pairs gave the opposite order, so this may change with more data. `signals.py` and the streaming pipeline stay in the
repository for that reason.

## Limits

- **Small sample.** Results come from at most 1,090 small-set patients with 84 positives; raw-signal results from 168
  matched records. Differences under about 0.03 are within noise.
- **No hidden-set score.** Everything is cross-validation or a local holdout on training data.
- **No Docker run.** Docker is not installed on the development PC. The entry was run in a clean Python 3.10
  environment with the pinned `requirements.txt`; the Docker image itself has not been built.
  `src/docker_loop.sh` builds it and repeats the official loop inside the container (untested so far).
- **Dataset shortcuts.** In the small set the positive rate rises with recording year (20% for 2016–2019 against 4–5%
  before 2014) and is 19.5% where BMI is recorded against 3.8% where it is missing. These reflect how the data were
  assembled. The model does not use recording date or BMI.

## Reproduce

```bash
# data: annotations for everyone + streamed raw-signal summaries for the matched subset (resumable)
.venv/Scripts/python.exe -I src/fetch.py 1 ABC
# version 2 model choices (logistic regularisation, variants) on all annotated patients
.venv/Scripts/python.exe -I src/exp_v2.py data/small 5 lr@0.002,lr@0.005,lr@0.05,base,gb,resid,with_age
# cross-validation of feature groups with the official metric (version 1 model)
.venv/Scripts/python.exe -I src/cv.py data/small out/cv pairs 10 pairs
# official train -> predict -> score loop in the pinned environment (version 2 entry)
bash src/run_loop.sh v2_all_s0 0 data/with_annotations_1090.csv nosignals
bash src/run_loop.sh v2_matched_s0 0 data/subset.csv nosignals
# the same loop inside the entry's Docker image (run in WSL/Linux with Docker installed)
sudo bash src/docker_loop.sh v2_all_s0
```

`submission/physionet2026_entry.zip` is the entry packaged as the repository root the organisers ask for.
