# Sleep-study features for predicting cognitive impairment (PhysioNet Challenge 2026 entry)

Entry for the George B. Moody PhysioNet Challenge 2026, in the layout of the official Python example.

- `team_code.py` — `train_model`, `load_model`, `run_model`.
- `features.py` — demographic and CAISR annotation features (sleep architecture, staging uncertainty, events).
- `signals.py` — raw-signal features: EEG spectra by sleep stage, heart-rate variability, oxygen saturation.
- `model.py` — rank-averaged logistic regression + shallow gradient-boosted trees.
- `train_model.py`, `run_model.py`, `evaluate_model.py`, `helper_code.py`, `channel_table.csv` — official, unedited.

Run:

    python train_model.py -d training_data -m model -v
    python run_model.py -d holdout_data -m model -o holdout_outputs -v
    python evaluate_model.py -d labels.csv -o holdout_outputs/demographics.csv -p training_data/demographics.csv

Missing or unreadable files give missing features, not errors, so every record gets a prediction.
