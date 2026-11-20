#!/usr/bin/env python

# Team code for the George B. Moody PhysioNet Challenge 2026.
# Required functions: train_model, load_model, run_model. Features live in features.py, the model in model.py.

import os

import joblib
import numpy as np
import pandas as pd

from features import feature_table, record_features
from model import Ensemble

DEMOGRAPHICS_FILE = 'demographics.csv'
MODEL_FILE = 'model.joblib'
LABEL = 'Cognitive_Impairment'


def _labels(df):
    return df[LABEL].astype(str).str.strip().str.lower().isin(['true', '1', '1.0']).astype(int).values


def train_model(data_folder, model_folder, verbose):
    if verbose:
        print('Finding the Challenge data...')
    demo = pd.read_csv(os.path.join(data_folder, DEMOGRAPHICS_FILE))
    demo = demo.drop_duplicates(subset=['BidsFolder', 'SiteID', 'SessionID']).reset_index(drop=True)
    if len(demo) == 0:
        raise FileNotFoundError('No data were provided.')

    if verbose:
        print(f'Extracting features from {len(demo)} records...')
    X = feature_table(data_folder, demo, verbose=verbose)
    y = _labels(demo)

    if verbose:
        print(f'Training the model on {len(y)} records ({int(y.sum())} positive)...')
    model = Ensemble().fit(X, y)

    os.makedirs(model_folder, exist_ok=True)
    joblib.dump({'model': model}, os.path.join(model_folder, MODEL_FILE), protocol=4)
    if verbose:
        print('Done.')


def load_model(model_folder, verbose):
    return joblib.load(os.path.join(model_folder, MODEL_FILE))


def run_model(model, record, data_folder, verbose):
    m = model['model']
    # The record only carries identifiers; read the rest of the row (age, sex, ...) from the demographics file.
    demo = pd.read_csv(os.path.join(data_folder, DEMOGRAPHICS_FILE))
    mask = (demo['BidsFolder'] == record['BidsFolder']) & (demo['SessionID'] == record['SessionID'])
    row = demo.loc[mask].iloc[0].to_dict() if mask.any() else dict(record)
    X = pd.DataFrame([record_features(data_folder, row)])
    probability = float(m.predict_proba(X)[0])
    if not np.isfinite(probability):
        probability = 0.0
    binary = bool(m.predict(X)[0])
    return binary, probability
