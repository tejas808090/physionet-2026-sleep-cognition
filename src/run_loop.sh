#!/bin/bash
# Full Challenge loop with the official, unedited scripts: train -> run on holdout -> score.
# Runs in the pinned Python 3.10 environment (same package versions as the entry's Dockerfile).
# Raw recordings are not kept on this PC, so the entry reads the compact summaries in out/signal_cache instead.
# Prevalence by age (for the Reward metric) comes from the full small training set's demographics.
# Usage: bash src/run_loop.sh [tag] [seed] [records_csv] [nosignals]
#   nosignals: ignore the signal summaries, so only demographics and annotation features are used
#   (needed when some records have no summary yet, otherwise 'has a summary' would leak the label).
set -e
cd /c/Personal/kaggle/physionet-2026
PY=$PWD/.venv-entry/Scripts/python.exe
TAG=${1:-loop}; SEED=${2:-0}; REC=${3:-data/pairs_with_signals.csv}
S=out/$TAG/split; O=out/$TAG
export PSG_SIGNAL_CACHE=$PWD/out/signal_cache PYTHONWARNINGS=ignore
[ "$4" = nosignals ] && export PSG_SIGNAL_CACHE=$PWD/out/no_cache
mkdir -p $O/model $O/outputs
$PY src/make_split.py data/small $S 0.25 $SEED $REC
cd entry
$PY train_model.py -d ../$S/train -m ../$O/model -v 2>&1 | grep -E "Training|Done" || true
$PY run_model.py -d ../$S/holdout -m ../$O/model -o ../$O/outputs
$PY evaluate_model.py -d ../$S/holdout_labels.csv -o ../$O/outputs/demographics.csv -p ../data/small/demographics.csv -s ../$O/scores.txt -t ../$O/table.tsv
cat ../$O/scores.txt
