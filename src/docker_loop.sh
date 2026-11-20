#!/bin/bash
# Build the entry's Docker image (the organisers' Dockerfile, unedited) and run the official
# train_model.py -> run_model.py -> evaluate_model.py loop inside it on one existing split.
# Run inside WSL Ubuntu after installing Docker:
#   sudo bash /mnt/c/Personal/kaggle/physionet-2026/src/docker_loop.sh [split_tag]
# split_tag names a folder made by src/run_loop.sh (default full_s0), so the Docker score can be compared
# with the score from the pinned Python environment for the same split (out/<split_tag>/scores.txt).
# The project is mounted at the same path inside the container so the split folders' links resolve.
set -e
P=/mnt/c/Personal/kaggle/physionet-2026
TAG=${1:-full_s0}
S=$P/out/$TAG/split
O=$P/out/docker_$TAG
[ -d "$S/train" ] || { echo "no split at $S; make one with src/run_loop.sh first"; exit 1; }
mkdir -p "$O/model" "$O/outputs"
docker build -t physionet2026-entry "$P/entry"
docker run --rm --network none -v "$P:$P" -e PSG_SIGNAL_CACHE="$P/out/signal_cache" -e PYTHONWARNINGS=ignore \
  physionet2026-entry sh -c "
    python train_model.py -d $S/train -m $O/model -v &&
    python run_model.py -d $S/holdout -m $O/model -o $O/outputs &&
    python evaluate_model.py -d $S/holdout_labels.csv -o $O/outputs/demographics.csv \
      -p $P/data/small/demographics.csv -s $O/scores.txt"
echo "== Docker ($TAG) =="; cat "$O/scores.txt"
echo "== pinned Python env ($TAG) =="; cat "$P/out/$TAG/scores.txt" 2>/dev/null || true
