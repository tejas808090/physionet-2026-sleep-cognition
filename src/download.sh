#!/bin/bash
# Download files of the small training set whose path starts with a prefix; skips files already present.
# Usage: bash src/download.sh <prefix> [parallel jobs]
cd /c/Personal/kaggle/physionet-2026
K=/c/Personal/kaggle/.venv/Scripts/kaggle.exe
grep "^$1" data/small_files.txt | cut -d' ' -f1 | while read f; do [ -f "data/small/$f" ] || echo "$f"; done |
  xargs -P "${2:-8}" -I{} $K datasets download physionet/physionetchallenge2026data -f {} -p data/small -q >/dev/null 2>&1
echo "present: $(find data/small/$1* -type f | wc -l) of $(grep -c "^$1" data/small_files.txt)"
