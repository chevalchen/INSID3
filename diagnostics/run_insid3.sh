#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
out=${1:?fresh output directory required}
mkdir "$out"
export CONDA_PREFIX=/data6/chensq/miniconda3/envs/insid3
export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHONPATH="$PWD/insid3_deps:$PWD/insid3_crf"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0
python -c 'import sys,torch,torchvision;print(sys.executable,torch.__version__,torchvision.__version__,flush=True)'
python -u diagnose.py "$out/smoke" --limit 8
python -u diagnose.py "$out/full"
cmp manifest0/manifest.jsonl "$out/full/manifest.jsonl"
python -u evaluator_replay.py "$out/full" > "$out/evaluator.log" 2>&1
python summary.py "$out/full" "$out/full"
python report.py "$out/full"
