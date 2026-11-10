#!/bin/bash
# Runs every benchmark sweep sequentially; results/<sweep>.jsonl, logs/bench_<sweep>.log
cd "$(dirname "$0")"
PY=${PYTHON:-python}
for s in iters batch horizon step tol scenes precision; do
  rm -f results/$s.jsonl
  $PY -I bench.py --sweep $s --out results/$s.jsonl > logs/bench_$s.log 2>&1
  echo "sweep $s done"
done
