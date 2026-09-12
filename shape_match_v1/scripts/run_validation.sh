#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
bash scripts/build.sh > results/build.log 2>&1
./build/test_match | tee results/test.log
./build/match_sample | tee results/sample.log
./build/accuracy_bench results/accuracy.csv | tee results/accuracy.log
./build/performance_bench 5 results/benchmark.json | tee results/benchmark.log
python3 scripts/report.py > results/report.log
echo "Validation complete: results/VALIDATION.md"
