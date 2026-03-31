#!/bin/bash

set -e

CONDA_ENV="py310"
RESULTS_DIR="benchmarks/results"
BENCHMARK_DIR="benchmarks"

mkdir -p "$RESULTS_DIR"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

run_benchmark() {
    script="$1"
    name="$(basename "$script" .py)"
    output="$RESULTS_DIR/${name}.tsv"
    
    if [ -f "$output" ]; then
        echo "SKIP: $output already exists, preserving previous results"
        return 0
    fi
    
    echo "Running: $name"
    if conda run -n "$CONDA_ENV" python "$script" 2>&1 | tee "$output"; then
        echo "Done: $name -> $output"
    else
        echo "WARN: $name failed, output saved to $output"
    fi
}

SKIP_SCRIPTS="benchmark_vram benchmark_vs_reference"

for script in "$BENCHMARK_DIR"/benchmark*.py; do
    if [ -f "$script" ]; then
        name="$(basename "$script" .py)"
        if echo "$SKIP_SCRIPTS" | grep -q "$name"; then
            echo "SKIP: $name (requires unavailable dependency)"
            continue
        fi
        run_benchmark "$script" || true
    fi
done

echo "All benchmarks completed!"
