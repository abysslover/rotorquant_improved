#!/bin/bash
# TurboQuant Facade 실행 래퍼 - torch import 전 LD_LIBRARY_PATH 설정 보장
#
# Usage:
#   ./run_turboquant_facade.sh [args]
#
# Example:
#   ./run_turboquant_facade.sh
#   ./run_turboquant_facade.sh --output-dir results

# 스크립트가 위치한 디렉토리
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 프로젝트 루트 디렉토리 (experiments 의 상위 디렉토리)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# conda 환경에서 실행
source /data/anaconda3/etc/profile.d/conda.sh

# PyTorch lib 경로 설정 (py310 환경)
# CUDA 라이브러리가 로드될 수 있도록 LD_LIBRARY_PATH 설정
export LD_LIBRARY_PATH="/data/anaconda3/envs/py310/lib/python3.10/site-packages/torch/lib:/data/anaconda3/envs/py310/lib/python3.10/site-packages/torch/lib/cuda/lib:${LD_LIBRARY_PATH:-}"
echo "[INFO] Set LD_LIBRARY_PATH to include CUDA libs"

# conda linker 문제 우회 - /usr/bin 가 먼저 오도록 PATH 설정
export PATH="/usr/bin:/usr/local/bin:$PATH"

cd "$PROJECT_ROOT"
conda run -n py310 python3 experiments/turboquant_facade.py "$@"
