#!/bin/bash
# Run a LEMSS competition on one GPU and keep a log of everything it prints.
# Usage: bash configs/run_lemss.sh <config_file> [gpu_id]
set -eo pipefail

CONFIG=$1
GPU=${2:-0}

source ~/PROGS/miniconda3/bin/activate lemss
cd ~/projects/LEMSS-Platform

mkdir -p run_logs
LOG=run_logs/$(basename "$CONFIG" .json)_$(date +%Y-%m-%d_%H-%M).log
echo "Config: $CONFIG | GPU: $GPU | Log: $LOG"

CUDA_VISIBLE_DEVICES=$GPU python main.py --config_file "$CONFIG" 2>&1 | tee "$LOG"
