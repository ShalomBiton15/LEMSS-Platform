#!/bin/bash
# Run every config in a folder one after another on one GPU; a failed run does not stop the rest.
# Usage (inside screen/tmux): bash configs/run_all_paper.sh [gpu_id] [folder]   (folder: paper or paper_test)
GPU=${1:-0}
DIR=$(cd "$(dirname "$0")" && pwd)
FOLDER=${2:-paper}

for CONFIG in "$DIR/$FOLDER"/*.json; do
    echo "=== $(date '+%F %T') start $(basename "$CONFIG")"
    bash "$DIR/run_lemss.sh" "$CONFIG" "$GPU" || echo "!!! $(basename "$CONFIG") failed"
done
echo "=== $(date '+%F %T') all done"
