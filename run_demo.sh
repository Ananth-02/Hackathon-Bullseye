#!/usr/bin/env bash
# Analyse both codebases and build the viewer. Add --git for commit evidence (slower, ~1 min each).
set -e
python -m bullseye analyze ../repos/FreeRTOS-Kernel --platform freertos-cm4f --label "FreeRTOS kernel" --git -o data/freertos.json
python -m bullseye analyze ../repos --platform cfs-mcp750-vxworks --label "NASA cFS (OSAL + PSP)" --git -o data/cfs.json
python -m bullseye viewer data/freertos.json data/cfs.json -o bullseye.html
python -m bullseye eval data/freertos.json eval/freertos.json || true
python -m bullseye eval data/cfs.json eval/cfs.json || true
echo "Open bullseye.html in a browser."
