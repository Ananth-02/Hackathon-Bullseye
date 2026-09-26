#!/usr/bin/env bash
# Clone the two test codebases next to this folder, with history (blobs fetched lazily) for git evidence.
set -e
mkdir -p ../repos && cd ../repos
[ -d FreeRTOS-Kernel ] || git clone --filter=blob:none https://github.com/FreeRTOS/FreeRTOS-Kernel.git
[ -d osal ] || git clone --filter=blob:none https://github.com/nasa/osal.git
[ -d PSP ]  || git clone --filter=blob:none https://github.com/nasa/PSP.git
