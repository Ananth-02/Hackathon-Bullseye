#!/usr/bin/env bash
# Clone the two test codebases next to this folder, with full history for git evidence.
#
# Do NOT add --filter=blob:none here. `git blame` over history needs the historical
# blobs, so a blobless clone turns every blame into a separate lazy fetch from the
# remote: an --git run spent minutes doing hundreds of round-trips instead of
# reading locally. Paying for the blobs once at clone time is far cheaper.
set -e
mkdir -p ../repos && cd ../repos
[ -d FreeRTOS-Kernel ] || git clone https://github.com/FreeRTOS/FreeRTOS-Kernel.git
[ -d osal ] || git clone https://github.com/nasa/osal.git
[ -d PSP ]  || git clone https://github.com/nasa/PSP.git
