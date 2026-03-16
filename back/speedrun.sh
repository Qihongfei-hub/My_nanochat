#!/bin/bash

# This script is configured to train your own GPT-2 grade LLM (pretraining + finetuning)
# It is designed to run on a blank 8XH100 GPU node and takes approximately 3 hours to complete.

# 1) Example launch (simplest):
# bash runs/speedrun.sh
# 2) Example launch in a screen session (because the run takes ~3 hours):
# screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh
# 3) Example launch with wandb logging, but see below for setting up wandb first:
# WANDB_RUN=speedrun screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# Ensure we run from repo root even if script is invoked from elsewhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

## added by qhf, 2024-06-17, for uv compatibility on windows, see
##orchrun 尝试使用 libuv 来创建分布式 rendezvous/TCP store，但你当前安装的 PyTorch 是在没有 libuv 支持下构建的，报错信息里说 "use_libuv was requested but PyTorch was built without libuv support"。因此无法建立 C10d store，导致 RendezvousConnectionError
export USE_LIBUV=0
# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
#command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
#[ -d ".venv" ] || uv venv
# install the repo dependencies
#uv sync --extra gpu
# activate venv so that `python` uses the project's venv instead of system python
#source .venv/bin/activate


# -----------------------------------------------------------------------------
# wandb setup
# If you wish to use wandb for logging (it's nice!, recommended).
# 1) Make sure to first log in to wandb, e.g. run:
#    `wandb login`
# 2) Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=d26 bash speedrun.sh`
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# Download the first ~2B characters of pretraining dataset
# each data shard is ~250M chars
# so we download 2e9 / 250e6 = 8 data shards at this point
# each shard is ~100MB of text (compressed), so this is about ~800MB of data on disk
# look at dev/repackage_data_reference.py for details on how this data was prepared
#python -m nanochat.dataset -n 8
# Immediately also kick off downloading more shards in the background while tokenizer trains
# Approximately 150 shards are needed for GPT-2 capability pretraining, add 20 for padding.
# The maximum total number of shards available in the entire dataset is 6542.
#python -m nanochat.dataset -n 170 &
#DATASET_DOWNLOAD_PID=$!
# train the tokenizer with vocab size 2**15 = 32768 on ~2B characters of data


python -m scripts.tok_train
python -m scripts.tok_eval


python -m scripts.base_train --depth=12 --save-every=500 --num-iterations=14000 --run=dummy --depth=8 --head-dim=64 --window-pattern=L  --max-seq-len=512  --device-batch-size=32 --total-batch-size=16384 --eval-tokens=524288 --core-metric-every=-1 --sample-every=100


# On Windows, avoid torchrun rendezvous entirely for single-process runs.
# This bypasses C10d/TCP store issues (libuv and localhost resolution).

if [[ "$OS" == "Windows_NT" ]]; then
    python scripts/base_train.py --depth=4 --target-param-data-ratio=4 --device-batch-size=32
else
    USE_LIBUV=0 torchrun --standalone scripts/base_train.py --depth=4 --target-param-data-ratio=2 --device-batch-size=1
fi
