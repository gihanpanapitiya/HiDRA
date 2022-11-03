#!/bin/bash

# arg 1 CUDA_VISIBLE_DEVICES
# arg 2 CANDLE_DATA_DIR
# arg 3 CANDLE_CONFIG

export CUDA_VISIBLE_DEVICES=$1
export CANDLE_DATA_DIR=$2
export CANDLE_CONFIG=$3

python HiDRA_FeatureGeneration.py
python HiDRA_training.py

