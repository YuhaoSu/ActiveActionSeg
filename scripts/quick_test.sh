#!/bin/bash
# Quick test run with 1 cycle and 10 epochs

python train_active.py \
    --dataset gtea \
    --num_active_cycle 1 \
    --epochs 10 \
    --clip_active_method random \
    --video_active_method random \
    --device cuda:0 \
    --data_dir ./data \
    --output_dir ./results
