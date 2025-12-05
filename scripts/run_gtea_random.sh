#!/bin/bash
# Run Random baseline on GTEA dataset

python train_active.py \
    --dataset gtea \
    --num_active_cycle 4 \
    --epochs 80 \
    --clip_active_method random \
    --video_active_method random \
    --device cuda:0 \
    --data_dir ./data \
    --output_dir ./results
