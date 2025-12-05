#!/bin/bash
# Run Proposed method (STW + Drop-DTW) on GTEA dataset

python train_active.py \
    --dataset gtea \
    --num_active_cycle 4 \
    --epochs 80 \
    --clip_active_method stw \
    --video_active_method drop_dtw \
    --device cuda:0 \
    --data_dir ./data \
    --output_dir ./results
