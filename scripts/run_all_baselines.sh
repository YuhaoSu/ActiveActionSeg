#!/bin/bash
# Run all baselines on GTEA dataset

echo "Running Random baseline..."
bash scripts/run_gtea_random.sh

echo "Running Entropy MC baseline..."
bash scripts/run_gtea_entropy.sh

echo "Running Proposed method (STW + Drop-DTW)..."
bash scripts/run_gtea_proposed.sh

echo "All experiments completed!"
