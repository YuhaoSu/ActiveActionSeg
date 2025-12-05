# Active Learning for Temporal Action Segmentation (AL-TAS)

Official PyTorch implementation of **"Two-stage Active Learning Framework for Temporal Action Segmentation"** (ECCV 2024)

**Paper**: [https://www.khoury.northeastern.edu/home/eelhami/publications/ATAS-ECCV24.pdf](https://www.khoury.northeastern.edu/home/eelhami/publications/ATAS-ECCV24.pdf)

## Highlights

- **95% of full-supervision performance with only 0.35% labeled frames**
- Two-stage active learning framework:
  - **Inter-video selection**: Drop-DTW for diverse video sampling
  - **Intra-video selection**: Summary Time Warping (STW) for representative frame selection
- Achieves state-of-the-art results on GTEA, 50Salads, and Breakfast datasets

## Installation

### Requirements

- Python 3.7+
- PyTorch 1.8.0+
- CUDA (recommended)

### Setup

1. Clone this repository:
```bash
git clone https://github.com/YuhaoSu/ActiveActionSeg.git
cd ActiveActionSeg
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create data directory:
```bash
mkdir -p data results
```

4. Prepare datasets (see [DATA.md](DATA.md) for detailed instructions)

## Quick Start

### Quick Test (1-2 cycles)

```bash
python train_active.py \
    --dataset gtea \
    --num_active_cycle 1 \
    --epochs 10 \
    --clip_active_method random \
    --video_active_method random \
    --device cuda:0 \
    --data_dir /path/to/your/data \
    --output_dir ./results
```

### Full Experiments

**Random Baseline:**
```bash
python train_active.py \
    --dataset gtea \
    --clip_active_method random \
    --video_active_method random \
    --device cuda:0
```

**Entropy Baseline (MC Dropout):**
```bash
python train_active.py \
    --dataset gtea \
    --clip_active_method entropy_mc \
    --video_active_method random \
    --device cuda:0
```

**Proposed Method (STW + Drop-DTW):**
```bash
python train_active.py \
    --dataset gtea \
    --device cuda:0
```

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--dataset` | Dataset name (gtea, 50salads, breakfast) | 50salads |
| `--num_active_cycle` | Number of active learning cycles | 4 |
| `--epochs` | Training epochs per cycle | 80 |
| `--clip_active_method` | Clip selection (random, entropy_mc, stw) | stw |
| `--video_active_method` | Video selection (random, drop_dtw) | drop_dtw |
| `--clip_size` | Fraction of clips to label per video | 0.25 |
| `--adding_ratio` | Fraction of videos added per cycle | 0.05 |
| `--device` | CUDA device | cuda:2 |
| `--data_dir` | Path to dataset directory | ./data |
| `--output_dir` | Path to save results | ./results |

## Project Structure

```
al-tas/
├── train_active.py          # Main entry point for active learning
├── train_supervised.py      # Full supervision baseline
├── trainer.py               # Training loop with active learning
├── model.py                 # ASFormer model + learnable prototypes
├── active_batch_gen.py      # Data loader with progressive labeling
├── drop_dtw.py              # Drop-DTW for inter-video selection
├── optimizer.py             # STW for intra-video selection
├── core.py                  # Utility functions
├── metric.py                # Evaluation metrics
├── new_eval.py              # Additional evaluation metrics
├── temp.py                  # Helper functions
├── pre_compute_stw.py       # Pre-compute STW cache (optional)
├── stw_save.py              # STW cache utilities
└── DATA.md                  # Dataset preparation guide
```

## Expected Results

### GTEA Dataset

| Method | F1@10 | F1@25 | F1@50 | Edit | Acc |
|--------|-------|-------|-------|------|-----|
| Random | ~65% | ~60% | ~45% | ~70% | ~75% |
| Entropy MC | ~70% | ~65% | ~50% | ~75% | ~78% |
| **STW + Drop-DTW (Ours)** | **~78%** | **~75%** | **~62%** | **~82%** | **~85%** |
| Full Supervision | ~82% | ~80% | ~68% | ~85% | ~88% |

Note: Results with 0.35% labeled frames (4 active learning cycles, 5% videos per cycle, 25% clips per video)

## Data Preparation

See [DATA.md](DATA.md) for detailed instructions on:
- Downloading datasets (GTEA, 50Salads, Breakfast)
- Extracting I3D features
- Preparing labels
- Setting up directory structure

## STW Cache (Optional - For Faster Training)

The STW (Summary Time Warping) method can work in two modes:

### Option 1: On-the-fly Computation (Default)
The code computes STW on-the-fly during training if no cache exists. This works out of the box but is slower.

### Option 2: Pre-computed Cache (Recommended for Multiple Runs)
To speed up training when running multiple experiments:

```bash
# Pre-compute STW cache once
python pre_compute_stw.py --dataset gtea --data_dir ./data

# This creates stw_cache/ folder with pre-computed files:
# stw_cache/gtea_gtea.pt
```

**Note**: The `stw_cache/` directory is not included in the git repository. Users should pre-compute it locally or let the code compute on-the-fly.

## Results

Results are saved to `./results/` directory with the following structure:

```
results/
└── {dataset}_{method}_{timestamp}/
    ├── metrics.csv          # Evaluation metrics per cycle
    ├── selected_videos.txt  # Videos selected in each cycle
    └── model_checkpoints/   # Saved model weights
```

## Reproducibility

To reproduce results from the paper:

1. Download datasets and extract I3D features (see [DATA.md](DATA.md))
2. Run experiments with default parameters:
```bash
# Our method
python train_active.py --dataset gtea

# Baselines
python train_active.py --dataset gtea --clip_active_method random --video_active_method random
python train_active.py --dataset gtea --clip_active_method entropy_mc --video_active_method random
```

## Troubleshooting

**Issue**: `FileNotFoundError: [Errno 2] No such file or directory: './data/gtea/mapping.txt'`

**Solution**: Make sure you've prepared the dataset following [DATA.md](DATA.md)

**Issue**: `RuntimeError: CUDA out of memory`

**Solution**: Reduce batch size or use a smaller dataset for testing

**Issue**: `ImportError: No module named 'torch'`

**Solution**: Install PyTorch: `pip install torch>=1.8.0`

## Citation

If you find this code useful, please cite our paper:

```bibtex
@inproceedings{su2024altaas,
  title={Two-stage Active Learning Framework for Temporal Action Segmentation},
  author={Su, Yuhao and Elhamifar, Ehsan},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- ASFormer backbone: [https://github.com/ChinaYi/ASFormer](https://github.com/ChinaYi/ASFormer)
- I3D features extraction tools
- GTEA, 50Salads, and Breakfast dataset providers

## Contact

For questions or issues, please:
- Open an issue on GitHub: https://github.com/YuhaoSu/ActiveActionSeg/issues

---

**Note**: This is a minimal release for public availability. Pre-extracted features and model checkpoints are not included. Please follow [DATA.md](DATA.md) for dataset preparation instructions.
