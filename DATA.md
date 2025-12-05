# Dataset Preparation

This guide explains how to download and prepare datasets for AL-TAS experiments.

## Supported Datasets

- **GTEA** - Georgia Tech Egocentric Activities
- **50Salads** - 50 Salads Dataset
- **Breakfast** - Breakfast Actions Dataset

## Directory Structure

The code expects data to be organized as follows:

```
data/
├── gtea/
│   ├── clip_features/        # I3D features (.npy files)
│   ├── clip_labels/           # Frame-level labels (.npy files)
│   └── mapping.txt            # Action class mapping
├── 50salads/
│   ├── clip_features/
│   ├── clip_labels/
│   └── mapping.txt
└── breakfast/
    ├── clip_features/
    ├── clip_labels/
    └── mapping.txt
```

## Downloading Datasets

### GTEA

1. Download from: http://cbs.ic.gatech.edu/fpv/
2. Extract videos to a folder

### 50Salads

1. Download from: http://cvip.computing.dundee.ac.uk/datasets/foodpreparation/50salads/
2. Extract videos and annotations

### Breakfast

1. Download from: https://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/
2. Extract videos and annotations

## Feature Extraction

This code uses **I3D features** (2048-dimensional) pre-extracted from videos.

### Option 1: Extract Features Yourself

You'll need to:
1. Install I3D feature extraction tools
2. Process each video with I3D model
3. Save features as `.npy` files in `clip_features/` folder
4. Each feature file should be named: `{video_name}_video_embeddings.npy`
5. Shape: `(2048, num_clips)`

### Option 2: Use Pre-extracted Features

If you have access to pre-extracted I3D features:
1. Download the feature files
2. Place them in the corresponding `clip_features/` folder

## Label Preparation

Labels should be:
- Frame-level annotations
- Saved as `.npy` files in `clip_labels/` folder
- Named: `{video_name}.npy`
- Shape: `(num_frames,)`
- Values: action class indices (0 to num_classes-1)

## Mapping File Format

`mapping.txt` should contain action class mappings, one per line:

```
0 background
1 take_bowl
2 pour_milk
3 stir_milk
...
```

## STW Cache (Optional)

For the STW (Summary Time Warping) method, you can optionally pre-compute STW cache:

```bash
python pre_compute_stw.py --dataset gtea
```

This creates cache files in `stw_cache/` folder to speed up training.

## Data Path Configuration

By default, the code looks for data in `./data/`. You can change this with:

```bash
python train_active.py --dataset gtea --data_dir /path/to/your/data
```

## Verification

After setup, verify your data structure:

```bash
ls data/gtea/clip_features/  # Should show .npy files
ls data/gtea/clip_labels/    # Should show .npy files
cat data/gtea/mapping.txt    # Should show class mappings
```

## Troubleshooting

**Error: "No such file or directory: ./data/gtea/mapping.txt"**
- Make sure you've created the data directory structure
- Verify mapping.txt exists and has correct format

**Error: "Cannot load features"**
- Check that feature files are in `.npy` format
- Verify filenames match pattern: `{video_name}_video_embeddings.npy`

**Error: "Shape mismatch"**
- I3D features should be (2048, num_clips)
- Labels should be (num_frames,)

## Contact

For questions about data preparation, please open an issue on GitHub.
