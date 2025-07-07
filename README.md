# Cell Challenge

A memory-efficient, transformer-based and baseline modeling pipeline for the Virtual Cell Challenge.

## Features
- Data exploration and visualization (`analysis/data_exploration.py`)
- Baseline model (mean gene expression, chunked, memory-safe)
- STATE-inspired transformer model for perturbation prediction
- Real data training pipeline with progress tracking and checkpointing
- Handles large single-cell datasets with chunking and .gitignore for big files

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Data Preparation
Place your data files in the `data/` directory:
- `adata_Training.h5ad`
- `gene_names.csv`
- `pert_counts_Validation.csv`

### 3. Data Exploration
```bash
python analysis/data_exploration.py
```

### 4. Run Baseline Model
```bash
python models/baseline_model.py
```

### 5. Run STATE Model on Real Data
```bash
python models/real_data_training.py
```

## Notes
- Large data and model files are excluded from git via `.gitignore`.
- Checkpoints and best models are saved in `models/`.
- For best results, use a machine with at least 16GB RAM.

## Repository
[https://github.com/adesaamit8desai/cell-challenge](https://github.com/adesaamit8desai/cell-challenge) 