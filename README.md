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

## Submission Flow & Memory Issues (2024 Update)

### Background
- The original approach (in-memory concatenation of all predictions) caused out-of-memory (OOM) errors, even for small test cases, due to the size of the data and AnnData's memory usage.
- The new approach writes each perturbation's predictions as a separate `.h5ad` file, then merges them in batches for the final submission.

### Steps for a Valid Submission
1. **Run the submission script:**
   ```bash
   python models/create_submission.py
   ```
   - This writes one `.h5ad` file per perturbation to a temp directory (path is printed at the end).
   - You can adjust the number of cells per perturbation in the script for resource constraints.

2. **Merge the `.h5ad` files:**
   ```bash
   python models/merge_h5ad_files.py --indir <temp_dir> --outfile submission.h5ad
   ```
   - `<temp_dir>` is the directory printed by the submission script.
   - This merges all files in batches (default 100 at a time) to avoid OOM errors.

3. **Prepare the final .vcc file:**
   ```bash
   cell-eval prep -i submission.h5ad --genes models/highly_variable_genes.csv
   ```
   - This checks format and produces the `.vcc` file for submission.

### Requirements Enforced
- All 18,080 genes, correct order, float32, log1p/integer counts.
- `target_gene` column in `.obs`.
- Controls (`non-targeting`) included.
- ≤100,000 total cells.

### Troubleshooting & Lessons Learned
- **If you see 'Killed' or OOM errors:** Lower the number of cells per perturbation, or merge in smaller batches.
- **Why not use AnnData backed mode?** It is less flexible for row-wise appends and more error-prone for this use case.
- **This approach is robust:** You can inspect, retry, or parallelize per-perturbation files, and merging is memory-safe.

### Where We Got Stuck
- In-memory merging failed even for tiny test cases due to AnnData's memory usage.
- The batch merge approach is the only reliable way to generate a valid submission on resource-constrained systems.

--- 