# Training Workflow

## Quick Training & Results Update

### 1. Run Training
```bash
# Small test
python models/scaled_training.py --strategy standard --genes 200 --cells 50 --epochs 2 --batch-size 8

# Medium test  
python models/scaled_training.py --strategy standard --genes 500 --cells 100 --epochs 3 --batch-size 16

# Production scale
python models/scaled_training.py --strategy standard --genes 3000 --cells 10000 --epochs 20 --batch-size 32
```

### 2. Update README with Results
```bash
# Automatically finds and updates with the most recent training run
python update_training_results.py
```

### 3. Check Results
- **Training Report**: `training_runs/[run_name]/training_report.md`
- **Training History**: `training_runs/[run_name]/training_history.json`
- **Selected Genes**: `training_runs/[run_name]/selected_genes.csv`
- **Trained Model**: `training_runs/[run_name]/scaled_model_final.pt`

## What Gets Updated

The `update_training_results.py` script automatically extracts and updates:

- **Configuration**: Strategy, genes, epochs
- **Performance**: Initial loss, final loss, improvement
- **Hardware**: Device used (CPU/GPU)
- **Gene Selection**: Breakdown by category (TF, metabolic, cell cycle, etc.)
- **Results Location**: Directory path for easy access

## Example Output

After running the update script, your README will show:

```
### Latest Training Run (2025-07-17)
**Configuration**: Standard strategy, 500 genes, 3 epochs
- **Initial Loss**: 354.51
- **Final Loss**: 294.76
- **Loss Improvement**: 59.74
- **Device**: cpu
- **Gene Categories**: 360 highly_variable, 46 transcription_factor, 40 cell_cycle, 32 metabolic, 22 pluripotency
- **Results Directory**: `training_runs/scaled_standard_20250717_173713/`
```

## Tips

1. **Always update after training**: Run `python update_training_results.py` after each training run
2. **Check the reports**: Look at `training_report.md` for detailed analysis
3. **Track progress**: The README now maintains a history of your training runs
4. **Scale gradually**: Start with small tests, then scale up to production runs 