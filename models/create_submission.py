"""
Virtual Cell Challenge Submission Script
========================================

This script implements a memory-efficient, competition-compliant submission flow:
- For each perturbation, predictions are written as a separate .h5ad file in a temp directory.
- After all are written, use a separate script or notebook to merge these files into a single .h5ad file for submission.
- This approach avoids all in-memory concatenation, which caused OOM errors even for small test cases.
- See README.md for a full discussion of the memory issues encountered and the rationale for this approach.

Steps:
1. Run this script to generate per-perturbation .h5ad files in a temp directory.
2. Use the provided merge script or notebook to combine them into a single .h5ad file.
3. Run cell-eval prep on the merged file to generate your .vcc submission.

Requirements enforced:
- Each .h5ad file contains predictions for all 18,080 genes (or a subset for testing), float32, log1p/integer.
- .obs includes 'target_gene'.
- Controls ('non-targeting') must be included in the merged file.
- Total cells <= 100,000.
- Gene order matches gene_names.csv.

See README.md for more details.
"""
import torch
import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc
from pathlib import Path
import subprocess # For running cell-eval prep
import tempfile
import shutil
import sys
import argparse

# Fix import paths
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import our model and config
from models.state_model import VirtualCellSTATE
from models.real_data_training import TrainingConfig, VirtualCellDataset # Re-use VirtualCellDataset for validation data loading

def generate_model_predictions_anndata(
    model,
    device,
    pert_counts_df,
    gene_names_list,
    training_adata_path, # For NTCs and potentially baseline expression
    max_cells=None
):
    # This function will simulate the prediction process for the validation set
    # It needs to create an AnnData object with predictions
    # For each perturbation in pert_counts_df:
    #   - Get target gene
    #   - Get number of cells (n_cells)
    #   - Create dummy/baseline expression for n_cells
    #   - Get target gene ID
    #   - Run model inference
    #   - Store predictions
    # Concatenate all predictions into a single AnnData

    # Placeholder for actual implementation
    # ... (detailed logic to follow) ...

    # For now, let's create a dummy AnnData as per the tutorial's random_predictor
    # This will be replaced by actual model inference
    
    # Load training adata to get gene order and NTCs (in backed mode for memory efficiency)
    try:
        print("Loading training data in backed mode for memory efficiency...")
        full_training_adata = sc.read_h5ad(training_adata_path, backed='r')
        
        # Extract non-targeting controls (NTCs) - limit to small sample for memory
        ntc_mask = full_training_adata.obs['target_gene'] == 'non-targeting'
        if ntc_mask.sum() > 0:
            # Get a small sample of NTCs to avoid memory issues
            ntc_indices = np.where(ntc_mask)[0]
            if len(ntc_indices) > 100:  # Limit to 100 NTCs
                np.random.seed(42)
                ntc_indices = np.random.choice(ntc_indices, 100, replace=False)
            
            # Load only the NTC subset
            ntc_adata = full_training_adata[ntc_indices, :].to_memory()
            
            # Ensure NTCs have the same gene order as the model expects
            ntc_adata = ntc_adata[:, gene_names_list].copy()
            
            print(f"✅ Extracted {ntc_adata.n_obs} non-targeting control cells.")
        else:
            print("No non-targeting controls found in training data.")
            ntc_adata = None

    except Exception as e:
        print(f"Could not load training adata for NTCs: {e}. Proceeding without NTCs from training data.")
        full_training_adata = None
        ntc_adata = None

    temp_dir = tempfile.mkdtemp(prefix="perturbation_preds_")
    temp_files = []

    # Add NTCs to the list of files to be merged
    if ntc_adata is not None:
        ntc_file = f"{temp_dir}/pert_non-targeting.h5ad"
        ntc_adata.write_h5ad(ntc_file)
        temp_files.append(ntc_file)
        print(f"  Writing NTC AnnData to {ntc_file} ...")
        print("  File written.")

    # Remove the test limits for full run
    # pert_counts_df = pert_counts_df.head(5)  # (REMOVE THIS LINE)
    # n_cells = min(n_cells, 100)  # (REMOVE THIS LINE)
    for idx, row in pert_counts_df.iterrows():
        pert_name = row['target_gene']
        n_cells = row['n_cells']
        if max_cells is not None:
            n_cells = min(n_cells, max_cells)
        print(f"Processing perturbation {pert_name} (idx {idx}) with {n_cells} cell(s) and {len(gene_names_list)} genes...")
        print("  Generating baseline expression...")
        baseline_expression_for_pert = torch.randn(n_cells, len(gene_names_list)).to(device)
        target_gene_id = torch.LongTensor([gene_names_list.index(pert_name) if pert_name in gene_names_list else 0]).to(device)
        target_gene_ids_for_pert = target_gene_id.repeat(n_cells)
        print("  Running model inference...")
        with torch.no_grad():
            model_predictions = model(baseline_expression_for_pert, target_gene_ids_for_pert)['expression']
        model_predictions = model_predictions.to(torch.float32)
        print("  Model inference complete.")
        print("  Creating AnnData object...")
        obs_df = pd.DataFrame({
            "target_gene": [pert_name] * n_cells,
            "n_cells": [n_cells] * n_cells
        }, index=[f"cell_{pert_name}_{i}" for i in range(n_cells)])
        var_df = pd.DataFrame(index=gene_names_list)
        predicted_adata = ad.AnnData(
            X=model_predictions.cpu().numpy(),
            obs=obs_df,
            var=var_df
        )
        print("  AnnData object created.")
        temp_file = f"{temp_dir}/pert_{idx}_{pert_name}.h5ad"
        print(f"  Writing AnnData to {temp_file} ...")
        predicted_adata.write_h5ad(temp_file)
        print("  File written.")
        temp_files.append(temp_file)
        del predicted_adata, model_predictions, baseline_expression_for_pert
    
    print(f"\nAll perturbation .h5ad files written to: {temp_dir}")
    print(f"Number of files: {len(temp_files)}")
    print("\nTo merge these into a single submission file, use the provided merge script:")
    print(f"python models/merge_h5ad_files.py --indir {temp_dir} --outfile submission.h5ad")
    print("\nAfter merging, run cell-eval prep as usual.")
    return None


def create_submission_file(start=0, end=None, max_cells=None):
    print("🚀 Preparing .h5ad submission file...")
    config = TrainingConfig()

    # Load necessary data
    pert_counts_path = Path("data/pert_counts_Validation.csv")
    if not pert_counts_path.exists():
        print(f"❌  {pert_counts_path} not found!")
        return
    pert_counts_df = pd.read_csv(pert_counts_path)
    if end is not None:
        pert_counts_df = pert_counts_df.iloc[start:end]
    else:
        pert_counts_df = pert_counts_df.iloc[start:]

    training_adata_path = Path("data/adata_Training.h5ad")
    if not training_adata_path.exists():
        print(f"❌  {training_adata_path} not found!")
        return

    # Load the list of highly variable genes used during training
    hvg_path = Path("models/highly_variable_genes.csv")
    if not hvg_path.exists():
        print(f"❌  {hvg_path} not found! Please run real_data_training.py first.")
        return
    gene_names_list = pd.read_csv(hvg_path, header=None).iloc[:, 0].tolist()

    # Load trained model
    model_path = Path("models/state_best_model.pt")
    if not model_path.exists():
        print("❌ Trained model not found! Please train the model first (run real_data_training.py).")
        return

    checkpoint = torch.load(model_path, map_location=config.device)
    n_genes_model = checkpoint['model_state']['gene_embedding.gene_embeddings.weight'].shape[0]

    model = VirtualCellSTATE(
        n_genes=n_genes_model, # Use n_genes from trained model
        embedding_dim=config.embedding_dim,
        n_heads=config.n_heads,
        n_layers=config.n_layers
    ).to(config.device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval() # Set model to evaluation mode

    # Generate predictions and create AnnData
    predicted_anndata = generate_model_predictions_anndata(
        model,
        config.device,
        pert_counts_df,
        gene_names_list,
        training_adata_path,
        max_cells=max_cells
    )

    if predicted_anndata is None:
        print("Failed to generate predicted AnnData.")
        return

    # Save the AnnData object
    output_h5ad_path = "submission.h5ad"
    predicted_anndata.write_h5ad(output_h5ad_path)
    print(f"✅ Predicted AnnData saved to {output_h5ad_path}")

    # Run cell-eval prep
    print("\nRunning cell-eval prep tool...")
    try:
        # Ensure cell-eval is in PATH or provide full path
        # Assuming cell-eval is installed and accessible
        subprocess.run([
            "cell-eval", "prep",
            "-i", output_h5ad_path,
            "--genes", str(hvg_path)
        ], check=True)
        print("✅ cell-eval prep completed successfully!")
        print(f"Your submission is ready at {output_h5ad_path.replace('.h5ad', '.prep.vcc')}")
    except FileNotFoundError:
        print("❌ Error: 'cell-eval' command not found. Please ensure cell-eval is installed and in your system's PATH.")
        print("   See: https://github.com/ArcInstitute/cell-eval?tab=readme-ov-file#installation")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running cell-eval prep: {e}")
        print(f"   Stdout: {e.stdout.decode()}")
        print(f"   Stderr: {e.stderr.decode()}")
    except Exception as e:
        print(f"❌ An unexpected error occurred during cell-eval prep: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch submission generator for Virtual Cell Challenge.")
    parser.add_argument('--start', type=int, default=0, help='Start index of perturbations to process (inclusive)')
    parser.add_argument('--end', type=int, default=None, help='End index of perturbations to process (exclusive)')
    parser.add_argument('--max_cells', type=int, default=None, help='Maximum number of cells per perturbation')
    args = parser.parse_args()
    create_submission_file(start=args.start, end=args.end, max_cells=args.max_cells)
