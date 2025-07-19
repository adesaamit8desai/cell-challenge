"""
Scaled Model Submission Generator
================================

This script uses the latest trained scaled model to generate predictions for the Virtual Cell Challenge.
It handles the mapping between the 500 genes the model was trained on and the full 18,080 gene set.
"""

import torch
import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc
from pathlib import Path
import subprocess
import tempfile
import shutil
import sys
import argparse
import os

# Fix import paths
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import our model and config
from models.memory_efficient_training import MemoryEfficientStateModel
from models.real_data_training import TrainingConfig

def load_trained_model(model_path, selected_genes_path):
    """
    Load the trained scaled model and selected genes
    """
    print(f"📂 Loading trained model from {model_path}...")
    
    # Load selected genes
    selected_genes_df = pd.read_csv(selected_genes_path)
    selected_genes = selected_genes_df['gene_name'].tolist()
    print(f"   Model was trained on {len(selected_genes)} selected genes")
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Check if checkpoint is wrapped in a dictionary or direct state dict
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model_state = checkpoint['model_state']
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
    else:
        # Direct state dict
        model_state = checkpoint
    
    # Extract number of genes from gene embeddings
    if 'gene_embeddings.weight' in model_state:
        n_genes_model = model_state['gene_embeddings.weight'].shape[0]
    else:
        # Fallback: use length of selected genes
        n_genes_model = len(selected_genes)
    
    # Create model with same architecture
    model = MemoryEfficientStateModel(
        n_genes=n_genes_model,
        embed_dim=512,  # From training report
        n_heads=8,
        n_layers=6,
        attention_type='sparse',
        use_checkpointing=True,
        mixed_precision=True
    )
    
    # Load trained weights
    model.load_state_dict(model_state)
    model.eval()
    
    print(f"   ✅ Model loaded successfully")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, selected_genes

def generate_predictions_for_perturbation(
    model,
    device,
    pert_name,
    n_cells,
    selected_genes,
    full_gene_list,
    max_cells=None,
    gene_mean_dict=None
):
    """
    Generate predictions for a single perturbation
    """
    if max_cells is not None:
        n_cells = min(n_cells, max_cells)
    
    print(f"   Processing {pert_name} with {n_cells} cells...")
    
    # Create baseline expression for selected genes only
    baseline_expression = torch.randn(n_cells, len(selected_genes)).to(device)
    
    # Check if target gene is in selected genes
    if pert_name in selected_genes:
        print(f"     ✅ Target gene {pert_name} found in selected genes")
    else:
        print(f"     ⚠️ Target gene {pert_name} not in selected genes, using baseline")
    
    # Run model inference (MemoryEfficientStateModel only takes x)
    with torch.no_grad():
        model_predictions = model(baseline_expression)
    
    # Convert to numpy
    predictions = model_predictions.cpu().numpy()
    
    # Create full gene predictions (18,080 genes)
    full_predictions = np.zeros((n_cells, len(full_gene_list)))
    
    # Map predictions from selected genes to full gene list
    for i, gene in enumerate(selected_genes):
        if gene in full_gene_list:
            full_idx = full_gene_list.index(gene)
            full_predictions[:, full_idx] = predictions[:, i]
    
    # For genes not in selected set, use NTC mean as baseline
    for i, gene in enumerate(full_gene_list):
        if gene not in selected_genes:
            if gene_mean_dict is not None and gene in gene_mean_dict:
                full_predictions[:, i] = gene_mean_dict[gene]
            else:
                full_predictions[:, i] = 0.0  # fallback
    
    return full_predictions

def concatenate_anndata_files(file_list, output_file):
    """Concatenate a list of AnnData .h5ad files along obs axis and write to output_file."""
    adatas = [ad.read_h5ad(f) for f in file_list]
    merged = ad.concat(adatas, axis=0, merge='same')
    merged.write_h5ad(output_file)
    for a in adatas:
        del a
    del merged

def create_scaled_submission(
    model_path,
    selected_genes_path,
    pert_counts_path,
    training_adata_path,
    output_h5ad_path="submission_scaled.h5ad",
    max_cells=None,
    batch_size=100
):
    """
    Create submission using the trained scaled model, processing each perturbation in batches.
    """
    print("🚀 Creating submission with scaled model...")
    
    # Load model and selected genes
    model, selected_genes = load_trained_model(model_path, selected_genes_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Load perturbation counts
    pert_counts_df = pd.read_csv(pert_counts_path)
    print(f"   Processing {len(pert_counts_df)} perturbations...")
    
    # Load full gene list
    gene_names_path = Path("data/gene_names.csv")
    if gene_names_path.exists():
        full_gene_list = pd.read_csv(gene_names_path, header=None).iloc[:, 0].tolist()
    else:
        # Fallback: use all genes from training data
        training_adata = sc.read_h5ad(training_adata_path, backed='r')
        full_gene_list = training_adata.var.index.tolist()
    print(f"   Full gene list: {len(full_gene_list)} genes")

    # --- Robust NTC mean calculation (as before) ---
    print("   Computing NTC mean for all genes as baseline...")
    gene_mean_dict = None
    try:
        training_adata = sc.read_h5ad(training_adata_path, backed='r')
        ntc_mask = training_adata.obs['target_gene'] == 'non-targeting'
        n_ntc = int(ntc_mask.sum())
        print(f"   Found {n_ntc} NTC cells in training data.")
        if n_ntc == 0:
            raise ValueError("No non-targeting controls found!")
        ntc_indices = np.where(ntc_mask)[0]
        # For memory, sample up to 50 NTCs if too many
        if len(ntc_indices) > 50:
            np.random.seed(42)
            ntc_indices = np.random.choice(ntc_indices, 50, replace=False)
        print(f"   Using {len(ntc_indices)} NTC cells for mean calculation.")
        # Read only the sampled NTCs into memory
        ntc_adata = training_adata[ntc_indices, :].to_memory()
        # Convert to dense if needed
        X = ntc_adata.X
        if not isinstance(X, np.ndarray):
            X = X.toarray()
        gene_means = np.asarray(X.mean(axis=0)).ravel()
        gene_mean_dict = {gene: gene_means[i] for i, gene in enumerate(ntc_adata.var.index)}
        del ntc_adata
        print("   ✅ NTC mean calculation successful.")
    except Exception as e:
        print(f"   ⚠️ NTC mean calculation failed: {e}")
        print("   Falling back to global mean across all cells...")
        try:
            training_adata = sc.read_h5ad(training_adata_path, backed='r')
            # For memory, sample up to 50 random cells
            n_cells = training_adata.n_obs
            if n_cells > 50:
                np.random.seed(42)
                cell_indices = np.random.choice(n_cells, 50, replace=False)
                sampled_adata = training_adata[cell_indices, :].to_memory()
            else:
                sampled_adata = training_adata[:,:].to_memory()
            X = sampled_adata.X
            if not isinstance(X, np.ndarray):
                X = X.toarray()
            gene_means = np.asarray(X.mean(axis=0)).ravel()
            gene_mean_dict = {gene: gene_means[i] for i, gene in enumerate(sampled_adata.var.index)}
            del sampled_adata
            print("   ✅ Global mean calculation successful.")
        except Exception as e2:
            print(f"   ❌ Global mean calculation failed: {e2}")
            print("   Using zeros as baseline for all unmodeled genes.")
            gene_mean_dict = {gene: 0.0 for gene in full_gene_list}
    # --- END NEW ---

    # Create temporary directory for per-perturbation files
    temp_dir = tempfile.mkdtemp(prefix="scaled_submission_")
    temp_files = []
    
    # Add NTC cells as a separate perturbation file (CRITICAL for cell-eval prep)
    print("   Adding non-targeting control cells...")
    try:
        training_adata = sc.read_h5ad(training_adata_path, backed='r')
        ntc_mask = training_adata.obs['target_gene'] == 'non-targeting'
        ntc_indices = np.where(ntc_mask)[0]
        
        if len(ntc_indices) > 0:
            # Sample up to 100 NTC cells to avoid memory issues
            if len(ntc_indices) > 100:
                np.random.seed(42)
                ntc_indices = np.random.choice(ntc_indices, 100, replace=False)
            
            # Load NTC cells
            ntc_adata = training_adata[ntc_indices, :].to_memory()
            
            # Ensure NTC cells have the same gene order as full gene list
            if len(ntc_adata.var) != len(full_gene_list):
                # Reorder genes to match full gene list
                ntc_adata = ntc_adata[:, full_gene_list].copy()
            
            # Update obs to match expected format
            ntc_adata.obs['target_gene'] = 'non-targeting'
            ntc_adata.obs['n_cells'] = ntc_adata.n_obs
            
            # Write NTC cells to temp file
            ntc_file = f"{temp_dir}/pert_ntc_non-targeting.h5ad"
            ntc_adata.write_h5ad(ntc_file)
            temp_files.append(ntc_file)
            print(f"   ✅ Added {ntc_adata.n_obs} NTC cells")
            del ntc_adata
        else:
            print("   ⚠️ No NTC cells found in training data")
    except Exception as e:
        print(f"   ⚠️ Failed to add NTC cells: {e}")
    
    # Process each perturbation
    for idx, row in pert_counts_df.iterrows():
        pert_name = row['target_gene']
        n_cells = row['n_cells']
        if max_cells is not None:
            n_cells = min(n_cells, max_cells)
        print(f"   [{idx+1}/{len(pert_counts_df)}] {pert_name} ({n_cells} cells)")
        
        # Batch processing
        batch_file_list = []
        n_batches = (n_cells + batch_size - 1) // batch_size
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, n_cells)
            batch_n = end - start
            print(f"     Batch {batch_idx+1}/{n_batches} ({batch_n} cells)...")
            # Generate predictions for this batch
            predictions = generate_predictions_for_perturbation(
                model, device, pert_name, batch_n, selected_genes, full_gene_list, None,
                gene_mean_dict=gene_mean_dict
            )
            # Create AnnData for this batch
            obs_df = pd.DataFrame({
                "target_gene": [pert_name] * batch_n,
                "n_cells": [batch_n] * batch_n
            }, index=[f"cell_{pert_name}_{start+i}" for i in range(batch_n)])
            var_df = pd.DataFrame(index=full_gene_list)
            batch_adata = ad.AnnData(
                X=predictions.astype(np.float32),
                obs=obs_df,
                var=var_df
            )
            batch_file = f"{temp_dir}/pert_{idx}_{pert_name}_batch{batch_idx}.h5ad"
            batch_adata.write_h5ad(batch_file)
            batch_file_list.append(batch_file)
            del batch_adata, predictions
        # Concatenate all batch files for this perturbation
        pert_file = f"{temp_dir}/pert_{idx}_{pert_name}.h5ad"
        concatenate_anndata_files(batch_file_list, pert_file)
        temp_files.append(pert_file)
        # Clean up batch files
        for f in batch_file_list:
            try:
                os.remove(f)
            except Exception:
                pass
        print(f"   ✅ Finished {pert_name} ({n_cells} cells, {n_batches} batches)")
    
    print(f"\n✅ Generated {len(temp_files)} perturbation files")
    print(f"📁 Temporary files: {temp_dir}")
    
    # Merge files using existing merge script
    print("\n🔄 Merging files into final submission...")
    try:
        subprocess.run([
            "python", "models/merge_h5ad_files.py",
            "--indir", temp_dir,
            "--outfile", output_h5ad_path
        ], check=True)
        print(f"✅ Submission saved to: {output_h5ad_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error merging files: {e}")
        return None
    
    # Clean up temporary files
    shutil.rmtree(temp_dir)
    
    return output_h5ad_path

def main():
    parser = argparse.ArgumentParser(description="Create submission using trained scaled model")
    parser.add_argument('--model-dir', type=str, 
                       default='training_runs/scaled_standard_20250717_173713',
                       help='Directory containing trained model')
    parser.add_argument('--output', type=str, default='submission_scaled.h5ad',
                       help='Output h5ad file path')
    parser.add_argument('--max-cells', type=int, default=None,
                       help='Maximum cells per perturbation (for testing)')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for per-perturbation processing')
    parser.add_argument('--prep', action='store_true',
                       help='Run cell-eval prep after creating submission')
    
    args = parser.parse_args()
    
    # Set up paths
    model_path = Path(args.model_dir) / "scaled_model_final.pt"
    selected_genes_path = Path(args.model_dir) / "selected_genes.csv"
    pert_counts_path = Path("data/pert_counts_Validation.csv")
    training_adata_path = Path("data/adata_Training.h5ad")
    
    # Check files exist
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return
    
    if not selected_genes_path.exists():
        print(f"❌ Selected genes file not found: {selected_genes_path}")
        return
    
    if not pert_counts_path.exists():
        print(f"❌ Perturbation counts file not found: {pert_counts_path}")
        return
    
    # Create submission
    output_path = create_scaled_submission(
        model_path=str(model_path),
        selected_genes_path=str(selected_genes_path),
        pert_counts_path=str(pert_counts_path),
        training_adata_path=str(training_adata_path),
        output_h5ad_path=args.output,
        max_cells=args.max_cells,
        batch_size=args.batch_size
    )
    
    if output_path and args.prep:
        print("\n🔧 Running cell-eval prep...")
        try:
            subprocess.run([
                "cell-eval", "prep",
                "-i", output_path,
                "--genes", str(selected_genes_path)
            ], check=True)
            print("✅ cell-eval prep completed successfully!")
        except FileNotFoundError:
            print("❌ Error: 'cell-eval' command not found")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running cell-eval prep: {e}")

if __name__ == "__main__":
    main() 