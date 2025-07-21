#!/usr/bin/env python3
"""
Biological Pathway Submission Script
===================================

- Loads the trained biological pathway model
- Predicts for 1,000 pathway genes
- Fills remaining genes with mean expression from training data
- Outputs a full submission .h5ad file with all 18,080 genes
- Filename includes date and time for traceability
- Memory-efficient: processes in batches, uses backed mode where possible
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import tempfile
import gc
from datetime import datetime
import os
from models.biological_pathway_approach import BiologicalPathwayModel
from models.state_model import VirtualCellSTATE

# Helper to get all gene names in correct order
def get_all_gene_names():
    gene_names_path = Path('data/gene_names.csv')
    if gene_names_path.exists():
        return pd.read_csv(gene_names_path, header=None)[0].tolist()
    adata = sc.read_h5ad('data/adata_Training.h5ad', backed='r')
    return list(adata.var.index)

# Helper to get gene indices for mapping
def get_gene_indices(gene_list, all_gene_names):
    return [all_gene_names.index(g) for g in gene_list if g in all_gene_names]

def main():
    print("🚀 Hybrid Biological Pathway + STATE Submission Script (Memory-Efficient)")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir = tempfile.mkdtemp(prefix=f"hybrid_preds_{timestamp}_")
    print(f"   Writing per-perturbation .h5ad files to: {temp_dir}")

    pert_counts_path = Path("data/pert_counts_Validation.csv")
    if not pert_counts_path.exists():
        print(f"❌ {pert_counts_path} not found!")
        return
    pert_counts_df = pd.read_csv(pert_counts_path)

    all_gene_names = get_all_gene_names()
    print(f"   Total genes in submission: {len(all_gene_names)}")

    # Load biological pathway model
    bio_ckpt = torch.load("training_runs/biological_pathway_20250719_214812/biological_pathway_model.pt", map_location='cpu')
    bio_gene_list = bio_ckpt['gene_list']
    bio_pathway_info = bio_ckpt['pathway_info']
    bio_config = bio_ckpt['model_config']
    bio_model = BiologicalPathwayModel(
        n_genes=bio_config['n_genes'],
        embed_dim=bio_config['embed_dim'],
        n_layers=bio_config['n_layers']
    )
    bio_model.load_state_dict(bio_ckpt['model_state_dict'])
    bio_model.eval()
    print(f"   Loaded biological model for {len(bio_gene_list)} genes.")

    # Load STATE model
    state_ckpt = torch.load("models/state_best_model.pt", map_location='cpu')
    state_n_genes = state_ckpt['model_state']['gene_embedding.gene_embeddings.weight'].shape[0]
    from models.real_data_training import TrainingConfig
    config = TrainingConfig()
    state_model = VirtualCellSTATE(
        n_genes=state_n_genes,
        embedding_dim=config.embedding_dim,
        n_heads=config.n_heads,
        n_layers=config.n_layers
    )
    state_model.load_state_dict(state_ckpt['model_state'])
    state_model.eval()
    print(f"   Loaded STATE model for {state_n_genes} genes.")

    # Indices for mapping
    bio_gene_indices = get_gene_indices(bio_gene_list, all_gene_names)
    state_gene_indices = list(range(len(all_gene_names)))

    batch_size = 32
    for idx, row in pert_counts_df.iterrows():
        pert_name = row['target_gene']
        n_cells = int(row['n_cells'])
        print(f"[{idx+1}/{len(pert_counts_df)}] Perturbation: {pert_name}, Cells: {n_cells}")
        cell_exprs = []
        obs = []
        for start in range(0, n_cells, batch_size):
            end = min(start + batch_size, n_cells)
            batch_n = end - start
            # Model input: random baseline for pathway genes
            baseline_bio = torch.randn(batch_n, len(bio_gene_list))
            baseline_state = torch.randn(batch_n, state_n_genes)
            # Biological model prediction
            with torch.no_grad():
                bio_preds = bio_model(baseline_bio, bio_gene_list, bio_pathway_info)
            bio_preds = bio_preds.numpy()
            # STATE model prediction
            pert_gene_id = all_gene_names.index(pert_name) if pert_name in all_gene_names else 0
            pert_ids = torch.LongTensor([pert_gene_id]*batch_n)
            with torch.no_grad():
                state_out = state_model(baseline_state, pert_ids)
            state_preds = state_out['expression'].numpy() if isinstance(state_out, dict) and 'expression' in state_out else state_out.numpy()
            # Assemble hybrid predictions
            for i in range(batch_n):
                cell_expr = np.array([0.0]*len(all_gene_names), dtype=np.float32)
                # Fill pathway genes from biological model
                for j, idx_gene in enumerate(bio_gene_indices):
                    cell_expr[idx_gene] = bio_preds[i, j]
                # Fill rest from STATE model
                for j in range(len(all_gene_names)):
                    if j not in bio_gene_indices:
                        cell_expr[j] = state_preds[i, j] if j < state_preds.shape[1] else 0.0
                cell_exprs.append(cell_expr)
                obs.append({"target_gene": pert_name})
            del bio_preds, baseline_bio, state_preds, baseline_state, state_out
            gc.collect()
        X = np.stack(cell_exprs)
        obs_df = pd.DataFrame(obs, index=[f"cell_{pert_name}_{i}" for i in range(n_cells)])
        var_df = pd.DataFrame(index=all_gene_names)
        adata = ad.AnnData(X=X, obs=obs_df, var=var_df)
        pert_file = os.path.join(temp_dir, f"pert_{idx}_{pert_name}.h5ad")
        adata.write_h5ad(pert_file)
        print(f"   Wrote {pert_file}")
        del adata, X, obs_df, var_df, cell_exprs, obs
        gc.collect()
    print(f"\nAll per-perturbation files written to: {temp_dir}")
    print(f"To merge into a single submission file, run:")
    print(f"python models/merge_h5ad_files.py --indir {temp_dir} --outfile submissions/hybrid_submission_{timestamp}.h5ad")
    print(f"After merging, run cell-eval prep as usual.")

if __name__ == "__main__":
    main() 