# Real Data Training for STATE Model - FIXED VERSION
# Phase 3: Train the proven architecture on actual challenge data

import torch
import torch.nn as nn
import torch.nn.functional as F # Added for F.l1_loss, F.binary_cross_entropy, F.cosine_similarity
import numpy as np
import pandas as pd
from pathlib import Path
import scanpy as sc
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# Import our proven STATE model
import sys
sys.path.append('.')
from models.state_model import VirtualCellSTATE, StateTrainer

# =============================================================================
# STEP 1: REAL DATA LOADER (FIXED)
# =============================================================================

class VirtualCellDataset(Dataset):
    """Dataset for loading real challenge data efficiently using backed mode"""
    
    def __init__(self, data_path, max_cells=None, max_genes=None):
        """
        Load real data in backed mode for memory efficiency.
        
        Args:
            data_path: Path to adata_Training.h5ad
            max_cells: Optional: Limit total cells for training (for debugging/faster runs)
            max_genes: Optional: Limit total genes for training (for debugging/faster runs)
        """
        print(f"📂 Loading real data in backed mode from: {data_path}")
        
        # Load the data in backed mode
        self.adata = sc.read_h5ad(data_path, backed='r')
        print(f"   Full dataset shape (backed): {self.adata.shape}")
        
        # Apply max_cells limit if specified and load to memory
        if max_cells is not None and max_cells < self.adata.n_obs:
            self.adata = self.adata[:max_cells, :].to_memory()
            print(f"   Subsetted to {self.adata.n_obs} cells (loaded to memory).")
        else:
            self.adata = self.adata.to_memory() # Load full data to memory if no subsetting
            print(f"   Loaded full dataset to memory: {self.adata.shape}")

        # Ensure data is dense and clean before further processing
        self.adata.X = self.adata.X.toarray() if hasattr(self.adata.X, 'toarray') else self.adata.X
        self.adata.X = np.nan_to_num(self.adata.X, nan=0.0, posinf=1e6, neginf=0.0)

        # Find most variable genes if max_genes is specified and less than total genes
        if max_genes is not None and max_genes < self.adata.n_vars:
            print(f"   Finding top {max_genes} highly variable genes...")
            try:
                sc.pp.highly_variable_genes(self.adata, n_top_genes=max_genes)
                # Filter genes and ensure it's not empty
                if np.any(self.adata.var.highly_variable):
                    self.adata = self.adata[:, self.adata.var.highly_variable].copy()
                    print(f"   Subsetted to {self.adata.n_vars} genes (highly variable).")
                else:
                    raise ValueError("No highly variable genes found.")
            except Exception as e:
                print(f"   Warning: Could not find highly variable genes: {e}")
                print("   Using top genes by variance instead...")
                # Fallback: use top genes by variance
                gene_vars = np.var(self.adata.X, axis=0) 
                top_gene_indices = np.argsort(gene_vars)[-max_genes:]
                
                # Ensure top_gene_indices is not empty
                if len(top_gene_indices) > 0:
                    self.adata = self.adata[:, top_gene_indices].copy()
                    print(f"   Subsetted to {self.adata.n_vars} genes (by variance).")
                else:
                    raise ValueError("No genes found after variance-based selection.")
        
        # Final check for empty AnnData after subsetting
        if self.adata.n_obs == 0 or self.adata.n_vars == 0:
            raise ValueError("Resulting AnnData object is empty after subsetting.")

        self.gene_names = self.adata.var_names.tolist()
        self.gene_to_idx = {gene: i for i, gene in enumerate(self.gene_names)}
        
        self.perturbation_info = self._extract_perturbation_info()
        print(f"   Perturbations found: {len(set(self.perturbation_info))}")
        
    def _extract_perturbation_info(self):
        """Extract perturbation targets from metadata"""
        possible_columns = ['target_gene', 'perturbation', 'guide_target', 'gene_target']
        
        perturbation_column = None
        for col in possible_columns:
            if col in self.adata.obs.columns:
                perturbation_column = col
                break
        
        if perturbation_column is None:
            print("⚠️ No perturbation column found, using dummy targets")
            return ['DUMMY_TARGET'] * self.adata.n_obs
        
        return self.adata.obs[perturbation_column].tolist()
    
    def __len__(self):
        return self.adata.n_obs
    
    def __getitem__(self, idx):
        """Get one training sample from backed data"""
        
        # Current expression (loads only one row from disk)
        expression = torch.FloatTensor(self.adata.X[idx, :])
        
        # Target gene ID
        target_gene = self.perturbation_info[idx]
        target_gene_id = self.gene_to_idx.get(target_gene, 0)  # Default to 0 if not found
        
        # For training targets, we'll use the actual expression as ground truth
        # In real usage, this would be the perturbed expression
        targets = {
            'expression': expression,  # Ground truth expression
            'differential_expression': (expression > expression.mean()).float(),  # Simple DE definition
            'perturbation_ids': torch.LongTensor([target_gene_id])
        }
        
        return {
            'expression': expression,
            'target_gene_ids': torch.LongTensor([target_gene_id]),
            'targets': targets
        }

# =============================================================================
# STEP 2: ADVANCED STATE TRAINER WITH FULL EVALUATION
# =============================================================================

class AdvancedStateTrainer:
    """Advanced trainer with comprehensive evaluation metrics"""
    
    def __init__(self, model, device='cpu', learning_rate=1e-4):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
    def _contrastive_loss(self, signatures, pert_ids):
        """Encourage different perturbations to have different signatures (dummy loss for now)"""
        return torch.tensor(0.0, device=self.device)

    def compute_metrics(self, predictions, targets):
        """Calculate all three competition metrics"""
        
        pred_expr = predictions['expression']
        true_expr = targets['expression']
        
        # 1. Mean Absolute Error (MAE)
        mae = F.l1_loss(pred_expr, true_expr).item()
        
        # 2. Differential Expression Score (DES)
        # We need to calculate true DE, not the simple version
        true_de = (true_expr - true_expr.mean(dim=1, keepdim=True)).abs() > 0.2 # Simplified DE
        pred_de_probs = predictions['differential_expression']
        des = F.binary_cross_entropy(pred_de_probs, true_de.float()).item()

        # 3. Perturbation Discrimination Score (PDS)
        # This requires comparing signatures across different perturbations
        signatures = predictions['signature']
        pert_ids = targets['perturbation_ids'].squeeze()
        
        # Group signatures by perturbation ID
        unique_pids, inverse_indices = torch.unique(pert_ids, return_inverse=True)
        
        if len(unique_pids) > 1:
            # Calculate mean signature for each perturbation
            mean_signatures = torch.zeros(len(unique_pids), signatures.size(1), device=self.device)
            mean_signatures.index_add_(0, inverse_indices, signatures)
            counts = torch.bincount(inverse_indices, minlength=len(unique_pids)).unsqueeze(1) # Ensure minlength
            mean_signatures /= (counts + 1e-6) # Add epsilon to prevent division by zero
            
            # Cosine similarity between mean signatures
            cosine_sim = F.cosine_similarity(mean_signatures.unsqueeze(1), mean_signatures.unsqueeze(0), dim=-1)
            
            # PDS is based on how well we can distinguish perturbations
            # We want low similarity for different perturbations (off-diagonal)
            # and high for same (diagonal, which is 1)
            pds_loss = cosine_sim[~torch.eye(len(unique_pids), dtype=bool)].mean()
        else:
            pds_loss = torch.tensor(0.0) # Cannot compute with one perturbation type

        return {
            'mae': mae,
            'des': des,
            'pds': pds_loss.item()
        }

    def compute_loss(self, predictions, targets):
        """Multi-task loss combining all metrics"""
        
        # Individual losses (kept as tensors for backprop)
        expr_loss = F.l1_loss(predictions['expression'], targets['expression'])
        de_loss = F.binary_cross_entropy(
            predictions['differential_expression'], 
            targets['differential_expression']
        )
        sig_loss = self._contrastive_loss(
            predictions['signature'], 
            targets['perturbation_ids']
        )
        
        # Combined loss (weighted sum)
        total_loss = expr_loss + 0.5 * de_loss + 0.3 * sig_loss
        
        return {
            'total': total_loss,
            'expression': expr_loss,
            'de': de_loss,
            'signature': sig_loss
        }

    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            # Move to device
            expression = batch['expression'].to(self.device)
            target_ids = batch['target_gene_ids'].to(self.device).squeeze()
            targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
            
            # Forward pass
            predictions = self.model(expression, target_ids)
            
            # Compute loss
            losses = self.compute_loss(predictions, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()
            self.optimizer.step()
            
            total_loss += losses['total'].item()
        
        return total_loss / len(dataloader)

# =============================================================================
# STEP 3: TRAINING CONFIGURATION (FIXED)
# =============================================================================

class TrainingConfig:
    """Configuration for training"""
    
    def __init__(self):
        # Data parameters - MUCH smaller for testing
        self.max_cells = 2000       # Start with only 2K cells
        self.max_genes = 500        # Only 500 genes for speed
        
        # Model parameters - smaller for memory efficiency
        self.embedding_dim = 64     # Smaller embedding
        self.n_heads = 4           # Fewer attention heads
        self.n_layers = 2          # Shallow transformer
        
        # Training parameters - very conservative
        self.batch_size = 8        # Tiny batches
        self.learning_rate = 1e-3  # Slightly higher for faster convergence
        self.num_epochs = 2        # Only 2 epochs for testing
        self.save_every = 50       # Save checkpoint every 50 batches
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {self.device}")

# =============================================================================
# STEP 4: TRAINING PIPELINE (FIXED)
# =============================================================================

def train_on_real_data():
    """Train STATE model on real challenge data"""
    
    print("🚀 Training STATE Model on Real Data - FIXED VERSION")
    print("="*60)
    
    config = TrainingConfig()
    
    # Load real data
    data_path = Path("data/adata_Training.h5ad")
    
    if not data_path.exists():
        print("❌ Training data not found!")
        return {'mae': 999, 'status': 'no_data'}
    
    try:
        # Create dataset
        dataset = VirtualCellDataset(data_path, config.max_cells, config.max_genes)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        
        # Create model (use actual gene count from data)
        n_genes = len(dataset.gene_names)
        print(f"🧠 Creating model for {n_genes} genes")
        
        model = VirtualCellSTATE(
            n_genes=n_genes,
            embedding_dim=config.embedding_dim,
            n_heads=config.n_heads,
            n_layers=config.n_layers
        ).to(config.device)
        
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Create trainer
        trainer = AdvancedStateTrainer(model, config.device, learning_rate=config.learning_rate)
        
        # Training loop with progress tracking
        print("\n🏋️ Starting training...")
        print(f"   Total batches per epoch: {len(dataloader)}")
        print(f"   Checkpointing every {config.save_every} batches")
        
        best_overall_score = float('inf')
        training_history = []
        
        for epoch in range(config.num_epochs):
            print(f"\n📊 Epoch {epoch+1}/{config.num_epochs}")
            
            model.train()
            epoch_loss = 0
            batch_count = 0
            
            for batch_idx, batch in enumerate(dataloader):
                try:
                    # Move to device
                    expression = batch['expression'].to(config.device)
                    target_ids = batch['target_gene_ids'].to(config.device).squeeze()
                    targets = {k: v.to(config.device) for k, v in batch['targets'].items()}
                    
                    # Forward pass
                    predictions = model(expression, target_ids)
                    
                    # Compute loss
                    losses = trainer.compute_loss(predictions, targets)
                    
                    # Backward pass
                    trainer.optimizer.zero_grad()
                    losses['total'].backward()
                    trainer.optimizer.step()
                    
                    epoch_loss += losses['total'].item()
                    batch_count += 1
                    
                    # Progress tracking
                    if batch_idx % 10 == 0:
                        current_loss = epoch_loss / batch_count
                        print(f"   Batch {batch_idx}/{len(dataloader)} - Loss: {current_loss:.4f}")
                    
                    # Checkpointing and evaluation
                    if batch_idx % config.save_every == 0 and batch_idx > 0:
                        model.eval()
                        with torch.no_grad():
                            metrics = trainer.compute_metrics(predictions, targets)
                            overall_score = metrics['mae'] + 0.5 * metrics['des'] + 0.3 * metrics['pds']
                        model.train()
                        
                        print(f"   ✅ Checkpoint {batch_idx//config.save_every} - MAE: {metrics['mae']:.4f}, DES: {metrics['des']:.4f}, PDS: {metrics['pds']:.4f}")
                        
                        # Save checkpoint
                        if overall_score < best_overall_score:
                            best_overall_score = overall_score
                            checkpoint = {
                                'epoch': epoch,
                                'batch': batch_idx,
                                'model_state': model.state_dict(),
                                'optimizer_state': trainer.optimizer.state_dict(),
                                'metrics': metrics
                            }
                            torch.save(checkpoint, 'models/state_best_model.pt')
                            print(f"   🏆 New best score: {overall_score:.4f}")
                    
                except Exception as e:
                    print(f"   ❌ Error in batch {batch_idx}: {e}")
                    continue
            
            avg_epoch_loss = epoch_loss / batch_count
            print(f"   📈 Epoch {epoch+1} complete - Avg Loss: {avg_epoch_loss:.4f}")
            training_history.append(avg_epoch_loss)
        
        # Final evaluation using best model
        print("\n📊 Final evaluation...")
        
        # Load best model
        if Path('models/state_best_model.pt').exists():
            checkpoint = torch.load('models/state_best_model.pt')
            model.load_state_dict(checkpoint['model_state'])
            final_metrics = checkpoint['metrics']
            print(f"   Loaded best model with MAE: {final_metrics['mae']:.4f}")
        else:
            # Fallback if no best model was saved
            model.eval()
            total_mae, total_des, total_pds = 0, 0, 0
            num_batches = 0
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    if batch_idx >= 5: break
                    expression = batch['expression'].to(config.device)
                    target_ids = batch['target_gene_ids'].to(config.device).squeeze()
                    targets = {k: v.to(config.device) for k, v in batch['targets'].items()}
                    predictions = model(expression, target_ids)
                    metrics = trainer.compute_metrics(predictions, targets)
                    total_mae += metrics['mae']
                    total_des += metrics['des']
                    total_pds += metrics['pds']
                    num_batches += 1
            final_metrics = {
                'mae': total_mae / num_batches if num_batches > 0 else 999,
                'des': total_des / num_batches if num_batches > 0 else 999,
                'pds': total_pds / num_batches if num_batches > 0 else 999
            }

        return {
            'metrics': final_metrics,
            'status': 'success',
            'epochs': config.num_epochs,
            'genes_used': n_genes,
            'cells_used': config.max_cells
        }
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        return {
            'mae': 999,
            'status': 'error',
            'error': str(e)
        }

# =============================================================================
# STEP 5: MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    results = train_on_real_data()
    
    print("\n" + "="*50)
    print("🎯 STATE MODEL TRAINING SUMMARY:")
    print("="*50)
    if results['status'] == 'success':
        metrics = results['metrics']
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"DES: {metrics['des']:.4f}")
        print(f"PDS: {metrics['pds']:.4f}")
        print(f"Status: {results['status']}")
        print(f"Genes: {results['genes_used']}")
        print(f"Cells: {results['cells_used']}")
    else:
        print(f"Status: {results['status']}")
        if 'error' in results:
            print(f"Error: {results['error']}")
    print("="*50)
    
    print("\n📋 Just tell Claude:")
    if results['status'] == 'success':
        print(f"→ MAE: {results['metrics']['mae']:.4f}, DES: {results['metrics']['des']:.4f}, PDS: {results['metrics']['pds']:.4f}")
    print(f"→ Status: {results['status']}")
