# STATE-Inspired Transformer Model for Virtual Cell Challenge
# Phase 2: Advanced architecture based on Arc's STATE model

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# STEP 1: GENE EMBEDDING LAYER (Like STATE's SE Module)
# =============================================================================

class GeneEmbedding(nn.Module):
    """
    Creates embeddings for genes similar to STATE's State Embedding (SE) module
    """
    def __init__(self, n_genes, embedding_dim=128):
        super().__init__()
        self.n_genes = n_genes
        self.embedding_dim = embedding_dim
        
        # Gene position embeddings (like word embeddings)
        self.gene_embeddings = nn.Embedding(n_genes, embedding_dim)
        
        # Expression value encoder
        self.expression_encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )
        
        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, expression_values, gene_indices):
        """
        Args:
            expression_values: (batch_size, n_genes) - expression levels
            gene_indices: (n_genes,) - gene position indices
        """
        batch_size, n_genes = expression_values.shape
        
        # Get gene embeddings
        gene_emb = self.gene_embeddings(gene_indices)  # (n_genes, embedding_dim)
        gene_emb = gene_emb.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, n_genes, embedding_dim)
        
        # Encode expression values
        expr_emb = self.expression_encoder(expression_values.unsqueeze(-1))  # (batch_size, n_genes, embedding_dim)
        
        # Combine gene and expression embeddings
        combined = gene_emb + expr_emb
        return self.layer_norm(combined)

# =============================================================================
# STEP 2: PERTURBATION ENCODER
# =============================================================================

class PerturbationEncoder(nn.Module):
    """Encodes which gene is being perturbed"""
    
    def __init__(self, n_genes, embedding_dim=128):
        super().__init__()
        self.target_embedding = nn.Embedding(n_genes + 1, embedding_dim)  # +1 for "no target"
        
    def forward(self, target_gene_ids):
        """
        Args:
            target_gene_ids: (batch_size,) - which gene is being perturbed
        """
        # Ensure target_gene_ids is 1D
        if target_gene_ids.dim() > 1:
            target_gene_ids = target_gene_ids.squeeze()
        
        # Get single embedding per sample
        embeddings = self.target_embedding(target_gene_ids)  # (batch_size, embedding_dim)
        return embeddings

# =============================================================================
# STEP 3: TRANSFORMER ENCODER (Like STATE's ST Module)
# =============================================================================

class StateTransformer(nn.Module):
    """
    Transformer that predicts how cells transition between states
    Similar to STATE's State Transition (ST) module
    """
    
    def __init__(self, embedding_dim=128, n_heads=8, n_layers=4, dropout=0.1):
        super().__init__()
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Additional processing layers
        self.pre_transformer = nn.Linear(embedding_dim, embedding_dim)
        self.post_transformer = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, gene_embeddings, perturbation_emb):
        """
        Args:
            gene_embeddings: (batch_size, n_genes, embedding_dim)
            perturbation_emb: (batch_size, embedding_dim)
        """
        batch_size, n_genes, embedding_dim = gene_embeddings.shape
        
        # Ensure perturbation_emb has correct shape (batch_size, embedding_dim)
        if perturbation_emb.dim() == 1:
            perturbation_emb = perturbation_emb.unsqueeze(0)  # Add batch dimension if missing
        elif perturbation_emb.dim() > 2:
            perturbation_emb = perturbation_emb.squeeze()  # Remove extra dimensions
        
        # Expand perturbation embedding to match gene embeddings
        pert_expanded = perturbation_emb.unsqueeze(1).expand(-1, n_genes, -1)  # (batch_size, n_genes, embedding_dim)
        combined = gene_embeddings + pert_expanded
        
        # Pre-process
        processed = self.pre_transformer(combined)
        
        # Transformer processing (attention over genes)
        transformed = self.transformer(processed)
        
        # Post-process
        output = self.post_transformer(transformed)
        
        return output

# =============================================================================
# STEP 4: MULTI-TASK OUTPUT HEADS
# =============================================================================

class MultiTaskHead(nn.Module):
    """
    Multiple output heads for the 3 challenge metrics:
    1. Expression prediction (MAE)
    2. Differential expression (which genes change)
    3. Perturbation signature (discrimination)
    """
    
    def __init__(self, embedding_dim=128, n_genes=18000):
        super().__init__()
        
        # Head 1: Expression prediction (continuous values)
        self.expression_head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()  # Expression values are non-negative
        )
        
        # Head 2: Differential expression (binary classification)
        self.de_head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Probability of being differentially expressed
        )
        
        # Head 3: Perturbation signature (for discrimination)
        self.signature_head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)  # Compact signature representation
        )
        
    def forward(self, transformed_genes):
        """
        Args:
            transformed_genes: (batch_size, n_genes, embedding_dim)
        """
        # Expression predictions
        expression_pred = self.expression_head(transformed_genes).squeeze(-1)  # (batch_size, n_genes)
        
        # DE predictions
        de_pred = self.de_head(transformed_genes).squeeze(-1)  # (batch_size, n_genes)
        
        # Signature (global cell state)
        signature = self.signature_head(transformed_genes).mean(dim=1)  # (batch_size, 32)
        
        return {
            'expression': expression_pred,
            'differential_expression': de_pred,
            'signature': signature
        }

# =============================================================================
# STEP 5: COMPLETE MODEL
# =============================================================================

class VirtualCellSTATE(nn.Module):
    """
    Complete STATE-inspired model for Virtual Cell Challenge
    """
    
    def __init__(self, n_genes=18000, embedding_dim=128, n_heads=8, n_layers=4):
        super().__init__()
        
        self.n_genes = n_genes
        self.embedding_dim = embedding_dim
        
        # Components
        self.gene_embedding = GeneEmbedding(n_genes, embedding_dim)
        self.perturbation_encoder = PerturbationEncoder(n_genes, embedding_dim)
        self.transformer = StateTransformer(embedding_dim, n_heads, n_layers)
        self.output_heads = MultiTaskHead(embedding_dim, n_genes)
        
        # Gene indices (fixed)
        self.register_buffer('gene_indices', torch.arange(n_genes))
        
    def forward(self, expression, target_gene_ids):
        """
        Args:
            expression: (batch_size, n_genes) - baseline expression
            target_gene_ids: (batch_size,) - which gene is perturbed
        """
        # Embed genes and expression
        gene_emb = self.gene_embedding(expression, self.gene_indices)
        
        # Encode perturbation
        pert_emb = self.perturbation_encoder(target_gene_ids)
        
        # Transform through attention
        transformed = self.transformer(gene_emb, pert_emb)
        
        # Generate predictions
        outputs = self.output_heads(transformed)
        
        return outputs

# =============================================================================
# STEP 6: TRAINING PIPELINE
# =============================================================================

class StateTrainer:
    """Training pipeline for STATE model"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
    def compute_loss(self, predictions, targets):
        """Multi-task loss computation"""
        
        # Expression loss (MAE)
        expr_loss = F.l1_loss(predictions['expression'], targets['expression'])
        
        # DE loss (binary classification)
        de_loss = F.binary_cross_entropy(
            predictions['differential_expression'], 
            targets['differential_expression']
        )
        
        # Signature loss (contrastive - encourage different perturbations to have different signatures)
        sig_loss = self._contrastive_loss(
            predictions['signature'], 
            targets['perturbation_ids']
        )
        
        # Combined loss
        total_loss = expr_loss + 0.5 * de_loss + 0.3 * sig_loss
        
        return {
            'total': total_loss,
            'expression': expr_loss,
            'de': de_loss,
            'signature': sig_loss
        }
    
    def _contrastive_loss(self, signatures, pert_ids):
        """Encourage different perturbations to have different signatures"""
        # Simple version: minimize cosine similarity between different perturbations
        # More sophisticated versions could use proper contrastive learning
        
        # For now, just return a small dummy loss
        return torch.tensor(0.0, device=self.device)
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            # Move to device
            expression = batch['expression'].to(self.device)
            target_ids = batch['target_gene_ids'].to(self.device)
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
# STEP 7: MEMORY-EFFICIENT IMPLEMENTATION
# =============================================================================

def run_state_model_pipeline():
    """
    Run the STATE-inspired model with memory management
    """
    print("🚀 Starting STATE-Inspired Model Pipeline")
    print("="*60)
    
    # Model parameters (smaller for memory efficiency)
    n_genes_subset = 1000  # Use top 1000 genes for initial testing
    embedding_dim = 64     # Smaller embedding for memory
    batch_size = 32        # Small batches
    
    print(f"📊 Using {n_genes_subset} genes, embedding_dim={embedding_dim}")
    
    # Create model
    model = VirtualCellSTATE(
        n_genes=n_genes_subset,
        embedding_dim=embedding_dim,
        n_heads=4,
        n_layers=2
    )
    
    print(f"🧠 Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # For now, create dummy data to test the architecture
    batch_size = 16
    dummy_expression = torch.randn(batch_size, n_genes_subset)
    dummy_targets = torch.randint(0, n_genes_subset, (batch_size,))
    
    # Test forward pass
    try:
        with torch.no_grad():
            outputs = model(dummy_expression, dummy_targets)
        
        print("✅ Forward pass successful!")
        print(f"   Expression output shape: {outputs['expression'].shape}")
        print(f"   DE output shape: {outputs['differential_expression'].shape}")
        print(f"   Signature output shape: {outputs['signature'].shape}")
        
        # Simple MAE calculation for comparison
        mae = F.l1_loss(outputs['expression'], dummy_expression).item()
        
        return {
            'mae': mae,
            'model_type': 'STATE_inspired_transformer',
            'status': 'success',
            'parameters': sum(p.numel() for p in model.parameters())
        }
        
    except Exception as e:
        print(f"❌ Error in forward pass: {e}")
        return {
            'mae': 999,
            'model_type': 'STATE_inspired_transformer',
            'status': 'error',
            'error': str(e)
        }

if __name__ == "__main__":
    results = run_state_model_pipeline()
    
    print("\n" + "="*50)
    print("🎯 STATE MODEL SUMMARY:")
    print("="*50)
    print(f"MAE: {results['mae']:.4f}")
    print(f"Status: {results['status']}")
    print("="*50)
    
    print("\n📋 Just tell Claude:")
    print(f"→ MAE: {results['mae']:.4f}")
    print(f"→ Status: {results['status']}")
    print("\nThat's all I need! 🚀")