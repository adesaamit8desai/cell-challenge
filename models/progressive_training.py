# Progressive Training Strategy for Virtual Cell Challenge
# Implements curriculum learning with gradual gene introduction

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
import scanpy as sc
from typing import List, Dict, Tuple
import json
import os
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import torch.nn.functional as F

from gene_selection import BiologicalGeneSelector
from memory_efficient_training import MemoryEfficientStateModel, ScalingOptimizations

class ProgressiveGeneTraining:
    """
    Implements curriculum learning by gradually introducing more genes during training
    """
    
    def __init__(self, total_genes=18080, final_target_genes=5000):
        self.total_genes = total_genes
        self.final_target_genes = final_target_genes
        self.gene_schedule = []
        self.current_phase = 0
        self.gene_selector = BiologicalGeneSelector()
        
    def create_gene_schedule(self, adata):
        """
        Create progressive curriculum starting with essential genes
        """
        print("📚 Creating progressive gene training curriculum...")
        
        schedule = []
        
        # Phase 1: Essential genes only (500 genes)
        print("   Phase 1: Essential gene foundation...")
        essential_genes = []
        
        # Core TFs and pluripotency genes (most critical)
        tf_genes = self.gene_selector.load_transcription_factors()
        pluri_genes = self.gene_selector.load_pluripotency_genes()
        core_genes = list(set(tf_genes + pluri_genes))
        available_core = [g for g in core_genes if g in adata.var.index]
        essential_genes.extend(available_core)
        
        # Add essential metabolic genes
        metabolic_genes = self.gene_selector.load_metabolic_genes()
        available_metabolic = [g for g in metabolic_genes if g in adata.var.index]
        essential_genes.extend(available_metabolic)
        
        essential_genes = list(set(essential_genes))[:500]  # Limit to 500
        
        schedule.append({
            'phase': 1,
            'epoch_range': (0, 8),
            'genes': essential_genes,
            'n_genes': len(essential_genes),
            'batch_size': 256,
            'learning_rate': 1e-3,
            'description': 'Essential genes foundation',
            'embed_dim': 512,  # Can afford larger embeddings with fewer genes
            'n_heads': 8,
            'n_layers': 6
        })
        
        # Phase 2: Add cell cycle and stress response (1200 genes)
        self.logger.info("   Phase 2: Adding cell cycle and regulatory genes...")
        phase2_genes = essential_genes.copy()
        
        cc_genes = self.gene_selector.load_cell_cycle_genes()
        available_cc = [g for g in cc_genes if g in adata.var.index and g not in phase2_genes]
        phase2_genes.extend(available_cc)
        
        # Add top HVGs to reach 1200
        remaining_slots = 1200 - len(phase2_genes)
        if remaining_slots > 0:
            hvg_genes = self.gene_selector.select_highly_variable_genes(adata, n_top_genes=remaining_slots * 2)
            available_hvgs = [g for g in hvg_genes if g in adata.var.index and g not in phase2_genes]
            phase2_genes.extend(available_hvgs[:remaining_slots])
        
        phase2_genes = phase2_genes[:1200]
        
        schedule.append({
            'phase': 2,
            'epoch_range': (8, 15),
            'genes': phase2_genes,
            'n_genes': len(phase2_genes),
            'batch_size': 128,
            'learning_rate': 8e-4,
            'description': 'Essential + cell cycle + regulatory',
            'embed_dim': 384,
            'n_heads': 6,
            'n_layers': 5
        })
        
        # Phase 3: Add perturbation-responsive genes (2500 genes)
        self.logger.info("   Phase 3: Adding perturbation-responsive genes...")
        phase3_genes = phase2_genes.copy()
        
        # Add perturbation-responsive genes
        responsive_genes = self.gene_selector.select_perturbation_responsive_genes(adata, n_genes=800)
        available_responsive = [g for g in responsive_genes if g in adata.var.index and g not in phase3_genes]
        phase3_genes.extend(available_responsive)
        
        # Fill remaining with HVGs
        remaining_slots = 2500 - len(phase3_genes)
        if remaining_slots > 0:
            hvg_genes = self.gene_selector.select_highly_variable_genes(adata, n_top_genes=remaining_slots * 2)
            available_hvgs = [g for g in hvg_genes if g in adata.var.index and g not in phase3_genes]
            phase3_genes.extend(available_hvgs[:remaining_slots])
        
        phase3_genes = phase3_genes[:2500]
        
        schedule.append({
            'phase': 3,
            'epoch_range': (15, 22),
            'genes': phase3_genes,
            'n_genes': len(phase3_genes),
            'batch_size': 64,
            'learning_rate': 5e-4,
            'description': 'Essential + regulatory + perturbation-responsive',
            'embed_dim': 320,
            'n_heads': 5,
            'n_layers': 4
        })
        
        # Phase 4: Final target gene set (up to 5000 genes)
        self.logger.info("   Phase 4: Full target gene set...")
        phase4_genes = self.gene_selector.create_biologically_informed_gene_set(adata)
        phase4_genes = phase4_genes[:self.final_target_genes]
        
        schedule.append({
            'phase': 4,
            'epoch_range': (22, 35),
            'genes': phase4_genes,
            'n_genes': len(phase4_genes),
            'batch_size': 32,
            'learning_rate': 2e-4,
            'description': 'Full biologically-informed gene set',
            'embed_dim': 256,
            'n_heads': 4,
            'n_layers': 4
        })
        
        self.gene_schedule = schedule
        
        # Print schedule summary
        self.logger.info("📋 Progressive training schedule:")
        for phase_config in schedule:
            self.logger.info(f"   Phase {phase_config['phase']}: {phase_config['description']}")
            self.logger.info(f"      Epochs {phase_config['epoch_range'][0]}-{phase_config['epoch_range'][1]}")
            self.logger.info(f"      Genes: {phase_config['n_genes']}")
            self.logger.info(f"      Batch size: {phase_config['batch_size']}")
            self.logger.info(f"      LR: {phase_config['learning_rate']}")
        
        return schedule
    
    def adapt_model_for_phase(self, model, phase_config, device):
        """
        Adapt model architecture for current phase
        """
        n_genes = phase_config['n_genes']
        
        # Create new model with appropriate size
        new_model = MemoryEfficientStateModel(
            n_genes=n_genes,
            embed_dim=phase_config['embed_dim'],
            n_heads=phase_config['n_heads'],
            n_layers=phase_config['n_layers'],
            attention_type='sparse',
            use_checkpointing=True,
            mixed_precision=True
        ).to(device)
        
        # Transfer knowledge from previous phase
        if model is not None:
            self.transfer_knowledge(model, new_model, phase_config)
        
        return new_model
    
    def transfer_knowledge(self, source_model, target_model, phase_config):
        """
        Transfer learned representations from previous phase
        """
        print(f"🔄 Transferring knowledge to Phase {phase_config['phase']}...")
        
        with torch.no_grad():
            # Transfer embedding weights for genes that exist in both models
            source_n_genes = source_model.gene_embeddings.num_embeddings
            target_n_genes = target_model.gene_embeddings.num_embeddings
            
            # Transfer overlapping gene embeddings
            min_genes = min(source_n_genes, target_n_genes)
            if min_genes > 0:
                target_model.gene_embeddings.weight[:min_genes] = source_model.gene_embeddings.weight[:min_genes]
                print(f"   ✅ Transferred embeddings for {min_genes} genes")
            
            # Transfer transformer layers if dimensions match
            if (source_model.embed_dim == target_model.embed_dim and 
                len(source_model.layers) <= len(target_model.layers)):
                
                for i, source_layer in enumerate(source_model.layers):
                    if i < len(target_model.layers):
                        # Transfer layer weights
                        target_layer = target_model.layers[i]
                        
                        # Transfer attention weights
                        try:
                            target_layer.attention.query.weight.data = source_layer.attention.query.weight.data.clone()
                            target_layer.attention.key.weight.data = source_layer.attention.key.weight.data.clone()
                            target_layer.attention.value.weight.data = source_layer.attention.value.weight.data.clone()
                            target_layer.attention.out.weight.data = source_layer.attention.out.weight.data.clone()
                            
                            # Transfer layer norms
                            target_layer.norm1.weight.data = source_layer.norm1.weight.data.clone()
                            target_layer.norm1.bias.data = source_layer.norm1.bias.data.clone()
                            target_layer.norm2.weight.data = source_layer.norm2.weight.data.clone()
                            target_layer.norm2.bias.data = source_layer.norm2.bias.data.clone()
                            
                            # Transfer feed forward
                            target_layer.ff[0].weight.data = source_layer.ff[0].weight.data.clone()
                            target_layer.ff[0].bias.data = source_layer.ff[0].bias.data.clone()
                            target_layer.ff[3].weight.data = source_layer.ff[3].weight.data.clone()
                            target_layer.ff[3].bias.data = source_layer.ff[3].bias.data.clone()
                            
                        except Exception as e:
                            print(f"   ⚠️ Could not transfer layer {i}: {e}")
                
                print(f"   ✅ Transferred {len(source_model.layers)} transformer layers")
            else:
                print(f"   ⚠️ Skipping layer transfer: dimensions don't match (source: {source_model.embed_dim}, target: {target_model.embed_dim})")
    
    def train_progressive(self, adata, output_dir="training_runs/progressive", device='cpu'):
        """
        Execute progressive training strategy
        """
        print("🚀 Starting progressive training...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create gene schedule
        self.create_gene_schedule(adata)
        
        # Save schedule
        schedule_path = os.path.join(output_dir, "training_schedule.json")
        with open(schedule_path, 'w') as f:
            # Convert to JSON-serializable format
            schedule_json = []
            for phase in self.gene_schedule:
                phase_json = phase.copy()
                phase_json['genes'] = phase_json['genes'][:50]  # Save first 50 genes as example
                schedule_json.append(phase_json)
            json.dump(schedule_json, f, indent=2)
        
        # Initialize training state
        model = None
        training_history = {
            'phases': [],
            'total_epochs': 0,
            'best_loss': float('inf'),
            'final_genes': []
        }
        
        # Setup mixed precision
        scaler = ScalingOptimizations.setup_mixed_precision()
        
        # Execute each phase
        for phase_idx, phase_config in enumerate(self.gene_schedule):
            self.logger.info(f"""
🎯 Phase {phase_config['phase']}: {phase_config['description']}
""")
            self.logger.info(f"   Genes: {phase_config['n_genes']}")
            self.logger.info(f"   Epochs: {phase_config['epoch_range'][0]} - {phase_config['epoch_range'][1]}")
            
            # Subset data to current gene set
            phase_genes = phase_config['genes']
            gene_mask = adata.var.index.isin(phase_genes)
            adata_phase = adata[:, gene_mask].copy()
            
            self.logger.info(f"   Data shape after subsetting: {adata_phase.shape}")
            self.logger.info(f"   Number of genes for model: {adata_phase.n_vars}")
            
            # Adapt model for this phase
            model = self.adapt_model_for_phase(model, phase_config, device)
            self.logger.info(f"   Model adapted for phase {phase_config['phase']}")
            
            # Setup optimizer
            optimizer = optim.AdamW(
                model.parameters(), 
                lr=phase_config['learning_rate'],
                weight_decay=1e-5
            )
            
            # Train this phase
            phase_history = self.train_phase(
                model=model,
                adata=adata_phase,
                phase_config=phase_config,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                output_dir=output_dir
            )
            
            # Update training history
            training_history['phases'].append(phase_history)
            training_history['total_epochs'] = phase_config['epoch_range'][1]
            
            if phase_history['best_loss'] < training_history['best_loss']:
                training_history['best_loss'] = phase_history['best_loss']
                training_history['final_genes'] = phase_genes
            
            # Save checkpoint
            checkpoint_path = os.path.join(output_dir, f"phase_{phase_config['phase']}_checkpoint.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'phase_config': phase_config,
                'training_history': training_history
            }, checkpoint_path)
            
            print(f"   ✅ Phase {phase_config['phase']} completed")
            print(f"   Best loss: {phase_history['best_loss']:.4f}")
        
        # Save final results
        results_path = os.path.join(output_dir, "progressive_training_results.json")
        with open(results_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        # Save final gene list
        final_genes_path = os.path.join(output_dir, "final_gene_set.csv")
        final_genes_df = pd.DataFrame({
            'gene_name': training_history['final_genes'],
            'gene_index': range(len(training_history['final_genes']))
        })
        final_genes_df.to_csv(final_genes_path, index=False)
        
        print(f"🎉 Progressive training completed!")
        print(f"   Total epochs: {training_history['total_epochs']}")
        print(f"   Best loss: {training_history['best_loss']:.4f}")
        print(f"   Final gene count: {len(training_history['final_genes'])}")
        print(f"   Results saved to: {output_dir}")
        
        return model, training_history
    
    def train_phase(self, model, adata, phase_config, optimizer, scaler, device, output_dir):
        """
        Train model for a single phase
        """
        self.logger.info(f"Starting training for phase {phase_config['phase']}...")
        self.logger.info(f"  Batch size: {phase_config['batch_size']}")
        self.logger.info(f"  Learning rate: {phase_config['learning_rate']}")
        
        model.train()
        
        # Prepare data
        X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        X = torch.FloatTensor(X).to(device)
        
        n_samples, n_genes = X.shape
        batch_size = phase_config['batch_size']
        
        # Training loop
        epoch_start, epoch_end = phase_config['epoch_range']
        phase_history = {
            'phase': phase_config['phase'],
            'epochs': [],
            'losses': [],
            'best_loss': float('inf')
        }
        
        for epoch in range(epoch_start, epoch_end):
            epoch_start_time = time.time()
            epoch_losses = []
            n_batches = (n_samples + batch_size - 1) // batch_size
            
            # Shuffle data
            indices = torch.randperm(n_samples)
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                batch_x = X[batch_indices]
                self.logger.info(f"       Batch {batch_idx}: batch_x shape = {batch_x.shape}")
                
                # Forward pass with mixed precision
                optimizer.zero_grad()
                
                with autocast():
                    outputs = model(batch_x)
                    self.logger.info(f"       Batch {batch_idx}: outputs shape = {outputs.shape}")
                    # Self-supervised loss: predict expression from masked input
                    loss = F.mse_loss(outputs, batch_x)
                    self.logger.info(f"       Batch {batch_idx}: loss = {loss.item():.4f}")
                
                # Backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                epoch_losses.append(loss.item())
                
                if batch_idx % 100 == 0: # Log every 100 batches
                    self.logger.info(f"     Batch {batch_idx}/{n_batches}: Loss = {loss.item():.4f}")
            
            # Epoch summary
            epoch_time = time.time() - epoch_start_time
            avg_loss = np.mean(epoch_losses)
            phase_history['epochs'].append(epoch)
            phase_history['losses'].append(avg_loss)
            
            if avg_loss < phase_history['best_loss']:
                phase_history['best_loss'] = avg_loss
                
                # Save best model for this phase
                best_model_path = os.path.join(output_dir, f"phase_{phase_config['phase']}_best_model.pt")
                torch.save(model.state_dict(), best_model_path)
            
            self.logger.info(f"   Epoch {epoch}: Loss = {avg_loss:.4f}, Time = {epoch_time:.2f}s")
        
        return phase_history

def run_progressive_training(data_path="data/adata_Training.h5ad", 
                           target_genes=5000, 
                           output_dir="training_runs/progressive"):
    """
    Main function to run progressive training
    """
    print("🚀 Initializing progressive training...")
    
    # Load data
    print("📂 Loading training data...")
    adata = sc.read_h5ad(data_path, backed='r')
    
    # Subset for testing (remove for full training)
    print("🔬 Using subset for testing...")
    adata = adata[:5000].to_memory()  # Use 5K cells for testing
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")
    
    # Initialize progressive trainer
    trainer = ProgressiveGeneTraining(
        total_genes=adata.n_vars,
        final_target_genes=target_genes
    )
    
    # Run progressive training
    model, history = trainer.train_progressive(
        adata=adata,
        output_dir=output_dir,
        device=device
    )
    
    return model, history

if __name__ == "__main__":
    # Run progressive training
    model, history = run_progressive_training(
        target_genes=3000,
        output_dir="training_runs/progressive_test"
    )