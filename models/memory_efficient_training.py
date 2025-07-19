# Memory-Efficient Training Optimizations for Virtual Cell Challenge
# Implements mixed precision, gradient accumulation, and efficient attention

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class MemoryEfficientAttention(nn.Module):
    """
    Memory-efficient attention mechanisms for large gene sets
    """
    
    def __init__(self, embed_dim, n_heads, attention_type='sparse', max_genes=5000):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.attention_type = attention_type
        self.max_genes = max_genes
        
        # Standard attention components
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)
        
        # Initialize attention implementation
        if attention_type == 'sparse':
            self.attention_impl = SparseAttention(embed_dim, n_heads, max_genes)
        elif attention_type == 'linear':
            self.attention_impl = LinearAttention(embed_dim, n_heads)
        elif attention_type == 'local':
            self.attention_impl = LocalAttention(embed_dim, n_heads, window_size=100)
        else:
            self.attention_impl = StandardAttention(embed_dim, n_heads)
    
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        # Generate Q, K, V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply efficient attention
        attention_output = self.attention_impl(Q, K, V)
        
        # Reshape and project output
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        
        return self.out(attention_output)

class SparseAttention(nn.Module):
    """
    Sparse attention that only computes attention for biologically relevant gene pairs
    """
    
    def __init__(self, embed_dim, n_heads, max_genes, sparsity_ratio=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.max_genes = max_genes
        self.sparsity_ratio = sparsity_ratio
        
        # Create sparse attention mask
        self.register_buffer('attention_mask', self.create_sparse_mask())
    
    def create_sparse_mask(self):
        """
        Create sparse attention mask based on biological relevance
        For now, use local + random connections
        """
        mask = torch.zeros(self.max_genes, self.max_genes)
        
        # Local connections (neighboring genes)
        for i in range(self.max_genes):
            # Connect to nearby genes (sliding window)
            window_size = 20
            start = max(0, i - window_size // 2)
            end = min(self.max_genes, i + window_size // 2 + 1)
            mask[i, start:end] = 1.0
        
        # Add random long-range connections
        n_random = int(self.sparsity_ratio * self.max_genes * self.max_genes)
        random_i = torch.randint(0, self.max_genes, (n_random,))
        random_j = torch.randint(0, self.max_genes, (n_random,))
        mask[random_i, random_j] = 1.0
        
        # Ensure self-attention
        mask.fill_diagonal_(1.0)
        
        return mask
    
    def forward(self, query, key, value):
        batch_size, n_heads, seq_len, head_dim = query.shape
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(head_dim, dtype=torch.float32))
        
        # Apply sparse mask (only for the sequence length we have)
        if seq_len <= self.max_genes:
            mask = self.attention_mask[:seq_len, :seq_len]
            scores = scores * mask.unsqueeze(0).unsqueeze(0)
            
            # Set masked positions to very negative value
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = torch.nan_to_num(attention_weights, 0.0)  # Handle inf from masking
        
        # Apply attention to values
        output = torch.matmul(attention_weights, value)
        
        return output

class LinearAttention(nn.Module):
    """
    Linear complexity attention using feature maps
    """
    
    def __init__(self, embed_dim, n_heads, feature_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.feature_dim = feature_dim
    
    def feature_map(self, x):
        """
        Apply feature map for linear attention
        Using ReLU feature map for simplicity
        """
        return F.relu(x)
    
    def forward(self, query, key, value):
        # Apply feature maps
        phi_q = self.feature_map(query)
        phi_k = self.feature_map(key)
        
        # Linear attention: O(n) complexity
        # Compute K^T V first
        kv = torch.matmul(phi_k.transpose(-2, -1), value)
        
        # Then Q (K^T V)
        output = torch.matmul(phi_q, kv)
        
        # Normalization
        normalizer = torch.matmul(phi_q, phi_k.sum(dim=-2, keepdim=True).transpose(-2, -1))
        normalizer = torch.clamp(normalizer, min=1e-6)
        output = output / normalizer
        
        return output

class LocalAttention(nn.Module):
    """
    Local attention with sliding window
    """
    
    def __init__(self, embed_dim, n_heads, window_size=100):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.window_size = window_size
    
    def forward(self, query, key, value):
        batch_size, n_heads, seq_len, head_dim = query.shape
        
        # For now, implement as standard attention with masking
        # In practice, you'd implement sliding window efficiently
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(head_dim, dtype=torch.float32))
        
        # Create local mask
        mask = torch.zeros(seq_len, seq_len, device=scores.device)
        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            mask[i, start:end] = 1.0
        
        # Apply mask
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = torch.nan_to_num(attention_weights, 0.0)
        
        output = torch.matmul(attention_weights, value)
        return output

class StandardAttention(nn.Module):
    """
    Standard attention for comparison
    """
    
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
    
    def forward(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, value)
        return output

class MemoryOptimizedTransformerLayer(nn.Module):
    """
    Memory-optimized transformer layer with gradient checkpointing
    """
    
    def __init__(self, embed_dim, n_heads, ff_dim, attention_type='sparse', use_checkpointing=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_checkpointing = use_checkpointing
        
        # Multi-head attention
        self.attention = MemoryEfficientAttention(embed_dim, n_heads, attention_type)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Feed forward
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(0.1)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        if self.use_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(self._forward, x)
        else:
            return self._forward(x)
    
    def _forward(self, x):
        # Self-attention with residual connection
        attn_output = self.attention(x)
        x = self.norm1(x + attn_output)
        
        # Feed forward with residual connection
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        
        return x

class ScalingOptimizations:
    """
    Practical optimizations for memory-efficient training
    """
    
    @staticmethod
    def setup_mixed_precision():
        """
        Setup mixed precision training for ~50% memory reduction
        """
        scaler = GradScaler()
        print("🔥 Mixed precision training enabled")
        return scaler
    
    @staticmethod
    def create_efficient_dataloader(dataset, batch_size=32, num_workers=2):
        """
        Create memory-efficient data loader
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
            shuffle=True
        )
    
    @staticmethod
    def gradient_accumulation_step(model, batch, scaler, optimizer, accumulation_steps=4, step_count=0):
        """
        Perform gradient accumulation to simulate larger batch sizes
        """
        with autocast():
            outputs = model(batch['input'])
            loss = F.mse_loss(outputs, batch['target'])
            loss = loss / accumulation_steps  # Scale loss
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Update weights every accumulation_steps
        if (step_count + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            return True, loss.item() * accumulation_steps
        
        return False, loss.item() * accumulation_steps
    
    @staticmethod
    def estimate_memory_usage(model, batch_size, seq_len, embed_dim):
        """
        Estimate memory usage for planning
        """
        # Model parameters
        param_memory = sum(p.numel() * 4 for p in model.parameters()) / 1e9  # GB
        
        # Activations (rough estimate)
        activation_memory = (batch_size * seq_len * embed_dim * 4) / 1e9  # GB
        
        # Gradients (same as parameters)
        gradient_memory = param_memory
        
        total_memory = param_memory + activation_memory + gradient_memory
        
        print(f"📊 Memory estimate:")
        print(f"   Parameters: {param_memory:.2f} GB")
        print(f"   Activations: {activation_memory:.2f} GB")
        print(f"   Gradients: {gradient_memory:.2f} GB")
        print(f"   Total: {total_memory:.2f} GB")
        
        return total_memory

class MemoryEfficientStateModel(nn.Module):
    """
    Memory-efficient version of the STATE model
    """
    
    def __init__(self, n_genes, embed_dim=256, n_heads=4, n_layers=4, 
                 attention_type='sparse', use_checkpointing=True, mixed_precision=True):
        super().__init__()
        self.n_genes = n_genes
        self.embed_dim = embed_dim
        self.mixed_precision = mixed_precision
        
        # Gene embeddings
        self.gene_embeddings = nn.Embedding(n_genes, embed_dim)
        self.expression_encoder = nn.Sequential(
            nn.Linear(1, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        self.input_norm = nn.LayerNorm(embed_dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            MemoryOptimizedTransformerLayer(
                embed_dim=embed_dim,
                n_heads=n_heads,
                ff_dim=embed_dim * 2,  # Smaller FF dimension
                attention_type=attention_type,
                use_checkpointing=use_checkpointing
            )
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 1)
        )
        
        print(f"🧠 Created memory-efficient model:")
        print(f"   Genes: {n_genes}")
        print(f"   Embed dim: {embed_dim}")
        print(f"   Heads: {n_heads}")
        print(f"   Layers: {n_layers}")
        print(f"   Attention: {attention_type}")
        print(f"   Checkpointing: {use_checkpointing}")
        print(f"   Mixed precision: {mixed_precision}")
    
    def forward(self, x):
        batch_size, n_genes = x.shape
        
        # Create gene indices
        gene_indices = torch.arange(n_genes, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        
        # Gene embeddings
        gene_emb = self.gene_embeddings(gene_indices)
        
        # Expression embeddings
        expr_emb = self.expression_encoder(x.unsqueeze(-1))
        
        # Combine embeddings
        combined_emb = gene_emb + expr_emb
        combined_emb = self.input_norm(combined_emb)
        
        # Apply transformer layers
        for layer in self.layers:
            combined_emb = layer(combined_emb)
        
        # Output projection
        output = self.output_norm(combined_emb)
        output = self.output_proj(output).squeeze(-1)
        
        return output

def create_memory_efficient_model(n_genes=3000, device='cpu'):
    """
    Create optimized model for memory efficiency
    """
    model = MemoryEfficientStateModel(
        n_genes=n_genes,
        embed_dim=256,
        n_heads=4,
        n_layers=4,
        attention_type='sparse',
        use_checkpointing=True,
        mixed_precision=True
    )
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Model statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {total_params * 4 / 1e6:.2f} MB")
    
    return model

if __name__ == "__main__":
    # Test memory efficient model
    print("🧪 Testing memory-efficient model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_memory_efficient_model(n_genes=3000, device=device)
    
    # Test forward pass
    batch_size = 8
    n_genes = 3000
    x = torch.randn(batch_size, n_genes, device=device)
    
    with autocast():
        output = model(x)
    
    print(f"✅ Forward pass successful: {x.shape} → {output.shape}")