# Advanced Gene Selection for Virtual Cell Challenge Scaling
# Implements biologically-informed gene selection strategies

import numpy as np
import pandas as pd
import scanpy as sc
from typing import List, Set, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class BiologicalGeneSelector:
    """
    Strategic gene selection combining biological knowledge with data-driven approaches
    """
    
    def __init__(self, target_genes: int = 3000):
        self.target_genes = target_genes
        self.selected_genes = set()
        self.gene_categories = {
            'transcription_factors': [],
            'metabolic': [],
            'cell_cycle': [],
            'pluripotency': [],
            'highly_variable': []
        }
    
    def select_highly_variable_genes(self, adata, n_top_genes: int = 2000):
        """
        Select genes with highest biological variability using Seurat v3 method
        """
        print(f"🧬 Selecting {n_top_genes} highly variable genes...")
        
        try:
            # Use Seurat v3 method for HVG selection
            sc.pp.highly_variable_genes(
                adata, 
                n_top_genes=n_top_genes,
                flavor='seurat_v3',
                subset=False  # Don't subset yet, just mark
            )
            
            # Get HVG gene names
            hvg_genes = adata.var[adata.var.highly_variable].index.tolist()
            self.gene_categories['highly_variable'] = hvg_genes
            
            print(f"   ✅ Found {len(hvg_genes)} highly variable genes")
            return hvg_genes
            
        except Exception as e:
            print(f"   ⚠️ HVG selection failed: {e}")
            # Fallback: select by variance
            gene_vars = np.var(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X, axis=0)
            top_var_indices = np.argsort(gene_vars)[-n_top_genes:]
            hvg_genes = adata.var.index[top_var_indices].tolist()
            self.gene_categories['highly_variable'] = hvg_genes
            
            print(f"   ✅ Used variance fallback: {len(hvg_genes)} genes")
            return hvg_genes
    
    def select_perturbation_responsive_genes(self, adata, n_genes: int = 1000):
        """
        Select genes that respond most to perturbations in training data
        """
        print(f"🎯 Selecting {n_genes} perturbation-responsive genes...")
        
        if 'perturbation' not in adata.obs.columns:
            print("   ⚠️ No perturbation column found, skipping...")
            return []
        
        gene_perturbation_scores = []
        perturbations = adata.obs['perturbation'].unique()
        
        for gene_idx in range(adata.n_vars):
            # Calculate variance of this gene across different perturbations
            gene_expr = adata.X[:, gene_idx].toarray().flatten() if hasattr(adata.X, 'toarray') else adata.X[:, gene_idx]
            
            # Group by perturbation and calculate variance
            pert_means = []
            for pert in perturbations:
                pert_mask = adata.obs['perturbation'] == pert
                if np.sum(pert_mask) > 0:
                    pert_mean = np.mean(gene_expr[pert_mask])
                    pert_means.append(pert_mean)
            
            # Score = variance across perturbation means
            if len(pert_means) > 1:
                variance_score = np.var(pert_means)
            else:
                variance_score = 0
            
            gene_perturbation_scores.append(variance_score)
        
        # Select top responsive genes
        top_genes_idx = np.argsort(gene_perturbation_scores)[-n_genes:]
        responsive_genes = adata.var.index[top_genes_idx].tolist()
        
        print(f"   ✅ Found {len(responsive_genes)} perturbation-responsive genes")
        return responsive_genes
    
    def load_transcription_factors(self) -> List[str]:
        """
        Load known transcription factor genes
        Using a curated list of important TFs for stem cells
        """
        # Core transcription factors important for stem cells and development
        core_tfs = [
            'SOX2', 'OCT4', 'POU5F1', 'NANOG', 'KLF4', 'MYC', 'LIN28A',
            'SOX1', 'SOX3', 'SOX9', 'SOX17', 'FOXO1', 'FOXO3', 'FOXP1',
            'GATA1', 'GATA2', 'GATA3', 'GATA4', 'GATA6', 'TBX3', 'TBX5',
            'HAND1', 'HAND2', 'NKX2-5', 'MESP1', 'MESP2', 'MIXL1',
            'EOMES', 'TBXT', 'GSC', 'FOXA2', 'HNF4A', 'CDX2', 'DLX5',
            'MSX1', 'MSX2', 'PAX6', 'OTX2', 'EN1', 'EN2', 'FGF8',
            'WNT3A', 'NODAL', 'LEFTY1', 'LEFTY2', 'CER1', 'CHRD'
        ]
        
        self.gene_categories['transcription_factors'] = core_tfs
        print(f"   📚 Loaded {len(core_tfs)} core transcription factors")
        return core_tfs
    
    def load_metabolic_genes(self) -> List[str]:
        """
        Load essential metabolic genes
        """
        metabolic_genes = [
            'GAPDH', 'ENO1', 'PKM', 'LDHA', 'LDHB', 'PGK1', 'ALDOA',
            'TPI1', 'PGAM1', 'GPI', 'HK1', 'HK2', 'PFKL', 'PFKM',
            'ACLY', 'FASN', 'SCD', 'SREBF1', 'PPARA', 'PPARG',
            'G6PD', 'TALDO1', 'TKT', 'RPIA', 'RPE', 'PRPS1',
            'IDH1', 'IDH2', 'CS', 'ACO2', 'SUCLA2', 'SUCLG1',
            'FH', 'MDH1', 'MDH2', 'OGDH', 'DLAT', 'DLD'
        ]
        
        self.gene_categories['metabolic'] = metabolic_genes
        print(f"   ⚡ Loaded {len(metabolic_genes)} essential metabolic genes")
        return metabolic_genes
    
    def load_cell_cycle_genes(self) -> List[str]:
        """
        Load cell cycle genes critical for stem cells
        """
        cell_cycle_genes = [
            'CCND1', 'CCND2', 'CCND3', 'CCNE1', 'CCNE2', 'CCNA1', 'CCNA2',
            'CCNB1', 'CCNB2', 'CDK1', 'CDK2', 'CDK4', 'CDK6',
            'CDKN1A', 'CDKN1B', 'CDKN2A', 'CDKN2B', 'RB1', 'RBL1', 'RBL2',
            'E2F1', 'E2F2', 'E2F3', 'TP53', 'MDM2', 'ATM', 'ATR',
            'CHEK1', 'CHEK2', 'WEE1', 'CDC25A', 'CDC25B', 'CDC25C',
            'PCNA', 'MCM2', 'MCM3', 'MCM4', 'MCM5', 'MCM6', 'MCM7'
        ]
        
        self.gene_categories['cell_cycle'] = cell_cycle_genes
        print(f"   🔄 Loaded {len(cell_cycle_genes)} cell cycle genes")
        return cell_cycle_genes
    
    def load_pluripotency_genes(self) -> List[str]:
        """
        Load extended pluripotency network genes
        """
        pluripotency_genes = [
            'OCT4', 'POU5F1', 'SOX2', 'NANOG', 'KLF4', 'MYC', 'LIN28A', 'LIN28B',
            'DPPA2', 'DPPA3', 'DPPA4', 'DPPA5', 'GDF3', 'LEFTY1', 'LEFTY2',
            'NODAL', 'TDGF1', 'ZFP42', 'SALL4', 'UTF1', 'DNMT3L',
            'STELLA', 'ESRRB', 'TBX3', 'STAT3', 'SMAD2', 'SMAD3',
            'CTNNB1', 'TCF7L1', 'LEF1', 'FGF2', 'FGF4', 'FGFR1', 'FGFR2'
        ]
        
        self.gene_categories['pluripotency'] = pluripotency_genes
        print(f"   🌱 Loaded {len(pluripotency_genes)} pluripotency network genes")
        return pluripotency_genes
    
    def create_biologically_informed_gene_set(self, adata) -> List[str]:
        """
        Combine multiple selection criteria for optimal gene set
        """
        print(f"🧠 Creating biologically-informed gene set ({self.target_genes} genes)...")
        
        selected_genes = set()
        
        # 1. Always include transcription factors
        tf_genes = self.load_transcription_factors()
        available_tfs = [g for g in tf_genes if g in adata.var.index]
        selected_genes.update(available_tfs)
        print(f"   ✅ Added {len(available_tfs)} transcription factors")
        
        # 2. Include essential metabolic genes
        metabolic_genes = self.load_metabolic_genes()
        available_metabolic = [g for g in metabolic_genes if g in adata.var.index]
        selected_genes.update(available_metabolic)
        print(f"   ✅ Added {len(available_metabolic)} metabolic genes")
        
        # 3. Include cell cycle genes
        cc_genes = self.load_cell_cycle_genes()
        available_cc = [g for g in cc_genes if g in adata.var.index]
        selected_genes.update(available_cc)
        print(f"   ✅ Added {len(available_cc)} cell cycle genes")
        
        # 4. Include pluripotency markers
        pluri_genes = self.load_pluripotency_genes()
        available_pluri = [g for g in pluri_genes if g in adata.var.index]
        selected_genes.update(available_pluri)
        print(f"   ✅ Added {len(available_pluri)} pluripotency genes")
        
        # 5. Add perturbation-responsive genes
        remaining_slots = max(0, self.target_genes - len(selected_genes))
        if remaining_slots > 500:
            responsive_genes = self.select_perturbation_responsive_genes(
                adata, n_genes=min(500, remaining_slots // 2)
            )
            available_responsive = [g for g in responsive_genes if g in adata.var.index and g not in selected_genes]
            selected_genes.update(available_responsive)
            print(f"   ✅ Added {len(available_responsive)} perturbation-responsive genes")
        
        # 6. Fill remaining slots with HVGs
        remaining_slots = max(0, self.target_genes - len(selected_genes))
        if remaining_slots > 0:
            hvg_genes = self.select_highly_variable_genes(adata, n_top_genes=remaining_slots * 2)
            available_hvgs = [g for g in hvg_genes if g in adata.var.index and g not in selected_genes]
            selected_genes.update(available_hvgs[:remaining_slots])
            print(f"   ✅ Added {len(available_hvgs[:remaining_slots])} highly variable genes")
        
        final_gene_list = list(selected_genes)
        print(f"🎯 Final gene set: {len(final_gene_list)} genes")
        
        # Save gene categories for analysis
        self.selected_genes = selected_genes
        
        return final_gene_list
    
    def save_gene_selection(self, gene_list: List[str], output_path: str):
        """
        Save selected genes to CSV for reproducibility
        """
        df = pd.DataFrame({
            'gene_name': gene_list,
            'gene_index': range(len(gene_list))
        })
        
        # Add category information
        categories = []
        for gene in gene_list:
            if gene in self.gene_categories['transcription_factors']:
                categories.append('transcription_factor')
            elif gene in self.gene_categories['metabolic']:
                categories.append('metabolic')
            elif gene in self.gene_categories['cell_cycle']:
                categories.append('cell_cycle')
            elif gene in self.gene_categories['pluripotency']:
                categories.append('pluripotency')
            elif gene in self.gene_categories['highly_variable']:
                categories.append('highly_variable')
            else:
                categories.append('other')
        
        df['category'] = categories
        df.to_csv(output_path, index=False)
        print(f"💾 Saved gene selection to {output_path}")
        
        # Print summary
        category_counts = df['category'].value_counts()
        print("📊 Gene category breakdown:")
        for category, count in category_counts.items():
            print(f"   {category}: {count} genes")

def select_optimal_genes(adata, target_genes: int = 3000, output_path: str = None):
    """
    Main function to select optimal gene set for training
    """
    selector = BiologicalGeneSelector(target_genes=target_genes)
    selected_genes = selector.create_biologically_informed_gene_set(adata)
    
    if output_path:
        selector.save_gene_selection(selected_genes, output_path)
    
    # Subset the data to selected genes
    gene_mask = adata.var.index.isin(selected_genes)
    adata_subset = adata[:, gene_mask].copy()
    
    print(f"📏 Data shape after gene selection: {adata_subset.shape}")
    print(f"📉 Memory reduction: {adata.n_vars} → {adata_subset.n_vars} genes ({adata_subset.n_vars/adata.n_vars:.1%})")
    
    return adata_subset, selected_genes

if __name__ == "__main__":
    # Test gene selection
    print("🧪 Testing gene selection...")
    
    # This would be run with actual data
    # adata = sc.read_h5ad('data/adata_Training.h5ad')
    # adata_subset, genes = select_optimal_genes(adata, target_genes=3000, output_path='selected_genes.csv')