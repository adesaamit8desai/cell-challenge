# Virtual Cell Challenge - Complete Analysis
# All Parts from Colab (1, 2, 3) - Adapted for Cursor

# =============================================================================
# PART 1: ENVIRONMENT SETUP
# =============================================================================

# Import all necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up plotting
plt.rcParams['figure.figsize'] = (10, 6)
sns.set_style("whitegrid")

print("✅ Environment setup complete!")
print(f"🐼 Pandas version: {pd.__version__}")
print(f"🔢 NumPy version: {np.__version__}")

# =============================================================================
# PART 2: LOAD AND INSPECT CHALLENGE DATA
# =============================================================================

# Step 1: Check if files are available
print(f"\n📋 STEP 1: Check Available Files")
print("="*50)

# Update paths for our folder structure
data_dir = Path("data")
expected_files = ['gene_names.csv', 'pert_counts_Validation.csv', 'adata_Training.h5ad']
available_files = []

for filename in expected_files:
    filepath = data_dir / filename
    if filepath.exists():
        file_size = filepath.stat().st_size / (1024*1024)  # Size in MB
        print(f"✅ {filename} - {file_size:.1f} MB")
        available_files.append(filename)
    else:
        print(f"❌ {filename} - Not found")

print(f"\nFound {len(available_files)}/{len(expected_files)} files")

# Step 2: Load and inspect gene_names.csv
print(f"\n🧬 STEP 2: Inspect Gene Names File")
print("="*50)

if 'gene_names.csv' in available_files:
    gene_names_df = pd.read_csv(data_dir / 'gene_names.csv')
    
    print(f"📊 Shape: {gene_names_df.shape}")
    print(f"📋 Columns: {list(gene_names_df.columns)}")
    print(f"\n🔍 First 10 rows:")
    print(gene_names_df.head(10))
    
    print(f"\n📈 Data info:")
    print(gene_names_df.info())
    
    if gene_names_df.shape[0] > 10:
        print(f"\n🔍 Last 5 rows:")
        print(gene_names_df.tail())
        
    # Check for duplicates
    if len(gene_names_df.columns) > 0:
        first_col = gene_names_df.columns[0]
        duplicates = gene_names_df[first_col].duplicated().sum()
        print(f"\n🔄 Duplicate entries in '{first_col}': {duplicates}")
else:
    print("❌ gene_names.csv not found")

# Step 3: Load and inspect pert_counts_Validation.csv
print(f"\n🎯 STEP 3: Inspect Validation Perturbation Counts")
print("="*50)

if 'pert_counts_Validation.csv' in available_files:
    pert_validation_df = pd.read_csv(data_dir / 'pert_counts_Validation.csv')
    
    print(f"📊 Shape: {pert_validation_df.shape}")
    print(f"📋 Columns: {list(pert_validation_df.columns)}")
    print(f"\n🔍 First 10 rows:")
    print(pert_validation_df.head(10))
    
    print(f"\n📈 Data info:")
    print(pert_validation_df.info())
    
    # Basic statistics if numeric columns exist
    numeric_cols = pert_validation_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\n📊 Summary statistics:")
        print(pert_validation_df[numeric_cols].describe())
    
    # Look for patterns in the data
    print(f"\n🔍 Unique values per column:")
    for col in pert_validation_df.columns:
        unique_count = pert_validation_df[col].nunique()
        print(f"   {col}: {unique_count} unique values")
        if unique_count <= 10:  # Show values if few unique ones
            print(f"      Values: {sorted(pert_validation_df[col].unique())}")
else:
    print("❌ pert_counts_Validation.csv not found")

# Step 4: Check for training data (without loading if large)
print(f"\n🧬 STEP 4: Check Training Data Status")
print("="*50)

training_file = data_dir / 'adata_Training.h5ad'
if training_file.exists():
    file_size_gb = training_file.stat().st_size / (1024**3)
    print(f"✅ adata_Training.h5ad found! ({file_size_gb:.2f} GB)")
    print(f"   (Will load this when needed for model training)")
else:
    print(f"❌ adata_Training.h5ad not found")

# =============================================================================
# PART 3: PERTURBATION ANALYSIS AND TRAINING DATA PREPARATION
# =============================================================================

# Step 1: Ensure data is loaded (reload if needed)
print(f"\n📂 STEP 1: Ensure Data is Available")
print("="*60)

# Check if we have the data, if not reload it
if 'pert_validation_df' not in locals() and 'pert_counts_Validation.csv' in available_files:
    print("⏳ Loading validation data...")
    pert_validation_df = pd.read_csv(data_dir / 'pert_counts_Validation.csv')
    print(f"✅ Loaded validation data: {pert_validation_df.shape}")

if 'gene_names_df' not in locals() and 'gene_names.csv' in available_files:
    print("⏳ Loading gene names...")
    gene_names_df = pd.read_csv(data_dir / 'gene_names.csv')
    print(f"✅ Loaded gene names: {gene_names_df.shape}")

print("📊 Data ready for analysis!")

# Step 2: Analyze the validation perturbations
print(f"\n🎯 STEP 2: Analyze Validation Perturbations")
print("="*60)

if 'pert_validation_df' in locals():
    print(f"📊 Validation Set Overview:")
    print(f"   - Number of perturbations: {len(pert_validation_df)}")
    print(f"   - Total cells across all perturbations: {pert_validation_df['n_cells'].sum():,}")
    print(f"   - Average cells per perturbation: {pert_validation_df['n_cells'].mean():.1f}")

    # Analyze cell count distribution
    print(f"\n📈 Cell Count Distribution:")
    print(f"   - Min cells per perturbation: {pert_validation_df['n_cells'].min()}")
    print(f"   - Max cells per perturbation: {pert_validation_df['n_cells'].max()}")
    print(f"   - Median cells per perturbation: {pert_validation_df['n_cells'].median()}")
    print(f"   - Std deviation: {pert_validation_df['n_cells'].std():.1f}")

    # Show the perturbation targets
    print(f"\n🧬 Validation Target Genes:")
    target_genes = sorted(pert_validation_df['target_gene'].tolist())
    print(f"   First 10: {target_genes[:10]}")
    print(f"   Last 10: {target_genes[-10:]}")

    # UMI analysis
    print(f"\n🔬 UMI per Cell Analysis:")
    print(f"   - Average median UMI: {pert_validation_df['median_umi_per_cell'].mean():.0f}")
    print(f"   - UMI range: {pert_validation_df['median_umi_per_cell'].min():.0f} - {pert_validation_df['median_umi_per_cell'].max():.0f}")

# Step 3: Analyze gene names structure
print(f"\n🧬 STEP 3: Analyze Gene Names")
print("="*60)

if 'gene_names_df' in locals():
    # Fix the gene names dataframe (the column name is the first gene)
    gene_names_fixed = gene_names_df.iloc[:, 0].tolist()  # Get the actual gene names
    print(f"📊 Gene Information:")
    print(f"   - Total genes: {len(gene_names_fixed):,}")
    print(f"   - First 10 genes: {gene_names_fixed[:10]}")
    print(f"   - Last 10 genes: {gene_names_fixed[-10:]}")

    # Look for different gene types
    mitochondrial_genes = [gene for gene in gene_names_fixed if gene.startswith('MT-')]
    ribosomal_genes = [gene for gene in gene_names_fixed if gene.startswith(('RPS', 'RPL'))]

    print(f"\n🔍 Gene Type Analysis:")
    print(f"   - Mitochondrial genes (MT-*): {len(mitochondrial_genes)}")
    print(f"   - Ribosomal genes (RPS*/RPL*): {len(ribosomal_genes)}")
    print(f"   - Other genes: {len(gene_names_fixed) - len(mitochondrial_genes) - len(ribosomal_genes)}")

    if mitochondrial_genes:
        print(f"   - Mitochondrial examples: {mitochondrial_genes[:5]}")

# Step 4: Check which validation genes are in our gene list
print(f"\n🔍 STEP 4: Validation Gene Coverage")
print("="*60)

if 'pert_validation_df' in locals() and 'gene_names_df' in locals():
    validation_targets = set(pert_validation_df['target_gene'].tolist())
    available_genes = set(gene_names_fixed)

    genes_in_list = validation_targets.intersection(available_genes)
    genes_missing = validation_targets - available_genes

    print(f"📊 Gene Coverage Analysis:")
    print(f"   - Validation targets: {len(validation_targets)}")
    print(f"   - Targets in gene list: {len(genes_in_list)} ({len(genes_in_list)/len(validation_targets)*100:.1f}%)")
    print(f"   - Missing targets: {len(genes_missing)}")

    if genes_missing:
        print(f"   - Missing genes: {sorted(list(genes_missing))}")

# Step 5: Create visualizations
print(f"\n📊 STEP 5: Create Validation Visualizations")
print("="*60)

if 'pert_validation_df' in locals():
    # Create comprehensive plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Validation Set Analysis', fontsize=16, fontweight='bold')

    # 1. Cell count distribution
    axes[0,0].hist(pert_validation_df['n_cells'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].set_xlabel('Number of Cells per Perturbation')
    axes[0,0].set_ylabel('Number of Perturbations')
    axes[0,0].set_title('Distribution of Cell Counts')
    axes[0,0].grid(True, alpha=0.3)

    # 2. UMI distribution
    axes[0,1].hist(pert_validation_df['median_umi_per_cell'], bins=15, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0,1].set_xlabel('Median UMI per Cell')
    axes[0,1].set_ylabel('Number of Perturbations')
    axes[0,1].set_title('Distribution of UMI Counts')
    axes[0,1].grid(True, alpha=0.3)

    # 3. Scatter plot: cells vs UMI
    axes[1,0].scatter(pert_validation_df['n_cells'], pert_validation_df['median_umi_per_cell'], 
                       alpha=0.7, c='green', s=60)
    axes[1,0].set_xlabel('Number of Cells')
    axes[1,0].set_ylabel('Median UMI per Cell')
    axes[1,0].set_title('Cells vs UMI per Perturbation')
    axes[1,0].grid(True, alpha=0.3)

    # 4. Top perturbations by cell count
    top_perts = pert_validation_df.nlargest(10, 'n_cells')
    y_pos = range(len(top_perts))
    axes[1,1].barh(y_pos, top_perts['n_cells'], color='orange', alpha=0.7)
    axes[1,1].set_yticks(y_pos)
    axes[1,1].set_yticklabels(top_perts['target_gene'], fontsize=8)
    axes[1,1].set_xlabel('Number of Cells')
    axes[1,1].set_title('Top 10 Perturbations by Cell Count')
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# Step 6: Summary of what we learned
print(f"\n🎯 STEP 6: Summary and Next Steps")
print("="*60)

print(f"📋 What We Know So Far:")
if 'gene_names_df' in locals():
    print(f"✅ Gene Universe: {len(gene_names_fixed):,} genes")
if 'pert_validation_df' in locals():
    print(f"✅ Validation Set: 50 perturbations, {pert_validation_df['n_cells'].sum():,} total cells")
    print(f"✅ Data Quality: ~{pert_validation_df['median_umi_per_cell'].mean():.0f} UMI per cell (good depth)")
if 'genes_in_list' in locals():
    print(f"✅ Gene Coverage: {len(genes_in_list)}/{len(validation_targets)} validation targets in gene list")

print(f"\n🔮 Expected Training Data:")
print(f"   - ~150 perturbations (3x validation)")
print(f"   - ~150,000 cells total (estimate)")
if 'gene_names_df' in locals():
    print(f"   - Same {len(gene_names_fixed):,} genes")
print(f"   - Similar UMI depth")

print(f"\n🚀 Next Steps:")
print(f"1. ✅ Data structure understood")
print(f"2. 🔄 Ready for model architecture planning")
print(f"3. 📊 Ready for feature engineering")
print(f"4. 🧬 Ready to think about perturbation prediction strategies")

print(f"\n💡 Ready for Part 4: Model Planning and Architecture!")
print(f"   Part 4 will cover: model approaches, feature engineering, and training strategy")