#!/usr/bin/env python3
"""
Submission Validation Script
===========================

This script validates submission files before running cell-eval prep to catch
common errors early and prevent wasted computation time.

Usage:
    python models/validate_submission.py submission.h5ad
"""

import sys
import argparse
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from pathlib import Path
import psutil
import os

def check_memory_usage():
    """Check if we have enough memory for operations."""
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    print(f"Available memory: {available_gb:.1f} GB")
    return available_gb > 2.0  # Need at least 2GB

def validate_submission(adata_path):
    """Validate a submission file against all requirements."""
    print(f"🔍 Validating submission: {adata_path}")
    
    # Check file exists
    if not Path(adata_path).exists():
        print(f"❌ File not found: {adata_path}")
        return False
    
    # Check file size
    file_size_gb = Path(adata_path).stat().st_size / (1024**3)
    print(f"📁 File size: {file_size_gb:.2f} GB")
    
    if file_size_gb > 10:
        print("⚠️  Warning: File is very large (>10GB)")
    
    # Load AnnData
    print("📖 Loading AnnData...")
    try:
        adata = sc.read_h5ad(adata_path)
        print(f"✅ Loaded AnnData: {adata.n_obs} cells × {adata.n_vars} genes")
    except Exception as e:
        print(f"❌ Failed to load AnnData: {e}")
        return False
    
    # 1. Check basic structure
    print("\n📋 Checking basic structure...")
    
    # Check dimensions
    if adata.n_obs == 0:
        print("❌ No cells in submission")
        return False
    
    if adata.n_vars == 0:
        print("❌ No genes in submission")
        return False
    
    print(f"✅ Dimensions: {adata.n_obs} cells × {adata.n_vars} genes")
    
    # 2. Check required columns in .obs
    print("\n📊 Checking .obs columns...")
    required_cols = ['target_gene']
    for col in required_cols:
        if col not in adata.obs.columns:
            print(f"❌ Missing required column: {col}")
            return False
    print("✅ All required .obs columns present")
    
    # 3. Check for NTC cells (CRITICAL)
    print("\n🎯 Checking for non-targeting control cells...")
    ntc_mask = adata.obs['target_gene'] == 'non-targeting'
    ntc_count = ntc_mask.sum()
    
    if ntc_count == 0:
        print("❌ CRITICAL ERROR: No non-targeting control cells found!")
        print("   This will cause cell-eval prep to fail.")
        return False
    elif ntc_count < 100:
        print(f"⚠️  Warning: Only {ntc_count} NTC cells (recommend at least 100)")
    else:
        print(f"✅ Found {ntc_count} non-targeting control cells")
    
    # 4. Check expression data
    print("\n🧬 Checking expression data...")
    
    # Check for NaN values
    nan_count = np.isnan(adata.X).sum()
    if nan_count > 0:
        print(f"❌ Found {nan_count} NaN values in expression data")
        return False
    print("✅ No NaN values found")
    
    # Check for infinite values
    inf_count = np.isinf(adata.X).sum()
    if inf_count > 0:
        print(f"❌ Found {inf_count} infinite values in expression data")
        return False
    print("✅ No infinite values found")
    
    # Check data type
    if adata.X.dtype != np.float32:
        print(f"⚠️  Warning: Expression data is {adata.X.dtype}, expected float32")
    else:
        print("✅ Expression data is float32")
    
    # 5. Check expression value ranges
    print("\n📈 Checking expression value ranges...")
    min_val = adata.X.min()
    max_val = adata.X.max()
    mean_val = adata.X.mean()
    
    print(f"   Min: {min_val:.4f}")
    print(f"   Max: {max_val:.4f}")
    print(f"   Mean: {mean_val:.4f}")
    
    if min_val < -10 or max_val > 20:
        print("⚠️  Warning: Expression values seem outside normal range")
    else:
        print("✅ Expression values in reasonable range")
    
    # 6. Check perturbation distribution
    print("\n🎯 Checking perturbation distribution...")
    pert_counts = adata.obs['target_gene'].value_counts()
    print(f"   Total perturbations: {len(pert_counts)}")
    print(f"   Most common: {pert_counts.head(3).to_dict()}")
    
    # 7. Check total cell count
    if adata.n_obs > 100000:
        print(f"❌ Too many cells: {adata.n_obs} > 100,000")
        return False
    else:
        print(f"✅ Cell count within limits: {adata.n_obs} ≤ 100,000")
    
    # 8. Memory check
    print("\n💾 Checking memory requirements...")
    if not check_memory_usage():
        print("⚠️  Warning: Low memory available for cell-eval prep")
    
    print("\n✅ All validation checks passed!")
    return True

def main():
    parser = argparse.ArgumentParser(description="Validate submission file before cell-eval prep")
    parser.add_argument("submission_file", help="Path to submission .h5ad file")
    args = parser.parse_args()
    
    success = validate_submission(args.submission_file)
    
    if success:
        print("\n🎉 Submission validation successful!")
        print("   You can now run cell-eval prep:")
        print(f"   cell-eval prep -i {args.submission_file}")
    else:
        print("\n❌ Submission validation failed!")
        print("   Please fix the issues above before running cell-eval prep.")
        sys.exit(1)

if __name__ == "__main__":
    main() 