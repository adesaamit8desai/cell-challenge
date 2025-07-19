#!/usr/bin/env python3
"""
Simple H5AD File Merger
=======================

This script merges multiple .h5ad files into a single submission file.
"""

import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import glob
import os

def merge_h5ad_files(input_dir, output_file):
    """
    Merge all .h5ad files in input_dir into a single output file.
    """
    print(f"🔍 Looking for .h5ad files in {input_dir}...")
    
    # Find all .h5ad files
    h5ad_files = glob.glob(os.path.join(input_dir, "*.h5ad"))
    h5ad_files.sort()  # Sort for consistent ordering
    
    if not h5ad_files:
        print(f"❌ No .h5ad files found in {input_dir}")
        return False
    
    print(f"📁 Found {len(h5ad_files)} files:")
    for f in h5ad_files:
        print(f"   {os.path.basename(f)}")
    
    # Load all files
    print("📖 Loading files...")
    adatas = []
    for i, file_path in enumerate(h5ad_files):
        print(f"   [{i+1}/{len(h5ad_files)}] Loading {os.path.basename(file_path)}...")
        try:
            adata = sc.read_h5ad(file_path)
            adatas.append(adata)
            print(f"   ✅ Loaded: {adata.shape}")
        except Exception as e:
            print(f"   ❌ Failed to load {file_path}: {e}")
            return False
    
    # Merge all files
    print("🔄 Merging files...")
    try:
        merged = ad.concat(adatas, axis=0, merge='same')
        print(f"✅ Merged shape: {merged.shape}")
        
        # Ensure data type is float32
        if not isinstance(merged.X, np.ndarray):
            merged.X = merged.X.toarray()
        merged.X = merged.X.astype(np.float32)
        
        # Save merged file
        print(f"💾 Saving to {output_file}...")
        merged.write_h5ad(output_file)
        print(f"✅ Successfully saved: {output_file}")
        
        # Clean up
        for adata in adatas:
            del adata
        del merged
        
        return True
        
    except Exception as e:
        print(f"❌ Error merging files: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Merge multiple .h5ad files")
    parser.add_argument("--indir", type=str, required=True,
                       help="Input directory containing .h5ad files")
    parser.add_argument("--outfile", type=str, required=True,
                       help="Output .h5ad file path")
    
    args = parser.parse_args()
    
    # Check input directory exists
    if not os.path.exists(args.indir):
        print(f"❌ Input directory does not exist: {args.indir}")
        return
    
    # Merge files
    success = merge_h5ad_files(args.indir, args.outfile)
    
    if success:
        print("🎉 Merge completed successfully!")
    else:
        print("❌ Merge failed!")
        exit(1)

if __name__ == "__main__":
    main() 