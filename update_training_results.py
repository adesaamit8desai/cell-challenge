#!/usr/bin/env python3
"""
Update Training Results in README
================================

This script automatically updates the README.md file with the latest training results.
It reads the training report and history from the most recent training run and
updates the "Latest Training Run" section in the README.

Usage:
    python update_training_results.py [training_results_dir]
    
If no directory is provided, it will find the most recent training run automatically.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
import argparse

def find_latest_training_run():
    """Find the most recent training run directory"""
    training_runs = Path("training_runs")
    if not training_runs.exists():
        return None
    
    # Find all directories that match the pattern
    run_dirs = [d for d in training_runs.iterdir() if d.is_dir() and d.name.startswith("scaled_")]
    
    if not run_dirs:
        return None
    
    # Sort by modification time (most recent first)
    run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return run_dirs[0]

def extract_training_info(results_dir):
    """Extract training information from the results directory"""
    results_path = Path(results_dir)
    
    # Read training report
    report_path = results_path / "training_report.md"
    if report_path.exists():
        with open(report_path, 'r') as f:
            report_content = f.read()
        
        # Extract key metrics
        config_match = re.search(r'- \*\*Strategy\*\*: (\w+)', report_content)
        genes_match = re.search(r'- \*\*Target Genes\*\*: (\d+)', report_content)
        final_loss_match = re.search(r'- \*\*Final Loss\*\*: ([0-9.]+)', report_content)
        device_match = re.search(r'- \*\*Device\*\*: (\w+)', report_content)
        
        strategy = config_match.group(1) if config_match else "unknown"
        target_genes = genes_match.group(1) if genes_match else "unknown"
        final_loss = final_loss_match.group(1) if final_loss_match else "unknown"
        device = device_match.group(1) if device_match else "unknown"
    else:
        strategy = target_genes = final_loss = device = "unknown"
    
    # Read training history
    history_path = results_path / "training_history.json"
    if history_path.exists():
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        epochs = len(history.get('epochs', []))
        losses = history.get('losses', [])
        if losses:
            initial_loss = losses[0]
            final_loss = losses[-1]
            loss_improvement = initial_loss - final_loss
        else:
            initial_loss = final_loss = loss_improvement = "unknown"
    else:
        epochs = initial_loss = final_loss = loss_improvement = "unknown"
    
    # Read selected genes
    genes_path = results_path / "selected_genes.csv"
    if genes_path.exists():
        import pandas as pd
        try:
            genes_df = pd.read_csv(genes_path)
            gene_categories = genes_df['category'].value_counts().to_dict()
            gene_breakdown = ", ".join([f"{count} {cat}" for cat, count in gene_categories.items()])
        except:
            gene_breakdown = "unknown"
    else:
        gene_breakdown = "unknown"
    
    # Get directory name and timestamp
    dir_name = results_path.name
    timestamp = results_path.stat().st_mtime
    date_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
    
    return {
        'date': date_str,
        'dir_name': dir_name,
        'strategy': strategy,
        'target_genes': target_genes,
        'epochs': epochs,
        'final_loss': final_loss,
        'initial_loss': initial_loss,
        'loss_improvement': loss_improvement,
        'device': device,
        'gene_breakdown': gene_breakdown
    }

def update_readme_with_results(training_info):
    """Update the README.md file with the latest training results"""
    readme_path = Path("README.md")
    
    if not readme_path.exists():
        print("❌ README.md not found!")
        return
    
    with open(readme_path, 'r') as f:
        content = f.read()
    
    # Create the new training results section
    new_section = f"""### Latest Training Run ({training_info['date']})
**Configuration**: {training_info['strategy'].title()} strategy, {training_info['target_genes']} genes, {training_info['epochs']} epochs
- **Training Time**: Auto-generated from training history
- **Initial Loss**: {training_info['initial_loss']}
- **Final Loss**: {training_info['final_loss']}
- **Loss Improvement**: {training_info['loss_improvement']}
- **Device**: {training_info['device']}
- **Gene Categories**: {training_info['gene_breakdown']}
- **Results Directory**: `training_runs/{training_info['dir_name']}/`

### Previous Test Run (2025-07-17)
**Configuration**: Standard strategy, 200 genes, 50 cells, 2 epochs
- **Training Time**: 1 minute 23 seconds
- **Final Loss**: 815.80
- **Memory Usage**: 0.11 GB
- **Gene Categories**: 60 HVG, 46 TF, 40 CC, 32 Metabolic, 22 Pluripotency
- **Results Directory**: `training_runs/scaled_standard_20250717_173540/`"""
    
    # Replace the existing "Latest Training Run" section
    pattern = r'### Latest Training Run.*?(?=### Previous Test Run)'
    replacement = new_section + '\n\n'
    
    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    else:
        # If pattern not found, add the section after the "🚀 Scaled Training Results" header
        pattern = r'(## 🚀 Scaled Training Results\n\n)'
        replacement = r'\1' + new_section + '\n\n### Previous Test Run (2025-07-17)\n**Configuration**: Standard strategy, 200 genes, 50 cells, 2 epochs\n- **Training Time**: 1 minute 23 seconds\n- **Final Loss**: 815.80\n- **Memory Usage**: 0.11 GB\n- **Gene Categories**: 60 HVG, 46 TF, 40 CC, 32 Metabolic, 22 Pluripotency\n- **Results Directory**: `training_runs/scaled_standard_20250717_173540/`\n\n'
        content = re.sub(pattern, replacement, content)
    
    # Write the updated content
    with open(readme_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Updated README.md with latest training results from {training_info['dir_name']}")

def main():
    parser = argparse.ArgumentParser(description='Update README with latest training results')
    parser.add_argument('results_dir', nargs='?', help='Training results directory (optional)')
    
    args = parser.parse_args()
    
    if args.results_dir:
        results_dir = Path(args.results_dir)
        if not results_dir.exists():
            print(f"❌ Results directory {results_dir} not found!")
            return
    else:
        results_dir = find_latest_training_run()
        if not results_dir:
            print("❌ No training results found in training_runs/")
            return
    
    print(f"📊 Extracting training info from {results_dir}...")
    training_info = extract_training_info(results_dir)
    
    print(f"📝 Updating README.md...")
    update_readme_with_results(training_info)
    
    print(f"🎉 Successfully updated README with training results!")

if __name__ == "__main__":
    main() 