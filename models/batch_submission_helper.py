import os
import shutil
import glob
import subprocess
from pathlib import Path

# Parameters
BATCH_SIZE = 10  # Number of perturbations per batch
MAX_CELLS = 200  # Max cells per perturbation (adjust as needed)
ALL_H5AD_DIR = 'all_h5ad_files'  # Directory to collect all .h5ad files

# Get number of perturbations
import pandas as pd
pert_counts_df = pd.read_csv('data/pert_counts_Validation.csv')
N = len(pert_counts_df)

# Create output directory
os.makedirs(ALL_H5AD_DIR, exist_ok=True)

for start in range(0, N, BATCH_SIZE):
    end = min(start + BATCH_SIZE, N)
    print(f'Processing perturbations {start} to {end-1}...')
    # Run the batch
    result = subprocess.run([
        'python', 'models/create_submission.py',
        '--start', str(start),
        '--end', str(end),
        '--max_cells', str(MAX_CELLS)
    ], capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    # Find the temp dir from output
    lines = result.stdout.splitlines()
    temp_dir = None
    for line in lines:
        if 'All perturbation .h5ad files written to:' in line:
            temp_dir = line.split(':', 1)[1].strip()
            break
    if temp_dir and os.path.isdir(temp_dir):
        for f in glob.glob(os.path.join(temp_dir, '*.h5ad')):
            shutil.copy(f, ALL_H5AD_DIR)
        print(f'Copied .h5ad files from {temp_dir} to {ALL_H5AD_DIR}')
    else:
        print('Could not find temp_dir for this batch!')

print('All batches complete. You can now merge all .h5ad files:')
print(f'python models/merge_h5ad_files.py --indir {ALL_H5AD_DIR} --outfile submission.h5ad') 