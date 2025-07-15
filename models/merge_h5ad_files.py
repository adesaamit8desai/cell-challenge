import anndata as ad
import glob
import argparse
import os

parser = argparse.ArgumentParser(description='Merge per-perturbation .h5ad files into a single AnnData file.')
parser.add_argument('--indir', type=str, required=True, help='Directory containing .h5ad files to merge')
parser.add_argument('--outfile', type=str, default='submission.h5ad', help='Output merged AnnData file')
parser.add_argument('--batch_size', type=int, default=100, help='Number of files to merge at once (for memory efficiency)')
args = parser.parse_args()

indir = args.indir
outfile = args.outfile
batch_size = args.batch_size

files = sorted(glob.glob(os.path.join(indir, '*.h5ad')))
print(f'Found {len(files)} .h5ad files in {indir}')

if len(files) == 0:
    print('No files to merge!')
    exit(1)

# Merge in batches if needed
def batch_merge(file_list, batch_size):
    merged = None
    for i in range(0, len(file_list), batch_size):
        batch_files = file_list[i:i+batch_size]
        print(f'Merging batch {i//batch_size+1}: {len(batch_files)} files')
        batch_adatas = [ad.read_h5ad(f) for f in batch_files]
        batch_merged = ad.concat(batch_adatas, axis=0, join='outer', fill_value=0)
        if merged is None:
            merged = batch_merged
        else:
            merged = ad.concat([merged, batch_merged], axis=0, join='outer', fill_value=0)
        # Optionally, free memory
        del batch_adatas
        del batch_merged
    return merged

merged = batch_merge(files, batch_size)
print(f'Writing merged AnnData to {outfile} ...')
merged.write_h5ad(outfile)
print('Done!') 