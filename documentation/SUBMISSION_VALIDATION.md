# Submission Validation Checklist

## Pre-Submission Validation Steps

### 1. **Format Requirements**
- [ ] Submission is in `.h5ad` format
- [ ] Contains exactly 18,080 genes (or subset for testing)
- [ ] Gene order matches `gene_names.csv`
- [ ] Expression values are float32, log1p/integer format
- [ ] Total cells ≤ 100,000

### 2. **Required Cell Types**
- [ ] **NON-TARGETING CONTROL CELLS MUST BE PRESENT**
  - [ ] At least 100 NTC cells included
  - [ ] NTC cells have `target_gene == 'non-targeting'`
  - [ ] NTC cells have realistic baseline expression (not zeros)

### 3. **Data Structure**
- [ ] `.obs` contains required columns:
  - [ ] `target_gene` (string)
  - [ ] `n_cells` (integer)
- [ ] `.var` contains gene information
- [ ] `.X` contains expression matrix (cells × genes)

### 4. **Content Validation**
- [ ] Each perturbation has correct number of cells
- [ ] Target genes exist in gene list
- [ ] No NaN or infinite values
- [ ] Expression values are reasonable (not all zeros, not extreme values)

## Testing Protocol

### Phase 1: Small Test Submission
1. Create submission with first 5 perturbations only
2. Run cell-eval prep on test submission
3. Verify no errors in prep step
4. Check output format is correct

### Phase 2: Medium Test Submission  
1. Create submission with first 50 perturbations
2. Include NTC cells
3. Run cell-eval prep
4. Verify memory usage is reasonable

### Phase 3: Full Submission
1. Only proceed after Phase 1 and 2 pass
2. Monitor memory usage during creation
3. Verify final file size is reasonable
4. Run cell-eval prep on full submission

## Common Error Prevention

### Memory Issues
- [ ] Use batch processing for large datasets
- [ ] Monitor memory usage during creation
- [ ] Use backed mode for large AnnData operations
- [ ] Clean up temporary objects

### Format Issues
- [ ] Verify gene order matches training data
- [ ] Check data types (float32 for expression)
- [ ] Ensure no missing values
- [ ] Validate cell and gene counts

### Missing Requirements
- [ ] **ALWAYS include NTC cells**
- [ ] Verify all required perturbations are present
- [ ] Check that target genes are valid
- [ ] Ensure expression values are biologically reasonable

## Validation Script

Run this before any submission:
```bash
python models/validate_submission.py submission.h5ad
```

## Emergency Recovery

If submission creation fails:
1. **Don't delete existing files** - they may be recoverable
2. **Check error logs** for specific issues
3. **Use smaller test cases** to isolate problems
4. **Verify requirements** before retrying 