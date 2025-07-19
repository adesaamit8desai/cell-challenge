# Submission Workflow with Error Prevention

## Overview
This workflow ensures submissions are created correctly and validated before running cell-eval prep, preventing wasted computation time.

## Step-by-Step Process

### Phase 1: Preparation
1. **Verify Requirements**
   ```bash
   # Check that we have all required files
   ls -la data/adata_Training.h5ad
   ls -la models/state_best_model.pt
   ls -la data/pert_counts_Validation.csv
   ```

2. **Check Memory**
   ```bash
   # Ensure sufficient memory for operations
   free -h
   ```

### Phase 2: Small Test (ALWAYS DO THIS FIRST)
1. **Create Small Test Submission**
   ```bash
   python models/create_scaled_submission.py --start 0 --end 5 --max_cells 10
   ```

2. **Validate Test Submission**
   ```bash
   python models/validate_submission.py submission_scaled_test.h5ad
   ```

3. **Test cell-eval prep**
   ```bash
   cell-eval prep -i submission_scaled_test.h5ad
   ```

4. **If test fails, fix issues before proceeding**

### Phase 3: Medium Test
1. **Create Medium Test Submission**
   ```bash
   python models/create_scaled_submission.py --start 0 --end 50 --max_cells 50
   ```

2. **Validate and test prep**
   ```bash
   python models/validate_submission.py submission_scaled_medium.h5ad
   cell-eval prep -i submission_scaled_medium.h5ad
   ```

### Phase 4: Full Submission (Only after tests pass)
1. **Create Full Submission**
   ```bash
   python models/create_scaled_submission.py
   ```

2. **Validate Full Submission**
   ```bash
   python models/validate_submission.py submission_scaled.h5ad
   ```

3. **Run cell-eval prep**
   ```bash
   cell-eval prep -i submission_scaled.h5ad
   ```

## Error Prevention Checklist

### Before Creating Any Submission
- [ ] Check available memory (>8GB recommended)
- [ ] Verify all input files exist
- [ ] Ensure model is trained and saved
- [ ] Check that NTC cells will be included

### After Creating Submission
- [ ] Run validation script
- [ ] Check file size is reasonable
- [ ] Verify NTC cells are present
- [ ] Test cell-eval prep on small subset

### Common Issues and Solutions

#### Issue: Missing NTC cells
**Solution**: Update submission script to always include NTC cells during creation

#### Issue: Memory errors during creation
**Solution**: Use batch processing and monitor memory usage

#### Issue: cell-eval prep fails
**Solution**: Run validation script first to identify issues

#### Issue: File too large
**Solution**: Check if all perturbations are necessary, consider subsetting

## Recovery Procedures

### If Submission Creation Fails
1. **Don't delete partial files** - they may be recoverable
2. **Check error logs** for specific issues
3. **Use smaller test cases** to isolate problems
4. **Verify requirements** before retrying

### If Validation Fails
1. **Fix the specific issues** identified by validation
2. **Re-run with smaller test case**
3. **Only proceed to full submission after test passes**

### If cell-eval prep Fails
1. **Run validation script** to identify issues
2. **Check error messages** carefully
3. **Fix issues** and re-test with small submission
4. **Only retry full submission** after small test passes

## Monitoring and Logging

### Memory Monitoring
```bash
# Monitor memory during creation
watch -n 1 'free -h'
```

### Progress Tracking
```bash
# Check file sizes during creation
ls -lh submission_*.h5ad
```

### Error Logging
```bash
# Capture all output for debugging
python models/create_scaled_submission.py 2>&1 | tee submission_creation.log
```

## Best Practices

1. **Always test with small subsets first**
2. **Validate before running cell-eval prep**
3. **Monitor memory usage during creation**
4. **Keep backup copies of working submissions**
5. **Document any issues and solutions**
6. **Use version control for submission scripts**

## Quick Commands

```bash
# Quick validation
python models/validate_submission.py submission.h5ad

# Quick test submission
python models/create_scaled_submission.py --start 0 --end 5 --max_cells 10

# Quick prep test
cell-eval prep -i submission_test.h5ad
``` 