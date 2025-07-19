# Submissions

This directory contains all generated submission files for the Virtual Cell Challenge.

## Current Submissions

### **submission.prep.vcc** (Original)
- **Size**: 2.3MB
- **Model**: Scaled transformer with sparse attention
- **Status**: ✅ Valid competition submission
- **Generation Date**: July 15, 2024
- **Validation**: Passes cell-eval prep validation

### **submission_v2.prep.vcc** (Version 2)
- **Size**: 2.3MB
- **Model**: Same as original (copy for versioning)
- **Status**: ✅ Valid competition submission
- **Generation Date**: July 18, 2024
- **Validation**: Passes cell-eval prep validation

## Submission Pipeline

### **Generation Process**
1. **Model Training**: `python models/scaled_training.py`
2. **Prediction Generation**: `python models/create_scaled_submission.py`
3. **Validation**: `python models/validate_submission.py`
4. **Final Format**: `cell-eval prep` → `.vcc` file

### **Validation Checklist**
- ✅ Contains NTC (non-targeting control) cells
- ✅ Correct gene order matching gene_names.csv
- ✅ Proper data types (float32, log1p/integer)
- ✅ Total cells ≤ 100,000
- ✅ Passes cell-eval prep validation

## Future Submissions

### **Naming Convention**
- `submission_v{version}.prep.vcc` - Versioned submissions
- `submission_{model_type}_v{version}.prep.vcc` - Model-specific submissions

### **Version Control**
- Each new model approach gets a new version
- All submissions are validated before committing
- Version history is documented in this README

## File Details

### **Format**
- **Extension**: `.prep.vcc` (Virtual Cell Challenge format)
- **Size**: ~2.3MB (compressed format)
- **Content**: Gene expression predictions for competition

### **Validation**
All submissions are validated using:
```bash
python models/validate_submission.py {submission_file}
```

### **Competition Submission**
Files in this directory are ready for direct submission to the Virtual Cell Challenge competition. 