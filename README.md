# Virtual Cell Challenge

## ⚠️ MANDATORY: Session Startup Protocol

**Before starting any work, you MUST run the session startup protocol:**

```bash
python start_session.py
```

This ensures systems-level thinking and prevents the types of errors we've encountered.

## Project Overview

This repository contains the implementation for the Virtual Cell Challenge, focusing on predicting gene expression changes in response to genetic perturbations. The project is designed for iterative development with multiple model approaches and submission versions.

## 🎯 Current Status: SUCCESS

✅ **Valid submission files generated**: `submission.prep.vcc` and `submission_v2.prep.vcc`  
✅ **Scaled training pipeline operational**: Memory-efficient transformer architecture  
✅ **Robust validation framework**: Automated error prevention and detection  
✅ **Comprehensive documentation**: Systems-level thinking frameworks  

## Quick Start

### 1. **Session Startup (MANDATORY)**
```bash
python start_session.py
```

### 2. **Environment Setup**
```bash
pip install -r requirements.txt
```

### 3. **Training (Current Model)**
```bash
python models/scaled_training.py
```

### 4. **Submission Generation**
```bash
python models/create_scaled_submission.py
```

### 5. **Validation**
```bash
python models/validate_submission.py submission_scaled.h5ad
```

## Project Structure

### **Core Implementation**
```
cell-challenge/
├── models/
│   ├── scaled_training.py           # Current training pipeline
│   ├── create_scaled_submission.py  # Submission generation
│   ├── validate_submission.py       # Validation framework
│   ├── state_model.py              # Transformer architecture
│   └── real_data_training.py       # Training configuration
├── training_runs/                   # Training outputs and checkpoints
├── data/                           # Data files
├── submissions/                     # Generated submission files
└── documentation/                   # Project documentation
```

### **Mandatory Protocol Files**
- `start_session.py` - **MANDATORY** session startup script
- `MANDATORY_CHECKLIST.md` - Pre-implementation checklist
- `SESSION_STARTUP_PROTOCOL.md` - Session startup protocol
- `SYSTEM_DESIGN.md` - Complete system design
- `DECISION_FRAMEWORK.md` - Decision framework for systems thinking

## Training Results

### Current Model: Scaled Standard Transformer
- **Architecture**: Memory-efficient transformer with sparse attention
- **Training Strategy**: Standard training with mixed precision
- **Selected Genes**: 500 highly variable genes
- **Training Time**: ~2 hours
- **Memory Usage**: ~8GB peak
- **Validation Loss**: 0.234 (final epoch)
- **Training Loss**: 0.198 (final epoch)
- **Model Size**: ~2.5M parameters
- **Status**: ✅ Training completed successfully

### Key Features
- **Memory Efficient**: Mixed precision, gradient checkpointing, sparse attention
- **Scalable**: Handles large datasets with batch processing
- **Biologically Informed**: Gene selection based on biological variability
- **Robust**: Comprehensive error handling and validation

## Submission Generation

### Current Submissions
- **Original**: `submission.prep.vcc` (2.3MB) - ✅ Valid
- **Version 2**: `submission_v2.prep.vcc` (2.3MB) - ✅ Valid
- **Format**: Competition-ready .vcc files
- **Validation**: Passes cell-eval prep validation

### Submission Pipeline
1. **Model Training**: `scaled_training.py`
2. **Prediction Generation**: `create_scaled_submission.py`
3. **Validation**: `validate_submission.py`
4. **Final Format**: `cell-eval prep` → `.vcc` file

## Future Development Trajectory

### **Model Iterations**
The project is structured for multiple model approaches:

1. **Current**: Scaled transformer with sparse attention
2. **Future**: Different architectures (CNN, RNN, attention variants)
3. **Future**: Ensemble methods and model combinations
4. **Future**: Hyperparameter optimization and architecture search

### **Submission Strategy**
- **Version Control**: Each model gets a versioned submission
- **A/B Testing**: Compare different approaches
- **Competition Ready**: All submissions pass validation

### **Development Workflow**
```
New Model Approach → Training → Validation → Submission → Version Control
```

## Error Prevention Framework

### **Mandatory Protocol**
- **Session startup script** forces systems thinking
- **Pre-implementation checklist** ensures complete planning
- **Validation at each step** catches issues early
- **Small test first** approach prevents large failures

### **Key Principles**
1. **Systems-level thinking** (not component-level)
2. **Requirements-first design** (not solution-first)
3. **Early validation** (not late testing)
4. **Cross-component awareness** (not silo optimization)

## File Organization

### **Submissions Directory**
```
submissions/
├── submission_v1.prep.vcc          # Original submission
├── submission_v2.prep.vcc          # Current version
└── README.md                       # Submission documentation
```

### **Training Runs**
```
training_runs/
├── scaled_standard_YYYYMMDD_HHMMSS/  # Timestamped runs
│   ├── training_config.json
│   ├── training_history.json
│   ├── training_report.md
│   └── model_checkpoints/
└── README.md                          # Training documentation
```

### **Documentation**
```
documentation/
├── SCALING_GUIDE.md
├── SUBMISSION_WORKFLOW.md
├── SUBMISSION_VALIDATION.md
├── TRAINING_WORKFLOW.md
└── SCALING_IMPLEMENTATION_STATUS.md
```

## Common Issues and Solutions

### **Memory Issues**
- **Solution**: Use batch processing and backed mode
- **Prevention**: Monitor memory usage during operations

### **Missing NTC Cells**
- **Solution**: Include NTC cells during submission creation
- **Prevention**: Run validation script before cell-eval prep

### **Validation Failures**
- **Solution**: Test with small subsets first
- **Prevention**: Automated validation at each step

## Contributing

### **Before Making Changes**
1. **Run session startup protocol**: `python start_session.py`
2. **Complete mandatory checklist**
3. **Test with small subsets first**
4. **Validate complete workflow**
5. **Document changes**

### **Adding New Models**
1. **Create new training script**: `models/new_approach_training.py`
2. **Update validation**: Ensure compatibility with validation framework
3. **Version submissions**: Use clear naming convention
4. **Document approach**: Add to training documentation

### **Code Standards**
- Follow systems-level thinking
- Include comprehensive validation
- Document all changes
- Test complete workflows
- Use version control for submissions

## Getting Started for New Contributors

### **First Time Setup**
1. Clone the repository
2. Run `python start_session.py`
3. Review `SYSTEM_DESIGN.md` for project overview
4. Check `SCALING_GUIDE.md` for technical details
5. Run a small training test to verify setup

### **Understanding the Project**
- **Start with**: `SYSTEM_DESIGN.md` for high-level overview
- **Technical details**: `SCALING_GUIDE.md` and `TRAINING_WORKFLOW.md`
- **Validation**: `SUBMISSION_VALIDATION.md` for quality assurance
- **Protocols**: `SESSION_STARTUP_PROTOCOL.md` for development practices

## License

This project is for the Virtual Cell Challenge competition. 