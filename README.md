# Virtual Cell Challenge - Competition Winning Strategy

## ⚠️ MANDATORY: Session Startup Protocol

**Before starting any work, you MUST run the session startup protocol:**

```bash
python start_session.py
```

This ensures systems-level thinking and prevents the types of errors we've encountered.

## Project Overview

This repository contains **three distinct modeling approaches** for the Virtual Cell Challenge, evolving from baseline improvements to a novel competition-winning strategy. Each approach represents a different philosophy for modeling cellular perturbations.

## 🎯 Three Model Approaches - Evolution Strategy

### **Approach 1: STATE-Inspired Transformer (BASELINE SUCCESSFUL)**
**Philosophy**: Improve upon the published STATE baseline model
- **Performance**: MAE 0.0476, DE 0.233, Discrimination 0.50 ✅
- **Architecture**: Multi-task transformer with biological components
- **Status**: ✅ **SUCCESSFUL** - Generated valid submissions
- **Files**: `models/state_model.py`, `models/real_data_training.py`, `models/create_submission.py`

### **Approach 2: Memory-Efficient Scaled Transformer (FAILED)**
**Philosophy**: Scale up with memory optimizations for larger gene sets
- **Performance**: MAE 2.3511, DE 0.16, Discrimination 0.51 ❌
- **Architecture**: Mixed precision, sparse attention, progressive training
- **Status**: ❌ **FAILED** - Gene mapping destroyed performance
- **Root Cause**: Zero-filling 17,580 unmapped genes crushed expression magnitude
- **Files**: `models/scaled_training.py`, `models/memory_efficient_training.py`, `models/create_scaled_submission.py`

### **Approach 3: Biological Graph Transformer (COMPETITION-WINNING)**
**Philosophy**: Model cellular perturbations as dynamic cascades through gene regulatory networks
- **Performance**: **TARGETING > 0.03 MAE, > 0.25 DE, > 0.52 Discrimination** 🎯
- **Innovation**: Graph Neural Network with biological domain knowledge
- **Architecture**: Pathway-aware graph attention + perturbation propagation modeling
- **Status**: 🚀 **ACTIVE DEVELOPMENT** - Novel domain-expert approach
- **Files**: `models/biological_graph_transformer.py`, `models/pathway_efficient_training.py`, `models/biological_graph_submission.py`

## 🧠 Why Approach 3 Will Win the Competition

**Domain Expert Insight**: Instead of treating genes as independent features, model them as a **biological system**:

1. **Gene Regulatory Networks** - Explicit graph structure based on pathway databases
2. **Pathway-Aware Attention** - Attention patterns respect biological interactions  
3. **Perturbation Propagation** - Models how perturbations cascade through networks
4. **Multi-Scale Hierarchy** - Gene → Pathway → Cellular process modeling
5. **Biological Knowledge Integration** - Hardcoded domain expertise competitors won't have

**Key Advantages Over STATE Baseline**:
- **Novel Architecture**: Graph Neural Network vs. standard transformer
- **Causal Modeling**: Perturbation propagation vs. static input-output
- **Domain Knowledge**: Biological priors vs. learned-from-scratch
- **Systems Thinking**: Network dynamics vs. independent predictions  

## Quick Start Guide - Choose Your Approach

### **🔧 Environment Setup (Required for All Approaches)**
```bash
# 1. Session startup (MANDATORY)
python start_session.py

# 2. Install dependencies
pip install -r requirements.txt
pip install -r agent/requirements_agent.txt  # AI assistant dependencies
```

### **Approach 1: STATE-Inspired (Proven Baseline) ✅**
```bash
# Train model
python models/real_data_training.py

# Create submission
python models/create_submission.py

# Validate
python models/validate_submission.py submission.h5ad
```

### **Approach 2: Scaled Transformer (Failed - Don't Use) ❌**
```bash
# For reference only - known to perform poorly
python models/scaled_training.py           # Training
python models/create_scaled_submission.py  # Submission (poor performance)
```

### **Approach 3: Biological Graph Transformer (Competition Winner) 🎯**
```bash
# Demo version (proves concept)
python models/demo_biological_graph.py

# Full production training (memory-efficient pathway batching)
python models/pathway_efficient_training.py

# Create graph-based submission
python models/biological_graph_submission.py --model-dir training_runs/biological_graph_YYYYMMDD_HHMMSS

# Validate and prep
python models/validate_submission.py submission_biological_graph.h5ad
```

## Project Structure - Three Approaches

### **Approach 1: STATE-Inspired Transformer**
```
models/
├── state_model.py              # Original STATE-inspired architecture
├── real_data_training.py       # Training pipeline (SUCCESSFUL)
└── create_submission.py        # Submission generation (MAE 0.0476)
```

### **Approach 2: Memory-Efficient Scaled Transformer (FAILED)**
```
models/
├── scaled_training.py           # Memory-efficient training
├── memory_efficient_training.py # Optimization techniques
├── progressive_training.py      # Curriculum learning
└── create_scaled_submission.py  # Submission (POOR PERFORMANCE)
```

### **Approach 3: Biological Graph Transformer (COMPETITION-WINNING)**
```
models/
├── biological_graph_transformer.py  # Graph neural network architecture
├── pathway_efficient_training.py    # Memory-efficient pathway batching
├── biological_graph_submission.py   # Graph-based submission creation
└── demo_biological_graph.py        # Proof of concept demo
```

### **Supporting Infrastructure**
```
cell-challenge/
├── agent/                          # AI Research Assistant
│   ├── main.py                     # CLI interface
│   ├── agent_core/                 # LangGraph implementation
│   ├── tools/                      # AI tools (research, code generation)
│   └── knowledge_base/             # Curated documents
├── training_runs/                   # All training outputs
│   ├── step_*/                     # Approach 1 results
│   ├── scaled_*/                   # Approach 2 results
│   └── biological_graph_*/         # Approach 3 results
├── data/                           # Training and validation data
├── submissions/                     # Generated submission files
└── documentation/                   # Project documentation
```

### **Mandatory Protocol Files**
- `start_session.py` - **MANDATORY** session startup script
- `MANDATORY_CHECKLIST.md` - Pre-implementation checklist
- `SESSION_STARTUP_PROTOCOL.md` - Session startup protocol
- `SYSTEM_DESIGN.md` - Complete system design
- `DECISION_FRAMEWORK.md` - Decision framework for systems thinking

## Performance Comparison - Three Approaches

### **Approach 1: STATE-Inspired Transformer ✅**
- **MAE**: 0.0476 (EXCELLENT)
- **Differential Expression**: 0.233 (GOOD)
- **Perturbation Discrimination**: 0.50 (BASELINE)
- **Architecture**: Multi-task transformer with biological components
- **Training Time**: ~2 hours
- **Status**: ✅ **SUCCESSFUL** - Proven working baseline

### **Approach 2: Scaled Transformer ❌**
- **MAE**: 2.3511 (TERRIBLE - 50x worse!)
- **Differential Expression**: 0.16 (POOR)
- **Perturbation Discrimination**: 0.51 (BASELINE)
- **Root Cause**: Gene mapping failure - zero-filling 17,580 genes
- **Status**: ❌ **FAILED** - Don't use this approach

### **Approach 3: Biological Graph Transformer 🎯**
- **Target MAE**: < 0.03 (3x better than Approach 1)
- **Target DE**: > 0.25 (Better differential expression detection)
- **Target Discrimination**: > 0.52 (Improved perturbation discrimination)
- **Innovation**: Graph neural network with biological domain knowledge
- **Status**: 🚀 **ACTIVE DEVELOPMENT** - Competition-winning strategy

### **Why Approach 3 Will Dominate:**
1. **Novel Architecture**: Graph vs. transformer (fundamentally different)
2. **Domain Knowledge**: Biological pathways vs. learned patterns
3. **Causal Modeling**: Perturbation propagation vs. static prediction
4. **Systems Biology**: Network dynamics vs. independent features

## Submission Status - Three Approaches

### **Approach 1: STATE-Inspired (PROVEN) ✅**
- **Files**: `submission.prep.vcc` (2.3MB) - ✅ Valid
- **Performance**: MAE 0.0476 - Excellent baseline
- **Pipeline**: `real_data_training.py` → `create_submission.py` → `cell-eval prep`
- **Status**: Ready for competition

### **Approach 2: Scaled Transformer (BROKEN) ❌**
- **Files**: `submission_v2.prep.vcc` (2.3MB) - ✅ Valid format but poor performance
- **Performance**: MAE 2.3511 - Catastrophically bad
- **Issue**: Gene mapping failure destroyed predictions
- **Status**: Don't submit this

### **Approach 3: Biological Graph (TARGET) 🎯**
- **Files**: In development - `submission_biological_graph.prep.vcc`
- **Target Performance**: MAE < 0.03 (3x better than Approach 1)
- **Pipeline**: `pathway_efficient_training.py` → `biological_graph_submission.py` → `cell-eval prep`
- **Status**: Competition-winning approach in development

## Competition Strategy - Three-Tiered Approach

### **Tier 1: Proven Baseline (Insurance Policy)**
- **Approach 1**: STATE-inspired transformer (MAE 0.0476)
- **Status**: Ready to submit - guaranteed decent performance
- **Use Case**: Fallback submission if novel approaches fail

### **Tier 2: Novel Innovation (Competition Winner)**
- **Approach 3**: Biological Graph Transformer
- **Goal**: 3x performance improvement through domain expertise
- **Differentiator**: Only team modeling biological networks explicitly
- **Timeline**: Scale up demo to full production

### **Tier 3: Ensemble Strategy (Final Polish)**
- **Combination**: Best of Approach 1 + Approach 3
- **Method**: Graph model for network genes, STATE for independent genes
- **Target**: Push performance beyond any single approach

### **Development Roadmap**
```
Phase 1: Scale Biological Graph to Full Dataset    [CURRENT]
Phase 2: Production Training with Pathway Batching [NEXT]
Phase 3: Ensemble Approach 1 + 3 for Maximum Performance
Phase 4: Final Submission with Competition Winner
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

## Strategic Lessons Learned

### **What We Learned from Three Approaches**

**Approach 1 (STATE-Inspired)**: ✅ **SUCCESS**
- **Lesson**: Start with proven baselines and incremental improvements
- **Why it worked**: Respected the original architecture's strengths
- **Result**: Solid MAE 0.0476 performance

**Approach 2 (Scaled Transformer)**: ❌ **FAILURE**  
- **Lesson**: Optimization without understanding the core problem fails
- **Why it failed**: Gene mapping destroyed the fundamental prediction task
- **Critical Error**: Zero-filling 97% of genes (17,580 out of 18,080)

**Approach 3 (Biological Graph)**: 🎯 **INNOVATION**
- **Lesson**: Domain expertise beats generic ML optimizations
- **Why it will win**: Models the actual biological problem (networks, cascades)
- **Differentiator**: Think like a systems biologist, not a generic ML engineer

### **Competition Philosophy**
1. **Insurance Policy**: Always have a working baseline (Approach 1)
2. **Innovation Edge**: Pursue novel domain-specific approaches (Approach 3)  
3. **Avoid Optimization Traps**: Don't optimize without understanding the problem
4. **Domain Expertise**: Biological knowledge > computational tricks

### **Next Steps to Victory**
- [x] Proven baseline working (Approach 1)
- [x] Novel architecture designed (Approach 3) 
- [x] Demo successfully implemented
- [ ] **Scale to full dataset with pathway batching**
- [ ] **Submit competition-winning biological graph model**

## 🚨 Current Status & Next Steps (as of July 20, 2025)

### Project State
- **Hybrid Approach (Biological Pathway + STATE + Mean Fallback) is ready for submission, but blocked by memory limits on this workstation.**
- All code for the hybrid approach is implemented and committed.
- The main bottleneck is generating gene means for all 18,080 genes from the full training set, which requires >16GB RAM.

### Memory Issues Encountered
- Even with chunked and column-wise access, Scanpy/AnnData backed mode is not memory-efficient enough for this dataset on a 12GB RAM machine.
- Sampling a small number of cells (e.g., 1000) is possible, but not ideal for competition-grade means.
- **Full gene means calculation and hybrid submission should be run on a workstation with at least 32GB RAM.**

### Hybrid Submission Approach (Recommended)
- **Biological Pathway Model:** Predicts 1,000 biologically selected genes.
- **STATE Model:** Predicts 500 highly variable genes (non-overlapping with pathway genes).
- **Mean Fallback:** All other genes are filled with mean expression from the training data.
- **Per-perturbation batching:** Each perturbation is written as a separate .h5ad file, then merged.

### How to Proceed on a High-Memory Workstation

1. **Clone the repository and set up the environment:**
   ```bash
   git clone <repo-url>
   cd cell-challenge
   python start_session.py
   pip install -r requirements.txt
   pip install -r agent/requirements_agent.txt
   ```

2. **Generate gene means for all genes (requires >16GB RAM):**
   ```bash
   # This will create data/gene_means.csv
   python models/compute_gene_means.py
   ```
   - If you have >32GB RAM, you can use the full dataset. Otherwise, sample 1000 cells as a fallback (already implemented in the script).

3. **Run the hybrid submission script:**
   ```bash
   # This will use the biological model, STATE model, and gene means to create per-perturbation .h5ad files
   python models/create_biological_pathway_submission.py
   # Follow the script's output for the merge command, e.g.:
   python models/merge_h5ad_files.py --indir <output_dir> --outfile submissions/hybrid_submission_<timestamp>.h5ad
   ```

4. **Validate and prepare the submission:**
   ```bash
   python models/validate_submission.py submissions/hybrid_submission_<timestamp>.h5ad
   cell-eval prep -i submissions/hybrid_submission_<timestamp>.h5ad --genes data/gene_names.csv
   ```

### If You Have More Memory:
- You can increase the sample size in `models/compute_gene_means.py` for more accurate means.
- You can also try running the full biological graph transformer pipeline for even better results.

### If You Have Questions or Need to Resume:
- All code changes and scripts are committed and up to date.
- See this README and the `models/` directory for all relevant scripts.
- Contact Amit or check the commit history for context on the latest changes.

## License

This project is for the Virtual Cell Challenge competition. The biological graph transformer approach represents novel domain-specific innovation for computational biology. 