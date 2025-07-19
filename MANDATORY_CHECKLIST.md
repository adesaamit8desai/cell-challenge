# MANDATORY PRE-IMPLEMENTATION CHECKLIST

## ⚠️ THIS CHECKLIST MUST BE COMPLETED BEFORE ANY IMPLEMENTATION

### **Step 1: End Goal Definition**
- [ ] **What is the final deliverable?** (competition submission, model, etc.)
- [ ] **What are all the requirements?** (functional, technical, competition)
- [ ] **What are the constraints?** (memory, time, data, etc.)
- [ ] **How does this fit into the complete workflow?**

### **Step 2: System Mapping**
- [ ] **Have I mapped the complete end-to-end workflow?**
- [ ] **What components does this affect?**
- [ ] **What components affect this?**
- [ ] **What are the cross-dependencies?**
- [ ] **What are the failure modes?**

### **Step 3: Requirements Validation**
- [ ] **Does this meet all competition requirements?**
- [ ] **Does this handle all technical constraints?**
- [ ] **Does this work with existing components?**
- [ ] **Can this be validated and tested?**

### **Step 4: Design for Whole System**
- [ ] **Does this solution work across the entire pipeline?**
- [ ] **Does this handle real-world constraints?**
- [ ] **Does this scale appropriately?**
- [ ] **Is this maintainable and reproducible?**

### **Step 5: Validation Strategy**
- [ ] **How will I test this works end-to-end?**
- [ ] **What are the validation checkpoints?**
- [ ] **How will I catch issues early?**
- [ ] **What are the recovery procedures?**

## **MANDATORY: Answer These Questions Before Starting**

### **1. What is the end goal?**
```
[Write the specific end goal here]
```

### **2. What are all the requirements?**
```
[List all functional, technical, and competition requirements]
```

### **3. What is the complete workflow?**
```
[Map the complete end-to-end workflow]
```

### **4. What are the dependencies?**
```
[List all components this affects and that affect this]
```

### **5. How will I validate this works?**
```
[Describe the validation strategy]
```

## **MANDATORY: Small Test First**
- [ ] **I will test with a small subset first**
- [ ] **I will validate the complete workflow**
- [ ] **I will only scale up after small test passes**
- [ ] **I will document any issues found**

## **MANDATORY: Systems Thinking Questions**
Before any implementation, I must answer:

1. **Does this solve the right problem?** (not just a symptom)
2. **Does this work with the complete system?** (not just in isolation)
3. **Does this meet all requirements?** (not just technical ones)
4. **Can this be validated?** (not just assumed to work)
5. **What are the failure modes?** (not just success path)

## **ANTI-PATTERNS: I WILL NOT DO THESE**
- ❌ **Jump to technical solutions** without understanding requirements
- ❌ **Optimize individual components** without considering the whole system
- ❌ **Test only at the end** instead of validating early
- ❌ **Focus on technical elegance** over meeting requirements
- ❌ **Assume unlimited resources** instead of designing for constraints

## **ENFORCEMENT: How to Use This Checklist**

### **Before Any Code Changes:**
1. **Complete this checklist in full**
2. **Answer all mandatory questions**
3. **Map the complete workflow**
4. **Plan validation strategy**
5. **Only then start implementation**

### **If I Skip This Checklist:**
- **Stop immediately**
- **Complete the checklist**
- **Re-evaluate the approach**
- **Start over with systems thinking**

## **EXAMPLE: The NTC Issue (What I Should Have Done)**

### **Step 1: End Goal Definition**
- **End goal:** Valid competition submission that passes cell-eval prep
- **Requirements:** Must include NTC cells, correct format, all genes
- **Constraints:** Memory efficient, scalable, reproducible
- **Workflow:** Training → Model → Predictions → Submission → Validation → Competition

### **Step 2: System Mapping**
- **Complete workflow:** Data → Training → Model → Predictions → Submission → cell-eval prep
- **Dependencies:** Training data → NTC cells, Model → Prediction format, Submission → cell-eval requirements
- **Failure modes:** Missing NTC cells, wrong format, memory issues

### **Step 3: Requirements Validation**
- **Competition requirements:** NTC cells required, specific format needed
- **Technical constraints:** Memory efficient, scalable
- **Cross-component:** Works with training data, model, validation

### **Step 4: Design for Whole System**
- **Include NTC cells during submission creation** (not post-hoc)
- **Design submission format to meet cell-eval requirements**
- **Use memory-efficient batch processing**
- **Validate at each step**

### **Step 5: Validation Strategy**
- **Test with small subsets first**
- **Validate submission format before cell-eval prep**
- **Check all requirements are met**
- **Plan recovery for failures**

## **RESULT: This Would Have Prevented the NTC Issue**
By following this checklist, I would have:
1. **Identified that NTC cells are required** (Step 3)
2. **Designed submission creation to include NTCs** (Step 4)
3. **Tested with small subsets first** (Step 5)
4. **Avoided the post-hoc merge issues** entirely

## **MANDATORY: I Must Reference This Checklist**
Before any implementation work, I must:
1. **Read this checklist**
2. **Complete all sections**
3. **Answer all questions**
4. **Only then proceed with implementation**

**This checklist is MANDATORY and non-negotiable.** 