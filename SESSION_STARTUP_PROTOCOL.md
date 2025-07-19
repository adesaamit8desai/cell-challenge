# SESSION STARTUP PROTOCOL

## ⚠️ MANDATORY: Follow This Protocol at the Start of Every Session

### **Step 1: Read the Mandatory Checklist**
```
I MUST read MANDATORY_CHECKLIST.md before starting any work
```

### **Step 2: Understand the Current State**
```
I MUST understand:
- What is the current goal?
- What has been done so far?
- What are the current issues?
- What is the next step?
```

### **Step 3: Map the Complete Context**
```
I MUST answer:
- What is the end goal?
- What are all the requirements?
- What are the constraints?
- How does this fit into the complete workflow?
```

### **Step 4: Plan the Approach**
```
I MUST plan:
- What is the complete workflow?
- What are the dependencies?
- How will I validate this works?
- What are the potential failure modes?
```

### **Step 5: Start with Small Tests**
```
I MUST:
- Test with small subsets first
- Validate the complete workflow
- Only scale up after small test passes
- Document any issues found
```

## **MANDATORY: Session Startup Questions**

### **Before Starting Any Work, I Must Answer:**

#### **1. What is the current goal?**
```
[Write the specific goal for this session]
```

#### **2. What is the end goal?**
```
[Write the final deliverable]
```

#### **3. What are all the requirements?**
```
[List all functional, technical, and competition requirements]
```

#### **4. What is the complete workflow?**
```
[Map the complete end-to-end workflow]
```

#### **5. What are the dependencies?**
```
[List all components this affects and that affect this]
```

#### **6. How will I validate this works?**
```
[Describe the validation strategy]
```

#### **7. What are the potential failure modes?**
```
[List potential issues and how to handle them]
```

## **MANDATORY: Systems Thinking Check**

### **Before Any Implementation, I Must Verify:**

- [ ] **I understand the complete context** (not just the immediate problem)
- [ ] **I have mapped the end-to-end workflow** (not just individual components)
- [ ] **I understand all requirements and constraints** (not just technical ones)
- [ ] **I have planned validation strategy** (not just assumed it will work)
- [ ] **I will test with small subsets first** (not just jump to full implementation)

## **MANDATORY: Anti-Pattern Prevention**

### **I Will NOT:**
- ❌ **Jump to technical solutions** without understanding requirements
- ❌ **Focus on individual components** without considering the whole system
- ❌ **Test only at the end** instead of validating early
- ❌ **Assume unlimited resources** instead of designing for constraints
- ❌ **Optimize in isolation** without considering cross-dependencies

### **I WILL:**
- ✅ **Start with the end goal** and work backwards
- ✅ **Map the complete workflow** before implementing
- ✅ **Validate early and often** with small tests
- ✅ **Design for the whole system** not just individual parts
- ✅ **Consider all requirements and constraints** before implementing

## **MANDATORY: Session Structure**

### **Every Session Must Follow This Structure:**

#### **Phase 1: Context Understanding (MANDATORY)**
1. **Read the mandatory checklist**
2. **Understand the current state**
3. **Map the complete context**
4. **Plan the approach**

#### **Phase 2: Small Test First (MANDATORY)**
1. **Test with small subsets**
2. **Validate the complete workflow**
3. **Document any issues**
4. **Only scale up after success**

#### **Phase 3: Implementation (Only After Phases 1 & 2)**
1. **Implement the solution**
2. **Validate at each step**
3. **Test the complete workflow**
4. **Document any issues**

#### **Phase 4: Validation (MANDATORY)**
1. **Test the complete workflow**
2. **Validate against all requirements**
3. **Check for any downstream issues**
4. **Document the results**

## **MANDATORY: Enforcement**

### **If I Skip This Protocol:**
- **Stop immediately**
- **Complete the protocol**
- **Re-evaluate the approach**
- **Start over with systems thinking**

### **If I Jump to Implementation:**
- **Stop immediately**
- **Complete the protocol**
- **Re-evaluate the approach**
- **Start over with systems thinking**

### **If I Focus on Individual Components:**
- **Stop immediately**
- **Map the complete system**
- **Re-evaluate the approach**
- **Start over with systems thinking**

## **MANDATORY: Session Documentation**

### **I Must Document:**
1. **What is the goal for this session?**
2. **What is the complete workflow?**
3. **What are the requirements and constraints?**
4. **How will I validate this works?**
5. **What are the potential failure modes?**
6. **What was the result?**

## **EXAMPLE: Session Startup**

### **Session Goal:** Fix the NTC issue in submission generation

### **End Goal:** Valid competition submission that passes cell-eval prep

### **Complete Workflow:** Training → Model → Predictions → Submission → Validation → Competition

### **Requirements:** Must include NTC cells, correct format, all genes, memory efficient

### **Dependencies:** Training data → NTC cells, Model → Prediction format, Submission → cell-eval requirements

### **Validation Strategy:** Test with small subsets first, validate submission format, check all requirements

### **Failure Modes:** Missing NTC cells, wrong format, memory issues, cell-eval prep failure

### **Approach:** Include NTC cells during submission creation, test with small subsets, validate complete workflow

## **MANDATORY: This Protocol is Non-Negotiable**

**I MUST follow this protocol at the start of every session.**
**I MUST complete all sections before starting any work.**
**I MUST validate the complete workflow before scaling up.**
**I MUST document the session results.**

**This protocol is MANDATORY and non-negotiable.** 