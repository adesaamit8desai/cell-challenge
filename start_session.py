#!/usr/bin/env python3
"""
Session Startup Script
=====================

This script forces me to complete the mandatory checklist at the start of every session.
It ensures I follow systems-level thinking before starting any work.

Usage:
    python start_session.py
"""

import os
import sys
from pathlib import Path

def print_header():
    print("=" * 80)
    print("🚀 SESSION STARTUP PROTOCOL")
    print("=" * 80)
    print("⚠️  MANDATORY: Complete this protocol before starting any work")
    print("=" * 80)

def check_mandatory_files():
    """Check that mandatory files exist."""
    mandatory_files = [
        "MANDATORY_CHECKLIST.md",
        "SESSION_STARTUP_PROTOCOL.md",
        "SYSTEM_DESIGN.md",
        "DECISION_FRAMEWORK.md"
    ]
    
    missing_files = []
    for file in mandatory_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing mandatory files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure all mandatory files are present.")
        return False
    
    print("✅ All mandatory files present")
    return True

def prompt_for_goal():
    """Prompt for session goal."""
    print("\n" + "=" * 80)
    print("📋 SESSION GOAL DEFINITION")
    print("=" * 80)
    
    goal = input("What is the current goal for this session? ").strip()
    if not goal:
        print("❌ Session goal is required!")
        return None
    
    print(f"✅ Session goal: {goal}")
    return goal

def prompt_for_end_goal():
    """Prompt for end goal."""
    print("\n" + "=" * 80)
    print("🎯 END GOAL DEFINITION")
    print("=" * 80)
    
    end_goal = input("What is the final deliverable/end goal? ").strip()
    if not end_goal:
        print("❌ End goal is required!")
        return None
    
    print(f"✅ End goal: {end_goal}")
    return end_goal

def prompt_for_requirements():
    """Prompt for requirements."""
    print("\n" + "=" * 80)
    print("📋 REQUIREMENTS DEFINITION")
    print("=" * 80)
    
    print("List all requirements (functional, technical, competition):")
    requirements = []
    while True:
        req = input("Requirement (or 'done' to finish): ").strip()
        if req.lower() == 'done':
            break
        if req:
            requirements.append(req)
    
    if not requirements:
        print("❌ At least one requirement is required!")
        return None
    
    print("✅ Requirements:")
    for i, req in enumerate(requirements, 1):
        print(f"   {i}. {req}")
    
    return requirements

def prompt_for_workflow():
    """Prompt for complete workflow."""
    print("\n" + "=" * 80)
    print("🔄 COMPLETE WORKFLOW MAPPING")
    print("=" * 80)
    
    workflow = input("Map the complete end-to-end workflow: ").strip()
    if not workflow:
        print("❌ Complete workflow is required!")
        return None
    
    print(f"✅ Complete workflow: {workflow}")
    return workflow

def prompt_for_dependencies():
    """Prompt for dependencies."""
    print("\n" + "=" * 80)
    print("🔗 DEPENDENCIES MAPPING")
    print("=" * 80)
    
    print("List all components this affects and that affect this:")
    dependencies = []
    while True:
        dep = input("Dependency (or 'done' to finish): ").strip()
        if dep.lower() == 'done':
            break
        if dep:
            dependencies.append(dep)
    
    if not dependencies:
        print("❌ At least one dependency is required!")
        return None
    
    print("✅ Dependencies:")
    for i, dep in enumerate(dependencies, 1):
        print(f"   {i}. {dep}")
    
    return dependencies

def prompt_for_validation():
    """Prompt for validation strategy."""
    print("\n" + "=" * 80)
    print("✅ VALIDATION STRATEGY")
    print("=" * 80)
    
    validation = input("How will you validate this works? ").strip()
    if not validation:
        print("❌ Validation strategy is required!")
        return None
    
    print(f"✅ Validation strategy: {validation}")
    return validation

def prompt_for_failure_modes():
    """Prompt for failure modes."""
    print("\n" + "=" * 80)
    print("⚠️  FAILURE MODES")
    print("=" * 80)
    
    print("List potential failure modes and how to handle them:")
    failure_modes = []
    while True:
        fm = input("Failure mode (or 'done' to finish): ").strip()
        if fm.lower() == 'done':
            break
        if fm:
            failure_modes.append(fm)
    
    if not failure_modes:
        print("❌ At least one failure mode is required!")
        return None
    
    print("✅ Failure modes:")
    for i, fm in enumerate(failure_modes, 1):
        print(f"   {i}. {fm}")
    
    return failure_modes

def prompt_for_small_test():
    """Prompt for small test plan."""
    print("\n" + "=" * 80)
    print("🧪 SMALL TEST PLAN")
    print("=" * 80)
    
    small_test = input("What small test will you run first? ").strip()
    if not small_test:
        print("❌ Small test plan is required!")
        return None
    
    print(f"✅ Small test plan: {small_test}")
    return small_test

def systems_thinking_check():
    """Verify systems thinking approach."""
    print("\n" + "=" * 80)
    print("🧠 SYSTEMS THINKING CHECK")
    print("=" * 80)
    
    checks = [
        "I understand the complete context (not just the immediate problem)",
        "I have mapped the end-to-end workflow (not just individual components)",
        "I understand all requirements and constraints (not just technical ones)",
        "I have planned validation strategy (not just assumed it will work)",
        "I will test with small subsets first (not just jump to full implementation)"
    ]
    
    for i, check in enumerate(checks, 1):
        response = input(f"{i}. {check} (y/n): ").strip().lower()
        if response != 'y':
            print(f"❌ Check {i} failed! Please reconsider your approach.")
            return False
    
    print("✅ All systems thinking checks passed!")
    return True

def anti_pattern_check():
    """Check for anti-patterns."""
    print("\n" + "=" * 80)
    print("🚫 ANTI-PATTERN CHECK")
    print("=" * 80)
    
    anti_patterns = [
        "Jump to technical solutions without understanding requirements",
        "Focus on individual components without considering the whole system",
        "Test only at the end instead of validating early",
        "Assume unlimited resources instead of designing for constraints",
        "Optimize in isolation without considering cross-dependencies"
    ]
    
    print("I will NOT do these anti-patterns:")
    for i, pattern in enumerate(anti_patterns, 1):
        print(f"   {i}. {pattern}")
    
    response = input("\nDo you commit to avoiding these anti-patterns? (y/n): ").strip().lower()
    if response != 'y':
        print("❌ You must commit to avoiding anti-patterns!")
        return False
    
    print("✅ Anti-pattern avoidance committed!")
    return True

def save_session_info(goal, end_goal, requirements, workflow, dependencies, validation, failure_modes, small_test):
    """Save session information."""
    session_info = f"""
# Session Information

## Session Goal
{goal}

## End Goal
{end_goal}

## Requirements
{chr(10).join(f"- {req}" for req in requirements)}

## Complete Workflow
{workflow}

## Dependencies
{chr(10).join(f"- {dep}" for dep in dependencies)}

## Validation Strategy
{validation}

## Failure Modes
{chr(10).join(f"- {fm}" for fm in failure_modes)}

## Small Test Plan
{small_test}

## Session Start Time
{__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open("session_info.md", "w") as f:
        f.write(session_info)
    
    print("✅ Session information saved to session_info.md")

def main():
    print_header()
    
    # Check mandatory files
    if not check_mandatory_files():
        sys.exit(1)
    
    # Complete mandatory checklist
    goal = prompt_for_goal()
    if not goal:
        sys.exit(1)
    
    end_goal = prompt_for_end_goal()
    if not end_goal:
        sys.exit(1)
    
    requirements = prompt_for_requirements()
    if not requirements:
        sys.exit(1)
    
    workflow = prompt_for_workflow()
    if not workflow:
        sys.exit(1)
    
    dependencies = prompt_for_dependencies()
    if not dependencies:
        sys.exit(1)
    
    validation = prompt_for_validation()
    if not validation:
        sys.exit(1)
    
    failure_modes = prompt_for_failure_modes()
    if not failure_modes:
        sys.exit(1)
    
    small_test = prompt_for_small_test()
    if not small_test:
        sys.exit(1)
    
    # Systems thinking check
    if not systems_thinking_check():
        sys.exit(1)
    
    # Anti-pattern check
    if not anti_pattern_check():
        sys.exit(1)
    
    # Save session information
    save_session_info(goal, end_goal, requirements, workflow, dependencies, validation, failure_modes, small_test)
    
    print("\n" + "=" * 80)
    print("🎉 SESSION STARTUP COMPLETE!")
    print("=" * 80)
    print("✅ You have completed the mandatory checklist")
    print("✅ You are ready to start work with systems thinking")
    print("✅ Session information saved to session_info.md")
    print("\nRemember:")
    print("- Test with small subsets first")
    print("- Validate the complete workflow")
    print("- Document any issues")
    print("- Only scale up after small test passes")
    print("=" * 80)

if __name__ == "__main__":
    main() 