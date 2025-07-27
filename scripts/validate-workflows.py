#!/usr/bin/env python3
"""
Workflow validation script for GitHub Actions
Validates YAML syntax and checks for common issues
"""

import os
import sys
import yaml
import glob
from pathlib import Path


def validate_yaml_file(file_path):
    """Validate YAML syntax of a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            yaml.safe_load(f)
        return True, None
    except yaml.YAMLError as e:
        return False, str(e)
    except UnicodeDecodeError as e:
        return False, f"Unicode decode error: {str(e)}"


def check_workflow_structure(workflow_data, file_path):
    """Check basic workflow structure"""
    issues = []
    
    # Check required fields - handle YAML parsing quirk where 'on:' becomes True
    required_fields = ['name', 'jobs']
    for field in required_fields:
        if field not in workflow_data:
            issues.append(f"Missing required field: {field}")
    
    # Check for 'on' field (might be parsed as True)
    if 'on' not in workflow_data and True not in workflow_data:
        issues.append("Missing required field: on")
    
    # Check jobs structure
    if 'jobs' in workflow_data:
        for job_name, job_data in workflow_data['jobs'].items():
            if not isinstance(job_data, dict):
                issues.append(f"Job '{job_name}' must be a dictionary")
                continue
                
            # Check required job fields
            if 'runs-on' not in job_data and 'uses' not in job_data:
                issues.append(f"Job '{job_name}' missing 'runs-on' or 'uses'")
    
    return issues


def check_security_best_practices(workflow_data, file_path):
    """Check for security best practices"""
    issues = []
    
    # Convert to string for pattern matching
    content = yaml.dump(workflow_data)
    
    # Check for hardcoded secrets
    if 'password' in content.lower() or 'token' in content.lower():
        if '${{ secrets.' not in content:
            issues.append("Potential hardcoded secrets detected")
    
    # Check for pull_request_target usage
    if 'pull_request_target' in content:
        issues.append("WARNING: pull_request_target can be dangerous - ensure proper validation")
    
    # Check for checkout with token
    if 'actions/checkout' in content and 'token:' in content:
        if '${{ secrets.GITHUB_TOKEN }}' not in content:
            issues.append("Custom token used with checkout - ensure it's necessary")
    
    return issues


def validate_workflow_file(file_path):
    """Validate a single workflow file"""
    print(f"\n🔍 Validating {file_path}")
    
    # Check YAML syntax
    is_valid, error = validate_yaml_file(file_path)
    if not is_valid:
        print(f"❌ YAML syntax error: {error}")
        return False
    
    # Load workflow data
    with open(file_path, 'r', encoding='utf-8') as f:
        workflow_data = yaml.safe_load(f)
    
    # Check workflow structure
    structure_issues = check_workflow_structure(workflow_data, file_path)
    if structure_issues:
        print("❌ Structure issues:")
        for issue in structure_issues:
            print(f"   - {issue}")
        return False
    
    # Check security best practices
    security_issues = check_security_best_practices(workflow_data, file_path)
    if security_issues:
        print("⚠️  Security considerations:")
        for issue in security_issues:
            print(f"   - {issue}")
    
    print("✅ Workflow validation passed")
    return True


def main():
    """Main validation function"""
    print("🚀 GitHub Actions Workflow Validator")
    print("=" * 50)
    
    # Find all workflow files
    workflow_dir = Path(".github/workflows")
    if not workflow_dir.exists():
        print("❌ .github/workflows directory not found")
        sys.exit(1)
    
    workflow_files = list(workflow_dir.glob("*.yml")) + list(workflow_dir.glob("*.yaml"))
    
    if not workflow_files:
        print("❌ No workflow files found")
        sys.exit(1)
    
    print(f"Found {len(workflow_files)} workflow files")
    
    all_valid = True
    for workflow_file in workflow_files:
        if not validate_workflow_file(workflow_file):
            all_valid = False
    
    print("\n" + "=" * 50)
    if all_valid:
        print("✅ All workflows validated successfully!")
        sys.exit(0)
    else:
        print("❌ Some workflows have issues")
        sys.exit(1)


if __name__ == "__main__":
    main()