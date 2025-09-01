#!/usr/bin/env python3
"""
Test script for the new password validation system
"""

import sys
import os

# Add the backend path to sys.path
sys.path.append('backend/user_service')

try:
    from utils.password_validator import PasswordValidator
    
    def test_password_validation():
        """Test various passwords with the new validation system"""
        validator = PasswordValidator()
        
        test_passwords = [
            "weak",                          # Too short, no complexity
            "password123",                   # Common password
            "Password123",                   # Good but no special chars
            "Password123!",                  # Strong password
            "MyStr0ng!P@ssw0rd",            # Very strong
            "abcdefgh",                      # Sequential
            "12345678",                      # Sequential numbers
            "Aa1!",                         # Too short but complex
            "VeryLongPasswordWithoutNumbers!", # Long but missing numbers
            "Admin123!",                     # Good password
        ]
        
        print("Password Validation Test Results:")
        print("=" * 80)
        print(f"{'Password':<30} {'Valid':<8} {'Strength':<10} {'Issues'}")
        print("=" * 80)
        
        for password in test_passwords:
            result = validator.validate_password(password)
            issues = []
            
            if not result['checks']['min_length']:
                issues.append("too short")
            if not result['checks']['has_uppercase']:
                issues.append("no uppercase")
            if not result['checks']['has_lowercase']:
                issues.append("no lowercase") 
            if not result['checks']['has_number']:
                issues.append("no numbers")
            if not result['checks']['has_special']:
                issues.append("no special chars")
            if not result['checks']['not_common']:
                issues.append("common password")
            if not result['checks']['no_sequences']:
                issues.append("has sequences")
                
            issues_str = ", ".join(issues) if issues else "none"
            
            print(f"{password:<30} {'✓' if result['is_valid'] else '✗':<8} "
                  f"{result['strength']}/7{'':<5} {issues_str}")
        
        print("=" * 80)
        
        # Test requirements endpoint format
        print("\nPassword Requirements (API format):")
        requirements = validator.get_requirements()
        for key, value in requirements.items():
            print(f"  {key}: {value}")
    
    if __name__ == "__main__":
        test_password_validation()
        
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the project root directory")
except Exception as e:
    print(f"Error: {e}")
