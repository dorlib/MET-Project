"""
Password validation utility for secure user authentication.
Implements comprehensive password strength requirements and validation.
"""

import re
from typing import Dict, List, Tuple
from datetime import datetime

class PasswordValidator:
    """Password validation class with configurable security requirements."""
    
    # Common weak passwords that should be rejected
    COMMON_PASSWORDS = [
        "password", "123456", "password123", "admin", "qwerty", "letmein",
        "welcome", "monkey", "1234567890", "password1", "123456789", 
        "12345678", "qwerty123", "abc123", "Password1", "password!", 
        "test123", "user123", "admin123", "root123", "guest", "demo",
        "temp123", "changeme", "default", "system", "manager", "service"
    ]
    
    # Minimum requirements
    MIN_LENGTH = 8
    MAX_LENGTH = 128
    
    @classmethod
    def validate_password(cls, password: str, email: str = None) -> Dict:
        """
        Comprehensive password validation.
        
        Args:
            password (str): The password to validate
            email (str, optional): User's email to check against password
            
        Returns:
            Dict: Validation result with success status, errors, and strength score
        """
        errors = []
        warnings = []
        strength_score = 0
        
        # Basic length check
        if len(password) < cls.MIN_LENGTH:
            errors.append(f"Password must be at least {cls.MIN_LENGTH} characters long")
        elif len(password) >= cls.MIN_LENGTH:
            strength_score += 1
            
        if len(password) > cls.MAX_LENGTH:
            errors.append(f"Password must be no more than {cls.MAX_LENGTH} characters long")
            
        # Check for uppercase letters
        if not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")
        else:
            strength_score += 1
            
        # Check for lowercase letters
        if not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")
        else:
            strength_score += 1
            
        # Check for digits
        if not re.search(r'\d', password):
            errors.append("Password must contain at least one number")
        else:
            strength_score += 1
            
        # Check for special characters
        if not re.search(r'[!@#$%^&*(),.?":{}|<>_+=\-\[\]\\;\'`~]', password):
            errors.append("Password must contain at least one special character (!@#$%^&*(),.?\":{}|<>_+-=[]\\;'`~)")
        else:
            strength_score += 1
            
        # Check for common/weak passwords
        if password.lower() in [p.lower() for p in cls.COMMON_PASSWORDS]:
            errors.append("Password is too common and easily guessable")
            
        # Check for sequential characters
        if cls._has_sequential_chars(password):
            warnings.append("Password contains sequential characters which may be less secure")
            
        # Check for repeated characters
        if cls._has_repeated_chars(password):
            warnings.append("Password contains many repeated characters which may be less secure")
            
        # Check if password contains email
        if email and cls._password_contains_email(password, email):
            errors.append("Password should not contain parts of your email address")
            
        # Additional strength checks for bonus points
        if len(password) >= 12:
            strength_score += 1  # Bonus for longer passwords
            
        if cls._has_mixed_case_and_special(password):
            strength_score += 1  # Bonus for good character variety
            
        # Calculate strength level
        strength_level = cls._calculate_strength_level(strength_score, len(errors))
        
        # Individual checks for frontend
        checks = {
            "min_length": len(password) >= cls.MIN_LENGTH,
            "has_uppercase": bool(re.search(r'[A-Z]', password)),
            "has_lowercase": bool(re.search(r'[a-z]', password)),
            "has_number": bool(re.search(r'\d', password)),
            "has_special": bool(re.search(r'[!@#$%^&*(),.?":{}|<>_+=\-\[\]\\;\'`~]', password)),
            "not_common": password.lower() not in [p.lower() for p in cls.COMMON_PASSWORDS],
            "no_sequences": not cls._has_sequential_chars(password)
        }
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "strength_score": strength_score,
            "strength_level": strength_level,
            "max_score": 7,
            "checks": checks
        }
    
    @classmethod
    def _has_sequential_chars(cls, password: str) -> bool:
        """Check for sequential characters like 'abc' or '123'."""
        sequential_count = 0
        for i in range(len(password) - 2):
            if (ord(password[i+1]) == ord(password[i]) + 1 and 
                ord(password[i+2]) == ord(password[i]) + 2):
                sequential_count += 1
                if sequential_count >= 2:  # Allow some sequential chars
                    return True
        return False
    
    @classmethod
    def _has_repeated_chars(cls, password: str) -> bool:
        """Check for too many repeated characters."""
        char_counts = {}
        for char in password:
            char_counts[char] = char_counts.get(char, 0) + 1
            
        # If any character appears more than 3 times, or more than 40% of password
        for count in char_counts.values():
            if count > 3 or count > len(password) * 0.4:
                return True
        return False
    
    @classmethod
    def _password_contains_email(cls, password: str, email: str) -> bool:
        """Check if password contains significant parts of email."""
        if not email:
            return False
            
        email_parts = email.lower().split('@')
        username = email_parts[0] if email_parts else ""
        
        # Check if password contains username or significant part of it
        if len(username) >= 4 and username.lower() in password.lower():
            return True
            
        return False
    
    @classmethod
    def _has_mixed_case_and_special(cls, password: str) -> bool:
        """Check for good variety of character types."""
        has_upper = bool(re.search(r'[A-Z]', password))
        has_lower = bool(re.search(r'[a-z]', password))
        has_digit = bool(re.search(r'\d', password))
        has_special = bool(re.search(r'[!@#$%^&*(),.?":{}|<>_+=\-\[\]\\;\'`~]', password))
        
        return sum([has_upper, has_lower, has_digit, has_special]) >= 3
    
    @classmethod
    def _calculate_strength_level(cls, score: int, error_count: int) -> str:
        """Calculate password strength level based on score and errors."""
        if error_count > 0:
            return "Invalid"
        elif score <= 2:
            return "Weak"
        elif score <= 4:
            return "Fair"
        elif score <= 5:
            return "Good"
        else:
            return "Strong"
    
    @classmethod
    def get_password_requirements(cls) -> Dict:
        """Get a dictionary of password requirements for frontend display."""
        return {
            "min_length": cls.MIN_LENGTH,
            "max_length": cls.MAX_LENGTH,
            "requirements": [
                "At least one uppercase letter (A-Z)",
                "At least one lowercase letter (a-z)", 
                "At least one number (0-9)",
                "At least one special character (!@#$%^&*(),.?\":{}|<>_+-=[]\\;'`~)",
                f"Between {cls.MIN_LENGTH} and {cls.MAX_LENGTH} characters long",
                "Should not be a common or easily guessable password",
                "Should not contain parts of your email address"
            ],
            "recommendations": [
                "Use 12+ characters for better security",
                "Mix different types of characters",
                "Avoid sequential characters (abc, 123)",
                "Avoid repeating the same character too many times",
                "Consider using a passphrase with special characters"
            ]
        }

class PasswordHistory:
    """Simple password history tracking to prevent password reuse."""
    
    @classmethod
    def can_reuse_password(cls, new_password: str, password_history: List[str], 
                          max_history: int = 5) -> bool:
        """
        Check if a password can be reused based on history.
        
        Args:
            new_password (str): The new password to check
            password_history (List[str]): List of previous password hashes
            max_history (int): Number of previous passwords to check
            
        Returns:
            bool: True if password can be used, False if it's too recent
        """
        import bcrypt
        
        # Check against recent passwords (limited by max_history)
        recent_passwords = password_history[-max_history:] if password_history else []
        
        for old_hash in recent_passwords:
            if bcrypt.checkpw(new_password.encode('utf-8'), old_hash.encode('utf-8')):
                return False
        
        return True
