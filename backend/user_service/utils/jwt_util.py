import jwt
import os
import datetime
from typing import Dict, Optional

# JWT configuration
JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', 'change-this-in-production')
JWT_ALGORITHM = 'HS256'
JWT_EXPIRATION_HOURS = 24  # Token valid for 24 hours

def generate_token(email: str, user_id: int) -> str:
    """
    Generate a JWT token for user authentication
    """
    payload = {
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=JWT_EXPIRATION_HOURS),
        'iat': datetime.datetime.utcnow(),
        'sub': email,
        'user_id': user_id
    }
    
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

def verify_token(token: str) -> Optional[Dict]:
    """
    Verify and decode JWT token
    Returns None if token is invalid
    """
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        print(f"DEBUG JWT: Token verified successfully, payload: {payload}")
        return payload
    except jwt.PyJWTError as e:
        print(f"DEBUG JWT: Token verification failed: {str(e)}")
        return None
