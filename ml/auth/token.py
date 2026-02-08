"""HMAC-Signed Session Tokens for MaxSight Provides stateless HMAC token generation and verification for secure session management. Tokens include expiration time and are signed with HMAC-SHA256."""

import base64
import hmac
import hashlib
import json
import time
import os
from typing import Dict, Optional


# Secret key from environment (must be set in production)
SECRET = os.environ.get("MAXSIGHT_SECRET_KEY", "change_me_in_production")
TTL = int(os.environ.get("MAXSIGHT_SESSION_TIMEOUT", 3600))  # Default 1 hour.


def make_token(payload: Dict) -> str:
    """Generate HMAC-signed session token with expiration."""
    payload = dict(payload)
    payload['exp'] = int(time.time()) + TTL
    
    # Encode payload as JSON.
    payload_b = json.dumps(payload, separators=(',', ':')).encode()
    
    # Generate HMAC signature.
    sig = hmac.new(SECRET.encode(), payload_b, hashlib.sha256).digest()
    
    # Encode both as URL-safe base64.
    token_payload = base64.urlsafe_b64encode(payload_b).rstrip(b'=')
    token_sig = base64.urlsafe_b64encode(sig).rstrip(b'=')
    
    # Combine: payload.signature.
    token = token_payload.decode() + '.' + token_sig.decode()
    
    return token


def verify_token(token: str) -> Dict:
    """Verify HMAC-signed token and return payload if valid."""
    try:
        # Split token into payload and signature.
        parts = token.split('.')
        if len(parts) != 2:
            raise ValueError("Invalid token format")
        
        p_b, s_b = parts
        
        # Decode payload (add padding if needed)
        payload_b = base64.urlsafe_b64decode(p_b + '==')
        
        # Decode signature (add padding if needed)
        sig = base64.urlsafe_b64decode(s_b + '==')
        
        # Verify signature.
        expected = hmac.new(SECRET.encode(), payload_b, hashlib.sha256).digest()
        if not hmac.compare_digest(expected, sig):
            raise ValueError("Bad signature - token tampered with")
        
        # Decode payload.
        payload = json.loads(payload_b)
        
        # Check expiration.
        if payload.get('exp', 0) < time.time():
            raise ValueError("Token expired")
        
        return payload
        
    except (ValueError, json.JSONDecodeError, base64.binascii.Error) as e:
        raise ValueError(f"Invalid token: {str(e)}") from e







