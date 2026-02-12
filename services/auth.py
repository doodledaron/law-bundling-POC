"""
Authentication service for the law document processing API.
Handles API key verification for protected endpoints.
"""
from fastapi import Header, HTTPException
from typing import Optional
import logging

from config import Config

logger = logging.getLogger(__name__)


def verify_api_key(x_api_key: Optional[str] = Header(None, alias=Config.API_KEY_HEADER)):
    """
    Verify API key for protected endpoints.
    
    Args:
        x_api_key: API key from request header
        
    Returns:
        bool: True if valid API key
        
    Raises:
        HTTPException: 401 if invalid or missing API key
    """
    if not Config.API_KEYS:
        # If no API keys configured, allow access (backward compatibility)
        logger.warning("No API keys configured - authentication disabled")
        return True
    
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "Missing API key",
                "message": f"Please provide a valid API key in the '{Config.API_KEY_HEADER}' header",
                "required_header": Config.API_KEY_HEADER
            }
        )
    
    if x_api_key not in Config.API_KEYS:
        logger.warning(f"Invalid API key attempted: {x_api_key[:8]}...")
        raise HTTPException(
            status_code=401,
            detail={
                "error": "Invalid API key",
                "message": "The provided API key is not valid",
                "required_header": Config.API_KEY_HEADER
            }
        )
    
    logger.info(f"Valid API key used: {x_api_key[:8]}...")
    return True
