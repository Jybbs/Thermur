"""
Token validation schemas for OAuth2 authentication.

This module defines Pydantic models for validating authentication tokens
from external services like Globus. Tokens are validated on instantiation
and provide type-safe access throughout the application.
"""
from pydantic import BaseModel, Field, field_validator
import time


class GlobusTokenModel(BaseModel, extra="forbid"):
    """
    Validated model for Globus OAuth2 tokens.
    
    This model ensures token integrity and provides type-safe access to token 
    components. The refresh token enables automatic renewal of expired access 
    tokens without user interaction.
    """
    access_token: str = Field(
        description = "Short-lived OAuth2 access token for API authorization."
    )
    expires_at: int = Field(
        description = "Unix timestamp indicating when the access token expires."
    )
    refresh_token: str = Field(
        description = "Long-lived token used to obtain new access tokens."
    )
    scope: str = Field(
        description = "Space-delimited OAuth2 scopes granted by this token."
    )
    
    @field_validator('expires_at')
    @classmethod
    def validate_expiry(cls, v: int) -> int:
        """
        Validate that expiry timestamp is reasonable.
        
        Args:
            v : Unix timestamp to validate
            
        Returns:
            Validated timestamp
            
        Raises:
            ValueError: If timestamp is negative or unreasonably far in future
        """
        if v < 0:
            raise ValueError("Expiry timestamp cannot be negative")
        
        # Tokens shouldn't expire more than 1 year in future
        max_future = time.time() + (365 * 24 * 60 * 60)
        if v > max_future:
            raise ValueError("Expiry timestamp too far in future")
            
        return v
    
    @property
    def is_expired(self) -> bool:
        """Check if the access token has expired."""
        return time.time() >= self.expires_at