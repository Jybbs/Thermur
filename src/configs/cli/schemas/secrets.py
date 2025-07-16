"""
Secure token storage using Pydantic BaseSettings.

This module provides secure storage and retrieval of OAuth2 tokens using
Pydantic's BaseSettings pattern. Tokens are stored in a dedicated secrets
directory with appropriate file permissions.
"""
from pathlib           import Path
from pydantic          import BaseSettings, Field, SecretStr, computed_field
from pydantic_settings import SettingsConfigDict
from time              import time
from typing            import Optional


class GlobusSecrets(BaseSettings):
    """
    Secure storage for Globus OAuth2 tokens.
    
    Uses Pydantic's BaseSettings to manage sensitive authentication tokens.
    Tokens are automatically loaded from and persisted to a secure directory
    with restricted permissions.
    """
    access_token: Optional[SecretStr] = Field(
        default=None,
        description="Short-lived OAuth2 access token for API authorization"
    )
    expires_at: Optional[int] = Field(
        default=None,
        description="Unix timestamp when the access token expires"
    )
    refresh_token: Optional[SecretStr] = Field(
        default=None,
        description="Long-lived token used to obtain new access tokens"
    )
    scope: Optional[str] = Field(
        default=None,
        description="Space-delimited OAuth2 scopes granted by this token"
    )
    
    model_config = SettingsConfigDict(
        secrets_dir=str(Path.home() / ".config/thermur/secrets"),
        env_prefix="THERMUR_GLOBUS_",
        case_sensitive=False
    )
    
    @computed_field
    @property
    def is_expired(self) -> bool:
        """
        Check if the access token has expired.
        """
        if self.expires_at is None:
            return True

        return time() >= self.expires_at
    
    @computed_field
    @property
    def is_valid(self) -> bool:
        """
        Check if all required token fields are present.
        """
        return all([
            self.access_token  is not None,
            self.expires_at    is not None,
            self.refresh_token is not None,
            self.scope         is not None
        ])
    def save(self) -> None:
        """
        Persist tokens to the secrets directory.
        
        Creates the secrets directory if needed and saves each token
        field as a separate file with restricted permissions.
        """
        secrets_dir = Path.home() / ".config/thermur/secrets"
        secrets_dir.mkdir(parents=True, exist_ok=True)
        
        if self.access_token:
            (secrets_dir / "access_token").write_text(
                self.access_token.get_secret_value()
            )
            (secrets_dir / "access_token").chmod(0o600)
            
        if self.expires_at is not None:
            (secrets_dir / "expires_at").write_text(str(self.expires_at))
            (secrets_dir / "expires_at").chmod(0o600)
            
        if self.refresh_token:
            (secrets_dir / "refresh_token").write_text(
                self.refresh_token.get_secret_value()
            )
            (secrets_dir / "refresh_token").chmod(0o600)
            
        if self.scope:
            (secrets_dir / "scope").write_text(self.scope)
            (secrets_dir / "scope").chmod(0o600)
    
    @classmethod
    def load(cls) -> Optional["GlobusSecrets"]:
        """
        Load tokens from the secrets directory.
        
        Attempts to load existing tokens. Returns None if the secrets
        directory doesn't exist or is missing required files.
        """
        secrets_dir = Path.home() / ".config/thermur/secrets"
        
        if not secrets_dir.exists():
            return None
            
        try:
            data = {}
            
            if (secrets_dir / "access_token").exists():
                data["access_token"] = (secrets_dir / "access_token").read_text().strip()
                
            if (secrets_dir / "expires_at").exists():
                data["expires_at"] = int((secrets_dir / "expires_at").read_text().strip())
                
            if (secrets_dir / "refresh_token").exists():
                data["refresh_token"] = (secrets_dir / "refresh_token").read_text().strip()
                
            if (secrets_dir / "scope").exists():
                data["scope"] = (secrets_dir / "scope").read_text().strip()
            
            if not data:
                return None
                
            return cls(**data)
        except (ValueError, IOError):
            return None