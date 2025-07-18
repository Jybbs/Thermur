"""
Secure token storage using Pydantic BaseSettings.

This module provides secure storage and retrieval of OAuth2 tokens using
Pydantic's BaseSettings pattern with automatic file persistence via the
secrets_dir feature.
"""
from pathlib           import Path
from pydantic          import computed_field, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing            import Optional

SECRETS_DIR = Path.home() / ".config" / "thermur" / "secrets"


class GlobusSecrets(BaseSettings):
    """
    Secure storage for Globus OAuth2 tokens.
    
    Uses Pydantic's BaseSettings with secrets_dir for automatic persistence.
    Each token field is stored as a separate file in the secrets directory,
    with the filename matching the field name.
    """
    refresh_token: Optional[SecretStr] = Field(
        default     = None,
        description = "Long-lived token used to obtain new access tokens"
    )
    scope: Optional[str] = Field(
        default     = None,
        description = "Space-delimited OAuth2 scopes granted by this token"
    )
    secrets_path: Path = Field(
        default     = SECRETS_DIR,
        description = "Directory for storing secret files"
    )
    
    model_config = SettingsConfigDict(
        case_sensitive = False,
        env_prefix     = "THERMUR_GLOBUS_",
        secrets_dir    = SECRETS_DIR
    )
    
    @computed_field
    @property
    def is_valid(self) -> bool:
        """
        Check if all required token fields are present.
        """
        return all(
            getattr(self, field) is not None 
            for field in ['refresh_token', 'scope']
        )
    
