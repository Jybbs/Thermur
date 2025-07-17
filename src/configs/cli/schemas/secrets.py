"""
Secure token storage using Pydantic BaseSettings.

This module provides secure storage and retrieval of OAuth2 tokens using
Pydantic's BaseSettings pattern with automatic file persistence via the
secrets_dir feature.
"""
from pathlib           import Path
from pydantic          import computed_field, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from time              import time
from typing            import Optional


class GlobusSecrets(BaseSettings):
    """
    Secure storage for Globus OAuth2 tokens.
    
    Uses Pydantic's BaseSettings with secrets_dir for automatic persistence.
    Each token field is stored as a separate file in the secrets directory,
    with the filename matching the field name.
    """
    access_token: Optional[SecretStr] = Field(
        description = "Short-lived OAuth2 access token for API authorization"
    )
    expires_at: Optional[int] = Field(
        description = "Unix timestamp when the access token expires"
    )
    refresh_token: Optional[SecretStr] = Field(
        description = "Long-lived token used to obtain new access tokens"
    )
    scope: Optional[str] = Field(
        description = "Space-delimited OAuth2 scopes granted by this token"
    )
    
    model_config = SettingsConfigDict(
        case_sensitive = False,
        env_prefix     = "THERMUR_GLOBUS_",
        secrets_dir    = Path.home() / ".config" / "thermur" / "secrets"
    )
    
    @computed_field
    @property
    def is_expired(self) -> bool:
        """
        Check if the access token has expired.
        """
        return self.expires_at is None or time() >= self.expires_at
    
    @computed_field
    @property
    def is_valid(self) -> bool:
        """
        Check if all required token fields are present.
        """
        return all(
            getattr(self, field) is not None 
            for field in ['access_token', 'expires_at', 'refresh_token', 'scope']
        )
    
    def save(self):
        """
        Write tokens to the secrets directory.
        
        Creates individual files for each token field in the secrets directory.
        The built-in secrets_dir functionality will automatically load these
        on the next instantiation.
        """
        self.model_config['secrets_dir'].mkdir(
            exist_ok = True,
            parents  = True
        )
        
        for field_name, value in self.model_dump(
            exclude_none = True, 
            mode         = 'python'
        ).items():
            if hasattr(value, 'get_secret_value'):
                value = value.get_secret_value()
            
            file_path = self.model_config['secrets_dir'] / field_name
            file_path.write_text(str(value))
            file_path.chmod(0o600)
