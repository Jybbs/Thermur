"""
Globus data transfer configuration schemas.

This module defines configuration for Globus authentication, endpoint access,
and dataset download management used by the GlobusManager helper.
"""
from pathlib           import Path
from pydantic          import BaseModel, computed_field, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing            import Literal, Optional

SECRETS_DIR = Path.home() / ".config" / "thermur" / "secrets"


class DownloadModel(BaseModel, extra="forbid"):
    """
    Download and management configuration for the CLI.
    
    This model contains settings for file downloads, display options, and
    caching behavior, separate from the dataset schema used for training.
    """
    globus_client_id: str = Field(
        default     = "ac349f52-8197-4a41-8d6d-5ae1c879273f",
        description = "Native app client ID for Globus OAuth2 authentication flow."
    )
    globus_dataset_path: str = Field(
        default     = "/1/published/publication_309/submitted_data",
        description = "Path to the WRF-Fire dataset within the FRDR Globus endpoint."
    )
    globus_endpoint_id: str = Field(
        default     = "f163c1b3-9c88-42f6-a7bb-5839ed6c4063",
        description = "UUID of the FRDR Globus endpoint hosting WRF-Fire simulations."
    )
    globus_scopes: str = Field(
        default     = "urn:globus:auth:scope:transfer.api.globus.org:all",
        description = "OAuth2 scopes required for Globus transfer operations."
    )
    recommended_files: list[dict[str, str]] = Field(
        default = [
            {
                "file" : "wrfout_W3F1R0",
                "desc" : "Light wind (3m/s) over short grass. Represents prescribed burns "
                         "or early-season fires in grasslands with stable atmospheric conditions."
            },
            {
                "file" : "wrfout_W5F7R4",  
                "desc" : "Moderate wind (5m/s) through brushy forest understory. Models typical "
                         "wildfire conditions with mixed vegetation and moderate atmospheric mixing."
            },
            {
                "file" : "wrfout_W8F13R6",
                "desc" : "Strong wind (8m/s) through heavy dead trees and branches. Simulates "
                         "post-logging or storm damage areas with deep atmospheric mixing."
            },
            {
                "file" : "wrfout_W12F4R8",
                "desc" : "Extreme wind (12m/s) in dense shrubland. Represents high-risk fire "
                         "weather in Mediterranean climates with strong temperature inversions."
            }
        ],
        description = "Recommended starter files with condition descriptions."
    )
    sample_data_path: Path = Field(
        default     = Path("data/samples/wrf_sample.nc"),
        description = "Local path where sample NetCDF file will be stored after extraction."
    )
    sample_data_url: str = Field(
        default     = "https://huggingface.co/datasets/Jybbs/sfire-samples/resolve/main/samples.tar.gz",
        description = "Hugging Face direct download URL for sample data tar.gz file."
    )
    sample_extract_dir: Path = Field(
        default     = Path("data"),
        description = "Directory where sample tar.gz will be extracted."
    )
    source: Literal["sample", "wrf-sfire", ""] = Field(
        default     = "",
        description = "Data source to download: 'sample' for quick start, 'wrf-sfire' for full dataset."
    )
    transfer_timeout: int = Field(
        default     = 86400,
        description = "Maximum seconds to wait for transfer completion (default: 24 hours)."
    )
    wrf_sfire_dir: Path = Field(
        default     = Path("data/wrf-sfire"),
        description = "Local directory for storing WRF-SFIRE dataset files from Globus."
    )


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
        secrets_dir    = str(SECRETS_DIR) if SECRETS_DIR.exists() else None
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