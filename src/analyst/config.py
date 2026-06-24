"""
analyst/config.py

Centralized, typed configuration for the project. Values load from a
.env file (see .env.example) so secrets and machine-specific paths
never get hardcoded into source files.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # --- SEC EDGAR ---
    # SEC requires every request to declare a User-Agent identifying who's
    # asking. This is NOT an API key or account — just a self-declared
    # "name + email" string. See: https://www.sec.gov/os/webmaster-faq#code-support
    SEC_USER_AGENT_NAME: str = "Adnan-Saifee"
    SEC_USER_AGENT_EMAIL: str = "adnan.saife2006@gmail.com"

    # --- Paths ---
    CONFIG_PATH: Path = Path(__file__).resolve() 
    PROJECT_ROOT: Path = CONFIG_PATH.parent.parent.parent
    RAW_DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"
    PROCESSED_DATA_DIR: Path = PROJECT_ROOT / "data" / "processed"
    CHROMA_DIR: Path = PROJECT_ROOT / "data" / "chroma_db"


settings = Settings()