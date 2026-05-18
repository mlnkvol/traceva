from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    APP_NAME: str = "Traceva"
    API_PREFIX: str = "/api"

    DEVICE: str = Field(default="cpu", alias="DEVICE")

    UPLOAD_DIR: Path = Field(default=Path("uploads"), alias="UPLOAD_DIR")
    RESULTS_DIR: Path = Field(default=Path("results"), alias="RESULTS_DIR")
    CHECKPOINTS_DIR: Path = Field(default=Path("checkpoints"), alias="CHECKPOINTS_DIR")

    SAM_CHECKPOINT: Path = Field(
        default=Path("checkpoints/sam2.1_hiera_large.pt"),
        alias="SAM_CHECKPOINT",
    )

    SAM_MODEL_CFG: str = Field(
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        alias="SAM_MODEL_CFG",
    )

    REDIS_URL: str = Field(
        default="redis://redis:6379/0",
        alias="REDIS_URL",
    )


settings = Settings()
