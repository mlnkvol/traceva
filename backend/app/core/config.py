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

    VLM_LAYER_NAMING_ENABLED: bool = Field(
        default=True,
        alias="VLM_LAYER_NAMING_ENABLED",
    )

    VLM_LAYER_NAMING_MODEL: str = Field(
        default="microsoft/Florence-2-base",
        alias="VLM_LAYER_NAMING_MODEL",
    )

    VLM_LAYER_NAMING_DEVICE: str = Field(
        default="cpu",
        alias="VLM_LAYER_NAMING_DEVICE",
    )

    VLM_LAYER_NAMING_PROMPT: str = Field(
        default="<CAPTION>",
        alias="VLM_LAYER_NAMING_PROMPT",
    )

    VLM_LAYER_NAMING_MAX_TOKENS: int = Field(
        default=12,
        alias="VLM_LAYER_NAMING_MAX_TOKENS",
    )

    VLM_LAYER_NAMING_CROP_SIZE: int = Field(
        default=384,
        alias="VLM_LAYER_NAMING_CROP_SIZE",
    )

    VLM_CACHE_DIR: Path = Field(
        default=Path("/app/.cache/huggingface"),
        alias="VLM_CACHE_DIR",
    )

    VLM_WARMUP_ON_START: bool = Field(
        default=True,
        alias="VLM_WARMUP_ON_START",
    )

    VLM_ON_DEMAND_LOAD_ENABLED: bool = Field(
        default=False,
        alias="VLM_ON_DEMAND_LOAD_ENABLED",
    )


settings = Settings()
