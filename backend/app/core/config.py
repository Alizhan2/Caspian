from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables without logging secrets."""

    cdse_client_id: str = ""
    cdse_client_secret: str = ""
    cdse_base_url: str = "https://sh.dataspace.copernicus.eu"
    cdse_token_url: str = (
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    )
    database_url: str = "postgresql://guardian:guardian@db:5432/guardian"
    redis_url: str = "redis://redis:6379/0"
    model_path: str = "backend/app/ml/weights/oil_unet.pt"
    storage_path: str = "./data"
    model_threshold: float = 0.58
    min_detection_area_km2: float = 0.005
    max_detection_area_km2: float = 250.0
    minio_endpoint: str = "minio:9000"
    minio_access_key: str = "guardian"
    minio_secret_key: str = "change-me-in-production"
    minio_bucket: str = "guardian-scenes"
    minio_secure: bool = False
    ollama_enabled: bool = False
    ollama_base_url: str = "http://ollama:11434"
    ollama_model: str = "qwen2.5:7b-instruct"
    monitoring_interval_minutes: int = 60
    webhook_timeout_seconds: int = 15
    analysis_mode: str = "oil_spill"
    log_level: str = "INFO"
    max_bbox_area_degrees: float = 8.0
    max_days_back: int = 30
    low_confidence_threshold: float = 0.55
    high_confidence_threshold: float = 0.75
    medium_area_km2: float = 0.05
    high_area_km2: float = 0.2

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
