import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

# Get the absolute path of the project root directory (one level up from app/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Settings(BaseSettings):
    DATABASE_URL: str
    REDIS_URL: str
    PUBLIC_BASE_URL: str
    LLM_PROVIDER: str
    GROQ_API_KEY: str = ""
    WHATSAPP_TOKEN: str = ""
    WHATSAPP_PHONE_NUMBER_ID: str = ""
    WHATSAPP_API_VERSION: str = "v26.0"
    WHATSAPP_MOCK: bool = True
    TWILIO_AUTH_TOKEN: str = ""
    DEEPGRAM_API_KEY: str = ""
    N8N_ORDER_WEBHOOK_URL: Optional[str] = None
    WEBHOOK_SHARED_SECRET: Optional[str] = None

    # Explicitly point to the absolute path of .env
    model_config = SettingsConfigDict(
        env_file=os.path.join(BASE_DIR, ".env"),
        extra="ignore"
    )

settings = Settings()