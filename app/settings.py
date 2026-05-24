from dataclasses import dataclass
from typing import List

from app.config import get_env, load_env


def _split_csv(value: str | None, default: List[str]) -> List[str]:
    if not value:
        return default
    return [v.strip() for v in value.split(",") if v.strip()]


@dataclass
class Settings:
    app_name: str
    app_version: str
    cors_origins: List[str]
    embedding_model: str
    groq_model: str
    groq_keys: List[str]
    groq_timeout: int
    max_context_chars: int
    chunk_size: int
    chunk_overlap: int
    default_top_k: int
    log_level: str
    data_dir: str


def get_settings() -> Settings:
    load_env()
    return Settings(
        app_name=get_env("APP_NAME", "LLM-Powered Intelligent Query Retrieval System") or "LLM-Powered Intelligent Query Retrieval System",
        app_version=get_env("APP_VERSION", "1.0.0") or "1.0.0",
        cors_origins=_split_csv(get_env("CORS_ORIGINS"), ["*"]),
        embedding_model=get_env("EMBEDDING_MODEL", "all-MiniLM-L6-v2") or "all-MiniLM-L6-v2",
        groq_model=get_env("GROQ_MODEL", "llama3-70b-8192") or "llama3-70b-8192",
        groq_keys=_split_csv(get_env("GROQ_KEYS"), ["gsk_key1", "gsk_key2", "gsk_key3"]),
        groq_timeout=int(get_env("GROQ_TIMEOUT", "30") or "30"),
        max_context_chars=int(get_env("MAX_CONTEXT_CHARS", "100000") or "100000"),
        chunk_size=int(get_env("CHUNK_SIZE", "700") or "700"),
        chunk_overlap=int(get_env("CHUNK_OVERLAP", "120") or "120"),
        default_top_k=int(get_env("DEFAULT_TOP_K", "5") or "5"),
        log_level=get_env("LOG_LEVEL", "INFO") or "INFO",
        data_dir=get_env("DATA_DIR", "data") or "data",
    )
