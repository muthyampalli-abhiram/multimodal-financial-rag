"""Configuration settings for the Multimodal Financial Intelligence RAG system."""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    # Google Gemini API Key
    GEMINI_API_KEY: Optional[str] = Field(
        default=None,
        description="Google Gemini API key for multimodal reasoning, chart extraction, and generation",
    )

    # Gemini Models
    GEMINI_MODEL_NAME: str = Field(
        default="gemini-1.5-pro-latest",
        description="Gemini model used for multimodal chart understanding and answer generation",
    )
    EMBEDDING_MODEL_NAME: str = Field(
        default="models/text-embedding-004",
        description="Gemini embedding model name used for indexing document elements",
    )

    # Vector Store Configuration
    VECTOR_STORE_PATH: str = Field(
        default="./data/vector_store",
        description="Path to local ChromaDB vector store directory",
    )

    # Ingestion & Chunking Configuration
    CHUNK_SIZE: int = Field(
        default=1000,
        description="Target chunk size for textual sections and narrative",
    )
    CHUNK_OVERLAP: int = Field(
        default=200,
        description="Token or character overlap between consecutive text chunks",
    )

    # API Server Configuration
    API_HOST: str = Field(
        default="0.0.0.0",
        description="Host interface to bind the FastAPI server",
    )
    API_PORT: int = Field(
        default=8000,
        description="Port number for the FastAPI server",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


# Global settings instance
settings = Settings()
