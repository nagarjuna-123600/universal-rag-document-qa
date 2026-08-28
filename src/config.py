import os
import streamlit as st
from dataclasses import dataclass


def get_groq_key():
    try:
        return st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        return os.getenv("GROQ_API_KEY", "")


@dataclass(frozen=True)
class Settings:
    groq_api_key: str = get_groq_key()
    llm_model: str = "openai/gpt-oss-120b"
    embedding_model: str = "BAAI/bge-m3"

    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k_per_file: int = 1

    max_file_mb: int = 50
    max_files_per_upload: int = 10

    retrieval_threshold: float = 1.15


settings = Settings()
