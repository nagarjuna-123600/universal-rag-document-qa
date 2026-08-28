import os
import streamlit as st
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def get_groq_key():
    # Streamlit Cloud
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
        if key:
            return key
    except Exception:
        pass

    # Local .env
    return os.getenv("GROQ_API_KEY", "")


groq_key = get_groq_key()


@dataclass(frozen=True)
class Settings:
    groq_api_key: str = groq_key
    llm_model: str = os.getenv(
        'LLM_MODEL',
        'llama-3.3-70b-versatile'
    )
    embedding_model: str = os.getenv(
        'EMBEDDING_MODEL',
        'BAAI/bge-m3'
    )
    chunk_size: int = int(
        os.getenv('CHUNK_SIZE', '900')
    )
    chunk_overlap: int = int(
        os.getenv('CHUNK_OVERLAP', '150')
    )
    top_k_per_file: int = int(
        os.getenv('TOP_K_PER_FILE', '1')
    )
    max_file_mb: int = int(
        os.getenv('MAX_FILE_MB', '50')
    )
    max_files_per_upload: int = int(
        os.getenv('MAX_FILES_PER_UPLOAD', '10')
    )
    retrieval_threshold: float = float(
        os.getenv('RETRIEVAL_THRESHOLD', '1.15')
    )


settings = Settings()
