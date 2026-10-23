"""
Embedding generation module using Google Gemini Embeddings.
"""

import os
from typing import Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.config import Config


def get_embedding_function(
    model: Optional[str] = None,
    api_key: Optional[str] = None
) -> GoogleGenerativeAIEmbeddings:
    """
    Instantiate GoogleGenerativeAIEmbeddings with configured model and credentials.
    """
    key = api_key or Config.GOOGLE_API_KEY
    if not key:
        raise ValueError(
            "Google API key is not configured. Please set GOOGLE_API_KEY in your .env file."
        )

    # Set in os.environ for compatibility with google-genai client
    os.environ["GOOGLE_API_KEY"] = key

    embedding_model = model or Config.GOOGLE_EMBEDDING_MODEL

    return GoogleGenerativeAIEmbeddings(
        model=embedding_model,
        google_api_key=key,
    )
