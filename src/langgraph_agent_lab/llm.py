"""LLM factory helper.

Provides a simple interface to create LLM clients for use in nodes.
This project is configured to use Gemini for every LLM call.

Usage in nodes:
    from .llm import get_llm
    llm = get_llm()
    response = llm.invoke("Hello")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def load_env_file() -> None:
    """Load .env values without overriding real environment variables."""
    current = Path.cwd()
    for directory in (current, *current.parents):
        env_path = directory / ".env"
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
        return


def get_llm(model: str | None = None, temperature: float = 0.0) -> Any:
    """Create a Gemini chat model from environment configuration."""
    load_env_file()
    if os.getenv("GEMINI_API_KEY"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-google-genai") from exc
        return ChatGoogleGenerativeAI(
            model=model or os.getenv("LLM_MODEL", "gemini-2.5-flash"),
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=temperature,
        )

    raise RuntimeError(
        "No Gemini API key found. Set GEMINI_API_KEY in .env"
    )
