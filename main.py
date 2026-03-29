"""
Uvicorn entrypoint: ``uvicorn main:app`` (used by Dockerfile and docker-compose).
Application logic lives under ``app/``.
"""

from pathlib import Path

# Optional: local dev loads .env.local; Docker usually uses -e / compose (no python-dotenv required).
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env.local")
except ImportError:
    pass

from app.application import create_app

app = create_app()
