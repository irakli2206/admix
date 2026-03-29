"""
Uvicorn entrypoint: ``uvicorn main:app`` (used by Dockerfile and docker-compose).
Application logic lives under ``app/``.
"""

from pathlib import Path

from dotenv import load_dotenv

# Local secrets (gitignored); does not override vars already set in the shell.
load_dotenv(Path(__file__).resolve().parent / ".env.local")

from app.application import create_app

app = create_app()
