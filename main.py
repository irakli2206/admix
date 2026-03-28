"""
Uvicorn entrypoint: ``uvicorn main:app`` (used by Dockerfile and docker-compose).
Application logic lives under ``app/``.
"""

from app.application import create_app

app = create_app()
