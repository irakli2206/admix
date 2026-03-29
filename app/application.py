"""FastAPI app factory: lifespan, middleware, routers."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import config
from app.routers import health


@asynccontextmanager
async def lifespan(app: FastAPI):
    if config.QPADM_ENABLED:
        from qpadm import consumer as qpadm_consumer
        from qpadm import store as qpadm_store

        qpadm_store.init_db()
        qpadm_consumer.start_background_task()
    yield
    if config.QPADM_ENABLED:
        from qpadm import consumer as qpadm_consumer

        await qpadm_consumer.stop_background_task()


def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    if config.CONVERSION_ENABLED:
        from app.routers import conversion

        app.include_router(conversion.router)
    if config.QPADM_ENABLED:
        from app.routers import qp_adm

        app.include_router(qp_adm.router)
    return app
