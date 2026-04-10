"""FastAPI app factory: lifespan, middleware, routers."""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import config
from app.routers import admixture, conversion, health, qp_adm
from admixture_jobs import consumer as admixture_consumer
from admixture_jobs import store as admixture_store
from qpadm import consumer as qpadm_consumer
from qpadm import store as qpadm_store
from qpadm.eigenstrat_subset import warm_page_cache


@asynccontextmanager
async def lifespan(app: FastAPI):
    if config.ADMIXTURE_ENABLED:
        admixture_store.init_db()
        admixture_consumer.start_background_task()
    if config.QPADM_ENABLED:
        qpadm_store.init_db()
        qpadm_consumer.start_background_task()
        geno = os.environ.get("QPADM_DEFAULT_GENO", "").strip()
        if geno and os.path.isfile(geno):
            asyncio.get_event_loop().run_in_executor(None, warm_page_cache, geno)
    yield
    if config.ADMIXTURE_ENABLED:
        await admixture_consumer.stop_background_task()
    if config.QPADM_ENABLED:
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
    app.include_router(conversion.router)
    app.include_router(admixture.router)
    app.include_router(qp_adm.router)
    return app
