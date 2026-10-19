"""FastAPI application for the LunarMind dashboard."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router
from backend.storage.db import init_db, reconcile_interrupted_jobs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("lunarmind.api")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    interrupted = reconcile_interrupted_jobs()
    if interrupted:
        logger.warning("marked interrupted experiments failed: %s", ", ".join(interrupted))
    yield


app = FastAPI(title="LunarMind", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_origin_regex=r"http://(127\.0\.0\.1|localhost):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
