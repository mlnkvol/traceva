import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import layer_namer, router
from app.core.config import settings


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    if (
        getattr(settings, "VLM_LAYER_NAMING_ENABLED", False)
        and getattr(settings, "VLM_WARMUP_ON_START", True)
    ):
        async def _warmup():
            ok = await asyncio.to_thread(layer_namer.warmup)
            logger.info("VLM warmup finished: ok=%s", ok)

        asyncio.create_task(_warmup())
    yield


app = FastAPI(
    title="Traceva API",
    description="Semantic gapless vectorization API",
    version="1.0.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(router, prefix="/api")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "app": settings.APP_NAME,
    }
