import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.services.outbox_dispatcher import start_outbox_worker
from app.api.voice import router as voice_router
from app.api.whatsapp import router as whatsapp_router
from app.api.owner import router as owner_router
from app.api.orders import router as orders_router

logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting application services...")
    # Spawn the outbox poller loop as an independent background task
    dispatcher_task = asyncio.create_task(start_outbox_worker(interval_seconds=3.0))
    yield
    # Clean shutdown on server termination
    logger.info("🛑 Stopping outbox dispatcher worker...")
    dispatcher_task.cancel()
    try:
        await dispatcher_task
    except asyncio.CancelledError:
        pass
    logger.info("Outbox worker stopped cleanly.")


app = FastAPI(
    title="Spice Villa Voice Agent API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include existing routers
app.include_router(voice_router)
app.include_router(whatsapp_router)
app.include_router(owner_router)

# Include order status router
app.include_router(orders_router)


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "spice-villa-backend"}