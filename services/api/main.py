"""
FastAPI Control Plane Application for Hawa Sorani Voice Studio.
"""

import traceback
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from packages.shared_config.database import Base, engine
from packages.shared_config.settings import settings
from packages.shared_config.telemetry import logger

from .routers.audio import router as audio_router
from .routers.audit import router as audit_router
from .routers.datasets import router as datasets_router
from .routers.deployments import router as deployments_router, models_router
from .routers.evaluations import router as evaluations_router
from .routers.speakers import router as speakers_router
from .routers.training import router as training_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup & Shutdown events."""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION} in {settings.ENVIRONMENT} mode...")
    
    # Auto-create tables for local development & SQLite
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database schema initialized successfully.")
    
    yield
    
    logger.info("Shutting down Hawa Voice Studio Control Plane...")
    await engine.dispose()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Production Control Plane for Kurdish Sorani Text-to-Speech (VoxCPM2 & CosyVoice3).",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)


# ==========================================
# Global Exception Handlers
# ==========================================

@app.exception_handler(ValidationError)
async def pydantic_validation_error_handler(request: Request, exc: ValidationError):
    """Return structured JSON for Pydantic validation failures."""
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "detail": exc.errors(),
            "message": "Request validation failed. Check the 'detail' field for specifics.",
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all handler to prevent raw stack traces from leaking to clients."""
    error_id = id(exc)
    logger.error(f"Unhandled exception [{error_id}]: {exc}\n{traceback.format_exc()}")
    
    detail = str(exc) if settings.DEBUG else "Internal server error"
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "error_id": error_id,
            "message": detail,
        },
    )


# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS if settings.ENVIRONMENT != "development" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routers
app.include_router(speakers_router)
app.include_router(datasets_router)
app.include_router(training_router)
app.include_router(evaluations_router)
app.include_router(audio_router)
app.include_router(deployments_router)
app.include_router(audit_router)
app.include_router(models_router)


@app.get("/")
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/v1/health"
    }


@app.get("/v1/health")
async def health_check():
    """Health check endpoint for load balancers and monitoring."""
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
