"""
Audit Logging and System Health Routers.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared_config.database import get_db
from packages.shared_config.settings import settings
from packages.shared_config.telemetry import metrics
from ..models.schema import AuditEvent

router = APIRouter(prefix="/v1", tags=["Audit & System"])


@router.get("/audit")
async def list_audit_events(limit: int = 50, db: AsyncSession = Depends(get_db)):
    """Fetch immutable audit log of speaker actions, approvals, revocations, and deployments."""
    stmt = select(AuditEvent).order_by(AuditEvent.created_at.desc()).limit(limit)
    events = (await db.scalars(stmt)).all()

    return [
        {
            "id": e.id,
            "action": e.action,
            "target_type": e.target_type,
            "target_id": e.target_id,
            "payload": e.payload_json,
            "timestamp": e.created_at,
        }
        for e in events
    ]


@router.get("/health")
async def health_check():
    """System health & telemetry metrics."""
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "metrics": metrics.get_summary(),
    }
