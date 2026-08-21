"""
Database Engine & Async Session Management.
Supports PostgreSQL (asyncpg) in production and SQLite (aiosqlite) in local development & automated tests.
"""

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator
from sqlalchemy import DateTime, String
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from .settings import settings


class Base(DeclarativeBase):
    """Base declarative class for all SQLAlchemy models."""
    pass


class TimestampMixin:
    """Standard created_at & updated_at timestamps."""
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False
    )


# Determine engine settings
db_url = settings.DATABASE_URL
if "sqlite" in db_url:
    Path("./data").mkdir(parents=True, exist_ok=True)
    engine = create_async_engine(db_url, connect_args={"check_same_thread": False})
else:
    engine = create_async_engine(
        db_url,
        pool_size=20,
        max_overflow=10,
        pool_pre_ping=True
    )

async_session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI Dependency for database session.
    
    NOTE: This dependency does NOT auto-commit. Routes must call
    `await db.commit()` explicitly after successful writes.
    This prevents double-commit bugs and gives routes full control
    over transaction boundaries.
    """
    async with async_session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
