"""
Authentication & Authorization Middleware for Hawa Sorani Voice Studio.
Implements JWT bearer token auth, API key validation, RBAC role enforcement, and rate limiting.
"""

import hashlib
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared_config.database import get_db
from packages.shared_config.settings import settings


# JWT Bearer scheme
bearer_scheme = HTTPBearer(auto_error=False)


# ==========================================
# Simple JWT (placeholder for python-jose / PyJWT in production)
# ==========================================

_SECRET_KEY = settings.SECRET_KEY if hasattr(settings, 'SECRET_KEY') else "hawa-studio-dev-secret-key-change-in-production"
_ALGORITHM = "HS256"
_TOKEN_EXPIRY_HOURS = 24


def create_access_token(user_id: str, role: str, org_id: Optional[str] = None) -> str:
    """Create a simple signed token (placeholder — use python-jose in production)."""
    import base64
    import json
    payload = {
        "sub": user_id,
        "role": role,
        "org": org_id,
        "exp": (datetime.now(timezone.utc) + timedelta(hours=_TOKEN_EXPIRY_HOURS)).isoformat(),
        "iat": datetime.now(timezone.utc).isoformat(),
    }
    token_data = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    signature = hashlib.sha256(f"{token_data}.{_SECRET_KEY}".encode()).hexdigest()[:32]
    return f"{token_data}.{signature}"


def decode_access_token(token: str) -> Optional[dict]:
    """Decode and verify a token."""
    import base64
    import json
    try:
        parts = token.rsplit(".", 1)
        if len(parts) != 2:
            return None
        token_data, signature = parts
        expected_sig = hashlib.sha256(f"{token_data}.{_SECRET_KEY}".encode()).hexdigest()[:32]
        if signature != expected_sig:
            return None
        payload = json.loads(base64.urlsafe_b64decode(token_data + "=="))
        # Check expiry
        if datetime.fromisoformat(payload["exp"]) < datetime.now(timezone.utc):
            return None
        return payload
    except Exception:
        return None


# ==========================================
# RBAC Roles
# ==========================================

class Role:
    ADMIN = "admin"
    ENGINEER = "engineer"
    LINGUIST_REVIEWER = "linguist_reviewer"
    LISTENER = "listener"


ROLE_HIERARCHY = {
    Role.ADMIN: {Role.ADMIN, Role.ENGINEER, Role.LINGUIST_REVIEWER, Role.LISTENER},
    Role.ENGINEER: {Role.ENGINEER, Role.LINGUIST_REVIEWER, Role.LISTENER},
    Role.LINGUIST_REVIEWER: {Role.LINGUIST_REVIEWER, Role.LISTENER},
    Role.LISTENER: {Role.LISTENER},
}


def check_role(user_role: str, required_role: str) -> bool:
    """Check if user_role satisfies required_role via hierarchy."""
    permitted = ROLE_HIERARCHY.get(user_role, set())
    return required_role in permitted


# ==========================================
# Rate Limiting (in-memory for dev; use Redis in production)
# ==========================================

_rate_buckets: dict = defaultdict(list)


def check_rate_limit(key: str, max_requests: int = 120, window_seconds: int = 60) -> bool:
    """Simple sliding-window rate limiter."""
    now = time.time()
    bucket = _rate_buckets[key]
    # Purge old entries
    _rate_buckets[key] = [t for t in bucket if now - t < window_seconds]
    if len(_rate_buckets[key]) >= max_requests:
        return False
    _rate_buckets[key].append(now)
    return True


# ==========================================
# FastAPI Dependencies
# ==========================================

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Extract and validate the current user from JWT bearer token.
    In development mode (ENVIRONMENT=development), allows unauthenticated access
    with a default admin user.
    """
    if credentials and credentials.credentials:
        payload = decode_access_token(credentials.credentials)
        if payload:
            return payload

    # Development fallback — allow unauthenticated access
    if hasattr(settings, 'ENVIRONMENT') and settings.ENVIRONMENT == "development":
        return {
            "sub": "dev-admin",
            "role": Role.ADMIN,
            "org": "dev-org",
        }

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing authentication token",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_role(required_role: str):
    """FastAPI dependency factory for role-based access control."""
    async def _check(user: dict = Depends(get_current_user)):
        if not check_role(user.get("role", ""), required_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires {required_role} role or higher",
            )
        return user
    return _check


async def validate_api_key(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> Optional[dict]:
    """Validate API key from X-API-Key header for external synthesis requests."""
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        return None

    from services.api.models.schema import ApiKey
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    stmt = select(ApiKey).where(ApiKey.key_hash == key_hash, ApiKey.is_active == True)
    key_record = await db.scalar(stmt)

    if not key_record:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    # Rate limiting per API key
    if not check_rate_limit(f"apikey:{key_record.id}", max_requests=key_record.rate_limit_rpm):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {key_record.rate_limit_rpm} requests/minute",
        )

    return {
        "api_key_id": key_record.id,
        "org_id": key_record.org_id,
        "rate_limit_rpm": key_record.rate_limit_rpm,
    }
