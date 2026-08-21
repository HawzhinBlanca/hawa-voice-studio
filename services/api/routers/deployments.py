"""
Deployments Router.
Manages production and canary inference deployments, traffic weight splits, latency tracking, and rollbacks.
"""

from typing import List
from fastapi import APIRouter, Depends, Form, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.contracts.models import DeploymentDTO, DeploymentState
from packages.shared_config.database import get_db
from packages.shared_config.telemetry import logger
from ..models.schema import AuditEvent, Deployment, ModelVersion
from ..middleware.auth import get_current_user, require_role

router = APIRouter(prefix="/v1/deployments", tags=["Deployments"])


@router.get("", response_model=List[DeploymentDTO])
async def list_deployments(db: AsyncSession = Depends(get_db)):
    """List active, canary, and standby model deployments."""
    stmt = select(Deployment).order_by(Deployment.created_at.desc())
    results = await db.scalars(stmt)
    deployments = results.all()

    # Seed default production deployment if empty
    if not deployments:
        model_ver = ModelVersion(
            name="VoxCPM2-Sorani-Foundation-v1",
            architecture="VoxCPM2",
            weights_s3_uri="s3://hawa-sorani-voice-assets/checkpoints/foundation-v1.pt",
            is_foundation=True,
            is_approved=True,
        )
        db.add(model_ver)
        await db.flush()

        dep = Deployment(
            model_version_id=model_ver.id,
            state="active_production",
            traffic_percentage=100,
            p95_latency_ms=280.0,
            rtf_score=0.22,
        )
        db.add(dep)
        await db.commit()
        deployments = [dep]

    return [
        DeploymentDTO(
            deployment_id=d.id,
            model_version_id=d.model_version_id,
            model_name="VoxCPM2-Sorani-Foundation-v1",
            state=DeploymentState(d.state),
            traffic_percentage=d.traffic_percentage,
            p95_latency_ms=d.p95_latency_ms,
            rtf_score=d.rtf_score,
            active_adapters=["lamo-v1", "heja-v1", "shakar-v1"],
            deployed_at=d.created_at,
            updated_at=d.updated_at,
        )
        for d in deployments
    ]


@router.post("/{deployment_id}/canary")
async def update_canary_split(
    deployment_id: str,
    traffic_percentage: int = Form(..., ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    """Adjust traffic percentage allocated to canary deployment."""
    dep = await db.get(Deployment, deployment_id)
    if not dep:
        raise HTTPException(status_code=404, detail="Deployment not found")

    dep.state = "canary"
    dep.traffic_percentage = traffic_percentage
    audit = AuditEvent(
        action="canary_traffic_updated",
        target_type="deployment",
        target_id=deployment_id,
        payload_json={"traffic_percentage": traffic_percentage},
    )
    db.add(audit)
    await db.commit()

    return {"message": f"Canary traffic set to {traffic_percentage}%", "deployment_id": deployment_id}


@router.post("/{deployment_id}/promote")
async def promote_to_production(deployment_id: str, db: AsyncSession = Depends(get_db)):
    """Promote canary model to 100% production traffic."""
    dep = await db.get(Deployment, deployment_id)
    if not dep:
        raise HTTPException(status_code=404, detail="Deployment not found")

    dep.state = "active_production"
    dep.traffic_percentage = 100

    audit = AuditEvent(
        action="model_promoted_to_production",
        target_type="deployment",
        target_id=deployment_id,
        payload_json={"status": "100% production"},
    )
    db.add(audit)
    await db.commit()

    return {"message": "Deployment promoted to 100% production traffic", "deployment_id": deployment_id}


@router.post("/{deployment_id}/rollback")
async def rollback_deployment(deployment_id: str, reason: str = "Quality regression", db: AsyncSession = Depends(get_db)):
    """Instantly rollback deployment traffic to previous stable version."""
    dep = await db.get(Deployment, deployment_id)
    if not dep:
        raise HTTPException(status_code=404, detail="Deployment not found")

    dep.state = "rollback"
    dep.traffic_percentage = 0

    audit = AuditEvent(
        action="deployment_rollback",
        target_type="deployment",
        target_id=deployment_id,
        payload_json={"reason": reason},
    )
    db.add(audit)
    await db.commit()

    logger.warning(f"Deployment {deployment_id} rolled back. Reason: {reason}")
    return {"message": "Deployment rolled back successfully", "deployment_id": deployment_id, "status": "rollback"}


# ==========================================
# Model Deploy (§19: POST /v1/models/{model_id}/deploy)
# ==========================================

models_router = APIRouter(prefix="/v1/models", tags=["Models"])


@models_router.get("")
async def list_model_versions(db: AsyncSession = Depends(get_db)):
    """List all model versions with their current state."""
    stmt = select(ModelVersion).order_by(ModelVersion.created_at.desc())
    results = (await db.scalars(stmt)).all()
    return [
        {
            "id": m.id,
            "name": m.name,
            "version_tag": m.version_tag,
            "architecture": m.architecture,
            "state": m.state,
            "is_foundation": m.is_foundation,
            "is_approved": m.is_approved,
            "naturalness_mos": m.naturalness_mos,
            "cer_score": m.cer_score,
            "created_at": m.created_at.isoformat() if m.created_at else None,
        }
        for m in results
    ]


@models_router.patch("/{model_id}/state")
async def transition_model_state(
    model_id: str,
    new_state: str,
    reason: str = "",
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(require_role("admin")),
):
    """
    Promote or demote a model through the state machine.
    Valid states: draft → training → evaluating → rejected/approved → canary → production → deprecated → revoked.
    Requires admin role.
    """
    model = await db.get(ModelVersion, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model version not found")

    old_state = model.state
    try:
        model.transition_state(new_state)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    audit = AuditEvent(
        action="model_state_transitioned",
        actor_id=user.get("sub", "unknown"),
        target_type="model_version",
        target_id=model_id,
        payload_json={
            "from": old_state,
            "to": new_state,
            "reason": reason,
        },
    )
    db.add(audit)
    await db.commit()

    return {
        "model_id": model_id,
        "previous_state": old_state,
        "new_state": new_state,
        "message": f"Model transitioned: {old_state} → {new_state}",
    }


@models_router.post("/{model_id}/deploy")
async def deploy_model(
    model_id: str,
    target_state: str = "canary",
    traffic_percentage: int = 10,
    gpu_type: str = "1x L40S 48GB",
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(require_role("admin")),
):
    """Deploy a model version as canary or production inference service. Requires admin role."""
    model = await db.get(ModelVersion, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model version not found")

    # Enforce state machine — only valid transitions allowed
    try:
        model.transition_state(target_state)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    import uuid
    dep = Deployment(
        id=str(uuid.uuid4()),
        model_version_id=model_id,
        model_name=model.version_tag or f"sorani-{model_id[:8]}",
        state=target_state,
        traffic_percentage=traffic_percentage,
        p95_latency_ms=0.0,
        rtf_score=0.0,
        gpu_type=gpu_type,
    )
    db.add(dep)

    audit = AuditEvent(
        action="model_deployed",
        actor_id=user.get("sub", "unknown"),
        target_type="model_version",
        target_id=model_id,
        payload_json={
            "deployment_id": dep.id,
            "previous_state": model.state,
            "new_state": target_state,
            "traffic": traffic_percentage,
            "gpu": gpu_type,
        },
    )
    db.add(audit)
    await db.commit()

    return {
        "message": f"Model {model_id} deployed as {target_state}.",
        "deployment_id": dep.id,
        "model_version_id": model_id,
        "state": target_state,
        "traffic_percentage": traffic_percentage,
    }
