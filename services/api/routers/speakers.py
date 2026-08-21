"""
Speakers Router.
Manages speaker profiles, legal consent records, reference audio clips, style presets, and revocation.
"""

from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from packages.contracts.models import (
    CreateSpeakerRequest,
    SpeakerConsentDTO,
    SpeakerProfileDTO,
    SpeakerReferenceClipDTO,
    SpeakerStatus,
    SpeakerStyleDTO,
)
from packages.shared_config.database import get_db
from packages.shared_config.telemetry import logger
from ..middleware.auth import get_current_user, require_role
from ..models.schema import AuditEvent, Speaker, SpeakerConsent, SpeakerReference, SpeakerStyle

router = APIRouter(prefix="/v1/speakers", tags=["Speakers"])


@router.post("", response_model=SpeakerProfileDTO, status_code=status.HTTP_201_CREATED)
async def create_speaker(req: CreateSpeakerRequest, db: AsyncSession = Depends(get_db), user: dict = Depends(get_current_user)):
    """Create a new speaker profile with draft/active status and legal consent record."""
    slug = req.name.lower().replace(" ", "-")
    
    # Check if slug exists
    stmt = select(Speaker).where(Speaker.slug == slug)
    existing = await db.scalar(stmt)
    if existing:
        slug = f"{slug}-{int(datetime.now().timestamp())}"

    speaker = Speaker(
        name=req.name,
        kurdish_name=req.kurdish_name,
        slug=slug,
        dialect=req.dialect.value if hasattr(req.dialect, "value") else str(req.dialect),
        gender=req.gender,
        age_bracket=req.age_bracket,
        voice_description=req.voice_description,
        status=SpeakerStatus.ACTIVE.value,
        naturalness_score=4.5,
        similarity_score=4.7,
        pronunciation_score=4.8,
    )
    db.add(speaker)
    await db.flush()

    # Create associated consent record
    consent = SpeakerConsent(
        speaker_id=speaker.id,
        consent_type=req.consent_type.value if hasattr(req.consent_type, "value") else str(req.consent_type),
        commercial_use_permitted=req.commercial_use_permitted,
        derivative_model_permitted=req.derivative_model_permitted,
    )
    db.add(consent)

    # Log audit event
    audit = AuditEvent(
        action="speaker_created",
        target_type="speaker",
        target_id=speaker.id,
        payload_json={"name": speaker.name, "consent_type": consent.consent_type},
    )
    db.add(audit)
    await db.commit()

    return await get_speaker(speaker.id, db)


@router.get("", response_model=List[SpeakerProfileDTO])
async def list_speakers(
    dialect: Optional[str] = None,
    status_filter: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """List all registered Kurdish speakers."""
    stmt = (
        select(Speaker)
        .options(
            selectinload(Speaker.consents),
            selectinload(Speaker.references),
            selectinload(Speaker.styles),
        )
        .order_by(Speaker.created_at.desc())
    )
    if dialect:
        stmt = stmt.where(Speaker.dialect == dialect)
    if status_filter:
        stmt = stmt.where(Speaker.status == status_filter)

    results = await db.scalars(stmt)
    speakers = results.all()

    profiles = []
    for spk in speakers:
        consent_dto = None
        if spk.consents:
            latest_consent = spk.consents[-1]
            consent_dto = SpeakerConsentDTO(
                consent_id=latest_consent.id,
                consent_type=latest_consent.consent_type,
                contract_signed_date=latest_consent.signed_date,
                expiry_date=latest_consent.expiry_date,
                commercial_use_permitted=latest_consent.commercial_use_permitted,
                derivative_model_permitted=latest_consent.derivative_model_permitted,
                prohibited_contexts=latest_consent.prohibited_contexts or [],
                document_uri=latest_consent.document_uri,
                revoked_at=latest_consent.revoked_at,
            )

        refs = [
            SpeakerReferenceClipDTO(
                reference_id=r.id,
                speaker_id=r.speaker_id,
                style_name=r.style_name,
                audio_uri=r.audio_uri,
                exact_transcript_raw=r.raw_transcript,
                exact_transcript_normalized=r.normalized_transcript,
                duration_seconds=r.duration_seconds,
                is_canonical=r.is_canonical,
                embedding_vector_cached=bool(r.embedding_cache_uri),
            )
            for r in spk.references
        ]

        styles = [
            SpeakerStyleDTO(
                style_id=s.id,
                speaker_id=s.speaker_id,
                name=s.name,
                instruction_prompt=s.instruction_prompt,
                recommended_speed=s.recommended_speed,
                canonical_reference_id=s.canonical_reference_id,
            )
            for s in spk.styles
        ]

        profiles.append(
            SpeakerProfileDTO(
                speaker_id=spk.id,
                name=spk.name,
                kurdish_name=spk.kurdish_name,
                dialect=spk.dialect,
                gender=spk.gender,
                status=spk.status,
                age_bracket=spk.age_bracket,
                voice_description=spk.voice_description,
                consent=consent_dto,
                references=refs,
                styles=styles,
                naturalness_score=spk.naturalness_score,
                similarity_score=spk.similarity_score,
                pronunciation_score=spk.pronunciation_score,
                created_at=spk.created_at,
                updated_at=spk.updated_at,
            )
        )

    return profiles


@router.get("/{speaker_id}", response_model=SpeakerProfileDTO)
async def get_speaker(speaker_id: str, db: AsyncSession = Depends(get_db)):
    """Fetch single speaker profile by UUID or slug."""
    stmt = (
        select(Speaker)
        .options(
            selectinload(Speaker.consents),
            selectinload(Speaker.references),
            selectinload(Speaker.styles),
        )
        .where((Speaker.id == speaker_id) | (Speaker.slug == speaker_id))
    )
    spk = await db.scalar(stmt)
    if not spk:
        raise HTTPException(status_code=404, detail="Speaker not found")

    consent_dto = None
    if spk.consents:
        latest_consent = spk.consents[-1]
        consent_dto = SpeakerConsentDTO(
            consent_id=latest_consent.id,
            consent_type=latest_consent.consent_type,
            contract_signed_date=latest_consent.signed_date,
            expiry_date=latest_consent.expiry_date,
            commercial_use_permitted=latest_consent.commercial_use_permitted,
            derivative_model_permitted=latest_consent.derivative_model_permitted,
            prohibited_contexts=latest_consent.prohibited_contexts or [],
            document_uri=latest_consent.document_uri,
            revoked_at=latest_consent.revoked_at,
        )

    refs = [
        SpeakerReferenceClipDTO(
            reference_id=r.id,
            speaker_id=r.speaker_id,
            style_name=r.style_name,
            audio_uri=r.audio_uri,
            exact_transcript_raw=r.raw_transcript,
            exact_transcript_normalized=r.normalized_transcript,
            duration_seconds=r.duration_seconds,
            is_canonical=r.is_canonical,
            embedding_vector_cached=bool(r.embedding_cache_uri),
        )
        for r in spk.references
    ]

    styles = [
        SpeakerStyleDTO(
            style_id=s.id,
            speaker_id=s.speaker_id,
            name=s.name,
            instruction_prompt=s.instruction_prompt,
            recommended_speed=s.recommended_speed,
            canonical_reference_id=s.canonical_reference_id,
        )
        for s in spk.styles
    ]

    return SpeakerProfileDTO(
        speaker_id=spk.id,
        name=spk.name,
        kurdish_name=spk.kurdish_name,
        dialect=spk.dialect,
        gender=spk.gender,
        status=spk.status,
        age_bracket=spk.age_bracket,
        voice_description=spk.voice_description,
        consent=consent_dto,
        references=refs,
        styles=styles,
        naturalness_score=spk.naturalness_score,
        similarity_score=spk.similarity_score,
        pronunciation_score=spk.pronunciation_score,
        created_at=spk.created_at,
        updated_at=spk.updated_at,
    )


@router.post("/{speaker_id}/revoke")
async def revoke_speaker(speaker_id: str, reason: str = "Owner requested revocation", db: AsyncSession = Depends(get_db), user: dict = Depends(require_role("admin"))):
    """Immediately revoke speaker consent and block all synthesis using this voice adapter. Requires admin role."""
    spk = await db.get(Speaker, speaker_id)
    if not spk:
        raise HTTPException(status_code=404, detail="Speaker not found")

    spk.status = SpeakerStatus.REVOKED.value
    
    # Revoke consent records
    stmt = select(SpeakerConsent).where(SpeakerConsent.speaker_id == speaker_id)
    consents = (await db.scalars(stmt)).all()
    now = datetime.now(timezone.utc)
    for c in consents:
        c.revoked_at = now
        c.commercial_use_permitted = False
        c.derivative_model_permitted = False

    audit = AuditEvent(
        action="speaker_revoked",
        target_type="speaker",
        target_id=speaker_id,
        payload_json={"reason": reason, "timestamp": now.isoformat()},
    )
    db.add(audit)
    await db.commit()

    logger.warning(f"Speaker {spk.name} ({speaker_id}) has been REVOKED. Reason: {reason}")
    return {"message": f"Speaker {spk.name} revoked successfully.", "status": "revoked"}


@router.patch("/{speaker_id}", response_model=SpeakerProfileDTO)
async def update_speaker(
    speaker_id: str,
    name: Optional[str] = None,
    kurdish_name: Optional[str] = None,
    dialect: Optional[str] = None,
    gender: Optional[str] = None,
    age_bracket: Optional[str] = None,
    voice_description: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """Update speaker profile fields."""
    spk = await db.get(Speaker, speaker_id)
    if not spk:
        raise HTTPException(status_code=404, detail="Speaker not found")

    if name is not None:
        spk.name = name
    if kurdish_name is not None:
        spk.kurdish_name = kurdish_name
    if dialect is not None:
        spk.dialect = dialect
    if gender is not None:
        spk.gender = gender
    if age_bracket is not None:
        spk.age_bracket = age_bracket
    if voice_description is not None:
        spk.voice_description = voice_description

    spk.updated_at = datetime.now(timezone.utc)

    audit = AuditEvent(
        action="speaker_updated",
        target_type="speaker",
        target_id=speaker_id,
        payload_json={"fields_updated": [f for f in ["name", "kurdish_name", "dialect", "gender", "age_bracket", "voice_description"] if locals().get(f) is not None]},
    )
    db.add(audit)
    await db.commit()

    return await get_speaker(speaker_id, db)


@router.post("/{speaker_id}/consents", response_model=SpeakerConsentDTO, status_code=status.HTTP_201_CREATED)
async def add_consent(
    speaker_id: str,
    consent_type: str = "commercial_non_exclusive",
    commercial_use_permitted: bool = True,
    derivative_model_permitted: bool = True,
    territories: Optional[str] = None,
    prohibited_contexts: Optional[str] = None,
    attribution_required: bool = False,
    document_uri: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """Add a new consent record to a speaker profile."""
    spk = await db.get(Speaker, speaker_id)
    if not spk:
        raise HTTPException(status_code=404, detail="Speaker not found")

    prohibited_list = [c.strip() for c in prohibited_contexts.split(",")] if prohibited_contexts else []

    consent = SpeakerConsent(
        speaker_id=speaker_id,
        consent_type=consent_type,
        commercial_use_permitted=commercial_use_permitted,
        derivative_model_permitted=derivative_model_permitted,
        prohibited_contexts=prohibited_list,
        document_uri=document_uri,
    )
    db.add(consent)

    audit = AuditEvent(
        action="consent_added",
        target_type="speaker",
        target_id=speaker_id,
        payload_json={"consent_type": consent_type},
    )
    db.add(audit)
    await db.commit()
    await db.refresh(consent)

    return SpeakerConsentDTO(
        consent_id=consent.id,
        consent_type=consent.consent_type,
        contract_signed_date=consent.signed_date,
        expiry_date=consent.expiry_date,
        commercial_use_permitted=consent.commercial_use_permitted,
        derivative_model_permitted=consent.derivative_model_permitted,
        prohibited_contexts=consent.prohibited_contexts or [],
        consent_document_uri=consent.document_uri,
        revoked_at=consent.revoked_at,
    )


@router.post("/{speaker_id}/references", response_model=SpeakerReferenceClipDTO, status_code=status.HTTP_201_CREATED)
async def add_reference(
    speaker_id: str,
    style_name: str = "neutral",
    raw_transcript: str = "",
    normalized_transcript: str = "",
    is_canonical: bool = False,
    audio_file: Optional[UploadFile] = None,
    db: AsyncSession = Depends(get_db),
):
    """Upload a canonical reference clip for a speaker."""
    spk = await db.get(Speaker, speaker_id)
    if not spk:
        raise HTTPException(status_code=404, detail="Speaker not found")

    import uuid as _uuid
    ref_id = str(_uuid.uuid4())
    audio_uri = f"s3://hawa-sorani-voice-assets/references/{speaker_id}/{ref_id}.wav"
    duration = 12.0  # Would be computed from actual audio

    if audio_file:
        audio_bytes = await audio_file.read()
        # In production: upload to S3, compute duration from WAV header
        duration = len(audio_bytes) / (48000 * 2)  # rough estimate 48kHz 16-bit mono

    ref = SpeakerReference(
        id=ref_id,
        speaker_id=speaker_id,
        style_name=style_name,
        audio_uri=audio_uri,
        raw_transcript=raw_transcript,
        normalized_transcript=normalized_transcript,
        duration_seconds=duration,
        is_canonical=is_canonical,
    )
    db.add(ref)

    audit = AuditEvent(
        action="reference_added",
        target_type="speaker",
        target_id=speaker_id,
        payload_json={"style": style_name, "is_canonical": is_canonical},
    )
    db.add(audit)
    await db.commit()
    await db.refresh(ref)

    return SpeakerReferenceClipDTO(
        reference_id=ref.id,
        speaker_id=ref.speaker_id,
        style_name=ref.style_name,
        audio_uri=ref.audio_uri,
        exact_transcript_raw=ref.raw_transcript,
        exact_transcript_normalized=ref.normalized_transcript,
        duration_seconds=ref.duration_seconds,
        is_canonical=ref.is_canonical,
        embedding_vector_cached=False,
    )


@router.post("/{speaker_id}/styles", response_model=SpeakerStyleDTO, status_code=status.HTTP_201_CREATED)
async def add_style(
    speaker_id: str,
    name: str = "warm_documentary",
    instruction_prompt: str = "calm, warm, assured documentary narration; measured pace",
    recommended_speed: float = 1.0,
    canonical_reference_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """Create a style preset for a speaker."""
    spk = await db.get(Speaker, speaker_id)
    if not spk:
        raise HTTPException(status_code=404, detail="Speaker not found")

    style = SpeakerStyle(
        speaker_id=speaker_id,
        name=name,
        instruction_prompt=instruction_prompt,
        recommended_speed=recommended_speed,
        canonical_reference_id=canonical_reference_id,
    )
    db.add(style)

    audit = AuditEvent(
        action="style_created",
        target_type="speaker",
        target_id=speaker_id,
        payload_json={"style_name": name},
    )
    db.add(audit)
    await db.commit()
    await db.refresh(style)

    return SpeakerStyleDTO(
        style_id=style.id,
        speaker_id=style.speaker_id,
        name=style.name,
        instruction_prompt=style.instruction_prompt,
        recommended_speed=style.recommended_speed,
        canonical_reference_id=style.canonical_reference_id,
    )
