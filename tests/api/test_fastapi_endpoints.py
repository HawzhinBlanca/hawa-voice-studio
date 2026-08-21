"""
FastAPI Control Plane Integration Tests.
"""

import io
import pytest
from fastapi.testclient import TestClient
from packages.audio_processing import AudioPipeline
from services.api.main import app


@pytest.fixture
def client():
    # Use SQLite in-memory / local test DB
    with TestClient(app) as c:
        yield c


def test_health_check(client):
    """Test health and telemetry metrics endpoint."""
    res = client.get("/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "metrics" in data


def test_speaker_crud_and_revocation(client):
    """Test speaker profile registration, retrieval, and immediate revocation."""
    # 1. Create speaker
    create_payload = {
        "name": "Test Actor",
        "kurdish_name": "ئەکتەری تاقیکاری",
        "dialect": "slemani",
        "gender": "male",
        "age_bracket": "adult",
        "voice_description": "Test voice",
        "consent_type": "commercial_non_exclusive",
        "commercial_use_permitted": True,
        "derivative_model_permitted": True
    }
    res = client.post("/v1/speakers", json=create_payload)
    assert res.status_code == 201
    speaker_data = res.json()
    speaker_id = speaker_data["speaker_id"]
    assert speaker_data["name"] == "Test Actor"

    # 2. List speakers
    list_res = client.get("/v1/speakers")
    assert list_res.status_code == 200
    assert any(s["speaker_id"] == speaker_id for s in list_res.json())

    # 3. Revoke speaker
    revoke_res = client.post(f"/v1/speakers/{speaker_id}/revoke")
    assert revoke_res.status_code == 200
    assert revoke_res.json()["status"] == "revoked"

    # 4. Attempt speech synthesis with revoked voice (must be rejected 403)
    synth_res = client.post("/v1/audio/speech", json={
        "model": "sorani-pro-v1",
        "input": "دەقی تاقیکاری",
        "voice": speaker_id,
        "format": "wav",
        "stream": False
    })
    assert synth_res.status_code == 403


def test_dataset_and_upload_flow(client):
    """Test dataset creation, audio upload with automated normalization, and dataset freeze."""
    # 1. Create dataset
    ds_res = client.post("/v1/datasets", data={
        "name": "Test Sorani Dataset",
        "description": "Integration test corpus",
        "source": "studio",
        "license": "Commercial"
    })
    assert ds_res.status_code == 201
    ds_id = ds_res.json()["dataset_id"]

    # 2. Upload audio utterance
    # Generate 1-sec test WAV
    wav_bytes = AudioPipeline.write_wav_bytes([0.1] * 16000, 16000, 2)
    upload_res = client.post(
        f"/v1/datasets/{ds_id}/uploads",
        data={
            "raw_transcript": "لە ساڵی ٢٠٢٦دا سەردانی $150م کرد.",
            "style_label": "neutral"
        },
        files={"audio_file": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")}
    )
    assert upload_res.status_code == 200
    utt = upload_res.json()
    assert "دۆلار" in utt["normalized_transcript"]  # Normalization verified
    utt_id = utt["utterance_id"]

    # 3. Review utterance (approve)
    review_res = client.patch(f"/v1/datasets/utterances/{utt_id}", data={"decision": "approved"})
    assert review_res.status_code == 200

    # 4. Freeze dataset
    freeze_res = client.post(f"/v1/datasets/{ds_id}/freeze", json={
        "dataset_id": ds_id,
        "version_tag": "v1.0-test-frozen",
        "notes": "Automated test snapshot"
    })
    assert freeze_res.status_code == 200
    assert "checksum_sha256" in freeze_res.json()


def test_speech_synthesis_and_normalization_endpoints(client):
    """Test speech synthesis, normalization API, and AudioSeal watermark verification."""
    # 1. Normalization API
    norm_res = client.post("/v1/audio/normalize", data={"text": "لە ١٤:٣٠دا نرخی نەوت بووە $75."})
    assert norm_res.status_code == 200
    norm_data = norm_res.json()
    assert "کاتژمێر چواردە و نیو" in norm_data["normalized_text"]
    assert "حەفتا و پێنج دۆلار" in norm_data["normalized_text"]

    # 2. Speech Synthesis
    synth_res = client.post("/v1/audio/speech", json={
        "model": "sorani-pro-v1",
        "input": "بەخێربێن بۆ ستۆدیۆی دەنگی هەوا لە ساڵی ٢٠٢٦.",
        "voice": "default",
        "style": "warm_documentary",
        "speed": 1.0,
        "format": "wav",
        "stream": False
    })
    assert synth_res.status_code == 200
    assert synth_res.headers["content-type"] == "audio/wav"
    assert int(synth_res.headers.get("X-Audio-Watermark-Id", "0")) > 0
